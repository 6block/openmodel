package health

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"strings"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"

	"openmodel/go-scheduler/internal/scheduler"
)

var (
	gpuStateGauge = prometheus.NewGauge(prometheus.GaugeOpts{
		Name: "sidecar_gpu_state",
		Help: "Current GPU state (1=available, 2=yielding, 3=window_post, 4=winning_post)",
	})
	yieldEventsTotal = prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "sidecar_yield_events_total",
		Help: "Total number of yield events",
	}, []string{"reason"})
	epochsUntilDeadline = prometheus.NewGauge(prometheus.GaugeOpts{
		Name: "sidecar_seconds_until_deadline",
		Help: "Seconds until next WindowPoSt deadline",
	})
)

func init() {
	prometheus.MustRegister(gpuStateGauge, yieldEventsTotal, epochsUntilDeadline)
}

// IncYieldEvent increments the yield events counter for the given reason.
// Reasons should be short strings like "window_post", "winning_post",
// "lotus_disconnected", "fault_cutoff".
func IncYieldEvent(reason string) {
	yieldEventsTotal.WithLabelValues(reason).Inc()
}

// Server serves health check and Prometheus metrics endpoints.
type Server struct {
	httpServer *http.Server
	sched      *scheduler.Scheduler
	logger     *slog.Logger
	ctx        context.Context
	token      string                                                       // if set, /ready and /debug/* require this Bearer token
	dynToken   func() string                                                // optional extra accepted token (SP self-registration; rotates)
	regStatus  func() any                                                   // optional: SP self-registration status for GET /registration
	reverify   func(ctx context.Context, model string) (int, []byte, error) // optional: relay to the gateway
	onGwPoll   func()                                                       // optional: called when the GATEWAY polls /ready with its issued token
}

// SetGatewayPollObserver wires a callback fired when a /ready poll authenticates with
// the self-registration token — the one signal that proves the gateway still honours
// this worker's registration. Static-token calls (local ops, curl) are excluded on
// purpose: they say nothing about the gateway. Call before Start.
func (s *Server) SetGatewayPollObserver(f func()) {
	s.onGwPoll = f
}

// SetRegistrationStatus wires the source for GET /registration, so an SP operator can
// ask "am I registered, with whom, and if not why" instead of grepping JSON logs or
// opening the file that holds the token. The provider must return a value that is safe
// to serialize — no full token. Call before Start; without it the route 404s.
func (s *Server) SetRegistrationStatus(f func() any) {
	s.regStatus = f
}

// SetDynamicToken registers an additional accepted Bearer token source — the
// per-worker token issued by gateway self-registration, which rotates on every
// (re-)registration. Auth is considered ENABLED once either a static token or a
// dynamic source is configured. Call before Start.
func (s *Server) SetDynamicToken(f func() string) {
	s.dynToken = f
}

// NewServer creates a new health/metrics server. token, if non-empty, gates /ready
// and the /debug/* endpoints (the gateway sends the per-worker token on its /ready
// polls); /health and /metrics stay open for local healthchecks / internal scrapes.
func NewServer(port int, sched *scheduler.Scheduler, logger *slog.Logger, ctx context.Context, token string) *Server {
	mux := http.NewServeMux()

	s := &Server{
		httpServer: &http.Server{
			Addr:    fmt.Sprintf(":%d", port),
			Handler: mux,
		},
		sched:  sched,
		logger: logger,
		ctx:    ctx,
		token:  token,
	}

	mux.HandleFunc("/health", s.handleHealth)
	mux.HandleFunc("/ready", s.requireToken(s.handleReady))
	mux.HandleFunc("/registration", s.requireToken(s.handleRegistration))
	mux.HandleFunc("/registration/reverify", s.requireToken(s.handleReverify))
	mux.Handle("/metrics", promhttp.Handler())

	// Debug endpoints — available in all modes; token-gated when configured so they
	// cannot be triggered by anyone reaching the port (e.g. /debug/trigger-winning-post).
	mux.HandleFunc("/debug/trigger-winning-post", s.requireToken(s.handleTriggerWinningPost))
	mux.HandleFunc("/debug/sector-cache", s.requireToken(s.handleSectorCache))
	mux.HandleFunc("/debug/winning-post-status", s.requireToken(s.handleWinningPostStatus))
	if token != "" {
		logger.Info("health server auth enabled: /ready and /debug/* require Bearer token")
	} else {
		logger.Warn("health server auth DISABLED — set metrics.auth_token (SCHEDULER_AUTH_TOKEN) or firewall this port; /debug/trigger-winning-post is otherwise open")
	}
	logger.Info("debug endpoints registered: POST /debug/trigger-winning-post, GET /debug/sector-cache, GET /debug/winning-post-status")

	return s
}

// requireToken wraps a handler so it requires `Authorization: Bearer <token>` when
// auth is configured (a static token and/or a dynamic self-registration token).
// No-op (open) when neither is set (trusted-LAN mode).
func (s *Server) requireToken(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if s.token != "" || s.dynToken != nil {
			got := strings.TrimSpace(strings.TrimPrefix(r.Header.Get("Authorization"), "Bearer "))
			if got == "" || !s.tokenAccepted(got) {
				http.Error(w, "unauthorized", http.StatusUnauthorized)
				return
			}
		}
		next(w, r)
	}
}

// noteGatewayPoll reports a /ready poll made with the self-registration token. It
// re-checks the header instead of reusing the requireToken result because only the
// DYNAMIC token identifies the caller as the gateway: a static-token /ready (an
// operator's curl, a local healthcheck) must never be mistaken for the gateway
// still holding this worker on its roster.
func (s *Server) noteGatewayPoll(r *http.Request) {
	if s.onGwPoll == nil || s.dynToken == nil {
		return
	}
	dt := s.dynToken()
	if dt == "" {
		return
	}
	if strings.TrimSpace(strings.TrimPrefix(r.Header.Get("Authorization"), "Bearer ")) == dt {
		s.onGwPoll()
	}
}

func (s *Server) tokenAccepted(got string) bool {
	if s.token != "" && got == s.token {
		return true
	}
	if s.dynToken != nil {
		if dt := s.dynToken(); dt != "" && got == dt {
			return true
		}
	}
	return false
}

func (s *Server) handleSectorCache(w http.ResponseWriter, r *http.Request) {
	cache := s.sched.SectorCacheSnapshot()
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"sector_cache":     cache,
		"gpu_state":        s.sched.CurrentState().String(),
		"active_deadlines": countActive(cache),
	})
}

func countActive(cache map[uint64]int) int {
	count := 0
	for _, sectors := range cache {
		if sectors > 0 {
			count++
		}
	}
	return count
}

func (s *Server) handleTriggerWinningPost(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "POST only", http.StatusMethodNotAllowed)
		return
	}

	s.sched.TriggerWinningPost(s.ctx)

	w.WriteHeader(http.StatusOK)
	fmt.Fprintln(w, `{"triggered": true, "message": "WinningPoSt triggered immediately"}`)
}

func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	w.WriteHeader(http.StatusOK)
	fmt.Fprintln(w, "ok")
}

// handleRegistration reports this worker's gateway self-registration state. Behind
// the same token gate as /ready: it names the gateway and payout address, which is
// operator information, not public. 404 when self-registration was never wired up.
// SetReverify installs the relay used by POST /registration/reverify.
func (s *Server) SetReverify(f func(ctx context.Context, model string) (int, []byte, error)) {
	s.reverify = f
}

// handleReverify: POST /registration/reverify {"model":"..."} — asks the
// gateway to re-examine one claimed model right away. The SP talks only to
// this local port; the worker token stays inside the scheduler.
func (s *Server) handleReverify(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, `{"error":"POST only"}`, http.StatusMethodNotAllowed)
		return
	}
	if s.reverify == nil {
		http.Error(w, `{"error":"self-registration not configured on this worker"}`, http.StatusNotFound)
		return
	}
	var req struct {
		Model string `json:"model"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil || req.Model == "" {
		http.Error(w, `{"error":"body must be {"model":"<claimed model id>"}"}`, http.StatusBadRequest)
		return
	}
	code, body, err := s.reverify(r.Context(), req.Model)
	if err != nil {
		http.Error(w, `{"error":"gateway unreachable: `+err.Error()+`"}`, http.StatusBadGateway)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code) // pass the gateway verdict through (200/404/409/429…)
	_, _ = w.Write(body)
}

func (s *Server) handleRegistration(w http.ResponseWriter, r *http.Request) {
	if s.regStatus == nil {
		http.Error(w, `{"error":"self-registration not configured on this worker"}`, http.StatusNotFound)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	if err := json.NewEncoder(w).Encode(s.regStatus()); err != nil {
		s.logger.Error("failed to encode registration status", "error", err)
	}
}

func (s *Server) handleReady(w http.ResponseWriter, r *http.Request) {
	s.noteGatewayPoll(r)
	state := s.sched.CurrentState()
	gpuStateGauge.Set(float64(state))

	// B1 predictive routing: expose how long the current GPU state is expected to
	// last. WindowPoSt deadlines are DETERMINISTIC on-chain, so while AVAILABLE this
	// is "seconds until the graceful yield begins" — the gateway de-prioritizes
	// workers about to yield instead of routing long streams into a known
	// interruption. While mining it is the estimated seconds until resume (feeds an
	// honest Retry-After). The field is placed BEFORE gpu_state= on purpose: older
	// gateways parse everything after "gpu_state=" as the state string, so appending
	// it there would break them.
	untilChange := int64(-1) // unknown
	if decision := s.sched.LatestDecision(); decision != nil {
		epochsUntilDeadline.Set(float64(decision.SecondsUntilNextChange))
		untilChange = decision.SecondsUntilNextChange
	}

	w.WriteHeader(http.StatusOK)
	if untilChange >= 0 {
		fmt.Fprintf(w, "ready, seconds_until_change=%d, gpu_state=%s\n", untilChange, state.String())
		return
	}
	fmt.Fprintf(w, "ready, gpu_state=%s\n", state.String())
}

func (s *Server) handleWinningPostStatus(w http.ResponseWriter, r *http.Request) {
	// Live check: call MinerGetBaseInfo via scheduler to verify the full path works.
	status := s.sched.CheckWinningPostNow(s.ctx)
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(status)
}

// Start begins serving.
func (s *Server) Start() {
	if err := s.httpServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		s.logger.Error("health server error", "error", err)
	}
}

// Stop gracefully shuts down the server.
func (s *Server) Stop() {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	s.httpServer.Shutdown(ctx)
}
