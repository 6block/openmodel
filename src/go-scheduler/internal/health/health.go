package health

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
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
}

// NewServer creates a new health/metrics server.
func NewServer(port int, sched *scheduler.Scheduler, logger *slog.Logger, ctx context.Context) *Server {
	mux := http.NewServeMux()

	s := &Server{
		httpServer: &http.Server{
			Addr:    fmt.Sprintf(":%d", port),
			Handler: mux,
		},
		sched:  sched,
		logger: logger,
		ctx:    ctx,
	}

	mux.HandleFunc("/health", s.handleHealth)
	mux.HandleFunc("/ready", s.handleReady)
	mux.Handle("/metrics", promhttp.Handler())

	// Debug endpoints — available in all modes
	mux.HandleFunc("/debug/trigger-winning-post", s.handleTriggerWinningPost)
	mux.HandleFunc("/debug/sector-cache", s.handleSectorCache)
	mux.HandleFunc("/debug/winning-post-status", s.handleWinningPostStatus)
	logger.Info("debug endpoints registered: POST /debug/trigger-winning-post, GET /debug/sector-cache, GET /debug/winning-post-status")

	return s
}


func (s *Server) handleSectorCache(w http.ResponseWriter, r *http.Request) {
	cache := s.sched.SectorCacheSnapshot()
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"sector_cache":    cache,
		"gpu_state":       s.sched.CurrentState().String(),
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

func (s *Server) handleReady(w http.ResponseWriter, r *http.Request) {
	state := s.sched.CurrentState()
	gpuStateGauge.Set(float64(state))

	if decision := s.sched.LatestDecision(); decision != nil {
		epochsUntilDeadline.Set(float64(decision.SecondsUntilNextChange))
	}

	w.WriteHeader(http.StatusOK)
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
