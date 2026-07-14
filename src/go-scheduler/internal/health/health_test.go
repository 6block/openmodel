package health

import (
	"context"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"openmodel/go-scheduler/internal/lotus"
	"openmodel/go-scheduler/internal/scheduler"
)

func hLog() *slog.Logger { return slog.New(slog.NewTextHandler(io.Discard, nil)) }

type fakeLotus struct{}

func (fakeLotus) GetProvingDeadline(context.Context) (*lotus.DeadlineInfo, error) { return nil, nil }
func (fakeLotus) GetDeadlineSectors(context.Context, uint64) (*lotus.DeadlineSectors, error) {
	return nil, nil
}
func (fakeLotus) GetMinerBaseInfo(context.Context, int64, []lotus.TipsetCID) (*lotus.MinerBaseInfo, error) {
	return nil, nil
}
func (fakeLotus) SubscribeChainHead(context.Context) (<-chan *lotus.ChainHead, error) {
	return nil, nil
}
func (fakeLotus) Close() error { return nil }

func newServer() *Server {
	sched := scheduler.New(fakeLotus{}, scheduler.YieldPolicy{}, hLog())
	return NewServer(9100, sched, hLog(), context.Background(), "")
}

func newServerWithToken(token string) *Server {
	sched := scheduler.New(fakeLotus{}, scheduler.YieldPolicy{}, hLog())
	return NewServer(9100, sched, hLog(), context.Background(), token)
}

// TestTokenGate: when a token is configured, /ready and /debug/* require it while
// /health stays open (local container healthcheck).
func TestTokenGate(t *testing.T) {
	s := newServerWithToken("wtok")

	if rr := req(s, http.MethodGet, "/ready"); rr.Code != http.StatusUnauthorized {
		t.Errorf("/ready without token: got %d, want 401", rr.Code)
	}
	if rr := reqAuth(s, http.MethodGet, "/ready", "Bearer wtok"); rr.Code == http.StatusUnauthorized {
		t.Errorf("/ready with valid token must not be 401")
	}
	if rr := req(s, http.MethodPost, "/debug/trigger-winning-post"); rr.Code != http.StatusUnauthorized {
		t.Errorf("/debug/trigger-winning-post without token: got %d, want 401", rr.Code)
	}
	// /health stays open regardless of token
	if rr := req(s, http.MethodGet, "/health"); rr.Code == http.StatusUnauthorized {
		t.Error("/health must stay open (local healthcheck)")
	}
}

func reqAuth(s *Server, method, path, auth string) *httptest.ResponseRecorder {
	r := httptest.NewRequest(method, path, nil)
	r.Header.Set("Authorization", auth)
	rr := httptest.NewRecorder()
	s.httpServer.Handler.ServeHTTP(rr, r)
	return rr
}

func req(s *Server, method, path string) *httptest.ResponseRecorder {
	r := httptest.NewRequest(method, path, nil)
	rec := httptest.NewRecorder()
	s.httpServer.Handler.ServeHTTP(rec, r)
	return rec
}

func TestHealthHandler(t *testing.T) {
	rec := req(newServer(), http.MethodGet, "/health")
	if rec.Code != 200 || !strings.Contains(rec.Body.String(), "ok") {
		t.Errorf("/health = %d %q", rec.Code, rec.Body.String())
	}
}

func TestReadyHandler(t *testing.T) {
	rec := req(newServer(), http.MethodGet, "/ready")
	if rec.Code != 200 || !strings.Contains(rec.Body.String(), "gpu_state=") {
		t.Errorf("/ready = %d %q", rec.Code, rec.Body.String())
	}
}

func TestTriggerWinningPostMethodGuard(t *testing.T) {
	if rec := req(newServer(), http.MethodGet, "/debug/trigger-winning-post"); rec.Code != http.StatusMethodNotAllowed {
		t.Errorf("GET /debug/trigger-winning-post = %d, want 405", rec.Code)
	}
}

func TestSectorCacheHandler(t *testing.T) {
	rec := req(newServer(), http.MethodGet, "/debug/sector-cache")
	if rec.Code != 200 {
		t.Errorf("/debug/sector-cache = %d, want 200", rec.Code)
	}
}

// B1: /ready must expose seconds_until_change BEFORE gpu_state (old gateways parse
// everything after "gpu_state=" as the state — appending there would break them).
func TestReadyExposesSecondsUntilChange(t *testing.T) {
	s := newServer()
	rr := req(s, http.MethodGet, "/ready")
	line := strings.TrimSpace(rr.Body.String())

	if i := strings.Index(line, "seconds_until_change="); i >= 0 {
		if j := strings.Index(line, "gpu_state="); j < i {
			t.Fatalf("seconds_until_change must PRECEDE gpu_state for backward compat: %q", line)
		}
	}
	// The legacy parser contract: everything after gpu_state= is a bare state string.
	parts := strings.SplitN(line, "gpu_state=", 2)
	if len(parts) != 2 || strings.ContainsAny(strings.TrimSpace(parts[1]), ", ") {
		t.Fatalf("gpu_state must be the LAST field, bare: %q", line)
	}
}
