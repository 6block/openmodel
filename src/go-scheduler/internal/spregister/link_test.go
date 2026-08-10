package spregister

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

func linkTestRegistrar(t *testing.T, gatewayURL string) *Registrar {
	t.Helper()
	r := New(Config{
		GatewayURL: gatewayURL, GatewayID: "gw-1",
		WorkerID: "sp-test", MinerID: "t01234", AutoRegister: true,
	}, nil, testLogger())
	r.token.Store("wk-abcdefghij0123456789")
	return r
}

// Holding no token means there is no link to judge — reporting it as a failure
// would alarm an operator who simply has not registered yet.
func TestLink_IdleWithoutToken(t *testing.T) {
	r := New(Config{GatewayURL: "http://gw:3000", AutoRegister: true}, nil, testLogger())
	if l := r.linkStatus(); l.Verdict != LinkIdle {
		t.Fatalf("no token should read idle, got %+v", l)
	}
}

// Right after start-up nothing has polled yet. That must NOT read as serving (we
// have no evidence) nor as broken (there has been no time to fail).
func TestLink_WaitingBeforeFirstPoll(t *testing.T) {
	r := linkTestRegistrar(t, "http://gw:3000")
	l := r.linkStatus()
	if l.Verdict != LinkWaiting {
		t.Fatalf("expected waiting before the first poll, got %+v", l)
	}
	if l.LastPolledAt != "" {
		t.Fatalf("no poll happened; last_polled_at must stay empty: %+v", l)
	}
}

func TestLink_ServingAfterPoll(t *testing.T) {
	r := linkTestRegistrar(t, "http://gw:3000")
	r.NoteGatewayPoll()

	l := r.linkStatus()
	if l.Verdict != LinkServing {
		t.Fatalf("a fresh poll must read serving, got %+v", l)
	}
	if l.LastPolledAt == "" || l.SilentForSec > 1 {
		t.Fatalf("poll timestamp not recorded: %+v", l)
	}
}

// The whole point of the split: a rejected re-registration (403) while the gateway
// keeps polling means "cannot rotate the token, but still earning". Reporting that
// as unregistered — the intuitive fix — would send the operator chasing a
// non-problem, e.g. after the gateway raises its admission floor without evicting
// the SPs already on its roster.
func TestLink_RejectedRegistrationStillServing(t *testing.T) {
	r := linkTestRegistrar(t, "http://gw:3000")
	r.NoteGatewayPoll()
	r.setStatus(func(s *Status) {
		s.LastError = "register: gateway returned 403: miner raw byte power below the admission floor"
	})

	s := r.CurrentStatus()
	if !s.Registered {
		t.Fatal("a held token must still read registered while re-registration is rejected")
	}
	if s.GatewayLink.Verdict != LinkServing {
		t.Fatalf("gateway is still polling; link must read serving: %+v", s.GatewayLink)
	}
	if s.LastError == "" {
		t.Fatal("the rejection must remain visible in last_error")
	}
}

// Silence + an unreachable gateway is a network problem: it resolves itself, so the
// verdict must not accuse the gateway of dropping the worker.
func TestLink_UnreachableGatewayReadsAsNetwork(t *testing.T) {
	// Port 1 on loopback: nothing listens, connection fails fast.
	r := linkTestRegistrar(t, "http://127.0.0.1:1")
	r.NoteGatewayPoll()
	r.lastPoll.Store(time.Now().Add(-10 * time.Minute).UnixNano())

	r.evaluateLink(context.Background())
	if l := r.linkStatus(); l.Verdict != LinkUnreachable {
		t.Fatalf("expected unreachable, got %+v", l)
	}
}

// Silence while the gateway answers is the case that actually needs a human: the
// registration is likely gone and the local token is stale.
func TestLink_ReachableButSilentReadsAsDropped(t *testing.T) {
	// 401 must still count as reachable — the probe is unauthenticated by design.
	gw := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "unauthorized", http.StatusUnauthorized)
	}))
	defer gw.Close()

	r := linkTestRegistrar(t, gw.URL)
	r.lastPoll.Store(time.Now().Add(-10 * time.Minute).UnixNano())

	r.evaluateLink(context.Background())
	l := r.linkStatus()
	if l.Verdict != LinkSilent {
		t.Fatalf("expected silent, got %+v", l)
	}
	if l.SilentForSec < 500 {
		t.Fatalf("silence duration not reported: %+v", l)
	}
}

// A poll arriving after an outage must clear the verdict immediately, without
// waiting for the 30s watcher tick — the operator refreshing the endpoint expects
// to see recovery the moment traffic resumes.
func TestLink_RecoversOnNextPoll(t *testing.T) {
	r := linkTestRegistrar(t, "http://127.0.0.1:1")
	r.lastPoll.Store(time.Now().Add(-10 * time.Minute).UnixNano())
	r.evaluateLink(context.Background())
	if r.linkStatus().Verdict != LinkUnreachable {
		t.Fatal("setup: expected a degraded link")
	}

	r.NoteGatewayPoll()
	if l := r.linkStatus(); l.Verdict != LinkServing || l.Note != "" {
		t.Fatalf("a new poll must restore serving and drop the note: %+v", l)
	}
}

// The watcher must not probe (or churn the verdict) while polls arrive normally.
func TestLink_WatcherQuietWhileHealthy(t *testing.T) {
	probes := 0
	gw := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		probes++
		w.WriteHeader(http.StatusOK)
	}))
	defer gw.Close()

	r := linkTestRegistrar(t, gw.URL)
	r.NoteGatewayPoll()
	r.evaluateLink(context.Background())

	if probes != 0 {
		t.Fatalf("a healthy link must generate no probe traffic, got %d", probes)
	}
	if r.linkStatus().Verdict != LinkServing {
		t.Fatal("expected serving")
	}
}
