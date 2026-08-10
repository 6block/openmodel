package spregister

import (
	"encoding/json"
	"strings"
	"testing"
)

func statusTestRegistrar(t *testing.T, tokenPath string) *Registrar {
	t.Helper()
	return New(Config{
		GatewayURL: "http://gw.example:3000", GatewayID: "gw-1",
		WorkerID: "sp-test", MinerID: "t01234",
		PayoutAddress: "0xAbC0000000000000000000000000000000000001",
		TokenPath:     tokenPath,
		AutoRegister:  true, // the normal case: self-registration is on
	}, nil, testLogger())
}

// The status is meant to be pasteable into a support request, so it must carry a
// token PREFIX and never the usable token.
func TestStatus_NeverLeaksToken(t *testing.T) {
	r := statusTestRegistrar(t, "")
	const secret = "wk-78415e1e6d90293f331df9c2e49ca9f58892f560859c296e"
	r.token.Store(secret)

	raw, err := json.Marshal(r.CurrentStatus())
	if err != nil {
		t.Fatal(err)
	}
	body := string(raw)
	if strings.Contains(body, secret) {
		t.Fatalf("full token leaked into status: %s", body)
	}
	// The prefix must be short enough to be useless and long enough to correlate
	// with the gateway's view of the same worker.
	if !strings.Contains(body, `"token_prefix":"wk-78415e1…"`) {
		t.Fatalf("expected a truncated token prefix, got: %s", body)
	}
}

func TestStatus_UnregisteredVsRegistered(t *testing.T) {
	r := statusTestRegistrar(t, "")

	s := r.CurrentStatus()
	if s.Registered {
		t.Fatal("a registrar holding no token must not report registered")
	}
	if !s.Enabled || s.GatewayURL != "http://gw.example:3000" || s.GatewayID != "gw-1" {
		t.Fatalf("configured fields should be reported even before success: %+v", s)
	}
	if s.PayoutAddress == "" || s.WorkerID != "sp-test" || s.MinerID != "t01234" {
		t.Fatalf("config fallbacks missing: %+v", s)
	}

	r.token.Store("wk-abcdefghij0123456789")
	if s := r.CurrentStatus(); !s.Registered || s.TokenPrefix == "" {
		t.Fatalf("holding a token should read as registered: %+v", s)
	}
}

// A failed attempt must surface WHY (that is the whole point for an SP debugging
// a rejection), and a later success must clear it.
func TestStatus_TracksErrorAndAttempts(t *testing.T) {
	r := statusTestRegistrar(t, "")

	r.setStatus(func(s *Status) { s.Attempts++ })
	r.setStatus(func(s *Status) { s.LastError = "gateway returned 403: below the admission floor" })
	s := r.CurrentStatus()
	if s.Attempts != 1 || !strings.Contains(s.LastError, "admission floor") {
		t.Fatalf("failure not recorded: %+v", s)
	}

	// simulate a success clearing the error
	r.setStatus(func(s *Status) {
		s.Registered = true
		s.RegisteredAt = "2026-07-27T04:00:00Z"
		s.LastError = ""
	})
	if s := r.CurrentStatus(); s.LastError != "" || s.RegisteredAt == "" {
		t.Fatalf("success should clear the error and stamp the time: %+v", s)
	}
}

// Disabled (no gateway configured) must be distinguishable from "configured but
// not yet registered" — otherwise an operator cannot tell a typo from a rejection.
func TestStatus_DisabledWhenNoGateway(t *testing.T) {
	r := New(Config{WorkerID: "sp-x"}, nil, testLogger())
	if s := r.CurrentStatus(); s.Enabled {
		t.Fatalf("no gateway URL should read as disabled: %+v", s)
	}
}

// auto_register:false with a gateway configured must report enabled:false while
// still naming the gateway — "configured but not auto-registering". Reporting it as
// unconfigured (the endpoint's 404) misleads an operator who did configure it.
func TestStatus_ConfiguredButAutoRegisterOff(t *testing.T) {
	r := New(Config{
		GatewayURL: "http://gw.example:3000", GatewayID: "gw-1",
		WorkerID: "sp-test", MinerID: "t01234", AutoRegister: false,
	}, nil, testLogger())

	s := r.CurrentStatus()
	if s.Enabled {
		t.Fatal("auto_register:false must read as enabled:false")
	}
	if s.GatewayURL != "http://gw.example:3000" || s.GatewayID != "gw-1" {
		t.Fatalf("the configured gateway must still be reported: %+v", s)
	}

	// A token held from an EARLIER registration must still show as registered —
	// this is the state that previously vanished (no Registrar → /ready stopped
	// accepting the token → the gateway polled the worker offline).
	r.token.Store("wk-previouslyissued123456")
	if s := r.CurrentStatus(); !s.Registered || s.Enabled {
		t.Fatalf("held token should read registered while auto-register stays off: %+v", s)
	}
}

func TestStatus_EnabledWhenAutoRegisterOn(t *testing.T) {
	r := New(Config{GatewayURL: "http://gw:3000", AutoRegister: true}, nil, testLogger())
	if s := r.CurrentStatus(); !s.Enabled {
		t.Fatalf("auto_register:true must read as enabled: %+v", s)
	}
}
