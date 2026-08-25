package spregister

import (
	"crypto/tls"
	"log/slog"
	"net/http"
	"testing"
)

func transportOf(t *testing.T, cfg Config) *tls.Config {
	t.Helper()
	r := &Registrar{cfg: cfg, logger: slog.Default(), hc: &http.Client{}}
	r.rebuildTransport()
	tr, ok := r.hc.Transport.(*http.Transport)
	if !ok {
		t.Fatalf("transport not armed (https URL expected to install one)")
	}
	return tr.TLSClientConfig
}

// Public-trust mode: nil RootCAs (= system store) and no ServerName override
// (= Go validates the dialed hostname), exactly like an ordinary HTTPS client.
func TestSystemCAModeUsesSystemStoreAndHostname(t *testing.T) {
	tc := transportOf(t, Config{
		GatewayURL:  "https://gateway.example.com",
		GatewayID:   "openmodel-mainnet",
		TLSSystemCA: true,
	})
	if tc.RootCAs != nil {
		t.Error("system-CA mode must leave RootCAs nil (system trust store)")
	}
	if tc.ServerName != "" {
		t.Errorf("system-CA mode must not pin ServerName (got %q) — hostname comes from the URL", tc.ServerName)
	}
}

// Private-CA mode is untouched: identity stays the gateway_id, never the address.
func TestPrivateCAModeStillPinsGatewayID(t *testing.T) {
	tc := transportOf(t, Config{
		GatewayURL: "https://36.0.0.1:18020",
		GatewayID:  "openmodel-mainnet",
		// no CA available in this test: pool stays empty, but the identity
		// pinning is what this test asserts
	})
	if tc.ServerName != "openmodel-mainnet" {
		t.Errorf("private-CA mode must pin ServerName=gateway_id, got %q", tc.ServerName)
	}
	if tc.RootCAs == nil {
		t.Error("private-CA mode must use an explicit pool, not the system store")
	}
}

// Plain http gateway URL: no transport is installed by rebuildTransport.
func TestHTTPURLInstallsNoTLSTransport(t *testing.T) {
	r := &Registrar{cfg: Config{GatewayURL: "http://10.0.0.1:3000", TLSSystemCA: true},
		logger: slog.Default(), hc: &http.Client{}}
	r.rebuildTransport()
	if r.hc.Transport != nil {
		t.Error("http URL should leave the default transport untouched")
	}
}
