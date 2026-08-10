package spregister

import (
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"log/slog"
	"math/big"
	"net/http"
	"os"
	"path/filepath"
	"testing"
	"time"
)

// Cold start: a worker that has never registered holds no cert-store CA, so an
// https gateway URL can only be trusted through gateway.tls_ca_path. This pins
// that the configured CA file actually lands in the transport's root pool with
// the gateway_id as the expected TLS identity — the bootstrap path the README
// documents for first registration over an untrusted network.
func TestRebuildTransport_ColdStartUsesTLSCAPath(t *testing.T) {
	key, _ := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	tmpl := &x509.Certificate{
		SerialNumber: big.NewInt(1), Subject: pkix.Name{CommonName: "test-ca"},
		NotBefore: time.Now().Add(-time.Hour), NotAfter: time.Now().Add(time.Hour),
		IsCA: true, KeyUsage: x509.KeyUsageCertSign, BasicConstraintsValid: true,
	}
	der, _ := x509.CreateCertificate(rand.Reader, tmpl, tmpl, &key.PublicKey, key)
	caPath := filepath.Join(t.TempDir(), "gateway-ca.crt")
	os.WriteFile(caPath, pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: der}), 0644)

	r := &Registrar{cfg: Config{
		GatewayURL: "https://203.0.113.7:18443", // arbitrary public mapping
		GatewayID:  "openmodel-mainnet",
		TLSCAPath:  caPath,
	}, hc: &http.Client{}, logger: slog.Default()}
	r.rebuildTransport()

	tr, ok := r.hc.Transport.(*http.Transport)
	if !ok || tr.TLSClientConfig == nil {
		t.Fatal("https gateway URL must arm a TLS transport")
	}
	if tr.TLSClientConfig.ServerName != "openmodel-mainnet" {
		t.Fatalf("expected identity must be the gateway_id, got %q", tr.TLSClientConfig.ServerName)
	}
	if tr.TLSClientConfig.RootCAs == nil {
		t.Fatal("tls_ca_path CA must land in the root pool (cold start has no cert store)")
	}
	if tr.TLSClientConfig.MinVersion != tls.VersionTLS12 {
		t.Fatal("minimum TLS version must be pinned")
	}
}

// An http:// gateway URL must leave the default transport untouched — TLS is
// armed only when the operator opted into https.
func TestRebuildTransport_PlainHTTPUntouched(t *testing.T) {
	r := &Registrar{cfg: Config{GatewayURL: "http://127.0.0.1:3200"},
		hc: &http.Client{}, logger: slog.Default()}
	r.rebuildTransport()
	if r.hc.Transport != nil {
		t.Fatal("plain http must not install a custom transport")
	}
}
