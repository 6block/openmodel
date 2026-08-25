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
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// Spin up a TLS server whose certificate is signed by a throwaway CA and
// carries SAN = gatewayID — the shape a real registration-CA gateway presents.
// Returns the server and the CA certificate PEM.
func newPrivateCAServer(t *testing.T, gatewayID string) (*httptest.Server, []byte) {
	t.Helper()
	caKey, _ := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	caTpl := &x509.Certificate{
		SerialNumber:          big.NewInt(1),
		Subject:               pkix.Name{CommonName: "test-reg-ca"},
		NotBefore:             time.Now().Add(-time.Hour),
		NotAfter:              time.Now().Add(time.Hour),
		IsCA:                  true,
		KeyUsage:              x509.KeyUsageCertSign,
		BasicConstraintsValid: true,
	}
	caDER, _ := x509.CreateCertificate(rand.Reader, caTpl, caTpl, &caKey.PublicKey, caKey)
	caCert, _ := x509.ParseCertificate(caDER)

	srvKey, _ := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	srvTpl := &x509.Certificate{
		SerialNumber: big.NewInt(2),
		Subject:      pkix.Name{CommonName: gatewayID},
		DNSNames:     []string{gatewayID}, // identity = gateway_id, not an address
		NotBefore:    time.Now().Add(-time.Hour),
		NotAfter:     time.Now().Add(time.Hour),
		KeyUsage:     x509.KeyUsageDigitalSignature,
		ExtKeyUsage:  []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
	}
	srvDER, _ := x509.CreateCertificate(rand.Reader, srvTpl, caCert, &srvKey.PublicKey, caKey)

	srv := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	srv.TLS = &tls.Config{Certificates: []tls.Certificate{{
		Certificate: [][]byte{srvDER},
		PrivateKey:  srvKey,
	}}}
	srv.StartTLS()
	t.Cleanup(srv.Close)
	return srv, pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: caDER})
}

func registrarFor(t *testing.T, cfg Config) *Registrar {
	t.Helper()
	r := &Registrar{cfg: cfg, logger: slog.Default(), hc: &http.Client{Timeout: 5 * time.Second}}
	r.rebuildTransport()
	return r
}

// Private-CA mode completes a real handshake against a gateway_id certificate.
func TestHandshake_PrivateCAModeSucceeds(t *testing.T) {
	srv, caPEM := newPrivateCAServer(t, "testgw-id")
	caPath := filepath.Join(t.TempDir(), "ca.crt")
	if err := os.WriteFile(caPath, caPEM, 0o600); err != nil {
		t.Fatal(err)
	}
	r := registrarFor(t, Config{
		GatewayURL: strings.Replace(srv.URL, "https://", "https://", 1),
		GatewayID:  "testgw-id",
		TLSCAPath:  caPath,
	})
	resp, err := r.hc.Get(srv.URL)
	if err != nil {
		t.Fatalf("private-CA handshake failed against a gateway_id cert: %v", err)
	}
	resp.Body.Close()
}

// System mode against the SAME private-CA server must fail the handshake:
// that certificate chains to no public root — proof the system store is what
// is actually consulted, not the private pool.
func TestHandshake_SystemModeRejectsPrivateCA(t *testing.T) {
	srv, _ := newPrivateCAServer(t, "testgw-id")
	r := registrarFor(t, Config{
		GatewayURL:  srv.URL,
		GatewayID:   "testgw-id",
		TLSSystemCA: true,
	})
	if _, err := r.hc.Get(srv.URL); err == nil {
		t.Fatal("system mode accepted a private-CA certificate — it is not using the system trust store")
	}
}

// Private-CA mode enforces identity: a server whose certificate says a
// DIFFERENT gateway_id is rejected even though the chain is trusted.
func TestHandshake_PrivateCAModeRejectsWrongGatewayID(t *testing.T) {
	srv, caPEM := newPrivateCAServer(t, "some-other-gateway")
	caPath := filepath.Join(t.TempDir(), "ca.crt")
	if err := os.WriteFile(caPath, caPEM, 0o600); err != nil {
		t.Fatal(err)
	}
	r := registrarFor(t, Config{
		GatewayURL: srv.URL,
		GatewayID:  "testgw-id", // expects this identity
		TLSCAPath:  caPath,
	})
	if _, err := r.hc.Get(srv.URL); err == nil {
		t.Fatal("private-CA mode accepted a certificate for the wrong gateway_id")
	}
}
