package tlsfront

import (
	"context"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"fmt"
	"io"
	"log/slog"
	"math/big"
	"net"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"
)

// miniPKI: CA + leaf issuance, in memory.
type miniPKI struct {
	caCert *x509.Certificate
	caKey  *ecdsa.PrivateKey
	pool   *x509.CertPool
}

func newMiniPKI(t *testing.T) *miniPKI {
	t.Helper()
	key, _ := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	tmpl := &x509.Certificate{
		SerialNumber: big.NewInt(1), Subject: pkix.Name{CommonName: "front-test-ca"},
		NotBefore: time.Now().Add(-time.Hour), NotAfter: time.Now().Add(time.Hour),
		IsCA: true, KeyUsage: x509.KeyUsageCertSign, BasicConstraintsValid: true,
	}
	der, _ := x509.CreateCertificate(rand.Reader, tmpl, tmpl, &key.PublicKey, key)
	cert, _ := x509.ParseCertificate(der)
	pool := x509.NewCertPool()
	pool.AddCert(cert)
	return &miniPKI{caCert: cert, caKey: key, pool: pool}
}

func (p *miniPKI) leaf(t *testing.T, cn string, eku x509.ExtKeyUsage) tls.Certificate {
	t.Helper()
	key, _ := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	tmpl := &x509.Certificate{
		SerialNumber: big.NewInt(time.Now().UnixNano()),
		Subject:      pkix.Name{CommonName: cn}, DNSNames: []string{cn},
		NotBefore: time.Now().Add(-time.Minute), NotAfter: time.Now().Add(time.Hour),
		KeyUsage: x509.KeyUsageDigitalSignature, ExtKeyUsage: []x509.ExtKeyUsage{eku},
	}
	der, _ := x509.CreateCertificate(rand.Reader, tmpl, p.caCert, &key.PublicKey, p.caKey)
	keyDER, _ := x509.MarshalECPrivateKey(key)
	pair, _ := tls.X509KeyPair(
		pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: der}),
		pem.EncodeToMemory(&pem.Block{Type: "EC PRIVATE KEY", Bytes: keyDER}))
	pair.Leaf, _ = x509.ParseCertificate(der)
	return pair
}

// swapMat is a Material whose cert/pool can be swapped mid-test (the hot-reload contract).
type swapMat struct {
	cert atomic.Value // *tls.Certificate
	pool atomic.Value // *x509.CertPool
}

func (m *swapMat) Certificate() *tls.Certificate {
	c, _ := m.cert.Load().(*tls.Certificate)
	return c
}
func (m *swapMat) ClientCAPool() *x509.CertPool {
	p, _ := m.pool.Load().(*x509.CertPool)
	return p
}

func testLogger() *slog.Logger {
	return slog.New(slog.NewTextHandler(io.Discard, nil))
}

func freePort(t *testing.T) int {
	t.Helper()
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer l.Close()
	return l.Addr().(*net.TCPAddr).Port
}

// The full loop: backend ← front ← mTLS client; then hot-swap the material and
// confirm the next handshake uses the new certificate with no restart.
func TestFront_MutualTLSAndHotSwap(t *testing.T) {
	pki := newMiniPKI(t)
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.WriteString(w, "backend-ok")
	}))
	defer backend.Close()

	mat := &swapMat{}
	port := freePort(t)
	f, err := New(mat, testLogger(), map[int]string{port: backend.URL})
	if err != nil {
		t.Fatal(err)
	}
	f.Start()
	t.Cleanup(func() { f.Stop(t.Context()) })

	gwCert := pki.leaf(t, "gateway", x509.ExtKeyUsageClientAuth)
	dial := func(serverName string, withCert bool) (*http.Response, error) {
		cfg := &tls.Config{RootCAs: pki.pool, ServerName: serverName}
		if withCert {
			cfg.Certificates = []tls.Certificate{gwCert}
		}
		c := &http.Client{Timeout: 5 * time.Second, Transport: &http.Transport{
			TLSClientConfig: cfg,
			DialContext: func(_ context.Context, _, _ string) (net.Conn, error) {
				return net.Dial("tcp", fmt.Sprintf("127.0.0.1:%d", port))
			},
		}}
		return c.Get(fmt.Sprintf("https://%s/", serverName))
	}

	// No material yet → handshake must fail cleanly.
	if _, err := dial("sp-a", true); err == nil {
		t.Fatal("handshake must fail before a certificate is installed")
	}

	// Install identity sp-a.
	certA := pki.leaf(t, "sp-a", x509.ExtKeyUsageServerAuth)
	mat.cert.Store(&certA)
	mat.pool.Store(pki.pool)

	resp, err := dial("sp-a", true)
	if err != nil {
		t.Fatalf("mTLS request should succeed: %v", err)
	}
	body, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if string(body) != "backend-ok" {
		t.Fatalf("proxied body = %q", body)
	}

	// Client WITHOUT a certificate → refused.
	if _, err := dial("sp-a", false); err == nil {
		t.Fatal("certless client must be refused")
	}

	// Hot swap to identity sp-b: next handshake serves the NEW cert (old name fails).
	certB := pki.leaf(t, "sp-b", x509.ExtKeyUsageServerAuth)
	mat.cert.Store(&certB)
	if _, err := dial("sp-a", true); err == nil {
		t.Fatal("stale identity must fail after the swap")
	}
	if resp, err := dial("sp-b", true); err != nil {
		t.Fatalf("new identity must serve immediately, no restart: %v", err)
	} else {
		resp.Body.Close()
	}
}
