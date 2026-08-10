package spregister

import (
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"math/big"
	"os"
	"path/filepath"
	"testing"
	"time"
)

// signCSR plays the gateway: parses a CSR and signs it with a throwaway CA.
type testCA struct {
	cert *x509.Certificate
	key  *ecdsa.PrivateKey
	pem  string
}

func newTestCA(t *testing.T) *testCA {
	t.Helper()
	key, _ := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	tmpl := &x509.Certificate{
		SerialNumber: big.NewInt(1), Subject: pkix.Name{CommonName: "cs-test-ca"},
		NotBefore: time.Now().Add(-time.Hour), NotAfter: time.Now().Add(24 * time.Hour),
		IsCA: true, KeyUsage: x509.KeyUsageCertSign, BasicConstraintsValid: true,
	}
	der, _ := x509.CreateCertificate(rand.Reader, tmpl, tmpl, &key.PublicKey, key)
	cert, _ := x509.ParseCertificate(der)
	return &testCA{cert: cert, key: key,
		pem: string(pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: der}))}
}

func (ca *testCA) sign(t *testing.T, csrPEM, cn string, validity time.Duration) string {
	t.Helper()
	block, _ := pem.Decode([]byte(csrPEM))
	csr, err := x509.ParseCertificateRequest(block.Bytes)
	if err != nil {
		t.Fatal(err)
	}
	tmpl := &x509.Certificate{
		SerialNumber: big.NewInt(time.Now().UnixNano()),
		Subject:      pkix.Name{CommonName: cn}, DNSNames: []string{cn},
		NotBefore: time.Now().Add(-time.Minute), NotAfter: time.Now().Add(validity),
		KeyUsage: x509.KeyUsageDigitalSignature, ExtKeyUsage: []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
	}
	der, err := x509.CreateCertificate(rand.Reader, tmpl, ca.cert, csr.PublicKey, ca.key)
	if err != nil {
		t.Fatal(err)
	}
	return string(pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: der}))
}

func TestCertStore_KeyPersistsAcrossRestarts(t *testing.T) {
	dir := t.TempDir()
	cs1, err := NewCertStore(dir, "sp-x")
	if err != nil {
		t.Fatal(err)
	}
	csr1, _ := cs1.CSR()

	cs2, err := NewCertStore(dir, "sp-x")
	if err != nil {
		t.Fatal(err)
	}
	csr2, _ := cs2.CSR()

	// Same key on disk → both CSRs carry the same public key.
	p1, _ := pem.Decode([]byte(csr1))
	p2, _ := pem.Decode([]byte(csr2))
	r1, _ := x509.ParseCertificateRequest(p1.Bytes)
	r2, _ := x509.ParseCertificateRequest(p2.Bytes)
	if !r1.PublicKey.(*ecdsa.PublicKey).Equal(r2.PublicKey) {
		t.Fatal("worker key must persist across restarts (identity churn otherwise)")
	}
	if fi, _ := os.Stat(filepath.Join(dir, "worker.key")); fi.Mode().Perm() != 0o600 {
		t.Fatalf("worker.key must be 0600, got %v", fi.Mode().Perm())
	}
}

func TestCertStore_InstallAndWarmStart(t *testing.T) {
	dir := t.TempDir()
	ca := newTestCA(t)
	cs, _ := NewCertStore(dir, "sp-x")
	if cs.Certificate() != nil {
		t.Fatal("no cert should be active before install")
	}
	if !cs.RenewDue() {
		t.Fatal("missing cert must read as renew-due")
	}

	csr, _ := cs.CSR()
	if err := cs.Install(ca.sign(t, csr, "sp-x", 7*24*time.Hour), ca.pem); err != nil {
		t.Fatal(err)
	}
	if cs.Certificate() == nil || cs.ClientCAPool() == nil {
		t.Fatal("install must activate cert and CA pool")
	}
	if cs.RenewDue() {
		t.Fatal("a fresh 7-day cert must not be renew-due")
	}

	// Warm start: a new store on the same dir serves the persisted cert at once.
	cs2, _ := NewCertStore(dir, "sp-x")
	if cs2.Certificate() == nil {
		t.Fatal("restart must warm-start from the persisted certificate")
	}
}

// A certificate for someone else's key (or identity) must be refused — this is
// what stops a confused gateway response from bricking the TLS front.
func TestCertStore_RejectsForeignCert(t *testing.T) {
	ca := newTestCA(t)
	cs, _ := NewCertStore(t.TempDir(), "sp-x")

	other, _ := NewCertStore(t.TempDir(), "sp-x") // different key, same CN
	otherCSR, _ := other.CSR()
	foreign := ca.sign(t, otherCSR, "sp-x", time.Hour)
	if err := cs.Install(foreign, ca.pem); err == nil {
		t.Fatal("cert for a different private key must be refused")
	}

	ownCSR, _ := cs.CSR()
	wrongCN := ca.sign(t, ownCSR, "sp-OTHER", time.Hour)
	if err := cs.Install(wrongCN, ca.pem); err == nil {
		t.Fatal("cert with a foreign CN must be refused")
	}
}

func TestCertStore_RenewDueAtHalfLife(t *testing.T) {
	ca := newTestCA(t)
	cs, _ := NewCertStore(t.TempDir(), "sp-x")
	csr, _ := cs.CSR()
	// 2-minute validity, already 90s in (NotBefore backdated 1min by sign +
	// validity 1min ahead): time.Until(NotAfter)=1min, total=2min → 1min < 1min?
	// Use explicit: validity 30s → half-life passes immediately after ~15s. To
	// keep the test instant, sign a cert whose NotAfter is 10s away: remaining
	// 10s < total(70s)/2 → due.
	if err := cs.Install(ca.sign(t, csr, "sp-x", 10*time.Second), ca.pem); err != nil {
		t.Fatal(err)
	}
	if !cs.RenewDue() {
		t.Fatal("cert with 10s left of a 70s life must be renew-due")
	}
}
