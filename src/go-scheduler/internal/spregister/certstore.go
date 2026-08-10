package spregister

import (
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"fmt"
	"os"
	"path/filepath"
	"sync/atomic"
	"time"
)

// CertStore owns the worker's mTLS material: a PERSISTENT private key that
// never leaves this machine, the short-lived certificate the gateway signs for
// it at registration/renewal, and the CA certificate the TLS front uses to
// verify the gateway's client certificate. Everything lives under one dir
// (0600 for the key) and the parsed state is swapped atomically so TLS
// handshakes always see a consistent pair without any reload.
type CertStore struct {
	dir      string
	workerID string

	key     *ecdsa.PrivateKey
	current atomic.Value // *tls.Certificate (nil until first install)
	caPool  atomic.Value // *x509.CertPool
	// NotAfter of the installed cert; renewal fires at half-life.
	notAfter atomic.Int64
}

// NewCertStore loads (or creates) the worker key under dir. The key is created
// exactly once and reused across registrations — rotating the certificate does
// not rotate the key, so a re-registration storm cannot churn the identity.
func NewCertStore(dir, workerID string) (*CertStore, error) {
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return nil, fmt.Errorf("certstore: mkdir: %w", err)
	}
	cs := &CertStore{dir: dir, workerID: workerID}

	keyPath := filepath.Join(dir, "worker.key")
	if raw, err := os.ReadFile(keyPath); err == nil {
		block, _ := pem.Decode(raw)
		if block == nil {
			return nil, fmt.Errorf("certstore: %s exists but holds no PEM", keyPath)
		}
		k, err := x509.ParseECPrivateKey(block.Bytes)
		if err != nil {
			return nil, fmt.Errorf("certstore: parse worker key: %w", err)
		}
		cs.key = k
	} else {
		k, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
		if err != nil {
			return nil, err
		}
		der, _ := x509.MarshalECPrivateKey(k)
		if err := os.WriteFile(keyPath, pem.EncodeToMemory(&pem.Block{Type: "EC PRIVATE KEY", Bytes: der}), 0o600); err != nil {
			return nil, fmt.Errorf("certstore: persist worker key: %w", err)
		}
		cs.key = k
	}

	// Best-effort warm start: a cert persisted by an earlier run keeps the TLS
	// front serving across restarts until the next registration rotates it.
	certPath, caPath := filepath.Join(dir, "worker.crt"), filepath.Join(dir, "ca.crt")
	if certPEM, err := os.ReadFile(certPath); err == nil {
		if caPEM, err2 := os.ReadFile(caPath); err2 == nil {
			_ = cs.install(string(certPEM), string(caPEM), false)
		}
	}
	return cs, nil
}

// CSR builds a fresh signing request for the worker identity. Only CN matters
// (the gateway ignores everything else by design).
func (cs *CertStore) CSR() (string, error) {
	der, err := x509.CreateCertificateRequest(rand.Reader, &x509.CertificateRequest{
		Subject: pkix.Name{CommonName: cs.workerID},
	}, cs.key)
	if err != nil {
		return "", err
	}
	return string(pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE REQUEST", Bytes: der})), nil
}

// Install validates and activates a certificate returned by the gateway,
// persisting both PEMs. The pair check (cert pubkey == our key) protects
// against a confused or malicious response installing someone else's cert.
func (cs *CertStore) Install(certPEM, caPEM string) error {
	return cs.install(certPEM, caPEM, true)
}

func (cs *CertStore) install(certPEM, caPEM string, persist bool) error {
	block, _ := pem.Decode([]byte(certPEM))
	if block == nil {
		return fmt.Errorf("certstore: worker cert is not PEM")
	}
	leaf, err := x509.ParseCertificate(block.Bytes)
	if err != nil {
		return fmt.Errorf("certstore: parse worker cert: %w", err)
	}
	pub, ok := leaf.PublicKey.(*ecdsa.PublicKey)
	if !ok || !pub.Equal(&cs.key.PublicKey) {
		return fmt.Errorf("certstore: certificate public key does not match our worker key")
	}
	if leaf.Subject.CommonName != cs.workerID {
		return fmt.Errorf("certstore: certificate CN %q != worker id %q", leaf.Subject.CommonName, cs.workerID)
	}
	pool := x509.NewCertPool()
	if !pool.AppendCertsFromPEM([]byte(caPEM)) {
		return fmt.Errorf("certstore: CA PEM parsed no certificates")
	}

	pair := &tls.Certificate{Certificate: [][]byte{block.Bytes}, PrivateKey: cs.key, Leaf: leaf}
	cs.current.Store(pair)
	cs.caPool.Store(pool)
	cs.notAfter.Store(leaf.NotAfter.Unix())

	if persist {
		if err := os.WriteFile(filepath.Join(cs.dir, "worker.crt"), []byte(certPEM), 0o644); err != nil {
			return err
		}
		if err := os.WriteFile(filepath.Join(cs.dir, "ca.crt"), []byte(caPEM), 0o644); err != nil {
			return err
		}
	}
	return nil
}

// Certificate returns the active pair (nil before the first install) — handed
// to tls.Config.GetCertificate, so every handshake picks up the newest cert.
func (cs *CertStore) Certificate() *tls.Certificate {
	c, _ := cs.current.Load().(*tls.Certificate)
	return c
}

// ClientCAPool returns the CA pool for verifying the GATEWAY's client cert.
func (cs *CertStore) ClientCAPool() *x509.CertPool {
	p, _ := cs.caPool.Load().(*x509.CertPool)
	return p
}

// RenewDue reports whether the active certificate has passed half its
// remaining life (or is missing entirely).
func (cs *CertStore) RenewDue() bool {
	c := cs.Certificate()
	if c == nil {
		return true
	}
	na := time.Unix(cs.notAfter.Load(), 0)
	total := na.Sub(c.Leaf.NotBefore)
	return time.Until(na) < total/2
}

// CAPEM returns the persisted registration-CA certificate (empty until a
// registration has installed one). Used as the trust anchor for the
// worker→gateway HTTPS direction — the same CA both sides already share.
func (cs *CertStore) CAPEM() []byte {
	pem, err := os.ReadFile(filepath.Join(cs.dir, "ca.crt"))
	if err != nil {
		return nil
	}
	return pem
}
