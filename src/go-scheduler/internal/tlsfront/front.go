// Package tlsfront is the worker-side mTLS terminator built into the
// scheduler: two TLS listeners reverse-proxying the local plaintext services
// (py-inference :8000, scheduler health :9090). It replaces the external nginx
// terminator so that certificate-at-registration needs NO reload plumbing —
// tls.Config callbacks fetch the current certificate and client-CA pool per
// handshake, so the certificate the gateway just issued is live on the very
// next connection.
//
// It refuses to serve until material exists (handshakes fail cleanly), and it
// REQUIRES a client certificate from the installed CA — the same posture as
// the nginx setup it replaces.
package tlsfront

import (
	"context"
	"crypto/tls"
	"crypto/x509"
	"fmt"
	"log/slog"
	"net/http"
	"net/http/httputil"
	"net/url"
	"time"
)

// Material is the mutable TLS state (implemented by spregister.CertStore).
type Material interface {
	Certificate() *tls.Certificate
	ClientCAPool() *x509.CertPool
}

// Front owns the two listeners.
type Front struct {
	servers []*http.Server
	logger  *slog.Logger
}

// New builds the front. Each mapping proxies one TLS port to one local
// plaintext backend.
func New(mat Material, logger *slog.Logger, mappings map[int]string) (*Front, error) {
	f := &Front{logger: logger}
	for port, backend := range mappings {
		target, err := url.Parse(backend)
		if err != nil {
			return nil, fmt.Errorf("tlsfront: bad backend %q: %w", backend, err)
		}
		proxy := httputil.NewSingleHostReverseProxy(target)
		proxy.FlushInterval = -1 // token-by-token streaming must flow

		tlsCfg := &tls.Config{
			MinVersion: tls.VersionTLS12,
			// Per-handshake callbacks — the heart of reload-free rotation.
			GetCertificate: func(*tls.ClientHelloInfo) (*tls.Certificate, error) {
				if c := mat.Certificate(); c != nil {
					return c, nil
				}
				return nil, fmt.Errorf("worker certificate not yet issued")
			},
			// ClientAuth must verify against the CURRENT CA pool. VerifyPeerCertificate
			// with RequireAnyClientCert lets the pool be dynamic (a static ClientCAs
			// snapshot would need a server restart on CA change).
			ClientAuth: tls.RequireAnyClientCert,
			VerifyPeerCertificate: func(rawCerts [][]byte, _ [][]*x509.Certificate) error {
				pool := mat.ClientCAPool()
				if pool == nil {
					return fmt.Errorf("client CA not installed yet")
				}
				if len(rawCerts) == 0 {
					return fmt.Errorf("client certificate required")
				}
				leaf, err := x509.ParseCertificate(rawCerts[0])
				if err != nil {
					return err
				}
				inter := x509.NewCertPool()
				for _, raw := range rawCerts[1:] {
					if c, err := x509.ParseCertificate(raw); err == nil {
						inter.AddCert(c)
					}
				}
				_, err = leaf.Verify(x509.VerifyOptions{
					Roots:         pool,
					Intermediates: inter,
					KeyUsages:     []x509.ExtKeyUsage{x509.ExtKeyUsageClientAuth},
				})
				return err
			},
		}
		f.servers = append(f.servers, &http.Server{
			Addr:              fmt.Sprintf(":%d", port),
			Handler:           proxy,
			TLSConfig:         tlsCfg,
			ReadHeaderTimeout: 15 * time.Second,
		})
	}
	return f, nil
}

// Start launches all listeners (non-blocking). Serve errors are logged, not
// fatal: a busy port must not take mining protection down with it.
func (f *Front) Start() {
	for _, srv := range f.servers {
		s := srv
		go func() {
			f.logger.Info("mTLS front listening", "addr", s.Addr)
			if err := s.ListenAndServeTLS("", ""); err != nil && err != http.ErrServerClosed {
				f.logger.Error("mTLS front stopped", "addr", s.Addr, "error", err)
			}
		}()
	}
}

// Stop shuts the listeners down gracefully.
func (f *Front) Stop(ctx context.Context) {
	for _, s := range f.servers {
		_ = s.Shutdown(ctx)
	}
}
