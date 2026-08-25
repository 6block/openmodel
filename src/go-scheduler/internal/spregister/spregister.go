// Package spregister implements the SP side of gateway self-registration: the
// scheduler proves control of its miner by signing a gateway challenge with the
// miner's worker (or owner) key from the local Lotus wallet, and receives the
// per-worker auth token in return. The token is persisted and handed to the
// inference service over the existing gRPC channel (ScheduleResponse.gateway_auth_token).
package spregister

import (
	"bytes"
	"context"
	"crypto/tls"
	"crypto/x509"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"time"
)

// LotusWallet is the slice of the Lotus API the registrar needs (implemented by
// lotus.RPCClient; faked in tests).
type LotusWallet interface {
	MinerControlAddrs(ctx context.Context) (ownerID, workerID string, err error)
	StateAccountKey(ctx context.Context, idAddr string) (string, error)
	WalletHas(ctx context.Context, addr string) (bool, error)
	WalletSign(ctx context.Context, addr string, data []byte) (sigType uint64, sig []byte, err error)
}

// Config carries everything needed to self-register with a gateway.
type Config struct {
	GatewayURL         string // e.g. "http://gateway-host:3000"; empty = registration disabled
	GatewayID          string // must equal the gateway's sp_registration.gateway_id (signature domain)
	WorkerID           string
	MinerID            string // t0/f0 ID address of this SP's miner
	PayoutAddress      string // EVM address earnings settle to (goes into the signed message)
	PublicEndpoint     string // this worker's py-inference URL as reachable FROM the gateway
	PublicSchedulerURL string // this scheduler's :9090 URL as reachable FROM the gateway
	GPUCount           int
	// SupportedModels is the FULL list this worker can serve — the loaded model
	// plus any it can switch to (weights present locally). Model-aware routing
	// can only switch toward a model it KNOWS about, and self-registration was
	// the one path that never reported this list, so switching silently never
	// applied to self-registered SPs.
	SupportedModels []string
	// TLSCAPath is the trust anchor for an https GatewayURL (cold start; once
	// registered, the cert store's CA is trusted as well).
	TLSCAPath string
	// TLSSystemCA: trust the system certificate store and validate the URL's
	// hostname instead of the registration CA + gateway_id — for gateways on a
	// public domain with a publicly-trusted certificate. See config.TLSSystemCA.
	TLSSystemCA bool
	TokenPath   string // where the issued token persists (0600); empty = not persisted
	// AutoRegister mirrors gateway.auto_register: whether registration runs on
	// start-up. A Registrar is built whenever a gateway URL is configured even if
	// this is false — the worker may still hold a valid token from an earlier
	// registration, which /ready must keep accepting, and the operator still wants
	// GET /registration to answer.
	AutoRegister bool
}

// Result is what a successful registration yields.
type Result struct {
	WorkerID  string `json:"worker_id"`
	AuthToken string `json:"auth_token"`
	MinerID   string `json:"miner_id"`
	Rotated   bool   `json:"rotated"`
	// Present when the gateway issues certificates and a CSR was submitted.
	WorkerCert       string `json:"worker_cert"`
	CACert           string `json:"ca_cert"`
	CertNotAfterUnix int64  `json:"cert_not_after_unix"`
}

type Registrar struct {
	cfg    Config
	lotus  LotusWallet
	hc     *http.Client
	logger *slog.Logger
	token  atomic.Value // string — current auth token for the gRPC push
	status atomic.Value // Status — observable registration state for GET /registration
	// Link observation: is the gateway still USING the token we hold? Holding a
	// token proves nothing on its own — it survives the gateway dropping us.
	certs     *CertStore   // nil = certificate-at-registration disabled
	lastPoll  atomic.Int64 // unix nano of the last /ready poll the gateway made with our token; 0 = none yet
	startedAt time.Time    // baseline for "silent for how long" before any poll arrives
	link      atomic.Value // Link — verdict computed by the background watcher
	selfView  atomic.Value // []byte — last /v1/worker/self JSON (admission progress)
}

// Verdicts for Link.Verdict.
const (
	LinkServing     = "serving"     // the gateway polled us recently — the token is live
	LinkWaiting     = "waiting"     // no poll seen yet, but not long enough to worry
	LinkUnreachable = "unreachable" // silent AND the gateway does not answer — a network problem
	LinkSilent      = "silent"      // silent BUT the gateway answers — we may have been dropped
	LinkIdle        = "idle"        // no token held; there is no link to judge
)

const (
	// The gateway polls /ready every 5s (workers.poll_interval_sec). 120s is 24
	// consecutive misses — well past any normal hiccup, still far below the point
	// where an operator would start guessing.
	pollSilenceThreshold = 120 * time.Second
	linkCheckInterval    = 30 * time.Second
)

// Link is the gateway-side view of the registration: not "do I have a token" but
// "is the gateway still using it". These differ in exactly the case that misleads
// an operator most — the gateway dropped the worker, the local token file is still
// there, and everything locally still looks registered.
//
// It deliberately separates the two ways a link can go quiet, because they call for
// opposite responses: a network problem fixes itself (the gateway keeps retrying and
// re-marks the worker online), being dropped does not.
type Link struct {
	Verdict      string `json:"verdict"`
	LastPolledAt string `json:"last_polled_at,omitempty"` // RFC3339; last poll accepted with our token
	SilentForSec int64  `json:"silent_for_sec"`
	Note         string `json:"note,omitempty"`
}

// SetCertStore arms certificate-at-registration: RegisterOnce attaches a CSR
// and installs the certificate the gateway returns; RunRenewLoop keeps it fresh.
func (r *Registrar) SetCertStore(cs *CertStore) {
	r.certs = cs
	r.rebuildTransport()
}

// rebuildTransport arms the HTTP client for an https gateway URL: trust is the
// registration CA (from the cert store once registered, or tls_ca_path on cold
// start) and the expected identity is the GATEWAY ID, not the address — the
// same address-decoupling worker certificates use, so operators can remap
// public ports freely without reissuing anything.
func (r *Registrar) rebuildTransport() {
	if !strings.HasPrefix(r.cfg.GatewayURL, "https://") {
		return
	}
	if r.cfg.TLSSystemCA {
		// Public-trust mode: the system store validates the chain and Go's
		// default hostname verification checks the dialed host — the gateway
		// sits behind a real domain with a publicly-trusted certificate, so
		// there is no CA file to place and gateway_id plays no TLS role here
		// (it remains the signature domain of the registration protocol).
		r.hc.Transport = &http.Transport{
			TLSClientConfig: &tls.Config{MinVersion: tls.VersionTLS12},
		}
		r.logger.Info("worker→gateway TLS armed", "trust", "system CA store",
			"identity", "URL hostname")
		return
	}
	pool := x509.NewCertPool()
	added := false
	if r.cfg.TLSCAPath != "" {
		if pem, err := os.ReadFile(r.cfg.TLSCAPath); err == nil && pool.AppendCertsFromPEM(pem) {
			added = true
		} else {
			r.logger.Warn("tls_ca_path unreadable or not a CA bundle", "path", r.cfg.TLSCAPath, "error", err)
		}
	}
	if r.certs != nil {
		if caPEM := r.certs.CAPEM(); len(caPEM) > 0 && pool.AppendCertsFromPEM(caPEM) {
			added = true
		}
	}
	if !added {
		r.logger.Error("https gateway_url configured but no CA available " +
			"(set gateway.tls_ca_path or register once over a trusted channel); TLS dials will fail closed")
	}
	r.hc.Transport = &http.Transport{
		TLSClientConfig: &tls.Config{
			RootCAs:    pool,
			ServerName: r.cfg.GatewayID, // identity = gateway_id, never the dialed address
			MinVersion: tls.VersionTLS12,
		},
	}
	r.logger.Info("worker→gateway TLS armed", "trust", "registration CA",
		"server_name", r.cfg.GatewayID)
}

// NoteGatewayPoll records that the gateway just authenticated a /ready poll with the
// token this registrar issued. Called on every poll (5s), so it stays a single store.
func (r *Registrar) NoteGatewayPoll() {
	r.lastPoll.Store(time.Now().UnixNano())
}

// Status is the SP-facing view of "am I registered, with whom, and if not, why".
// Without it an operator has to read JSON logs or a file that holds the secret; this
// is what GET /registration serves. The token is never included — only a prefix, so
// the answer can be shared in a bug report.
type Status struct {
	// Enabled reports whether registration runs automatically (gateway.auto_register).
	// false with a gateway_url present means "configured but not auto-registering" —
	// distinct from the endpoint 404, which means no gateway is configured at all.
	Enabled bool `json:"enabled"`
	// Registered means a token is HELD LOCALLY — not that the gateway still honours
	// it. It stays true after the gateway deregisters this worker (nothing tells the
	// SP), and it stays true while re-registration is being rejected (a gateway that
	// raises its admission floor does not evict the SPs already on its roster, so a
	// 403 here routinely coexists with a live registration). Read GatewayLink for
	// whether the token is actually in use.
	Registered    bool   `json:"registered"`
	WorkerID      string `json:"worker_id,omitempty"`
	MinerID       string `json:"miner_id,omitempty"`
	GatewayURL    string `json:"gateway_url,omitempty"`
	GatewayID     string `json:"gateway_id,omitempty"`
	PayoutAddress string `json:"payout_address,omitempty"`
	RegisteredAt  string `json:"registered_at,omitempty"` // RFC3339; this process's last success
	TokenPrefix   string `json:"token_prefix,omitempty"`  // e.g. "wk-78415e…" — never the full token
	Rotated       bool   `json:"rotated,omitempty"`       // last success replaced an existing registration
	LastError     string `json:"last_error,omitempty"`    // why the most recent attempt failed
	Attempts      int    `json:"attempts,omitempty"`      // attempts made in this process
	// GatewayLink answers the question Registered cannot: is the gateway still
	// polling us with this token. A failing LastError next to GatewayLink.serving
	// means "cannot re-register, but still working" — a real and non-alarming state.
	GatewayLink Link `json:"gateway_link"`
	// ModelAdmission mirrors the gateway's verdict on this worker's claimed
	// models (verified / pending / failing, fail counts, the floor) — fetched
	// on the link-watch cadence so an SP can see WHY a claim gets no traffic
	// and how close a failing one is to a ban, from this box alone.
	ModelAdmission json.RawMessage `json:"model_admission,omitempty"`
	// Note: a scheduler self-registers with exactly ONE gateway (the `gateway:`
	// config block is singular). A worker can still serve additional gateways that
	// were set up with a static token — those do not appear here.
}

// CurrentStatus returns the observable registration state.
func (r *Registrar) CurrentStatus() Status {
	s, _ := r.status.Load().(Status)
	s.Enabled = r.cfg.AutoRegister
	s.GatewayURL = r.cfg.GatewayURL
	s.GatewayID = r.cfg.GatewayID
	s.PayoutAddress = r.cfg.PayoutAddress
	if s.WorkerID == "" {
		s.WorkerID = r.cfg.WorkerID
	}
	if s.MinerID == "" {
		s.MinerID = r.cfg.MinerID
	}
	if tok := r.CurrentToken(); tok != "" {
		s.Registered = true
		s.TokenPrefix = tokenPrefix(tok)
	}
	s.GatewayLink = r.linkStatus()
	s.ModelAdmission = r.SelfView()
	return s
}

// linkStatus renders the current link view. It never does I/O — the reachability
// probe that distinguishes "unreachable" from "silent" runs in WatchLink — so the
// endpoint answers instantly. Elapsed time is recomputed here rather than read from
// the watcher's snapshot, which would be up to linkCheckInterval stale.
func (r *Registrar) linkStatus() Link {
	if r.CurrentToken() == "" {
		return Link{Verdict: LinkIdle, Note: "no token held; nothing to poll"}
	}

	lp := r.lastPoll.Load()
	if lp == 0 {
		// Never polled in this process. Measure from start-up, otherwise a worker
		// that has NEVER been reachable would report a comfortable "0s silent".
		silent := time.Since(r.startedAt)
		l, _ := r.link.Load().(Link)
		if silent < pollSilenceThreshold {
			return Link{
				Verdict:      LinkWaiting,
				SilentForSec: int64(silent.Seconds()),
				Note:         "holding a token but the gateway has not polled yet; normal right after start-up",
			}
		}
		// Past the threshold the watcher's probe result decides which kind of quiet
		// this is; fall back to the neutral wording until the first probe lands.
		if l.Verdict == LinkUnreachable || l.Verdict == LinkSilent {
			l.SilentForSec = int64(silent.Seconds())
			return l
		}
		return Link{
			Verdict:      LinkWaiting,
			SilentForSec: int64(silent.Seconds()),
			Note:         "no poll seen since start-up; checking whether the gateway is reachable",
		}
	}

	last := time.Unix(0, lp)
	silent := time.Since(last)
	l, _ := r.link.Load().(Link)
	l.LastPolledAt = last.UTC().Format(time.RFC3339)
	l.SilentForSec = int64(silent.Seconds())
	if silent < pollSilenceThreshold {
		// A recent poll is proof the token is live, whatever the watcher last thought.
		l.Verdict = LinkServing
		l.Note = ""
	}
	return l
}

// WatchLink runs the probe that separates "the network is down" from "the gateway
// dropped us" — indistinguishable locally, since both simply stop the polls, yet
// opposite in what the operator should do. It only probes once the link has been
// quiet past the threshold, so a healthy worker generates no extra traffic.
//
// It deliberately does NOT re-register on its own when the verdict is `silent`.
// Deregistration is an operator action on the gateway side (an SP cannot deregister
// itself), so silently registering back would be a worker overriding a human
// decision. Bans do not land here at all — banned workers keep being polled.
func (r *Registrar) WatchLink(ctx context.Context) {
	go func() {
		t := time.NewTicker(linkCheckInterval)
		defer t.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-t.C:
				r.evaluateLink(ctx)
				r.refreshSelfView(ctx)
			}
		}
	}()
}

func (r *Registrar) evaluateLink(ctx context.Context) {
	if r.CurrentToken() == "" {
		return
	}
	since := r.startedAt
	if lp := r.lastPoll.Load(); lp != 0 {
		since = time.Unix(0, lp)
	}
	if time.Since(since) < pollSilenceThreshold {
		if prev, _ := r.link.Load().(Link); prev.Verdict != LinkServing && prev.Verdict != "" {
			r.logger.Info("gateway link recovered", "worker_id", r.cfg.WorkerID)
		}
		r.link.Store(Link{Verdict: LinkServing})
		return
	}

	verdict, note := LinkUnreachable, "gateway is not answering from this host — most likely a transient network problem; the gateway retries on its own and the worker comes back online without action here"
	if r.gatewayReachable(ctx) {
		verdict, note = LinkSilent, "gateway is reachable but has stopped polling this worker — the registration may have been removed on the gateway side, which would make the token held here stale"
	}

	prev, _ := r.link.Load().(Link)
	r.link.Store(Link{Verdict: verdict, Note: note})
	if prev.Verdict != verdict {
		// Only on transition: this loop ticks every 30s and an outage can last hours.
		r.logger.Warn("gateway link degraded",
			"verdict", verdict, "worker_id", r.cfg.WorkerID,
			"silent_for", time.Since(since).Truncate(time.Second).String(), "detail", note)
	}
}

// gatewayReachable asks only "does something answer HTTP at the gateway URL".
// ANY status code counts as reachable — 401 and 404 both prove the gateway is up,
// and the endpoint used needs no credentials. Only a transport error means down.
func (r *Registrar) gatewayReachable(ctx context.Context) bool {
	probeCtx, cancel := context.WithTimeout(ctx, 8*time.Second)
	defer cancel()
	url := strings.TrimRight(r.cfg.GatewayURL, "/") + "/v1/models"
	req, err := http.NewRequestWithContext(probeCtx, http.MethodGet, url, nil)
	if err != nil {
		return false
	}
	resp, err := r.hc.Do(req)
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	io.Copy(io.Discard, io.LimitReader(resp.Body, 1<<10))
	return true
}

// tokenPrefix keeps enough to correlate with the gateway's view without leaking a
// usable credential.
func tokenPrefix(tok string) string {
	if len(tok) <= 10 {
		return "…"
	}
	return tok[:10] + "…"
}

func (r *Registrar) setStatus(mutate func(*Status)) {
	s, _ := r.status.Load().(Status)
	mutate(&s)
	r.status.Store(s)
}

func New(cfg Config, wallet LotusWallet, logger *slog.Logger) *Registrar {
	if logger == nil {
		logger = slog.Default()
	}
	r := &Registrar{
		cfg:       cfg,
		lotus:     wallet,
		hc:        &http.Client{Timeout: 20 * time.Second},
		logger:    logger,
		startedAt: time.Now(),
	}
	r.token.Store("")
	r.status.Store(Status{})
	r.link.Store(Link{})
	// A previously-issued token keeps the worker authenticated across scheduler
	// restarts until re-registration succeeds and rotates it.
	if tok, meta := r.loadPersistedToken(); tok != "" {
		r.token.Store(tok)
		r.setStatus(func(s *Status) {
			s.Registered = true
			s.WorkerID = meta.WorkerID
			s.MinerID = meta.MinerID
			s.RegisteredAt = meta.RegisteredAt // when the token we hold was issued
		})
	}
	r.rebuildTransport() // https gateway URLs verify gateway_id against the registration CA
	return r
}

// CurrentToken returns the auth token to advertise to the inference service
// (empty until a registration — this run or a persisted earlier one — provides one).
func (r *Registrar) CurrentToken() string {
	tok, _ := r.token.Load().(string)
	return tok
}

// RunBackground registers with retries until it succeeds or ctx ends. It runs
// beside the scheduler's mining-protection loop and must never block it — a
// gateway being down only delays inference traffic, never mining safety.
func (r *Registrar) RunBackground(ctx context.Context) {
	go func() {
		backoff := 15 * time.Second
		const maxBackoff = 5 * time.Minute
		for {
			res, err := r.RegisterOnce(ctx)
			if err == nil {
				r.logger.Info("sp self-registration complete",
					"worker_id", res.WorkerID, "miner", res.MinerID, "rotated", res.Rotated)
				return
			}
			if ctx.Err() != nil {
				return
			}
			r.setStatus(func(s *Status) { s.LastError = err.Error() })
			r.logger.Warn("sp self-registration failed; will retry",
				"error", err, "retry_in", backoff)
			select {
			case <-ctx.Done():
				return
			case <-time.After(backoff):
			}
			if backoff *= 2; backoff > maxBackoff {
				backoff = maxBackoff
			}
		}
	}()
}

// RegisterOnce performs one full challenge→sign→register round.
func (r *Registrar) RegisterOnce(ctx context.Context) (*Result, error) {
	r.setStatus(func(s *Status) { s.Attempts++ })
	signer, err := r.pickSigner(ctx)
	if err != nil {
		return nil, err
	}

	challenge, err := r.fetchChallenge(ctx)
	if err != nil {
		return nil, err
	}

	msg := buildRegistrationMessage(r.cfg.MinerID, r.cfg.GatewayID, r.cfg.WorkerID,
		r.cfg.PayoutAddress, r.cfg.PublicEndpoint, r.cfg.PublicSchedulerURL, challenge)
	sigType, sig, err := r.lotus.WalletSign(ctx, signer, []byte(msg))
	if err != nil {
		return nil, fmt.Errorf("sign registration message: %w", err)
	}

	res, err := r.submitRegistration(ctx, challenge, signer, sigType, sig)
	if err != nil {
		return nil, err
	}

	r.token.Store(res.AuthToken)
	r.setStatus(func(s *Status) {
		s.Registered = true
		s.WorkerID = res.WorkerID
		s.MinerID = res.MinerID
		s.Rotated = res.Rotated
		s.RegisteredAt = time.Now().UTC().Format(time.RFC3339)
		s.LastError = "" // a success clears the previous failure reason
	})
	if err := r.persistToken(res); err != nil {
		// The registration itself succeeded; a persistence failure only costs
		// restart continuity. Loud log, no failure.
		r.logger.Error("failed to persist gateway auth token", "path", r.cfg.TokenPath, "error", err)
	}
	return res, nil
}

// pickSigner resolves which locally-held key can prove miner control: the worker
// key when the local wallet has it, else the owner key.
func (r *Registrar) pickSigner(ctx context.Context) (string, error) {
	ownerID, workerID, err := r.lotus.MinerControlAddrs(ctx)
	if err != nil {
		return "", err
	}
	workerKey, err := r.lotus.StateAccountKey(ctx, workerID)
	if err != nil {
		return "", err
	}
	var probeErr error
	if has, err := r.lotus.WalletHas(ctx, workerKey); err != nil {
		probeErr = err
	} else if has {
		return workerKey, nil
	}
	ownerKey, err := r.lotus.StateAccountKey(ctx, ownerID)
	if err != nil {
		return "", err
	}
	if has, err := r.lotus.WalletHas(ctx, ownerKey); err != nil {
		if probeErr == nil {
			probeErr = err
		}
	} else if has {
		return ownerKey, nil
	}
	if probeErr != nil {
		// A failed wallet PROBE is not an absent key. The classic cause is a
		// missing/under-privileged Lotus API token — WalletHas needs write
		// permission — and swallowing that error here reported "key not in
		// wallet" while the key was present, sending an operator to debug the
		// wrong thing (it happened during the alpha clean-room run).
		return "", fmt.Errorf("could not query the local Lotus wallet (lotus.api_token missing or lacking permission?): %w", probeErr)
	}
	return "", fmt.Errorf("neither the worker key (%s) nor the owner key (%s) of miner %s is in the local Lotus wallet — import one, or register manually",
		workerKey, ownerKey, r.cfg.MinerID)
}

// buildRegistrationMessage is the EXACT text signed by the miner key. It MUST
// stay byte-identical to the gateway's buildSPRegistrationMessage
// (sp-state-agent internal/gateway/spregister.go) — the server reconstructs and
// verifies this text verbatim.
func buildRegistrationMessage(minerID, gatewayID, workerID, payout, endpoint, schedulerURL, challenge string) string {
	return "OpenModel SP registration\n" +
		"miner: " + minerID + "\n" +
		"gateway: " + gatewayID + "\n" +
		"worker_id: " + workerID + "\n" +
		"payout: " + payout + "\n" +
		"endpoint: " + endpoint + "\n" +
		"scheduler_url: " + schedulerURL + "\n" +
		"challenge: " + challenge
}

func (r *Registrar) fetchChallenge(ctx context.Context) (string, error) {
	body, _ := json.Marshal(map[string]string{"miner_id": r.cfg.MinerID})
	var resp struct {
		Challenge string `json:"challenge"`
	}
	if err := r.postJSON(ctx, "/v1/sp/challenge", body, &resp); err != nil {
		return "", fmt.Errorf("fetch challenge: %w", err)
	}
	if resp.Challenge == "" {
		return "", fmt.Errorf("fetch challenge: empty challenge in response")
	}
	return resp.Challenge, nil
}

func (r *Registrar) submitRegistration(ctx context.Context, challenge, signer string, sigType uint64, sig []byte) (*Result, error) {
	payload := map[string]any{
		"miner_id":         r.cfg.MinerID,
		"worker_id":        r.cfg.WorkerID,
		"payout_address":   r.cfg.PayoutAddress,
		"endpoint":         r.cfg.PublicEndpoint,
		"scheduler_url":    r.cfg.PublicSchedulerURL,
		"gpu_count":        r.cfg.GPUCount,
		"supported_models": r.cfg.SupportedModels,
		"challenge":        challenge,
		"signer":           signer,
		"signature": map[string]any{
			"type": sigType,
			"data": base64.StdEncoding.EncodeToString(sig),
		},
	}
	// Certificate-at-registration: same admission, same round-trip. The CSR
	// carries only our public key; the private key stays in the cert store.
	if r.certs != nil {
		if csr, err := r.certs.CSR(); err == nil {
			payload["csr"] = csr
		} else {
			r.logger.Warn("could not build CSR; registering without certificate", "error", err)
		}
	}
	body, _ := json.Marshal(payload)
	var res Result
	if err := r.postJSON(ctx, "/v1/sp/register", body, &res); err != nil {
		return nil, fmt.Errorf("register: %w", err)
	}
	if res.AuthToken == "" {
		return nil, fmt.Errorf("register: gateway returned no auth token")
	}
	if r.certs != nil && res.WorkerCert == "" {
		// mtls_front is on but the gateway issued nothing (no issuer configured).
		// Without this warning the failure is maddeningly indirect: registration
		// succeeds, yet the https endpoint has no certificate to serve, every
		// gateway poll dies in the handshake, the worker sits offline, and
		// gateway_link misreads the silence as "deregistered".
		r.logger.Warn("mTLS front is enabled but the gateway issued no certificate — " +
			"the https endpoint cannot serve; configure issuer_ca_cert/key on the gateway " +
			"or point public_endpoint back to http")
	}
	if r.certs != nil && res.WorkerCert != "" {
		if err := r.certs.Install(res.WorkerCert, res.CACert); err != nil {
			// The registration itself is fine (token works, plaintext serves);
			// a bad certificate only degrades the mTLS front.
			r.logger.Warn("issued certificate rejected by local validation", "error", err)
		} else {
			r.logger.Info("worker certificate installed",
				"not_after", time.Unix(res.CertNotAfterUnix, 0).UTC().Format(time.RFC3339))
		}
	}
	return &res, nil
}

// RunRenewLoop keeps the worker certificate fresh: it checks hourly and renews
// once the cert passes half-life, using POST /v1/sp/renew-cert with the worker
// token. Renewal refused (ban/deregistration) is only logged — the certificate
// then ages out, which is the designed escalation path, and the operator sees
// why in both logs. Runs beside mining protection; never blocks it.
func (r *Registrar) RunRenewLoop(ctx context.Context) {
	if r.certs == nil {
		return
	}
	go func() {
		// The due-check is pure memory, so a tight tick costs nothing and keeps
		// the loop correct for ANY validity (an hourly tick would overshoot the
		// half-life of short drill certificates entirely).
		t := time.NewTicker(1 * time.Minute)
		defer t.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-t.C:
				if !r.certs.RenewDue() {
					continue
				}
				if err := r.renewOnce(ctx); err != nil {
					r.logger.Warn("certificate renewal failed; will retry next tick", "error", err)
				}
			}
		}
	}()
}

func (r *Registrar) renewOnce(ctx context.Context) error {
	tok := r.CurrentToken()
	if tok == "" {
		return fmt.Errorf("no worker token held")
	}
	csr, err := r.certs.CSR()
	if err != nil {
		return err
	}
	body, _ := json.Marshal(map[string]string{"csr": csr})
	u := strings.TrimRight(r.cfg.GatewayURL, "/") + "/v1/sp/renew-cert"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, u, bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+tok)
	resp, err := r.hc.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("gateway returned %d: %s", resp.StatusCode, string(raw))
	}
	var out struct {
		WorkerCert       string `json:"worker_cert"`
		CACert           string `json:"ca_cert"`
		CertNotAfterUnix int64  `json:"cert_not_after_unix"`
	}
	if err := json.Unmarshal(raw, &out); err != nil {
		return err
	}
	if err := r.certs.Install(out.WorkerCert, out.CACert); err != nil {
		return err
	}
	r.logger.Info("worker certificate renewed",
		"not_after", time.Unix(out.CertNotAfterUnix, 0).UTC().Format(time.RFC3339))
	return nil
}

func (r *Registrar) postJSON(ctx context.Context, path string, body []byte, out any) error {
	u := strings.TrimRight(r.cfg.GatewayURL, "/") + path
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, u, bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := r.hc.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return err
	}
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("gateway returned %d: %.300s", resp.StatusCode, data)
	}
	return json.Unmarshal(data, out)
}

// persistToken writes the issued token (0600) so a scheduler restart keeps the
// worker authenticated even before re-registration completes.
func (r *Registrar) persistToken(res *Result) error {
	if r.cfg.TokenPath == "" {
		return nil
	}
	if err := os.MkdirAll(filepath.Dir(r.cfg.TokenPath), 0o700); err != nil {
		return err
	}
	data, err := json.MarshalIndent(map[string]any{
		"worker_id":     res.WorkerID,
		"auth_token":    res.AuthToken,
		"miner_id":      res.MinerID,
		"gateway_url":   r.cfg.GatewayURL,
		"registered_at": time.Now().UTC().Format(time.RFC3339),
	}, "", "  ")
	if err != nil {
		return err
	}
	tmp := r.cfg.TokenPath + ".tmp"
	if err := os.WriteFile(tmp, data, 0o600); err != nil {
		return err
	}
	return os.Rename(tmp, r.cfg.TokenPath)
}

// loadPersistedToken returns the stored token plus the metadata that describes it,
// so a restarted scheduler can report WHEN the token it is holding was issued — not
// just that it has one.
func (r *Registrar) loadPersistedToken() (string, persistedMeta) {
	if r.cfg.TokenPath == "" {
		return "", persistedMeta{}
	}
	data, err := os.ReadFile(r.cfg.TokenPath)
	if err != nil {
		return "", persistedMeta{}
	}
	var rec struct {
		AuthToken  string `json:"auth_token"`
		GatewayURL string `json:"gateway_url"`
		persistedMeta
	}
	if err := json.Unmarshal(data, &rec); err != nil {
		return "", persistedMeta{}
	}
	// A token from a DIFFERENT gateway must not leak into this one's polls.
	if rec.GatewayURL != r.cfg.GatewayURL {
		return "", persistedMeta{}
	}
	return rec.AuthToken, rec.persistedMeta
}

type persistedMeta struct {
	WorkerID     string `json:"worker_id"`
	MinerID      string `json:"miner_id"`
	RegisteredAt string `json:"registered_at"`
}
