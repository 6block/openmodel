// selfview.go — surfaces the gateway's admission verdict on THIS worker's
// claimed models, and forwards the SP's "re-examine me now" request.
//
// The verdict (which claims are verified / failing / pending, and how close a
// failing one is to a ban) lives on the gateway; the SP operates the scheduler.
// Without this bridge the SP's only signals were indirect — no traffic on a
// model, or a ban after the fact. The scheduler polls the self-view on the
// link-watch cadence and republishes it under GET /registration, and relays
// POST /registration/reverify with the worker token so the SP never handles
// the token directly.
package spregister

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

// refreshSelfView fetches /v1/worker/self with the current worker token and
// caches the raw JSON for the status endpoint. Called from the link watcher, so
// staleness is bounded by linkCheckInterval.
func (r *Registrar) refreshSelfView(ctx context.Context) {
	tok := r.CurrentToken()
	if tok == "" || r.cfg.GatewayURL == "" {
		return
	}
	reqCtx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(reqCtx, http.MethodGet,
		r.cfg.GatewayURL+"/v1/worker/self", nil)
	if err != nil {
		return
	}
	req.Header.Set("Authorization", "Bearer "+tok)
	resp, err := r.hc.Do(req)
	if err != nil {
		return // link trouble is the link watcher's story; keep the last view
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return // older gateway without the endpoint (404) — nothing to show
	}
	body, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil || !json.Valid(body) {
		return
	}
	r.selfView.Store(body)
}

// SelfView returns the last fetched admission self-view (nil until one landed).
func (r *Registrar) SelfView() json.RawMessage {
	if v, ok := r.selfView.Load().([]byte); ok {
		return v
	}
	return nil
}

// Reverify relays the SP's re-examination request for one claimed model.
// The gateway enforces ownership (token → worker) and the rate limit; this
// just carries the message and returns the gateway's HTTP status and body.
func (r *Registrar) Reverify(ctx context.Context, model string) (int, []byte, error) {
	tok := r.CurrentToken()
	if tok == "" {
		return 0, nil, fmt.Errorf("no worker token held; register first")
	}
	payload, _ := json.Marshal(map[string]string{"model": model})
	reqCtx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(reqCtx, http.MethodPost,
		r.cfg.GatewayURL+"/v1/worker/reverify", bytes.NewReader(payload))
	if err != nil {
		return 0, nil, err
	}
	req.Header.Set("Authorization", "Bearer "+tok)
	req.Header.Set("Content-Type", "application/json")
	resp, err := r.hc.Do(req)
	if err != nil {
		return 0, nil, err
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	return resp.StatusCode, body, nil
}
