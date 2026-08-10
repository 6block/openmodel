package spregister

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// fakeWallet implements LotusWallet.
type fakeWallet struct {
	ownerID, workerID   string
	ownerKey, workerKey string
	has                 map[string]bool
	hasErr              error
	signedMsgs          []string
	sigType             uint64
	sig                 []byte
}

func (f *fakeWallet) MinerControlAddrs(context.Context) (string, string, error) {
	return f.ownerID, f.workerID, nil
}
func (f *fakeWallet) StateAccountKey(_ context.Context, id string) (string, error) {
	switch id {
	case f.ownerID:
		return f.ownerKey, nil
	case f.workerID:
		return f.workerKey, nil
	}
	return "", fmt.Errorf("unknown id %s", id)
}
func (f *fakeWallet) WalletHas(_ context.Context, addr string) (bool, error) {
	if f.hasErr != nil {
		return false, f.hasErr
	}
	return f.has[addr], nil
}
func (f *fakeWallet) WalletSign(_ context.Context, addr string, data []byte) (uint64, []byte, error) {
	f.signedMsgs = append(f.signedMsgs, string(data))
	return f.sigType, f.sig, nil
}

func testLogger() *slog.Logger {
	return slog.New(slog.NewTextHandler(io.Discard, nil))
}

// TestMessageGolden pins the exact signed text. The gateway side
// (sp-state-agent internal/gateway/spregister.go buildSPRegistrationMessage)
// carries the SAME golden test — if either side drifts, its golden test fails
// before any cross-machine debugging is needed.
func TestMessageGolden(t *testing.T) {
	got := buildRegistrationMessage(
		"t01000", "openmodel-test-gw", "sp-t01000",
		"0x9965507D1a55bcC2695C58ba16FB37d819B0A4dc",
		"http://198.51.100.7:8000", "http://198.51.100.7:8090", "abc123")
	want := "OpenModel SP registration\n" +
		"miner: t01000\n" +
		"gateway: openmodel-test-gw\n" +
		"worker_id: sp-t01000\n" +
		"payout: 0x9965507D1a55bcC2695C58ba16FB37d819B0A4dc\n" +
		"endpoint: http://198.51.100.7:8000\n" +
		"scheduler_url: http://198.51.100.7:8090\n" +
		"challenge: abc123"
	if got != want {
		t.Fatalf("message drifted from the gateway contract:\n got: %q\nwant: %q", got, want)
	}
}

func TestPickSigner(t *testing.T) {
	w := &fakeWallet{
		ownerID: "t0100", workerID: "t0101",
		ownerKey: "t1owner", workerKey: "t3worker",
		has: map[string]bool{"t3worker": true, "t1owner": true},
	}
	r := New(Config{MinerID: "t01000"}, w, testLogger())

	// Worker key preferred.
	signer, err := r.pickSigner(context.Background())
	if err != nil || signer != "t3worker" {
		t.Fatalf("signer = %q, %v (want worker key)", signer, err)
	}

	// Owner fallback when the worker key is absent.
	w.has = map[string]bool{"t1owner": true}
	signer, err = r.pickSigner(context.Background())
	if err != nil || signer != "t1owner" {
		t.Fatalf("signer = %q, %v (want owner key)", signer, err)
	}

	// Neither key present → actionable error.
	w.has = map[string]bool{}
	if _, err = r.pickSigner(context.Background()); err == nil {
		t.Fatal("expected error when no key is locally held")
	}
}

// fakeGateway mimics the gateway's challenge/register endpoints and records the
// register payload for assertions.
type fakeGateway struct {
	challenge    string
	lastRegister map[string]any
	failRegister int // fail this many register calls with 500 before succeeding
	token        string
}

func (fg *fakeGateway) handler(t *testing.T) http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/sp/challenge", func(w http.ResponseWriter, r *http.Request) {
		var req map[string]string
		_ = json.NewDecoder(r.Body).Decode(&req)
		if req["miner_id"] == "" {
			http.Error(w, "no miner", 400)
			return
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"challenge": fg.challenge, "expires_at": 9999999999})
	})
	mux.HandleFunc("/v1/sp/register", func(w http.ResponseWriter, r *http.Request) {
		if fg.failRegister > 0 {
			fg.failRegister--
			http.Error(w, "boom", 500)
			return
		}
		body, _ := io.ReadAll(r.Body)
		fg.lastRegister = map[string]any{}
		_ = json.Unmarshal(body, &fg.lastRegister)
		_ = json.NewEncoder(w).Encode(map[string]any{
			"worker_id":      fg.lastRegister["worker_id"],
			"auth_token":     fg.token,
			"miner_id":       fg.lastRegister["miner_id"],
			"payout_address": fg.lastRegister["payout_address"],
			"state":          "registered",
			"rotated":        false,
		})
	})
	return mux
}

func TestRegisterOnceHappyPath(t *testing.T) {
	fg := &fakeGateway{challenge: "cafebabe", token: "wk-issued-token"}
	srv := httptest.NewServer(fg.handler(t))
	defer srv.Close()

	wallet := &fakeWallet{
		ownerID: "t0100", workerID: "t0101",
		ownerKey: "t1owner", workerKey: "t3worker",
		has:     map[string]bool{"t3worker": true},
		sigType: 2, sig: []byte("fake-bls-signature"),
	}
	tokenPath := filepath.Join(t.TempDir(), "sub", "gateway-token.json")
	r := New(Config{
		GatewayURL:         srv.URL,
		GatewayID:          "openmodel-test-gw",
		WorkerID:           "sp-t01000",
		MinerID:            "t01000",
		PayoutAddress:      "0x9965507D1a55bcC2695C58ba16FB37d819B0A4dc",
		PublicEndpoint:     "http://198.51.100.7:8000",
		PublicSchedulerURL: "http://198.51.100.7:8090",
		GPUCount:           1,
		TokenPath:          tokenPath,
	}, wallet, testLogger())

	res, err := r.RegisterOnce(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if res.AuthToken != "wk-issued-token" || r.CurrentToken() != "wk-issued-token" {
		t.Fatalf("token not adopted: %+v / %q", res, r.CurrentToken())
	}

	// The signed message must be the canonical text with the server challenge.
	if len(wallet.signedMsgs) != 1 {
		t.Fatalf("signed %d messages", len(wallet.signedMsgs))
	}
	wantMsg := buildRegistrationMessage("t01000", "openmodel-test-gw", "sp-t01000",
		"0x9965507D1a55bcC2695C58ba16FB37d819B0A4dc",
		"http://198.51.100.7:8000", "http://198.51.100.7:8090", "cafebabe")
	if wallet.signedMsgs[0] != wantMsg {
		t.Fatalf("signed message mismatch:\n got %q\nwant %q", wallet.signedMsgs[0], wantMsg)
	}

	// Register payload carries the signer + Lotus-form signature.
	if fg.lastRegister["signer"] != "t3worker" || fg.lastRegister["challenge"] != "cafebabe" {
		t.Fatalf("register payload: %+v", fg.lastRegister)
	}
	sig := fg.lastRegister["signature"].(map[string]any)
	if sig["type"].(float64) != 2 {
		t.Fatalf("signature type: %+v", sig)
	}
	if raw, _ := base64.StdEncoding.DecodeString(sig["data"].(string)); string(raw) != "fake-bls-signature" {
		t.Fatalf("signature data: %+v", sig)
	}

	// Token persisted 0600 and reloadable across a restart.
	st, err := os.Stat(tokenPath)
	if err != nil {
		t.Fatal(err)
	}
	if st.Mode().Perm() != 0o600 {
		t.Fatalf("token file mode %v, want 0600", st.Mode().Perm())
	}
	r2 := New(Config{GatewayURL: srv.URL, TokenPath: tokenPath}, wallet, testLogger())
	if r2.CurrentToken() != "wk-issued-token" {
		t.Fatalf("persisted token not reloaded: %q", r2.CurrentToken())
	}

	// A token persisted for a DIFFERENT gateway must not be adopted.
	r3 := New(Config{GatewayURL: "http://other-gateway:3000", TokenPath: tokenPath}, wallet, testLogger())
	if r3.CurrentToken() != "" {
		t.Fatal("token from another gateway leaked in")
	}
}

func TestRegisterOnceSurfacesGatewayRejection(t *testing.T) {
	fg := &fakeGateway{challenge: "c1", token: "wk-x", failRegister: 1}
	srv := httptest.NewServer(fg.handler(t))
	defer srv.Close()

	wallet := &fakeWallet{
		ownerID: "t0100", workerID: "t0101",
		ownerKey: "t1owner", workerKey: "t3worker",
		has:     map[string]bool{"t3worker": true},
		sigType: 2, sig: []byte("s"),
	}
	r := New(Config{GatewayURL: srv.URL, GatewayID: "g", WorkerID: "w", MinerID: "t01000"}, wallet, testLogger())

	if _, err := r.RegisterOnce(context.Background()); err == nil {
		t.Fatal("expected first attempt to surface the gateway 500")
	}
	if r.CurrentToken() != "" {
		t.Fatal("token set despite failure")
	}
	// Next attempt (fresh challenge) succeeds — the retry loop's contract.
	if _, err := r.RegisterOnce(context.Background()); err != nil {
		t.Fatal(err)
	}
	if r.CurrentToken() != "wk-x" {
		t.Fatal("token not adopted after retry")
	}
}

// A failed wallet probe (e.g. Lotus API token missing → HTTP 401) must surface
// as a probe failure, NOT as "key not in wallet": during the alpha clean-room
// run the swallowed 401 pointed the operator at a key that was provably present.
func TestPickSignerReportsProbeFailure(t *testing.T) {
	f := &fakeWallet{
		ownerID: "t0100", workerID: "t0101",
		ownerKey: "t3owner", workerKey: "t3worker",
		hasErr: fmt.Errorf("WalletHas(t3worker): HTTP 401"),
	}
	r := &Registrar{lotus: f, cfg: Config{MinerID: "t01000"}, logger: testLogger()}
	_, err := r.pickSigner(context.Background())
	if err == nil {
		t.Fatal("expected an error")
	}
	if !strings.Contains(err.Error(), "could not query the local Lotus wallet") ||
		!strings.Contains(err.Error(), "api_token") {
		t.Fatalf("probe failure misreported: %v", err)
	}
	if strings.Contains(err.Error(), "is in the local Lotus wallet") {
		t.Fatalf("probe failure still reads as a missing key: %v", err)
	}

	// The genuine absent-key path (probes succeed, both keys missing) keeps its
	// original actionable message.
	f2 := &fakeWallet{ownerID: "t0100", workerID: "t0101",
		ownerKey: "t3owner", workerKey: "t3worker", has: map[string]bool{}}
	r2 := &Registrar{lotus: f2, cfg: Config{MinerID: "t01000"}, logger: testLogger()}
	if _, err2 := r2.pickSigner(context.Background()); err2 == nil ||
		!strings.Contains(err2.Error(), "is in the local Lotus wallet") {
		t.Fatalf("absent-key message changed: %v", err2)
	}
}
