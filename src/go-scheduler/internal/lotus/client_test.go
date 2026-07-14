package lotus

import (
	"context"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

func testLog() *slog.Logger { return slog.New(slog.NewTextHandler(io.Discard, nil)) }

func newTestClient(t *testing.T, handler http.HandlerFunc) (*RPCClient, *httptest.Server) {
	t.Helper()
	srv := httptest.NewServer(handler)
	// NewRPCClient leaves an http:// URL unchanged, so it points at the test server.
	return NewRPCClient(srv.URL, "tok", "t0182063", testLog()), srv
}

func rpcResult(result string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintf(w, `{"jsonrpc":"2.0","result":%s,"id":1}`, result)
	}
}

func TestNewRPCClientURLRewrite(t *testing.T) {
	cases := map[string]string{
		"ws://host:1234/rpc/v0":  "http://host:1234/rpc/v0",
		"wss://host:1234/rpc/v0": "https://host:1234/rpc/v0",
		"http://host:1234/rpc":   "http://host:1234/rpc", // unchanged
	}
	for in, want := range cases {
		if got := NewRPCClient(in, "", "", testLog()).httpURL; got != want {
			t.Errorf("NewRPCClient(%q).httpURL = %q, want %q", in, got, want)
		}
	}
}

func TestConnect(t *testing.T) {
	// 200 → ok, and the auth header is sent
	c, srv := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer tok" {
			t.Errorf("missing/wrong auth header: %q", r.Header.Get("Authorization"))
		}
		fmt.Fprint(w, `{"jsonrpc":"2.0","result":{"Height":1},"id":1}`)
	})
	defer srv.Close()
	if err := c.Connect(context.Background(), time.Second); err != nil {
		t.Errorf("Connect 200 should succeed, got %v", err)
	}

	// non-200 → error
	c2, srv2 := newTestClient(t, func(w http.ResponseWriter, r *http.Request) { w.WriteHeader(503) })
	defer srv2.Close()
	if err := c2.Connect(context.Background(), time.Second); err == nil {
		t.Error("Connect should error on HTTP 503")
	}
}

func TestGetProvingDeadline(t *testing.T) {
	c, srv := newTestClient(t, rpcResult(`{"CurrentEpoch":10,"Open":20,"Close":30}`))
	defer srv.Close()
	info, err := c.GetProvingDeadline(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if info.CurrentEpoch != 10 || info.Open != 20 || info.Close != 30 {
		t.Errorf("got %+v", info)
	}
}

func TestGetDeadlineSectors(t *testing.T) {
	// two partitions → Partitions/Sectors == 2 (partition-count proxy)
	c, srv := newTestClient(t, rpcResult(`[{},{}]`))
	defer srv.Close()
	ds, err := c.GetDeadlineSectors(context.Background(), 3)
	if err != nil {
		t.Fatal(err)
	}
	if ds.Partitions != 2 || ds.Sectors != 2 || !ds.HasSectors() || ds.Deadline != 3 {
		t.Errorf("got %+v", ds)
	}

	// empty → no sectors
	c2, srv2 := newTestClient(t, rpcResult(`[]`))
	defer srv2.Close()
	ds2, _ := c2.GetDeadlineSectors(context.Background(), 0)
	if ds2.HasSectors() {
		t.Error("empty partitions should not HasSectors")
	}
}

func TestGetMinerBaseInfo(t *testing.T) {
	c, srv := newTestClient(t, rpcResult(`{"EligibleForMining":true,"HasMinPower":true}`))
	defer srv.Close()
	info, err := c.GetMinerBaseInfo(context.Background(), 100, []TipsetCID{{Root: "bafy"}})
	if err != nil {
		t.Fatal(err)
	}
	if !info.EligibleForMining || !info.HasMinPower {
		t.Errorf("got %+v", info)
	}
}

func TestCallRPCError(t *testing.T) {
	c, srv := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, `{"jsonrpc":"2.0","error":{"code":-32000,"message":"boom"},"id":1}`)
	})
	defer srv.Close()
	_, err := c.GetProvingDeadline(context.Background())
	if err == nil {
		t.Fatal("expected an rpc error")
	}
}

func TestCallMalformedJSON(t *testing.T) {
	c, srv := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		io.WriteString(w, "this is not json")
	})
	defer srv.Close()
	if _, err := c.GetProvingDeadline(context.Background()); err == nil {
		t.Fatal("expected an unmarshal error for malformed JSON")
	}
}

func TestSubscribeChainHeadImmediate(t *testing.T) {
	c, srv := newTestClient(t, rpcResult(`{"Height":42}`))
	defer srv.Close()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	ch, err := c.SubscribeChainHead(ctx)
	if err != nil {
		t.Fatal(err)
	}
	select {
	case head := <-ch:
		if head.Height != 42 {
			t.Errorf("height = %d, want 42", head.Height)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("no chain head received from immediate poll")
	}
}
