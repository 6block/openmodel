package lotus

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strings"
	"sync/atomic"
	"time"
)

// Client defines the interface for communicating with a Lotus Miner node.
type Client interface {
	GetProvingDeadline(ctx context.Context) (*DeadlineInfo, error)
	GetDeadlineSectors(ctx context.Context, deadlineIdx uint64) (*DeadlineSectors, error)
	GetMinerBaseInfo(ctx context.Context, epoch int64, tsk []TipsetCID) (*MinerBaseInfo, error)
	SubscribeChainHead(ctx context.Context) (<-chan *ChainHead, error)
	Close() error
}

// RPCClient communicates with a real Lotus node.
// Regular RPC calls use HTTP POST (reliable, no compression issues).
// ChainNotify subscription uses WebSocket (needs server push).
type RPCClient struct {
	httpURL      string // http://localhost:1234/rpc/v0
	apiToken     string
	minerAddress string
	httpClient   *http.Client
	requestID    atomic.Int64
	logger       *slog.Logger
}

// NewRPCClient creates a new Lotus RPC client.
func NewRPCClient(apiURL, apiToken, minerAddress string, logger *slog.Logger) *RPCClient {
	// Derive HTTP URL from WebSocket URL
	httpURL := apiURL
	httpURL = strings.Replace(httpURL, "ws://", "http://", 1)
	httpURL = strings.Replace(httpURL, "wss://", "https://", 1)

	return &RPCClient{
		httpURL:      httpURL,
		apiToken:     apiToken,
		minerAddress: minerAddress,
		httpClient: &http.Client{
			Timeout: 30 * time.Second,
		},
		logger: logger,
	}
}

// Connect verifies connectivity to the Lotus node via HTTP.
func (c *RPCClient) Connect(ctx context.Context, timeout time.Duration) error {
	// Test with a simple ChainHead call
	reqBody := jsonRPCRequest{
		JSONRPC: "2.0",
		Method:  "Filecoin.ChainHead",
		Params:  []interface{}{},
		ID:      1,
	}
	body, _ := json.Marshal(reqBody)

	req, err := http.NewRequestWithContext(ctx, "POST", c.httpURL, bytes.NewReader(body))
	if err != nil {
		return fmt.Errorf("connect to lotus: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	if c.apiToken != "" {
		req.Header.Set("Authorization", "Bearer "+c.apiToken)
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("connect to lotus: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		return fmt.Errorf("connect to lotus: HTTP %d", resp.StatusCode)
	}

	c.logger.Info("connected to lotus miner", "url", c.httpURL)
	return nil
}

type jsonRPCRequest struct {
	JSONRPC string        `json:"jsonrpc"`
	Method  string        `json:"method"`
	Params  []interface{} `json:"params"`
	ID      int64         `json:"id"`
}

type jsonRPCResponse struct {
	JSONRPC string          `json:"jsonrpc"`
	Result  json.RawMessage `json:"result"`
	Error   *jsonRPCError   `json:"error,omitempty"`
	ID      int64           `json:"id"`
}

type jsonRPCError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
}

// call makes an HTTP JSON-RPC call to Lotus.
func (c *RPCClient) call(ctx context.Context, method string, params []interface{}, result interface{}) error {
	id := c.requestID.Add(1)

	if params == nil {
		params = []interface{}{}
	}

	reqBody := jsonRPCRequest{
		JSONRPC: "2.0",
		Method:  method,
		Params:  params,
		ID:      id,
	}
	body, err := json.Marshal(reqBody)
	if err != nil {
		return fmt.Errorf("marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", c.httpURL, bytes.NewReader(body))
	if err != nil {
		return fmt.Errorf("create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	if c.apiToken != "" {
		req.Header.Set("Authorization", "Bearer "+c.apiToken)
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("http request: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("read response: %w", err)
	}

	var rpcResp jsonRPCResponse
	if err := json.Unmarshal(respBody, &rpcResp); err != nil {
		return fmt.Errorf("unmarshal response: %w", err)
	}

	if rpcResp.Error != nil {
		return fmt.Errorf("rpc error %d: %s", rpcResp.Error.Code, rpcResp.Error.Message)
	}

	if result != nil {
		if err := json.Unmarshal(rpcResp.Result, result); err != nil {
			return fmt.Errorf("unmarshal result: %w", err)
		}
	}
	return nil
}

// lotusCallTimeout bounds each Lotus RPC so a hung (connected-but-unresponsive)
// daemon triggers the fail-safe within seconds instead of stalling the poll tick up
// to the 30s HTTP client timeout (audit MEDIUM fix). Context cancellation aborts the
// in-flight request (call() uses http.NewRequestWithContext).
const lotusCallTimeout = 5 * time.Second

func (c *RPCClient) GetProvingDeadline(ctx context.Context) (*DeadlineInfo, error) {
	ctx, cancel := context.WithTimeout(ctx, lotusCallTimeout)
	defer cancel()
	var info DeadlineInfo
	err := c.call(ctx, "Filecoin.StateMinerProvingDeadline", []interface{}{c.minerAddress, nil}, &info)
	if err != nil {
		return nil, fmt.Errorf("get proving deadline: %w", err)
	}
	return &info, nil
}

func (c *RPCClient) GetDeadlineSectors(ctx context.Context, deadlineIdx uint64) (*DeadlineSectors, error) {
	ctx, cancel := context.WithTimeout(ctx, lotusCallTimeout)
	defer cancel()
	var partitions []json.RawMessage
	err := c.call(ctx, "Filecoin.StateMinerPartitions",
		[]interface{}{c.minerAddress, deadlineIdx, nil}, &partitions)
	if err != nil {
		return nil, fmt.Errorf("get deadline sectors: %w", err)
	}

	// Note: actual sector count requires parsing RLE+ bitfields from each
	// partition, which is complex. Since Sectors is only used for > 0 checks
	// (HasSectors), partition count is a correct proxy: partitions > 0
	// implies sectors > 0.
	return &DeadlineSectors{
		Deadline:   deadlineIdx,
		Partitions: len(partitions),
		Sectors:    len(partitions), // proxy: partitions > 0 ↔ sectors > 0
		Faults:     0,
	}, nil
}

func (c *RPCClient) GetMinerBaseInfo(ctx context.Context, epoch int64, tsk []TipsetCID) (*MinerBaseInfo, error) {
	var info MinerBaseInfo
	var tskParam interface{} = nil
	if len(tsk) > 0 {
		tskParam = tsk
	}
	err := c.call(ctx, "Filecoin.MinerGetBaseInfo", []interface{}{c.minerAddress, epoch, tskParam}, &info)
	if err != nil {
		return nil, fmt.Errorf("get miner base info: %w", err)
	}
	return &info, nil
}

// SubscribeChainHead polls Filecoin.ChainHead via HTTP to detect new epochs.
// This avoids WebSocket compression issues with Lotus's permessage-deflate.
// Polls every 15 seconds (half an epoch) to catch new blocks promptly.
func (c *RPCClient) SubscribeChainHead(ctx context.Context) (<-chan *ChainHead, error) {
	ch := make(chan *ChainHead, 16)

	go func() {
		defer close(ch)

		var lastHeight int64
		ticker := time.NewTicker(15 * time.Second)
		defer ticker.Stop()

		// Immediate first check
		if head, err := c.getChainHead(ctx); err == nil {
			lastHeight = head.Height
			select {
			case ch <- head:
			case <-ctx.Done():
				return
			}
		}

		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				head, err := c.getChainHead(ctx)
				if err != nil {
					c.logger.Warn("chain head poll failed", "error", err)
					continue
				}
				if head.Height > lastHeight {
					lastHeight = head.Height
					select {
					case ch <- head:
					case <-ctx.Done():
						return
					}
				}
			}
		}
	}()

	c.logger.Info("chain head polling started (HTTP, interval=15s)")
	return ch, nil
}

// getChainHead fetches the current chain head height and tipset key via HTTP.
func (c *RPCClient) getChainHead(ctx context.Context) (*ChainHead, error) {
	var result ChainHead
	err := c.call(ctx, "Filecoin.ChainHead", []interface{}{}, &result)
	if err != nil {
		return nil, err
	}
	return &result, nil
}

func (c *RPCClient) Close() error {
	// HTTP client doesn't need explicit close
	return nil
}
