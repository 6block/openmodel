package lotus

import (
	"context"
	"encoding/base64"
	"fmt"
)

// wallet.go adds the wallet/state calls SP self-registration needs: resolving
// the miner's owner/worker keys and signing the registration message with the
// local Lotus wallet (`Filecoin.WalletSign`). The Lotus API token configured in
// lotus.api_token must carry sign permission (an admin/full token does).

// MinerControlAddrs returns the miner's owner and worker as ID addresses (t0/f0).
func (c *RPCClient) MinerControlAddrs(ctx context.Context) (ownerID, workerID string, err error) {
	var out struct {
		Owner  string `json:"Owner"`
		Worker string `json:"Worker"`
	}
	if err := c.call(ctx, "Filecoin.StateMinerInfo", []interface{}{c.minerAddress, nil}, &out); err != nil {
		return "", "", fmt.Errorf("StateMinerInfo(%s): %w", c.minerAddress, err)
	}
	if out.Owner == "" || out.Worker == "" {
		return "", "", fmt.Errorf("StateMinerInfo(%s): empty owner/worker", c.minerAddress)
	}
	return out.Owner, out.Worker, nil
}

// StateAccountKey resolves an ID address (t0…) to its public-key address (t1/t3).
func (c *RPCClient) StateAccountKey(ctx context.Context, idAddr string) (string, error) {
	var out string
	if err := c.call(ctx, "Filecoin.StateAccountKey", []interface{}{idAddr, nil}, &out); err != nil {
		return "", fmt.Errorf("StateAccountKey(%s): %w", idAddr, err)
	}
	if out == "" {
		return "", fmt.Errorf("StateAccountKey(%s): empty result", idAddr)
	}
	return out, nil
}

// WalletHas reports whether the local Lotus wallet holds the key for addr.
func (c *RPCClient) WalletHas(ctx context.Context, addr string) (bool, error) {
	var out bool
	if err := c.call(ctx, "Filecoin.WalletHas", []interface{}{addr}, &out); err != nil {
		return false, fmt.Errorf("WalletHas(%s): %w", addr, err)
	}
	return out, nil
}

// WalletSign signs data with the local wallet key for addr. Returns the Lotus
// signature type (1 = secp256k1, 2 = bls) and the raw signature bytes — exactly
// what the gateway's registration endpoint verifies.
func (c *RPCClient) WalletSign(ctx context.Context, addr string, data []byte) (sigType uint64, sig []byte, err error) {
	var out struct {
		Type uint64 `json:"Type"`
		Data string `json:"Data"`
	}
	if err := c.call(ctx, "Filecoin.WalletSign", []interface{}{addr, base64.StdEncoding.EncodeToString(data)}, &out); err != nil {
		return 0, nil, fmt.Errorf("WalletSign(%s): %w", addr, err)
	}
	raw, err := base64.StdEncoding.DecodeString(out.Data)
	if err != nil {
		return 0, nil, fmt.Errorf("WalletSign(%s): bad signature encoding: %w", addr, err)
	}
	if len(raw) == 0 {
		return 0, nil, fmt.Errorf("WalletSign(%s): empty signature", addr)
	}
	return out.Type, raw, nil
}
