# OpenModel SP Guide — running a provider day to day

The operations reference for Storage Providers already serving inference:
checking your status, understanding and withdrawing earnings, the platform's
verification rules, and what to do when something looks wrong. For getting
connected in the first place, follow **[ONBOARDING-ALPHA.md](ONBOARDING-ALPHA.md)** — this guide starts
where that one ends.

Hosted mainnet deployment (alpha) used in the examples:

| | |
|---|---|
| Gateway (API, web UI, public queries — publicly-trusted certificate) | `https://openmodel.filfox.info` |
| Settlement contract ([Filfox-verified](https://filfox.info/en/address/0x60D41baEaBe1ABE061AE82c44425debc35bA524A)) | `0x60D41baEaBe1ABE061AE82c44425debc35bA524A` |

---

## 1. Checking your status

### 1.1 On your own machine

The scheduler's status port answers without any gateway round-trip:

```bash
curl -s -H "Authorization: Bearer $SCHED_TOKEN" http://127.0.0.1:9090/registration
```

Fields worth watching: `registered` (did the last registration succeed),
`rotated` (certificate re-issued on re-registration), `attempts` and
`last_error` (empty when healthy), `gateway_link` — `serving` means the
gateway's polls are reaching you; `silent` means it has never managed to, or
stopped (see §7).

### 1.2 As the gateway sees you

Your worker token (issued at registration, in `/data/gateway-token.json`)
authenticates a self-view:

```bash
curl -s https://openmodel.filfox.info/v1/worker/self \
  -H "Authorization: Bearer wk-YOURTOKEN"
```

Returns your `worker_id`, whether the admission gate is on, and per-model
verification status: `verified` (routable), `pending` (not yet probed),
`failing` (accumulating failed checks — see §5). Worker states you may see in
operator dashboards: `idle` / `busy` (servable), `mining` (yielded to
WindowPoSt/WinningPoSt — normal, never penalised), `loading` (model switch in
progress), `offline` (unreachable — §7).

---

## 2. How you earn

1. Every billable request routed to you earns USD at the model's catalog price
   (`GET /v1/catalog`, authenticated with any user API key, shows per-model USD
   per million tokens; the web UI shows the same table).
2. About every 20 minutes the gateway settles finished requests on-chain, one
   batch item per (user wallet, your payout address, token). The FIL/USD rate
   is snapshotted per batch.
3. Credited earnings sit in a **freeze window — 24 h on this deployment** —
   which doubles as the dispute window (§5). After it they mature and become
   withdrawable.
4. The platform fee is `platform_fee_bps` in your earnings detail (currently
   **0** — the alpha takes nothing).

Mining never costs you: requests interrupted by a yield fail over to another
provider and the tokens you did deliver are settled to you.

## 3. Watching your earnings

Public, no auth, keyed by your payout address:

```bash
curl "https://openmodel.filfox.info/api/v1/sp-earnings-detail/0xYOURPAYOUT?limit=100"
```

Top-level: `settled_count` / `settled_earning_usd` (on-chain already),
`pending_count` / `pending_earning_usd` (billed, next batch),
`total_earning_usd`, `platform_fee_bps`. Each item is one request: token
counts, `earning_usd`, and once settled the `tx_hash` + on-chain block of the
batch that carried it. The response never includes the paying client's
identity — earnings transparency is not client surveillance.

The chain is the authority the detail must reconcile with:

- `getTotalEarnings(yourPayout, token)` — credited total (frozen + mature)
- `getWithdrawableEarnings(yourPayout, token)` — what a withdrawal right now
  would pay

Both readable on the Filfox "Read Contract" tab; `token` is the zero address
for FIL.

## 4. Withdrawing earnings

Only the payout wallet itself can withdraw (the contract pays `msg.sender`),
so the wallet needs a little FIL for gas — the whole transaction costs on the
order of **0.000005 FIL** (measured on this deployment: 18.6M gas units).

**Zero-code path**: open the contract on Filfox → "Write Contract" → connect
the payout wallet → `withdrawEarnings(token)` with
`0x0000000000000000000000000000000000000000` for FIL. One call withdraws
everything mature; still-frozen earnings stay and mature on schedule.

**Tooling path** (ethers v6):

```js
const { ethers } = require("ethers");
const c = new ethers.Contract("0x60D41baEaBe1ABE061AE82c44425debc35bA524A", [
  "function withdrawEarnings(address)",
  "function getWithdrawableEarnings(address,address) view returns (uint256)",
], payoutWallet);
console.log("withdrawable:", ethers.formatEther(
  await c.getWithdrawableEarnings(payoutWallet.address, ethers.ZeroAddress)));
await (await c.withdrawEarnings(ethers.ZeroAddress)).wait();
```

Afterwards `getWithdrawableEarnings` reads 0, `getTotalEarnings` keeps only
the still-frozen remainder, and your wallet's native balance grows by the
withdrawn amount minus gas. The `SPWithdrawn` event on the transaction records
the exact figure.

## 5. Platform rules: probes, bans, seizures

The gateway continuously verifies that providers actually serve what they
claim. As an honest SP the only thing you will notice is occasional short
question batches; they are **not billed to anyone and earn nothing** — treat
them as background noise.

- **Admission**: when the admission gate is on, each model you claim must pass
  one probe (a batch of deterministically-scored questions) before routing
  trusts it. A freshly registered worker typically waits one probe interval
  (~5 min) before its first traffic.
- **Spot checks**: at most one routine probe per worker every **3 days**
  (admission and expiry re-checks are separate). Evidence expires after 7 days
  and is silently re-confirmed in place — you keep serving throughout.
- **Verdicts pool across runs**: one bad run cannot convict you; a ban needs
  the pooled score of several windows below the model's floor, plus an
  independent confirmation run. Transport errors (including a mid-probe mining
  yield) are never counted against you.
- **Bans are temporary** (default 1 h) and lift automatically. If your model
  shows `failing` in the self-view after you fixed a misconfiguration, trigger
  an immediate re-check (rate-limited to one per model per 10 min):

```bash
curl -s -X POST https://openmodel.filfox.info/v1/worker/reverify \
  -H "Authorization: Bearer wk-YOURTOKEN" \
  -H "Content-Type: application/json" -d '{"model":"/models/Qwen--Qwen3-4B-Instruct-2507"}'
```

- **Seizures**: only **still-frozen** earnings can ever be confiscated, only
  by the arbiter role, and every seizure must commit an `evidenceHash` to a
  published evidence bundle — it is publicly attributable on-chain. Matured
  earnings can never be taken; the 24 h freeze *is* the dispute window.

## 6. Verifying the operator

The same receipts users verify protect you: every request you served carries
your worker's ed25519 signature, and each settlement batch commits a Merkle
root over those receipts into its on-chain `detailsHash`. To confirm a
request in your earnings detail was settled exactly as recorded:

```bash
curl "https://openmodel.filfox.info/api/v1/receipt-proof/req-THEREQUESTID"
```

and run the five-check offline verifier (`verify-receipt.py` in
[openmodel-contracts](https://github.com/6block/openmodel-contracts)) against
it. The operator cannot inflate or drop your serving record without breaking
signatures it cannot forge.

## 7. Troubleshooting

**`gateway_link: silent`, log shows `remote error: tls: bad certificate`** —
the gateway's connection reaches your scheduler but a TLS side rejects the
other. Verify locally that your worker certificate chain is intact
(`openssl verify -CAfile /data/gateway-ca.crt` against your worker.crt is NOT
the right check — worker certs are signed by the gateway's registration CA,
delivered at registration). In every case seen so far the fix was
configuration on one side or the other, not re-registration; collect
`/registration` output and the exact log line before escalating to the
operator.

**Certificate expiry** — worker certificates live 7 days and the scheduler
renews them automatically at half-life. If `last_error` shows renewal
failures, check that your machine can still reach the gateway URL; a renewed
registration (`rotated: true`) also re-issues the certificate.

**`offline` in the gateway while your stack looks healthy** — the gateway
marks a worker offline after 5 consecutive failed polls (~25 s). Check that
your public endpoint/scheduler port mappings still point at the machine and
that nothing re-bound the ports.

**Model shows `pending` forever** — the auditor probes unverified claims
first, typically within one interval (~5 min). If it stays pending, your
worker may be reporting a `loading`/`mining` state that defers probing, or
the model name you claim does not match the catalog entry byte-for-byte.

**Payout went to the wrong address** — the payout address is bound at
registration, signed by your miner key; re-register with the corrected
address (same worker_id = legitimate rotation) — future settlements follow
the new address, already-settled earnings stay with the old one.

**`docker compose` cannot stop the inference container** — the error reads
`PID <n> is zombie and can not be killed`, and `docker ps` may still report
the container `Up (healthy)` after its main process has already died. An
engine subprocess exited without being reaped, leaving the container's PID 1
a zombie. The compose file ships `init: true` (tini as PID 1) for exactly
this; if you are on an older compose file, add it under the `inference` and
`foc-bridge` services and recreate. To recover an already-stuck container:

```bash
docker rm -f openmodel-inference
docker compose up -d --no-deps inference
```

`--no-deps` keeps the scheduler running (it holds your gateway registration,
so restarting it unnecessarily re-registers the worker). Confirm the fix took
with `docker exec openmodel-inference ps -p 1 -o comm=` — it should print
`docker-init`, not `python`.

## Where to go deeper

- **[ONBOARDING-ALPHA.md](ONBOARDING-ALPHA.md)** — install & register.
- **[settlement-api.md](https://github.com/6block/openmodel-gateway/blob/main/docs/settlement-api.md)** — the full settlement/billing API surface.
- **[USER-GUIDE.md](https://github.com/6block/openmodel-gateway/blob/main/docs/USER-GUIDE.md)** — what your clients see (billing, receipts, refunds).
- **[openmodel-contracts](https://github.com/6block/openmodel-contracts)** — contract source & verifier.
