# OpenModel SP Guide — running a provider day to day

The operations reference for Storage Providers already serving inference:
checking your status, understanding and withdrawing earnings, the platform's
verification rules, and what to do when something looks wrong. For getting
connected in the first place, follow **[GETTING-STARTED-SP.md](GETTING-STARTED-SP.md)** — this guide starts
where that one ends.

Hosted mainnet deployment (alpha) used in the examples:

| | |
|---|---|
| Gateway (API, web UI, public queries — publicly-trusted certificate) | `https://openmodel.filfox.info` |
| Settlement contract (v1.4, [on Filfox](https://filfox.info/en/address/0x29b994f63D91e447945a1cDeA2776167B7415c26)) | `0x29b994f63D91e447945a1cDeA2776167B7415c26` |

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
stopped (see §8).

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
progress), `offline` (unreachable — §8).

---

## 2. How you earn

1. Every billable request routed to you earns USD at the model's catalog
   price, in the model's billing unit:

   | `price_unit` | Models | Quantity billed |
   |---|---|---|
   | `token` | text and image understanding | prompt + completion tokens (images count as prompt tokens) |
   | `image` | image generation | images delivered |
   | `audio_second` | speech recognition | input audio duration, rounded up to whole seconds (minimum 1) |
   | `character` | speech synthesis | input characters (Unicode code points) |

   `GET /v1/catalog`, authenticated with any user API key, lists every model
   with its `modality`, `price_unit` and price: USD per million tokens for
   token models, `unit_price_usd` per unit for the others. Read prices there;
   they are set per model by the gateway operator.
2. About every 20 minutes the gateway settles finished requests on-chain, one
   batch item per (user wallet, your payout address, token). The FIL/USD rate
   is snapshotted per batch. Each batch also records its volume per unit
   (token / image / audio_second / character) on-chain, in the
   `SettlementExecuted` event (`volumeByUnit`) and the network-wide
   `cumulativeVolume` counters.
3. Credited earnings sit in a **freeze window** — its length is
   `earningsFreezeSec()` on the contract — which doubles as the dispute window (§5). After it they mature and become
   withdrawable.
4. The platform fee is `platform_fee_bps` in your earnings detail (read from
   the contract's `platformFeeBps`).

Mining never costs you: requests interrupted by a yield fail over to another
provider and the tokens you did deliver are settled to you. An image render
or speech request cut off by a yield delivers nothing, so it bills nothing
and is retried elsewhere.

## 3. Watching your earnings

Public, no auth, keyed by your payout address:

```bash
curl "https://openmodel.filfox.info/api/v1/sp-earnings-detail/0xYOURPAYOUT?limit=100"
```

Top-level: `settled_count` / `settled_earning_usd` (on-chain already),
`pending_count` / `pending_earning_usd` (billed, next batch),
`total_earning_usd`, `platform_fee_bps`. Each item is one request: `model`,
token counts, `earning_usd`, and once settled the `tx_hash` + on-chain block
of the batch that carried it. Image and speech requests show token counts of
0 — their quantity and unit are in the request's receipt proof (§7). The response never includes the paying client's
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
order of **0.000005 FIL** (about 18.6M gas units).

**Zero-code path**: open the contract on Filfox → "Write Contract" → connect
the payout wallet → `withdrawEarnings(token)` with
`0x0000000000000000000000000000000000000000` for FIL. One call withdraws
everything mature; still-frozen earnings stay and mature on schedule.

**Tooling path** (ethers v6):

```js
const { ethers } = require("ethers");
const c = new ethers.Contract("0x29b994f63D91e447945a1cDeA2776167B7415c26", [
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

- **Quantized weights use their own model id**: if the weights in a model
  directory are quantized (their `config.json` carries a `quantization_config`),
  the id you serve them under must name the quantization — e.g.
  `Qwen--Qwen3-32B-AWQ`, `…-FP8`, `…-AWQ-4bit`. The worker refuses to start
  quantized weights under a full-precision id (`quantized models must be served
  under their own id`), because users pay that id's price for that model's
  quality. Models published only in a quantized form (`gpt-oss`, MXFP4) keep their
  official id.
- **Admission**: when the admission gate is on, each model you claim must pass
  one probe (a batch of deterministically-scored items, 16 by default) before
  routing trusts it. A freshly registered worker typically waits one probe
  interval (~5 min) before its first traffic.
- **What a probe checks, per modality**: the probe travels the same endpoint,
  headers and transport as paying traffic, every item is generated fresh, and
  scoring is a fixed rule — no model acts as judge. The score is compared with
  the model's floor, shown in `/registration`.

  | Modality | Endpoint | Each item |
  |---|---|---|
  | text | `/v1/chat/completions` | a freshly generated reasoning or arithmetic question with a computed answer |
  | vision | `/v1/chat/completions` | a freshly rendered image (solid background, 1–4 circles) sent as an `image_url` part; the answer must name the background colour or the circle count |
  | image generation | `/v1/images/generations` | one 1024×1024 image at 24 steps with a random seed, for a fixed prompt (a red circle on a white background); the image must decode, have the requested size, not be a single colour, and show a light background with one connected red region; repeated or replayed images lower the round's score |
  | speech recognition | `/v1/audio/transcriptions` | a 16 kHz WAV of four random spoken digits; the transcript must contain those digits in order (numerals or words) |
  | speech synthesis | `/v1/audio/speech` | a random English sentence with `response_format: wav` and no voice (your `tts_voice` default); the audio must decode, last plausibly long for the text, not be silent or a flat tone, and differ between inputs; when the gateway has an admitted speech-recognition worker, the audio is transcribed and must contain the requested words |

  Practical consequences: an image-generation pool must render 1024×1024
  within its card's memory (a round of 16 renders takes a few minutes), and a
  TTS pool's default voice must speak English.
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

A model that is already verified answers `"status": "already_verified"`: there is
nothing to re-examine, and the request does not use up the 10-minute allowance.

- **Seizures**: only **still-frozen** earnings can ever be confiscated, only
  by the arbiter role, and every seizure must commit an `evidenceHash` to a
  published evidence bundle — it is publicly attributable on-chain. Matured
  earnings can never be taken; the freeze period (`earningsFreezeSec()` on the
  contract) *is* the dispute window.

## 6. Serving several models: pools and model switching

One worker can run several **pools**, each on its own cards, each with one
engine — `vllm` (text and image understanding), `diffusion` (image generation),
`asr` (speech recognition) or `tts` (speech synthesis). A pool serves one model
at a time; its `supported_models` lists the other models it may switch to. A
trimmed example (`inference:` section of the sidecar config; the complete,
ready-to-fill version with a TTS pool is `config/sidecar-multipool-example.yaml`):

```yaml
inference:
  pools:
    - name: text
      engine: "vllm"
      model: "/models/Qwen--Qwen3-32B-AWQ"
      device_ids: [0, 1, 2, 3]
      mode: "tensor_parallel"
      supported_models: ["/models/Qwen--Qwen3-32B-AWQ", "/models/Qwen--Qwen3-VL-8B-Instruct"]
    - name: image
      engine: "diffusion"
      model: "/models/stabilityai--stable-diffusion-xl-base-1.0"
      device_ids: [4]
      supported_models: ["/models/stabilityai--stable-diffusion-xl-base-1.0", "/models/segmind--SSD-1B"]
    - name: asr
      engine: "asr"
      model: "/models/Systran--faster-whisper-small"
      device_ids: [5]
      supported_models: ["/models/Systran--faster-whisper-small", "/models/Systran--faster-whisper-medium"]
```

- **One pool per card set**: pools use disjoint cards; two pools listing the
  same card stop the worker at startup. Small speech engines may share a card
  when every pool on that card sets `share_devices: true` (the example template
  puts its `asr` and `tts` pools on one card this way).
- **Shared settings**: keys written in the `inference:` block outside `pools`
  (for example `api_port`) apply to every pool that does not set its own;
  `engine`, `model`, `device_ids` and `mode` are always per pool.
- **Claim every model**: `gateway.supported_models` must list each model you
  serve — every pool's resident model and every switch target. Each is admitted
  separately (§5); admitting a non-resident model makes its pool switch to it for
  the probe.
- **How a switch runs**: a request for a pool's switch target makes that pool
  unload its model and load the requested one. Only that pool's models wait
  meanwhile; the other pools keep serving. Image and speech pools reload in a
  few seconds, a vLLM pool in tens of seconds.
- **Same engine kind only**: a pool's switch targets must be models its own
  engine loads. The worker reads the kind from the weights on disk —
  `model_index.json` is a diffusion model; `model.bin` with `vocabulary.txt` or
  `vocabulary.json` is a speech-recognition model; a `.pth` file with a `voices/`
  directory is a speech-synthesis model; a `config.json` naming an architecture is
  a vLLM model. A mismatch stops the worker at startup when the weights are
  already downloaded, and refuses the switch (the pool keeps its current model)
  otherwise, with:
  `model '…' is a diffusion model and cannot be loaded by this pool's 'vllm' engine; list it only in a pool whose engine is 'diffusion'`.
- **A failed load rolls back**: if the requested model cannot be loaded, the pool
  reloads the model it had and the request gets `500 Model switch failed: …`.
- **Speech synthesis**: the `tts` engine loads Kokoro v1.0 weights
  (`kokoro-v1_0.pth`), so a TTS pool switches only between models in that layout.
- **Mining yields every pool**: a WindowPoSt or WinningPoSt yield pauses all
  pools together, and the worker reports `paused` only once every pool has
  released its cards. After the proof, image and speech pools come back first
  (seconds), then vLLM pools one at a time.

After editing pools or claims, recreate the inference service, then restart the
scheduler — it reports the claim list to the gateway when it registers:

```bash
docker compose up -d inference
docker compose restart scheduler
```

Each pool — its `engine_state`, `devices`, resident `models`, `features`,
`limits` and switch targets (`supported_models`) — is listed under
`multi_gpu.pools` in the worker's health output; top-level `features` and
`loaded_models` cover all pools together:

```bash
curl -s http://127.0.0.1:8000/health
```

## 7. Verifying the operator

The same receipts users verify protect you: every request you served carries
your worker's ed25519 signature, and each settlement batch commits a Merkle
root over those receipts into its on-chain `detailsHash`. Token requests
attest their token counts; image and speech requests attest a `quantity`
(images, seconds or characters) and an `asset_hash` — the sha256 of the
delivered PNG or WAV, or of the uploaded audio for a transcription — and
their proof carries the `price_unit` the leaf was committed under. To confirm a
request in your earnings detail was settled exactly as recorded:

```bash
curl "https://openmodel.filfox.info/api/v1/receipt-proof/req-THEREQUESTID"
```

and run the five-check offline verifier (`verify-receipt.py` in
[openmodel-contracts](https://github.com/6block/openmodel-contracts)) against
it. The operator cannot inflate or drop your serving record without breaking
signatures it cannot forge.

## 8. Troubleshooting

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
For an image or speech model, also check that its weights are on disk in the
layout its engine loads (§6) and that `GET /health` lists it in a pool.

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

- **[GETTING-STARTED-SP.md](GETTING-STARTED-SP.md)** — install & register.
- **[settlement-api.md](https://github.com/6block/openmodel-gateway/blob/main/docs/settlement-api.md)** — the full settlement/billing API surface.
- **[USER-GUIDE.md](https://github.com/6block/openmodel-gateway/blob/main/docs/USER-GUIDE.md)** — what your clients see (billing, receipts, refunds).
- **[openmodel-contracts](https://github.com/6block/openmodel-contracts)** — contract source & verifier.
