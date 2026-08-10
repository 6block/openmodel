# OpenModel v1.3.0 — Update Notes

Worker-stack release accompanying the M4 alpha. The headline: a worker now
**onboards itself** — self-registration with the gateway, certificates issued at
registration, encrypted transport in both directions, and a self-service view of
its admission status. Plus a long-standing inference correctness fix (chat
template rendering) that noticeably improves answer quality.

## Breaking / upgrade notes — read first

- **GPU driver ≥ 580 required.** Inference images are now built on
  `nvidia/cuda:13.0` (vLLM 0.22). Hosts on 5xx drivers older than 580 fail at
  container start with `nvidia-container-cli: requirement error: unsatisfied
  condition: cuda>=13.0`. Verified: 580.159 runs, 560.35 does not. Either
  upgrade the driver or stay on v1.2.0 images.
- **Upgrade scheduler and inference together.** The internal proto gained the
  dynamic worker-token push. A v1.2.0 inference container cannot receive tokens
  a v1.3.0 scheduler obtains from a gateway; mixed versions leave the worker
  serving only its static token.
- **Chat output changes (for the better).** Conversations are now rendered with
  the model's own chat template. Previously the transcript was joined as bare
  `role: content` text, which instruction-tuned models treat as text to
  continue — visible as the model completing the user's sentence, inventing
  `Assistant:` markers, or (Qwen3-class models) leaking raw `<think>` blocks.
  If you had client-side workarounds for those artifacts, they are no longer
  needed. Reasoning-model "thinking" is disabled at the template level for
  direct, fast answers.

## New: self-registration (worker → gateway)

Configure the `gateway:` block in your sidecar config and the scheduler
registers this worker on start-up — no operator action on the gateway side:

1. Scheduler requests a challenge from the gateway.
2. It signs the challenge with the **miner's own key** via Lotus
   (`Filecoin.WalletSign`) — the registration proves control of the miner ID.
3. The gateway verifies the signature and admission rules (miner exists, meets
   the configured power/balance floor), then issues a **per-worker bearer
   token** (rotated on re-registration) and — when the gateway runs a
   registration CA — a **worker TLS certificate** in the same round-trip.
4. The token is pushed to the inference container automatically (dynamic token;
   accepted alongside the static `INFERENCE_API_TOKEN`, either works).

Registration state is observable locally:

```bash
curl -s -H "Authorization: Bearer $SCHEDULER_TOKEN" http://127.0.0.1:9090/registration
```

Key fields: `gateway_link.verdict` (`serving` / `waiting` / `unreachable` /
`silent`) tells you whether the gateway is actually polling you — a failing
re-registration next to `serving` means "cannot re-register, still working",
which is normal when a gateway raises its admission floor.

## New: encrypted transport, both directions, no public PKI

- **Gateway → worker** (dispatch, polling): the scheduler runs a built-in TLS
  front (`:38443` for inference, `:39443` for the scheduler port) using the
  certificate issued at registration. Identity is the **worker_id** (SAN), not
  the address — operators can remap public ports freely; certificates renew
  automatically at half-life.
- **Worker → gateway** (registration, token issuance, self-view): set
  `gateway.url` to `https://…` and the scheduler verifies the gateway against
  the same registration CA, expecting identity = **gateway_id** (again
  address-decoupled). Trust comes from the certificate store once registered;
  for the very first registration over an untrusted network, point
  `gateway.tls_ca_path` at the CA certificate distributed with your onboarding
  materials. No CA available → TLS dials fail closed.

## New: claimed models, switching, and admission self-service

- `gateway.supported_models` lists every model this worker can serve (the
  loaded one plus switch targets). The gateway can route a request for any
  claimed model and trigger the switch on demand. Weights are resolved by
  their local path (`/models/Org--Name`).
- **Prefetch-on-claim.** At start-up the worker fetches, in the background,
  every `supported_models` entry missing from disk — via its FOC registry
  entry when configured, otherwise (HF-form ids) from Hugging Face. A
  gateway-triggered switch no longer embeds a multi-minute download (measured:
  17.8 s to switch into a prefetched 4B on a 10 GB card), and admission probes
  stop running into download-induced timeouts. The boot model
  (`inference.model`) downloads at first start (blocking; pre-place to skip).
  When the FOC fetch fails, catalog-shaped entries fall back to Hugging Face
  (canonical name transform, same directory, `HF_FALLBACK=0` to disable) —
  logged loudly, as that copy is outside the FOC-pinned digest.
- Gateways with the admission gate enabled hold each claimed model to a scored
  probe before routing real traffic to it (**probe requests are not billed and
  earn nothing** — they are the exam, indistinguishable from user traffic on
  purpose). Your progress is visible in `/registration` under
  `model_admission`: per-model `status` (`verified` / `pending` / `failing`),
  the exact score floor you are held to, consecutive fails (3 escalate, after
  an independent confirmation re-test), and evidence-window progress.
- Fixed something, want re-examination now instead of waiting for the auditor?

  ```bash
  curl -s -X POST -H "Authorization: Bearer $SCHEDULER_TOKEN" \
    -H "Content-Type: application/json" \
    -d '{"model":"/models/Qwen--Qwen2.5-3B-Instruct"}' \
    http://127.0.0.1:9090/registration/reverify
  ```

  Rate-limited to one request per model per 10 minutes; a reset clears the
  accumulated evidence too, so retrying can never stack luck.

## New: production model catalog (v1)

Four entries, declared per-SP via `gateway.supported_models` and verified per
entry before routing (see the catalog section in README): `qwen3-4b`,
`qwen3-8b`, `gpt-oss-20b`, `qwen3-32b`. Quantisation is pinned per entry;
context is capped at 32k network-wide. All four entries are probe-verified on
network hardware at release.

## Fixes

- **Chat template rendering** (see upgrade notes above) — the most visible
  quality change in this release.
- **Path-named model switching**: switching to a model advertised by its local
  path (`/models/X`) previously mangled the path into a directory that cannot
  exist and tried to download instead. Switches now resolve local weights
  first.
- **Truncation-safe defaults**: gateway-side default `max_tokens` now actually
  reaches the worker (answers were silently cut at the worker's own 256
  default), and verbose reasoning models get enough budget to finish.
- **Idle connections no longer break the next request.** The worker's HTTP
  server closed keep-alive connections after 5 s while the gateway kept pooling
  them for 90 s. A request arriving on a connection the worker had already
  closed failed with `502 upstream error: net/http: timeout awaiting response
  headers` — invisible under continuous load (connections never idle long
  enough) but reliably hit by real conversational traffic, where a user reads
  the answer before sending the next message. The worker now holds idle
  connections for 120 s, longer than the pool keeps them, so the two ends can
  no longer disagree about whether a connection is alive.

## Versions

- Compatible gateway: openmodel-gateway v2.1+ (self-registration, certificate
  issuance, admission probes). Against older gateways the worker still serves
  with a static token; the new surfaces stay dormant.
- Images: `openmodel-scheduler` ~28 MB, `openmodel-inference` ~13 GB (CUDA 13 +
  PyTorch + vLLM 0.22), `openmodel-foc-bridge` ~253 MB.
