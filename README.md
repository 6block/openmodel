# OpenModel — SP-side AI Inference Stack

OpenModel turns idle GPU time on Filecoin Storage Provider (SP) machines into AI
inference capacity. The stack serves an OpenAI-compatible API and **automatically
yields the GPUs to mining** (WindowPoSt / WinningPoSt) whenever proofs need them,
then resumes inference once the proof window closes. Mining always wins.

This repository is the **worker (SP-side) stack** — one deployment per GPU
machine. The cross-SP routing gateway with billing and on-chain settlement lives
in [openmodel-gateway](https://github.com/6block/openmodel-gateway); the
settlement smart contract and billing verifier live in
[openmodel-contracts](https://github.com/6block/openmodel-contracts).

## Repository layout

```
openmodel/
├── README.md               # This document (deployment guide)
├── ONBOARDING-ALPHA.md     # Join the hosted mainnet alpha as a provider
├── SP-GUIDE.md             # Day-2 operations: status, earnings, withdrawal, rules
├── UPDATE-NOTES.md         # Release update notes (read before upgrading)
├── docker-compose.yml      # Compose file (pre-built images)
├── .env.example            # Environment variable template
├── config/                 # Scenario configs (single GPU / 8-GPU / FOC / mainnet alpha)
├── src/                    # Source code (go-scheduler, py-inference, foc-bridge, proto)
└── release/                # Staging area for image tarballs (uploaded to GitHub Releases)
```

## Architecture

Three containerized services, one `docker compose up -d`:

| Service | Image | Role | Port |
|---|---|---|---|
| Scheduler | `openmodel-scheduler` | Watches the miner's proving deadlines (Lotus RPC) and WinningPoSt elections (Curio); orders the GPUs to yield/resume; self-registers with a gateway and runs the worker's TLS front | 9090 (+39443 TLS) |
| Inference | `openmodel-inference` | vLLM engine, OpenAI-compatible REST API; one engine per GPU (multi-instance) or one model across GPUs (tensor parallel) | 8000 (+38443 TLS) |
| FOC Bridge | `openmodel-foc-bridge` | Downloads model weights from Filecoin SP retrieval URLs, with streaming sha256 integrity verification | 3100 |

## Requirements

| Item | Requirement |
|---|---|
| OS | Ubuntu 22.04+ |
| GPU | NVIDIA, 10 GB+ VRAM each, **driver 580+** (images are CUDA 13; older 5xx drivers fail at container start — verified: 580.159 runs, 560.35 does not) |
| Docker | 24+ with Compose v2 and NVIDIA Container Toolkit |
| Mining | A running Lotus node (HTTP RPC) and Curio/YugabyteDB |

## Deploy from images

```bash
# 1. Download the image tarballs from GitHub Releases, verify, and load
sha256sum -c SHA256SUMS.txt
docker load -i openmodel-scheduler.tar.gz
docker load -i openmodel-foc-bridge.tar.gz
cat openmodel-inference.tar.gz.part-* | docker load     # large image ships split

# 2. Configure
cp .env.example .env       # set LOTUS_API_TOKEN, SIDECAR_CONFIG, MODEL_CACHE_DIR

# 3. Pick a scenario config (config/, selected via SIDECAR_CONFIG)
#    sidecar-prod-test.yaml    single GPU (default)
#    sidecar-8gpu-multi.yaml   8 GPUs, one engine per GPU (small models, max throughput)
#    sidecar-8gpu-tensor.yaml  8 GPUs, tensor parallel (models too big for one card)
#    sidecar-foc.yaml          single GPU, weights fetched from a Filecoin SP
#    sidecar-mainnet-alpha.yaml join the hosted mainnet alpha (see ONBOARDING-ALPHA.md)

# 4. (Optional but recommended) Fill in the gateway block — see next section

# 5. Launch
docker compose up -d
```

### Verify

```bash
curl http://localhost:9090/health        # scheduler
curl http://localhost:8000/health        # inference (model load takes 1–3 min)

curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer $INFERENCE_API_TOKEN" -H "Content-Type: application/json" \
  -d '{"model":"default","messages":[{"role":"user","content":"hi"}],"max_tokens":16}'
```

## Production model catalog

The network bills per model, so the servable set is a **fixed catalog** — and
which catalog entries a worker offers is the SP's own choice, declared in
`gateway.supported_models` and proven per entry by the admission probe before
any traffic routes to it. The v1 catalog (five entries, one family as the
backbone, context capped at 32k):

| canonical-id | HF repo | quant | weights | local path under /models | typical hardware |
|---|---|---|---|---|---|
| `qwen3-4b` | Qwen/Qwen3-4B-Instruct-2507 | fp16 | ~8 GB | `Qwen--Qwen3-4B-Instruct-2507` | 1× 24 GB, or 2× 10 GB (TP=2) |
| `qwen3-8b` | Qwen/Qwen3-8B | fp16 | ~16 GB | `Qwen--Qwen3-8B` | 1× 24 GB, or 4× 10 GB (TP=4) |
| `gpt-oss-20b` | openai/gpt-oss-20b | mxfp4 (native) | ~13 GB | `openai--gpt-oss-20b` | 1× 24 GB, or 2–4× 10 GB |
| `qwen3-32b` | Qwen/Qwen3-32B-AWQ | awq-int4 | ~19 GB | `Qwen--Qwen3-32B-AWQ` | 1× 24 GB (tight), 4–8× 10 GB (TP) |
| `qwen3.8-27b` | Qwen/Qwen3.8-27B-FP8 | fp8 (Marlin W8A16 on Ampere) | ~28 GB | `Qwen--Qwen3.8-27B-FP8` | 8× 10 GB (TP=8) |

Rules that keep the catalog honest:

- **Quantisation is part of the entry.** Deploy exactly the repo and quant
  listed; a same-name different-quant deployment fails its verification.
- **Claiming is not serving.** Every entry you list in `supported_models` gets
  probed (load + capability) before the gateway routes real traffic to it, and
  is re-checked periodically afterwards. Gated HF repos (none in the current
  catalog) would additionally need `HF_TOKEN` set before first download.
- On 10 GB-class cards the 32k context is tight (tensor-parallel + eager mode
  required); 24 GB-class cards are comfortable. Smaller `max_model_len`
  configurations are a per-SP choice.

## Join a gateway (self-registration)

Since v1.3.0 the worker registers **itself** — no action on the gateway side.
Fill in the `gateway:` block of your chosen sidecar config:

```yaml
gateway:
  url: https://<gateway-host>:<port>   # https strongly recommended over public networks
  gateway_id: <the gateway's advertised id>   # TLS identity of the gateway (not its address)
  worker_id: my-sp-worker-1            # 1–64 chars, starts alphanumeric, then letters/digits/._-
  payout_address: "0x…"                # EVM address your earnings settle to
  public_endpoint: https://<this-host>:38443       # your inference TLS front, as the gateway reaches it
  public_scheduler_url: https://<this-host>:39443  # your scheduler TLS front
  gpu_count: 8
  supported_models:                    # everything you can serve; entries absent on disk are prefetched at start-up
    - "/models/Qwen--Qwen3-4B-Instruct-2507"
  auto_register: true
  # token_path is WRITTEN BY the scheduler (the issued token persists here, 0600)
  # — nothing to prepare. tls_ca_path is the one file YOU place: the CA
  # certificate from your gateway operator's onboarding materials; needed only
  # for the very first registration over https (afterwards the cert store's CA
  # is trusted). Both live under ./data on the host (see docker-compose.yml).
  # Note: this CA is the identity layer of the gateway→worker direction (the
  # mTLS front below) regardless of how the gateway is fronted. For the
  # worker→gateway direction it is one of two trust modes — see tls_system_ca.
  token_path: /data/gateway-token.json
  tls_ca_path: /data/gateway-ca.crt
  # Gateway on a public domain with a real certificate? Drop tls_ca_path and
  # set tls_system_ca: true instead — system trust store, no CA file to place.
  mtls_front:
    enabled: true
    cert_dir: /data/mtls
```

What happens on start-up: the scheduler asks the gateway for a challenge, signs
it with the **miner's key via Lotus** (proof of miner control), and receives a
per-worker bearer token plus — when the gateway runs a registration CA — a
worker TLS certificate, in one round-trip. The token is pushed to the inference
container automatically; certificates renew themselves at half-life. Both TLS
directions verify **identities, not addresses** (worker_id / gateway_id in the
SAN), so public port mappings can change freely without reissuing anything.

Claimed weights are **prefetched at start-up**: every `supported_models` entry
absent from local disk is fetched in the background — through the FOC
piece-CID registry when the entry is configured there, otherwise (HF-form ids
like `Org/Name`) from Hugging Face — so a gateway-triggered switch never
embeds a multi-minute download; it only pays the load time. Two limits:
path-form entries (`/models/…`) without an FOC registry entry cannot be
prefetched (place those weights yourself), and the boot model
(`inference.model`) is fetched before the API comes up — a slow first boot;
pre-place it to skip the wait.

When the FOC fetch fails — e.g. the retrieval provider is down — catalog-shaped
entries **fall back to Hugging Face** via the canonical name transform
(`/models/Org--Name` → `Org/Name`), fetched into the same directory and
honoring `HF_ENDPOINT` / `HF_TOKEN`. The fallback is logged loudly because that
copy is not covered by the FOC-pinned digest; set `HF_FALLBACK=0` to disable it.

Watch your own state — including the gateway's admission verdict on every model
you claim — from this machine alone. `$SCHEDULER_TOKEN` is `metrics.auth_token`
from your sidecar config if you set one; otherwise (once registered) use the
gateway-issued worker token — the `auth_token` field inside the file at
`gateway.token_path` (`/data/gateway-token.json` in the sample; it is 0600 and
container-owned, so read it as
`docker exec openmodel-scheduler cat /data/gateway-token.json`). With a dynamic
token present these endpoints always require auth; they are open only when the
worker has neither a static token nor a registration:

```bash
curl -s -H "Authorization: Bearer $SCHEDULER_TOKEN" http://127.0.0.1:9090/registration
```

- `gateway_link.verdict`: `serving` (gateway is polling you) / `waiting` /
  `unreachable` / `silent`. A rejected re-registration next to `serving` means
  "cannot re-register, but still working" — normal when a gateway raises its
  admission floor; existing registrations are not evicted.
- `model_admission.models[]`: per claimed model — `status`
  (`verified` / `pending` / `failing`), the exact score `floor` you are held
  to, `consecutive_fails` (3 escalate, only after an independent confirmation
  re-test), and evidence-window progress. Gateways probe claimed models with
  exam requests that are **not billed and earn nothing**, and are deliberately
  indistinguishable from user traffic.
- Fixed the underlying problem and want re-examination now:

  ```bash
  curl -s -X POST -H "Authorization: Bearer $SCHEDULER_TOKEN" \
    -H "Content-Type: application/json" \
    -d '{"model":"/models/Qwen--Qwen3-4B-Instruct-2507"}' \
    http://127.0.0.1:9090/registration/reverify
  ```

  One request per model per 10 minutes; the reset clears accumulated evidence
  too, so retrying never stacks luck.

## GPU yield behaviour

- **WindowPoSt** (predictable on-chain deadline): graceful yield starts 5 minutes
  before the proving window, hard stop at 2 minutes; inference resumes when proof
  completion is detected (Curio log watcher, DB polling as fallback).
- **WinningPoSt** (sporadic block election): immediate yield on election; the
  whole yield → proof → resume cycle is about 35 seconds.
- While yielded, the API answers `503` with an honest `Retry-After` estimate.
- `GET /ready` reports `seconds_until_change` — while servable, the seconds until
  the next scheduled yield begins; while mining, the estimated seconds until
  resume. Gateways use this to route long requests away from imminent yields.

## What's new in v1.3.0

See [UPDATE-NOTES.md](UPDATE-NOTES.md) for the full list and upgrade caveats.
Highlights:

- **Self-registration**: challenge → miner-key signature → per-worker token,
  fully worker-initiated; token rotation on re-registration.
- **Certificate-at-registration + built-in TLS fronts** (38443/39443):
  encrypted gateway→worker transport with identities decoupled from addresses;
  automatic renewal at half-life.
- **Worker→gateway HTTPS**: `gateway.url: https://…` verified against the same
  registration CA (identity = gateway_id); `tls_ca_path` bootstraps the first
  registration; fail-closed without a CA.
- **Dynamic per-worker tokens** pushed to the inference container; accepted
  alongside the static token.
- **Claimed model list + on-demand switching**: `supported_models` reaches the
  gateway's model-aware router; local path-named weights resolve correctly.
- **Admission self-service**: per-model verification status, the exact floor,
  and a rate-limited re-verification trigger — all from the worker's own box.
- **Chat template rendering** (fix): conversations render with the model's own
  template — no more transcript-continuation artifacts or leaked reasoning
  blocks; noticeably better answers, especially on Qwen3-class models.

## Build from source

Source lives under `src/`. Images are built per component:

```bash
docker build -t openmodel-scheduler:latest  src/go-scheduler
docker build -t openmodel-inference:latest  src/py-inference
docker build -t openmodel-foc-bridge:latest src/foc-bridge
```

Note: build the inference image on an x86_64 host with NVIDIA tooling; the
resulting image is ~17 GB (CUDA 13 + PyTorch + vLLM 0.22); on GitHub
Releases it ships gzip-compressed as six ~1.5 GB split volumes.

## Troubleshooting

- **`nvidia-container-cli: requirement error: unsatisfied condition: cuda>=13.0`**:
  host driver older than 580. Upgrade the driver, or deploy v1.2.0 images.
- **GPU not detected**: check `nvidia-smi` on the host and that the NVIDIA
  Container Toolkit is installed (`docker info | grep -i nvidia`).
- **Inference stuck loading**: first model load downloads weights; check
  `docker logs openmodel-inference` and your `HF_ENDPOINT` reachability.
- **Scheduler unhealthy**: verify `LOTUS_API_TOKEN` and that the Lotus RPC
  (port 1234) is reachable from the container (`network_mode: host`).
- **503 responses**: the miner is proving; this is by design. Honor
  `Retry-After`.
- **401 from inference after a restart**: the dynamic token is pushed by the
  scheduler within a few seconds of (re)registration; if it persists, check
  `/registration` → `gateway_link` and that scheduler and inference are both
  v1.3.0 (mixed versions cannot exchange dynamic tokens).
- **`gateway_link: unreachable` vs `silent`**: `unreachable` means this worker
  cannot reach the gateway (network/mapping); `silent` means the gateway is
  reachable but no longer polling this worker (deregistered?) — re-register or
  contact the gateway operator.

## Versions

- **v1.4.0** (this release): per-request thinking mode
  (`enable_thinking` → `reasoning_content`), `qwen3.8-27b` catalog entry,
  `tls_system_ca` domain trust mode for worker→gateway, `init: true` in the
  compose file (PID-1 zombie protection), config templates rebuilt around the
  production catalog, fail-fast on engine-shape config errors.
- v1.3.0: see UPDATE-NOTES.md.
- v1.2.0: per-worker auth, signed billing receipts, stream continuation,
  weight integrity, predictive readiness.
- v1.1.1: accurate token accounting, SSE streaming fixes, multi-partition
  WindowPoSt detection.
- v1.0.0: initial release.

Compatible gateway: openmodel-gateway v2.1+ (self-registration, certificate
issuance, admission probes). Older gateways work; the new surfaces stay dormant
and the worker serves with its static token.
