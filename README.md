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
├── docker-compose.yml      # Compose file (pre-built images)
├── .env.example            # Environment variable template
├── config/                 # Scenario configs (single GPU / 8-GPU / FOC download)
├── src/                    # Source code (go-scheduler, py-inference, foc-bridge, proto)
└── release/                # Staging area for image tarballs (uploaded to GitHub Releases)
```

## Architecture

Three containerized services, one `docker compose up -d`:

| Service | Image | Role | Port |
|---|---|---|---|
| Scheduler | `openmodel-scheduler` | Watches the miner's proving deadlines (Lotus RPC) and WinningPoSt elections (Curio); orders the GPUs to yield/resume | 9090 |
| Inference | `openmodel-inference` | vLLM engine, OpenAI-compatible REST API; one engine per GPU (multi-instance) or one model across GPUs (tensor parallel) | 8000 |
| FOC Bridge | `openmodel-foc-bridge` | Downloads model weights from Filecoin SP retrieval URLs, with streaming sha256 integrity verification | 3100 |

## Requirements

| Item | Requirement |
|---|---|
| OS | Ubuntu 22.04+ |
| GPU | NVIDIA, 10 GB+ VRAM each, driver 535+ |
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
#    sidecar-8gpu-multi.yaml   8 GPUs, one engine per GPU (3–7B models, max throughput)
#    sidecar-8gpu-tensor.yaml  8 GPUs, tensor parallel (14B+ models)
#    sidecar-foc.yaml          single GPU, weights fetched from a Filecoin SP

# 4. Launch
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

To attach this worker to a gateway, register it from the gateway's admin API with
this machine's inference/scheduler URLs and the same worker token configured here
(see the gateway repository's API docs).

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

## What's new in v1.2.0

All changes are backward-compatible; an older gateway simply ignores the new
surfaces (capabilities are negotiated via `/health`).

- **Per-worker authentication**: inference `/v1/*` and scheduler `/ready` +
  `/debug/*` can require a Bearer token (`INFERENCE_API_TOKEN` env / scheduler
  `metrics.auth_token`), so a worker exposed to a public network cannot be used
  around the gateway. Empty token keeps the previous open LAN behaviour.
- **Signed billing receipts**: every served request is attested by a worker-held
  ed25519 key — sha256 of the exact request body, sha256 of the generated text,
  and the token counts, signed and attached to the response. The public key is
  advertised on `/health` (`receipt_pubkey`); the key persists at
  `/models/.openmodel/receipt-ed25519.key`. This makes every charge independently
  verifiable end-to-end (see the openmodel-contracts repository).
- **Stream continuation**: the worker can resume another worker's interrupted
  stream mid-generation, so a mining yield no longer breaks a client's stream —
  the gateway re-dispatches the remainder transparently.
- **Model weight integrity**: FOC downloads verify a pinned sha256 while
  streaming; a mismatch returns 422 and deletes the file, so a corrupted or
  tampered weight never reaches the engine.
- **Predictive readiness**: the `/ready` field described above, kept live
  continuously (not only at state changes).
- **Generation stall guard**: if an engine dies mid-generation, in-flight
  requests are aborted with a retryable error after a per-token stall limit
  (default 60 s) instead of hanging until client timeout.
- **Robustness**: multi-partition WindowPoSt proof detection; engine reload
  retries with process self-restart as a last resort; gRPC hardening against
  malformed status reports.

## Build from source

Source lives under `src/`. Images are built per component:

```bash
docker build -t openmodel-scheduler:latest  src/go-scheduler
docker build -t openmodel-inference:latest  src/py-inference
docker build -t openmodel-foc-bridge:latest src/foc-bridge
```

Note: build the inference image on an x86_64 host with NVIDIA tooling; the
resulting image is ~13 GB (CUDA + PyTorch + vLLM).

## Troubleshooting

- **GPU not detected**: check `nvidia-smi` on the host and that the NVIDIA
  Container Toolkit is installed (`docker info | grep -i nvidia`).
- **Inference stuck loading**: first model load downloads weights; check
  `docker logs openmodel-inference` and your `HF_ENDPOINT` reachability.
- **Scheduler unhealthy**: verify `LOTUS_API_TOKEN` and that the Lotus RPC
  (port 1234) is reachable from the container (`network_mode: host`).
- **503 responses**: the miner is proving; this is by design. Honor
  `Retry-After`.

## Versions

- **v1.2.0** (this release): see "What's new" above.
- v1.1.1: accurate token accounting, SSE streaming fixes, multi-partition
  WindowPoSt detection.
- v1.0.0: initial release.

Compatible gateway: openmodel-gateway v2.x (older gateways work; new billing
features stay dormant without gateway support).
