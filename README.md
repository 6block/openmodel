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
├── GETTING-STARTED-SP.md   # Join the hosted mainnet alpha as a provider
├── SP-GUIDE.md             # Day-2 operations: status, earnings, withdrawal, rules
├── UPDATE-NOTES.md         # Release update notes (read before upgrading)
├── docker-compose.yml      # Compose file (pre-built images)
├── .env.example            # Environment variable template
├── config/                 # Scenario configs (single GPU / 8-GPU / FOC / mainnet alpha / multi-pool)
├── src/                    # Source code (go-scheduler, py-inference, foc-bridge, proto)
└── release/                # Staging area for image tarballs (uploaded to GitHub Releases)
```

## Architecture

Three containerized services, one `docker compose up -d`:

| Service | Image | Role | Port |
|---|---|---|---|
| Scheduler | `openmodel-scheduler` | Watches the miner's proving deadlines (Lotus RPC) and WinningPoSt elections (Curio); orders the GPUs to yield/resume; self-registers with a gateway and runs the worker's TLS front | 9090 (+39443 TLS) |
| Inference | `openmodel-inference` | OpenAI-compatible REST API over four engine kinds — vLLM (text and image understanding), diffusion (image generation), ASR (speech recognition), TTS (speech synthesis); one engine, or several pools each on its own cards (see [Multimodal workers](#multimodal-workers-engines-and-pools)) | 8000 (+38443 TLS) |
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
#    sidecar-mainnet-alpha.yaml join the hosted mainnet alpha (see GETTING-STARTED-SP.md)
#    sidecar-multipool-example.yaml text + image generation + speech pools on one worker

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

On a multi-pool worker, check that every pool came up on its cards:

```bash
curl -s http://localhost:8000/health | python3 -c 'import json,sys; h=json.load(sys.stdin); print(h["engine_state"], h["features"]); [print(p["name"], p["engine_state"], p["devices"], p["models"]) for p in h["multi_gpu"]["pools"]]'
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
- **Quantized weights are served under an id that names the quantization**
  (`…-AWQ`, `…-FP8`, `…-4bit`, …). The worker reads `quantization_config` from
  the model's `config.json` and refuses to start quantized weights under a
  full-precision id (`quantized models must be served under their own id`).
  Models published only in a quantized form (`gpt-oss`, MXFP4) keep their
  official id.

The table lists the text entries whose Filecoin piece registry ships in
`config/sidecar-mainnet-alpha.yaml`. The network's catalog also carries
image-understanding, image-generation and speech models; the live list, each
entry's `modality` and `price_unit`, and the prices are served by the gateway
at `GET /v1/catalog` (or `GET /v1/models`) — read them there before choosing
what to claim.

## Multimodal workers: engines and pools

The inference service runs four engine kinds. Each serves its own endpoint
and is billed in its own unit:

| `engine` | Serves | Endpoint | Billing unit (quantity) | Weights on disk |
|---|---|---|---|---|
| `vllm` | text and image-understanding chat | `/v1/chat/completions` (images as `image_url` content parts) | `token` (prompt + completion; images count as prompt tokens) | `config.json` naming an architecture + weights |
| `diffusion` | image generation | `/v1/images/generations` | `image` (images delivered) | diffusers layout (`model_index.json` + component directories) |
| `asr` | speech recognition | `/v1/audio/transcriptions` (multipart) | `audio_second` (input duration, rounded up, minimum 1) | faster-whisper / CTranslate2 (`model.bin` + `vocabulary.txt` or `vocabulary.json`) |
| `tts` | speech synthesis | `/v1/audio/speech` (WAV, MP3, Opus, AAC, FLAC or PCM) | `character` (input Unicode code points) | Kokoro v1.0 (`kokoro-v1_0.pth` + `voices/`) |

A worker runs either **one engine** (the `inference:` block as in every
single-engine template) or **several pools** under `inference.pools`. Each
pool is a complete engine configuration on its own cards — `engine`, `model`,
`device_ids`, `mode` and its own limits — with a `supported_models` list of
the models it may switch to. Settings written once in the `inference:` block
apply to every pool that does not set its own. A ready-to-fill example lives
in `config/sidecar-multipool-example.yaml`; [SP-GUIDE.md §6](SP-GUIDE.md)
covers switching rules and operations.

- **Disjoint cards.** Two pools listing the same card stop the worker at
  start-up (`GPU <n> is assigned to both pool '<a>' and pool '<b>'`). Small
  speech engines may share a card when **every** pool on it sets
  `share_devices: true`.
- **Routing by model.** A request goes to the pool that has the requested
  model resident or listed in its `supported_models`; a model no pool serves
  is refused (`no pool on this worker serves model '…'`). Requests that name
  no model (`"default"`) go to the first pool.
- **Per-pool limits.** `max_images_per_prompt` / `max_image_pixels` (vision),
  `max_images_per_generation` (image generation, default 1), `max_steps`
  (denoising steps one image request may ask for, default 50) and
  `default_steps` (steps for a request that names none; default by model —
  4 for distilled turbo / lightning / schnell models, 25 otherwise), `max_audio_sec`
  (ASR, default 600), `max_chars` and `tts_voice` (TTS, defaults 4000 and
  `af_heart`). Set them no lower than the per-model caps the gateway
  publishes in `/v1/catalog`, or in-catalog requests are refused here.
- **Status.** `GET /health` reports `features` as the union over pools
  (`vision`, `image_gen`, `asr`, `tts`), every resident model under
  `loaded_models`, and per pool — state, devices, models, features, limits
  and switch targets — under `multi_gpu.pools`.

**Yield and resume across pools.** Mining takes every card, so a yield
pauses all pools at once, and the worker reports `paused` (`engine_state`)
only when **every** pool has released its cards; until then it reports
`unloading`. A graceful yield lets running requests finish first; an
immediate yield stops an image render between denoising steps and speech
work at a segment boundary, and the interrupted request gets a `503` with
`Retry-After` and is not billed. On resume the image and speech pools
reload first (seconds each), then the vLLM pools one at a time; a new yield
arriving mid-resume stops the resume before the next pool loads.

**Receipts for every unit.** Every billed response carries an ed25519-signed
receipt for the gateway (the `X-OM-Receipt` header on non-streaming responses;
the public key is on `/health` as `receipt_pubkey`). Token-billed responses
attest the token counts. Image, speech-recognition and speech-synthesis
responses attest a `quantity` and an `asset_hash`: the sha256 of the delivered
PNG bytes (concatenated in order when several images are returned), of the
returned WAV, or of the uploaded audio for a transcription. The receipt does
not name its unit — the unit comes from the model's catalog entry on the
gateway, which re-measures the quantity and re-hashes the bytes itself and
refuses a receipt that disagrees.

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
  too, so retrying never stacks luck. A model that is already verified answers
  `"status": "already_verified"` — there is nothing to re-examine, and the
  10-minute allowance is not used.

## GPU yield behaviour

- **WindowPoSt** (predictable on-chain deadline): graceful yield starts 5 minutes
  before the proving window, hard stop at 2 minutes; inference resumes when proof
  completion is detected (Curio log watcher, DB polling as fallback).
- **WinningPoSt** (sporadic block election): immediate yield on election; the
  whole yield → proof → resume cycle is about 35 seconds.
- While yielded, the API answers `503` with an honest `Retry-After` estimate.
- `GET /ready` reports `seconds_until_change` — while servable, the seconds until
  the next scheduled yield begins; while mining, the estimated seconds until
  resume. The field is omitted when no estimate exists (for example right after a
  proof completes). Gateways use this to route long requests away from imminent yields.

## What's new in v1.5.0

See [UPDATE-NOTES.md](UPDATE-NOTES.md) for the full list and upgrade steps.
Highlights:

- **Multimodal engines**: image understanding through vLLM, image generation
  (`diffusion`), speech recognition (`asr`) and speech synthesis (`tts`), each
  on its OpenAI-shaped endpoint.
- **Multi-pool workers**: several pools on disjoint cards in one worker, each
  with its own model, limits and switch targets; opt-in card sharing for small
  speech engines; one all-or-nothing mining yield.
- **Asset receipts**: image and speech responses are signed with a
  `quantity` and the sha256 of the delivered (or uploaded) bytes, for billing
  in `image`, `audio_second` and `character` units.
- **Model checks at start-up and switch**: switch targets must match their
  pool's engine kind; quantized weights must be served under an id naming the
  quantization; a failed switch reloads the previous model.
- **Stalled-chain fail-safe**: a Lotus node that answers RPC but whose chain
  head stops advancing (`scheduler.chain_stale_after_sec`, default 600) is
  treated as disconnected, so the GPUs still yield under
  `fail_safe_on_disconnect`.

## Build from source

Source lives under `src/`. Images are built per component:

```bash
docker build -t openmodel-scheduler:latest  src/go-scheduler
docker build -t openmodel-inference:latest  src/py-inference
docker build -t openmodel-foc-bridge:latest src/foc-bridge
```

Note: build the inference image on an x86_64 host with NVIDIA tooling; the
resulting image is ~19 GB (CUDA 13 + PyTorch + vLLM 0.22 + the image-generation and
speech libraries); on GitHub Releases it ships gzip-compressed as split volumes
(see `release/SHA256SUMS.txt` for the list).

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
  `/registration` → `gateway_link` and that scheduler and inference run the
  same release (mixed versions cannot exchange dynamic tokens).
- **`GPU <n> is assigned to both pool '<a>' and pool '<b>'`** at start-up: two
  pools list the same card. Give each pool its own cards, or set
  `share_devices: true` on every pool sharing it (small speech engines only).
- **`model '…' is a <kind> model and cannot be loaded by this pool's '<engine>' engine`**:
  a model is listed in a pool whose engine cannot load it. Move it to a pool
  with the named engine.
- **`quantized models must be served under their own id`**: the weights are
  quantized but the model id names a full-precision model. Serve them under
  the quantized repository's name.
- **Log line `lotus chain head is not advancing`**: Lotus answers RPC but its
  chain has not moved for `scheduler.chain_stale_after_sec` (default 600 s);
  the worker yields as if Lotus were disconnected. Check `lotus sync status`.
- **`gateway_link: unreachable` vs `silent`**: `unreachable` means this worker
  cannot reach the gateway (network/mapping); `silent` means the gateway is
  reachable but no longer polling this worker (deregistered?) — re-register or
  contact the gateway operator.

## Versions

- **v1.5.0** (this release): multimodal engines (image understanding, image
  generation, speech recognition, speech synthesis), multi-pool workers,
  asset receipts, engine-kind and quantization-id checks, switch rollback,
  stalled-chain fail-safe.
- v1.4.0: per-request thinking mode
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

Compatible gateway: openmodel-gateway v2.2+ (asset receipts, image and
speech routing, per-modality admission probes). Text serving works against
v2.1; older gateways work too, with the newer surfaces dormant and the worker
serving with its static token.

## License

Copyright 2026 6block. Licensed under the [Apache License, Version 2.0](LICENSE).
