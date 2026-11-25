# OpenModel — Update Notes

## v1.5.0

Worker release for multimodal serving. A worker can now serve image
understanding, image generation, speech recognition and speech synthesis
alongside text, from one registration, in several pools on separate cards.
Image and speech responses are signed with asset receipts so the gateway can
bill them in their own units. Text-only workers upgrade by swapping images —
no configuration change is needed.

### Upgrade steps

1. Download the v1.5.0 image tarballs from GitHub Releases into `release/`,
   verify and load them (image names are unchanged):

   ```bash
   cd release
   sha256sum -c SHA256SUMS.txt
   docker load -i openmodel-scheduler.tar.gz
   docker load -i openmodel-foc-bridge.tar.gz
   cat openmodel-inference.tar.gz.part-* | docker load
   cd ..
   ```

2. Recreate the services (upgrade scheduler and inference together):

   ```bash
   docker compose up -d
   ```

3. Confirm the worker is back:

   ```bash
   curl -s http://127.0.0.1:8000/health
   ```

`docker-compose.yml` and `.env.example` are unchanged; your current sidecar
config keeps working. The single-GPU templates `sidecar-prod-test.yaml` and
`sidecar-foc.yaml` now set `enforce_eager: true`: without it a 10 GB card has
too little memory left for the KV cache once CUDA graphs are captured, and the
engine stops at start-up with `… KV cache is needed, which is larger than the
available KV cache memory`. If your config was copied from one of those two
templates and runs on a 10 GB card, add the same line under `inference:`. Every new configuration
key is optional and defaults to the v1.4.0 behaviour: without
`inference.pools` the worker runs the single engine of the `inference:`
block exactly as before.

### What a text-only worker will notice

Nothing breaks for a worker serving the text catalog as published. Four
behaviours are new and apply to every worker:

- **Quantized weights need a quantized id.** At start-up the worker reads
  `quantization_config` from each model's `config.json`; quantized weights
  served under an id that does not name the quantization stop the worker with
  `quantized models must be served under their own id`. The catalog ids
  (`Qwen--Qwen3-32B-AWQ`, `Qwen--Qwen3.8-27B-FP8`) already name theirs, and
  `gpt-oss` (published only as MXFP4) keeps its official id. Affected only if
  you renamed a quantized directory to a full-precision name.
- **Claimed models must suit the engine.** Claimed models already on disk are
  checked against the engine kind at start-up (and the rest at switch time);
  a text worker claiming an image or speech model stops with
  `model '…' is a <kind> model and cannot be loaded by this pool's 'vllm' engine`.
- **A failed switch rolls back.** If a requested model cannot be loaded, the
  worker reloads the model it had and the request gets
  `500 Model switch failed: …`.
- **Stalled-chain fail-safe.** A Lotus node that answers RPC but whose chain
  head has not advanced for `scheduler.chain_stale_after_sec` (default 600;
  negative disables) is treated as disconnected: with
  `fail_safe_on_disconnect: true` the GPUs yield until the chain moves again.
  The scheduler logs `lotus chain head is not advancing` once when this
  starts, and once when it ends.

### New: multimodal engines

| `engine` | Serves | Endpoint | Billing unit |
|---|---|---|---|
| `vllm` | text, and image understanding (OpenAI content arrays with `image_url` parts) | `/v1/chat/completions` | `token` |
| `diffusion` | image generation (PNG, base64) | `/v1/images/generations` | `image` |
| `asr` | speech recognition and translation into English (multipart upload: WAV, FLAC, MP3, OGG, WebM or M4A) | `/v1/audio/transcriptions`, `/v1/audio/translations` | `audio_second` |
| `tts` | speech synthesis (WAV, MP3, Opus, AAC, FLAC or PCM) | `/v1/audio/speech` | `character` |

New per-engine limits: `max_images_per_prompt` and `max_image_pixels`
(vision), `max_images_per_generation` (image generation, default 1),
`max_steps` (default 50) and `default_steps` (default by model: 4 for distilled
turbo / lightning / schnell models, 25 otherwise),
`max_audio_sec` (default 600), `max_chars` (default 4000) and `tts_voice`
(default `af_heart`). The engine kind of a model is read from its weights on
disk: `model_index.json` (diffusion), `model.bin` with `vocabulary.txt` or
`vocabulary.json` (ASR), a `.pth` file with `voices/` (TTS, Kokoro v1.0),
otherwise a `config.json` naming an architecture (vLLM).

### New: multi-pool workers

`inference.pools` runs several engines in one worker, each pool on its own
cards with its own resident model, limits and `supported_models` (the models
it may switch to). A request goes to the pool that serves its model; a switch
reloads only that pool while the others keep serving. Pools use disjoint
cards; small speech engines may share one card when every pool on it sets
`share_devices: true`. A complete example is
`config/sidecar-multipool-example.yaml` (new); SP-GUIDE.md §6 covers the
rules. Its `foc.real.piece_cid_registry` lists the Filecoin mainnet pieces
and archive sha256 of every model the template claims, so a fresh worker
fetches them from Filecoin with the digest checked.

Mining yields remain all-or-nothing: a yield pauses every pool, and the
worker reports `paused` only once every pool has released its cards. On
resume the image and speech pools reload first (seconds), then the vLLM pools
one at a time.

`GET /health` gains `loaded_models` (every resident model), `features` now
includes `vision`, `image_gen`, `asr` and `tts` as served, and on a
multi-pool worker `multi_gpu.pools` lists each pool's state, devices, models,
features, limits and switch targets.

### New: asset receipts

Image and speech responses carry a signed receipt with a `quantity`
(images, input seconds rounded up, or input characters as Unicode code
points) and an `asset_hash` — the sha256 of the delivered PNG bytes or WAV,
or of the uploaded audio for a transcription. Token receipts are unchanged.
The receipt does not name its unit; the gateway takes it from the model's
catalog entry and re-measures the quantity and the hash itself. Asset
billing needs openmodel-gateway v2.2.0; settlement of the new units needs the
v1.4 settlement contract (openmodel-contracts v1.4.0).

### Other changes

- Model directories in the diffusers layout and Kokoro `.pth` weights are
  recognised as complete downloads.
- FOC piece-CID registry entries accept `parts` (the uploader's
  `foc-registry.json` name) as well as `cids`; the uploader prints the
  archive's sha256 to pin alongside them.
- The uploader (`src/foc-bridge/upload.js`, run from source) takes
  `--provider-id <id>[,<id>…]` to store the parts with the named storage
  providers, one copy each, instead of letting the SDK choose; naming a
  provider skips the SDK's `/pdp/ping` health check.
- `GET /ready` omits `seconds_until_change` when the scheduler has no
  estimate.

### Versions

- Compatible gateway: openmodel-gateway v2.2.0 (multimodal routing, asset
  receipts, per-modality admission probes). Against older gateways the worker
  serves text as before; image and speech endpoints are not routed to it.
- Settlement contract: openmodel-contracts v1.4.0.

---

## v1.3.0

Worker-stack release accompanying the mainnet alpha. The headline: a worker now
**onboards itself** — self-registration with the gateway, certificates issued at
registration, encrypted transport in both directions, and a self-service view of
its admission status. Plus a long-standing inference correctness fix (chat
template rendering) that noticeably improves answer quality.

### Breaking / upgrade notes — read first

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
  direct, fast answers. (Since superseded on main: thinking is now a
  per-request opt-in via `enable_thinking`, still off by default.)

### New: self-registration (worker → gateway)

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

### New: encrypted transport, both directions, no public PKI

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

### New: claimed models, switching, and admission self-service

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
    -d '{"model":"/models/Qwen--Qwen3-4B-Instruct-2507"}' \
    http://127.0.0.1:9090/registration/reverify
  ```

  Rate-limited to one request per model per 10 minutes; a reset clears the
  accumulated evidence too, so retrying can never stack luck.

### New: production model catalog (v1)

Four entries, declared per-SP via `gateway.supported_models` and verified per
entry before routing (see the catalog section in README): `qwen3-4b`,
`qwen3-8b`, `gpt-oss-20b`, `qwen3-32b` — as of this v1.3.0 release; a fifth
entry, `qwen3.8-27b`, was added later on main. Quantisation is pinned per
entry; context is capped at 32k network-wide. All four entries are
probe-verified on network hardware at release.

### Fixes

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

### Versions

- Compatible gateway: openmodel-gateway v2.1+ (self-registration, certificate
  issuance, admission probes). Against older gateways the worker still serves
  with a static token; the new surfaces stay dormant.
- Images: `openmodel-scheduler` ~28 MB, `openmodel-inference` ~17 GB (CUDA 13 +
  PyTorch + vLLM 0.22), `openmodel-foc-bridge` ~253 MB.
