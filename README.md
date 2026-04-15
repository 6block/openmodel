# OpenModel Docker Deployment Guide

## Deliverables

```
openmodel-delivery/
├── README.md                  # This document
├── docker-compose.yml         # Compose file (pre-built images)
├── .env.example               # Environment variable template
├── config/                    # Configuration files
│   ├── sidecar-prod-test.yaml     # Single GPU production test
│   ├── sidecar-8gpu-multi.yaml    # 8 GPU multi-instance (recommended for small models)
│   ├── sidecar-8gpu-tensor.yaml   # 8 GPU tensor parallel (for large 14B+ models)
│   └── sidecar-foc.yaml           # FOC model download test
└── images/                    # Docker image tar archives
    ├── openmodel-scheduler.tar
    ├── openmodel-inference.tar
    └── openmodel-foc-bridge.tar
```

## 1. Prerequisites

| Item | Requirement |
|------|-------------|
| OS | Ubuntu 22.04+ or compatible Linux distribution |
| GPU | NVIDIA GPU with 10GB+ VRAM (RTX 3080 or higher) |
| NVIDIA Driver | 535+ |
| Docker | 24+ |
| NVIDIA Container Toolkit | Installed and configured |
| Lotus Miner | Curio node running with accessible API |
| YugabyteDB | Curio DB accessible |

### Installing NVIDIA Container Toolkit (if not installed)

```bash
# Add NVIDIA repository
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Verify
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

## 2. Import Images

```bash
# Download image tar files from the GitHub Release assets into the images/ directory
# The inference image is split into multiple parts due to size constraints

# Reassemble the inference image
cat images/openmodel-inference.tar.part_* > images/openmodel-inference.tar

# Load all three images
docker load -i images/openmodel-scheduler.tar
docker load -i images/openmodel-inference.tar
docker load -i images/openmodel-foc-bridge.tar

# Verify images are loaded
docker images | grep openmodel

# Clean up split files (optional)
rm -f images/openmodel-inference.tar.part_*
```

## 3. Configure Environment Variables

```bash
# Copy the template
cp .env.example .env

# Edit environment variables
vim .env
```

### Required Variables

| Variable | Description |
|----------|-------------|
| `LOTUS_API_TOKEN` | Lotus daemon API token |
| `SIDECAR_CONFIG` | Configuration filename from `config/` directory |
| `MODEL_CACHE_DIR` | Model cache directory (shared by inference and foc-bridge) |

### Optional Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_ENDPOINT` | `https://huggingface.co` | HuggingFace endpoint |
| `HF_TOKEN` | empty | HuggingFace token (required for gated models) |
| `HF_CACHE_DIR` | `~/.cache/huggingface` | HuggingFace cache directory |
| `FOC_PRIVATE_KEY` | empty | FOC chain private key |
| `FOC_RPC_URL` | calibration testnet | FOC RPC URL |
| `FOC_BRIDGE_PORT` | `3100` | FOC Bridge port |

## 4. Choose Configuration

Configuration files under `config/` are designed for different scenarios:

| Config File | Scenario | Description |
|-------------|----------|-------------|
| `sidecar-prod-test.yaml` | Single GPU test | Uses GPU 0, suitable for deployment verification |
| `sidecar-8gpu-multi.yaml` | 8 GPU multi-instance | One inference engine per GPU, high throughput (recommended for small models) |
| `sidecar-8gpu-tensor.yaml` | 8 GPU tensor parallel | Multi-GPU collaboration, suitable for 14B+ large models |
| `sidecar-foc.yaml` | FOC test | Model download functionality test |

You can copy and modify configuration files to match your environment (GPU count, model selection, Lotus port, Curio log path, yield thresholds, etc.).

### Key Configuration Items

- `lotus.miner_address`: Your miner address
- `lotus.api_url`: Lotus daemon WebSocket address
- `curio.dsn`: YugabyteDB connection string
- `curio.log_path`: Curio log path (required for WinningPoSt detection)
- `inference.model`: Model to load (e.g., `Qwen/Qwen2.5-3B-Instruct`)
- `inference.gpu_memory_utilization`: GPU memory utilization (0.80-0.90)
- `inference.multi_gpu.mode`: `tensor_parallel` or `multi_instance`
- `inference.multi_gpu.device_ids`: Which GPUs to use

## 5. Start Services

```bash
# Confirm Curio is running and the log file exists
ls -l /tmp/curio.log

# Start all services
docker compose up -d

# Or start with a specific configuration
SIDECAR_CONFIG=sidecar-8gpu-multi.yaml docker compose up -d
```

## 6. Verify Deployment

```bash
# Check container status (all should be running/healthy)
docker compose ps

# Check scheduler health
curl http://localhost:9090/health

# Check inference service (model loading may take 1-3 minutes)
curl http://localhost:8000/health

# Check FOC Bridge
curl http://localhost:3100/health

# Send a test inference request
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-3B-Instruct",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## 7. Operations

### View Logs

```bash
# View all logs
docker compose logs -f

# View individual service logs
docker compose logs -f scheduler
docker compose logs -f inference
docker compose logs -f foc-bridge

# View logs from the last 5 minutes
docker compose logs --since 5m
```

### Monitor GPU

```bash
watch -n 2 nvidia-smi
```

### Switch Configuration

No need to re-import images. Change `SIDECAR_CONFIG` in `.env` and restart:

```bash
# After modifying SIDECAR_CONFIG in .env
docker compose restart

# Or specify directly
SIDECAR_CONFIG=sidecar-8gpu-tensor.yaml docker compose up -d
```

### Stop Services

```bash
docker compose down
```

## 8. Architecture

```
+----------------+    JSON-RPC/HTTP    +----------------+     gRPC      +----------------+
| Lotus Miner    |<------------------>| Go Scheduler    |<------------>|    Python       |
|   (existing)   | StateMinerProving  |                | Schedule     | Inference Svc   |
|                | MinerGetBaseInfo   |  :50051 gRPC   | Events       |                |
+----------------+                    |  :9090 metrics |              |  :8000 REST    |
                                      +----------------+              +--------+-------+
+----------------+                                                             |
| Curio DB       |  proof completion                                       REST API
|  (YugabyteDB)  |  detection                                        (OpenAI compatible)
+----------------+                                                             |
                                      +----------------+              +--------v-------+
                                      | FOC Bridge     |              | AI Consumers    |
                                      |  :3100 HTTP    |              +----------------+
                                      +----------------+
```

### Service Ports

| Service | Port | Purpose |
|---------|------|---------|
| scheduler | 50051 | gRPC (internal communication) |
| scheduler | 9090 | Health check / Metrics |
| inference | 8000 | REST API (OpenAI compatible) |
| foc-bridge | 3100 | HTTP API |

## 9. API Endpoints

### OpenAI-Compatible REST API (`:8000`)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Chat completions |
| `/v1/completions` | POST | Text completions |
| `/v1/models` | GET | List loaded models |
| `/health` | GET | Engine status + multi-GPU details (JSON) |

During GPU yield, the inference API returns `503 Service Unavailable`:

```json
{
  "error": {
    "message": "GPU yielded to mining (paused). Retry later.",
    "type": "service_unavailable"
  }
}
```

The response includes a `Retry-After` header (15s during model loading, 60s during mining yield).

### Scheduler Endpoints (`:9090`)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Scheduler health status (returns `ok`) |
| `/ready` | GET | Current GPU state |
| `/metrics` | GET | Prometheus metrics (see below) |

### Debug / Test Endpoints (`:9090`)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/debug/trigger-winning-post` | POST | Manually trigger WinningPoSt yield (for testing) |
| `/debug/sector-cache` | GET | View sector cache, GPU state, active deadlines |
| `/debug/winning-post-status` | GET | Query recent WinningPoSt block wins |

Yield test example:

```bash
# Trigger WinningPoSt yield
curl -X POST http://localhost:9090/debug/trigger-winning-post
# Returns: {"triggered": true, "message": "WinningPoSt triggered immediately"}

# Check GPU state (should be GPU_STATE_WINNING_POST)
curl http://localhost:9090/debug/sector-cache

# Check inference status (should be paused)
curl http://localhost:8000/health

# Wait ~35 seconds for automatic recovery, inference resumes normally
```

## 10. Prometheus Metrics

The scheduler exposes Prometheus-format metrics at `:9090/metrics`:

| Metric | Type | Description |
|--------|------|-------------|
| `sidecar_gpu_state` | Gauge | Current GPU state (1=available, 2=yielding, 3=WindowPoSt, 4=WinningPoSt) |
| `sidecar_yield_events_total` | Counter | Cumulative yield events, labeled by `reason` (`window_post`, `winning_post`, `lotus_disconnected`, `fault_cutoff`) |
| `sidecar_seconds_until_deadline` | Gauge | Seconds until the next WindowPoSt deadline |

### Grafana Integration Example

```bash
# View current metrics
curl -s http://localhost:9090/metrics | grep sidecar_

# Example output:
# sidecar_gpu_state 1
# sidecar_yield_events_total{reason="window_post"} 12
# sidecar_yield_events_total{reason="winning_post"} 3
# sidecar_seconds_until_deadline 1847
```

## 11. Mining Yield Mechanism

### WindowPoSt (Periodic, Predictable)

```
API response: 200 200 200 200 | 200(last) | 503 503 503 ... 503 | 200 200 200
                               |           |                      |
Phase:         AVAILABLE       | YIELDING  |    WINDOW_POST       | AVAILABLE
Timeline: --------------------+-----------+----------------------+----------->
                            -5min       -2min     0 ~ 30min      +1min
                         Graceful     Hard stop   Mining window   Buffer
                          yield                                   recovery
```

| Phase | Threshold | Behavior |
|-------|-----------|----------|
| Graceful yield | 5 min before deadline | Complete current inference batch, then pause |
| Hard stop | 2 min before deadline | Abort in-flight requests, stop immediately |
| WindowPoSt active | During deadline window | All GPUs 100% dedicated to mining |
| Resume inference | 1 min after window closes | Restart AI inference |

### WinningPoSt (Sporadic, Unpredictable)

```
API response: 200 200 200 | 503 503 503 ... 503 | 200 200 200
                           |                     |
Phase:       AVAILABLE     |    WINNING_POST      | AVAILABLE
Timeline: ----------------+---------------------+----------->
                        Trigger               ~35s later
                      (IMMEDIATE)      (proof ~3s + recovery delay 30s)
```

- **Primary detection**: Curio log watcher tails `/tmp/curio.log` for `WinPostTask won election`
- **Fallback**: Polls Curio DB `mining_tasks.won=true` every 5s
- **Fail-safe**: Lotus disconnection triggers immediate yield of all GPUs (mining-first priority)

## 12. Multi-GPU Mode Comparison

| Feature | Tensor Parallel (tensor_parallel) | Multi-Instance (multi_instance) |
|---------|----------------------------------|--------------------------------|
| Use case | Large models (14B+) | High throughput with small models (3B-7B) |
| Engine count | 1 engine across N GPUs | N independent engines, one per GPU |
| Load balancing | Not needed (single engine) | round_robin / least_busy |
| Yield behavior | Pause 1 engine | Concurrently pause N engines |
| Per-request latency | Lower (combined compute) | Same as single GPU |
| Total throughput | Moderate | Higher (N-fold concurrency) |
| VRAM requirement | Sum of all GPU VRAM | Single GPU VRAM |
| Notes | 10GB VRAM requires `enforce_eager: true` | 8-engine resume takes ~2.5 min |

## 13. Troubleshooting

### GPU Not Detected

```bash
# Confirm GPU is visible
nvidia-smi

# Confirm Docker can access GPU
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

### Lotus Connection Failed

```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","method":"Filecoin.ChainHead","params":[],"id":1}' \
  http://localhost:1234/rpc/v0
```

### Inference Returns 503

This is expected behavior during WindowPoSt/WinningPoSt — the GPU has been yielded to mining. Check scheduler state:

```bash
curl http://localhost:9090/health
docker compose logs scheduler --since 5m | grep "state changed"
```

### VRAM Not Released on Resume

```bash
nvidia-smi                          # Check if VRAM is still occupied
pkill -9 -f vllm                    # Clean up residual vLLM processes
docker compose restart inference    # Restart inference service
```
