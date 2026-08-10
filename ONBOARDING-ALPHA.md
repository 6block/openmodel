# SP Onboarding — Mainnet Alpha

Complete steps to register a test SP with the OpenModel mainnet gateway. All
gateway-side parameters are prefilled in config/sidecar-mainnet-alpha.yaml;
only SP-specific fields need filling.

Hands-on time is about 20 minutes, plus the boot model's automatic download
(6.1 GB; 10-60 minutes depending on bandwidth). No weights need to be prepared
in advance.

Conventions: the repository directory is the runtime directory. Whether you
git-clone (yields openmodel), download a GitHub ZIP (yields openmodel-main),
or extract the internal onboarding bundle (yields openmodel-m1-update-v1.3),
everything happens inside that one directory. Examples below use ~/openmodel;
if your directory name or location differs, substitute the actual path.

---

## 1. Prerequisites

### 1.1 The miner key is in the local Lotus wallet

Registration signs a gateway challenge with the miner's worker or owner key to
prove control of the miner. (Use your host's actual lotus binary path
throughout.)

Note: miner-info prints ID-form addresses (t0...) while the wallet list shows
pubkey form (t1/t3...) — they never match directly; convert with
state lookup --reverse:

```bash
lotus state miner-info <miner-id> | grep -E "Owner|Worker"
```

```bash
lotus state lookup --reverse <t0-address-from-previous-step>
```

```bash
lotus wallet list
```

The pubkey address from the second command (worker or owner, either) must
appear in the wallet list. If not, import it first:

```bash
lotus wallet import <key-file-path>
```

When missing, registration fails with:
neither the worker key ... nor the owner key ... is in the local Lotus wallet

### 1.2 Public port mappings

The gateway actively connects to two worker ports for dispatch and status
polling. Map them before onboarding:

| Purpose | Local worker port | Requirement |
|---|---|---|
| Inference | 38443 | any public port → local 38443 |
| Scheduler status | 39443 | any public port → local 39443 |

Public port numbers may differ from local ones. Identity is verified by the
worker_id inside the certificate, not by address; mappings can change freely
later.

### 1.3 Admission floor

- Raw power ≥ 1 TiB
- Miner balance ≥ 1 FIL

---

## 2. Onboarding steps

### Step 1: materials

| Material | Location (inside the repository directory) |
|---|---|
| Image tarballs | three .tar.gz files under release/ |
| Gateway CA certificate | gateway-ca.crt |
| Config template | config/sidecar-mainnet-alpha.yaml |

Enter the repository directory (all steps run from here):

```bash
cd ~/openmodel
```

Verify the CA fingerprint:

```bash
openssl x509 -in gateway-ca.crt -noout -fingerprint -sha256 -subject
```

Expected output:

```
sha256 Fingerprint=49:1A:4F:DF:F7:6C:A2:56:7C:D3:15:0F:71:CE:C6:B9:C3:18:63:75:09:9E:B4:8C:1A:5E:2C:51:C7:4A:1F:73
subject=CN = OpenModel Mainnet CA
```

### Step 2: load the images

```bash
docker load -i release/openmodel-scheduler.tar.gz
docker load -i release/openmodel-inference.tar.gz
docker load -i release/openmodel-foc-bridge.tar.gz
```

### Step 3: data/models directories and .env

The compose file and config directory ship with the repository; only the data
directories, the CA copy, and .env need to be created:

```bash
mkdir -p ~/openmodel/models ~/openmodel/data
cp ~/openmodel/gateway-ca.crt ~/openmodel/data/gateway-ca.crt
cp ~/openmodel/.env.example ~/openmodel/.env
```

Get a Lotus API token (registration's WalletSign needs sign permission or
higher; the miner host's existing token file usually works):

```bash
cat ~/.lotus/token
```

On hosts with a custom Lotus data directory:

```bash
cat "$LOTUS_PATH/token"
```

Or mint a new one:

```bash
lotus auth create-token --perm sign
```

Edit .env:

```bash
vi ~/openmodel/.env
```

Change two entries, leave the rest at defaults:

- SIDECAR_CONFIG=sidecar-mainnet-alpha.yaml (it must live in .env: a value
  passed only on the command line is lost on the next plain compose command,
  which then fails with: failed to load config: no such file or directory)
- LOTUS_API_TOKEN=<the token from above>

The CA goes into the data directory (config: tls_ca_path /data/gateway-ca.crt),
not config.

### Step 4: model weights (fetched automatically)

Weights are fetched from Filecoin mainnet automatically (the template embeds
the catalog's piece registry):

- The **boot model** (inference.model) downloads at **first start** — the API
  is not up yet and the worker is invisible to the gateway meanwhile. The
  default 4B is ~6.1 GB; download time varies widely with bandwidth (measured
  10-60 minutes), plus ~1 minute to extract and ~30 s to load.
- The **other claimed entries** prefetch in the background after start-up;
  admission probing for them simply defers until ready (timeouts are not
  counted against you).
- If the Filecoin retrieval provider is down, fetches **fall back to Hugging
  Face** automatically (logged as: falling back to Hugging Face). No action
  needed.
- To skip the first-boot wait, pre-place weights under ~/openmodel/models/
  (directory names must match exactly):

| Directory (under ~/openmodel/models/) | HF repo | Quant | Size | Reference hardware |
|---|---|---|---|---|
| Qwen--Qwen3-4B-Instruct-2507 | Qwen/Qwen3-4B-Instruct-2507 | fp16 | ~8 GB | 1x24G, or 2x10G |
| Qwen--Qwen3-8B | Qwen/Qwen3-8B | fp16 | ~16 GB | 1x24G, or 4x10G |
| openai--gpt-oss-20b | openai/gpt-oss-20b | mxfp4 | ~13 GB | 1x24G, or 2-4x10G |
| Qwen--Qwen3-32B-AWQ | Qwen/Qwen3-32B-AWQ | awq-int4 | ~19 GB | 1x24G (tight), 4-8x10G |

Quantisation is part of the catalog entry; a same-name different-quant
deployment cannot pass verification. Disk budget: every claimed entry is
downloaded to this machine — sum the table.

### Step 5: fill in the config

```bash
vi ~/openmodel/config/sidecar-mainnet-alpha.yaml
```

Fill the fields marked <<<:

| Field | Value |
|---|---|
| lotus.miner_address | miner id, e.g. f01234 |
| inference.model | the catalog entry this machine serves |
| inference.multi_gpu.gpu_count / device_ids | per actual GPUs |
| gateway.worker_id | 1-64 chars, alphanumeric/-/_, e.g. sp-alpha-01 |
| gateway.payout_address | EVM address earnings settle to (0x...) |
| gateway.public_endpoint | https://<public-ip>:<mapped-inference-port> |
| gateway.public_scheduler_url | https://<public-ip>:<mapped-scheduler-port> |

Everything else (gateway address, gateway_id, CA path, mode: prod, mtls ports)
is prefilled — leave unchanged. Hardware notes:

- 4B on a 10 GB card: keep the template defaults gpu_memory_utilization 0.93,
  max_model_len 4096, enforce_eager true. Weights take ~7.6 GiB, leaving only
  ~1.04 GiB for the KV cache — above ~7500 tokens the engine refuses to start
  (ValueError: ... is larger than the available KV cache memory).
- 24 GB cards: max_model_len can go to 32768.

#### Multi-GPU: tensor_parallel or multi_instance?

Two ways to use several GPUs. The choice is usually forced by whether the model
fits on ONE card:

- tensor_parallel — one engine, the model split across the GPUs. Use it when the
  model does not fit on a single card.
- multi_instance — one independent engine per GPU, each holding a full copy.
  Only possible for models that fit on one card; multiplies throughput.

Worked example, 8x RTX 3080 (10 GB each), measured on our own machine:

| Model | Fits one 10 GB card? | Mode | Notes |
|---|---|---|---|
| Qwen3-32B-AWQ | no (~19 GB) | tensor_parallel, device_ids 0-7 | needs enforce_eager true; max_model_len 8192 |
| gpt-oss-20b (mxfp4) | no | tensor_parallel, device_ids 0-7 | 1.81 GiB weights + 6.34 GiB KV per card |
| Qwen3-4B (fp16) | barely (7.6 GB) | either | one engine per card needs max_model_len 4096, KV only ~1 GiB |
| Qwen2.5-3B / 1.5B | yes | multi_instance | 8 engines, ~8x the throughput |

So for a 32B model on 10 GB cards the answer is tensor_parallel — not a
preference: the weights do not fit any other way.

Whichever mode, list every model this machine can serve under
gateway.supported_models — the gateway switches between them on demand, and a
model absent from that list is never routed here.

### Step 6: verify registration on its own

Run registration only, without starting the service:

```bash
docker run --rm --network host --env-file ~/openmodel/.env \
  -v ~/openmodel/config/sidecar-mainnet-alpha.yaml:/etc/sidecar/config.yaml:ro \
  -v ~/openmodel/data:/data \
  --entrypoint /app/scheduler openmodel-scheduler:latest \
  -config /etc/sidecar/config.yaml -register-only
```

(If the full stack is already running, log lines like
mTLS front ... bind: address already in use are expected noise — the ports are
held by the running service; the registration itself is unaffected.)

Success criteria: a registration-success log line, and
~/openmodel/data/gateway-token.json exists.

### Step 7: start the stack

```bash
cd ~/openmodel
docker compose up -d
```

(The config filename comes from SIDECAR_CONFIG in .env; later restart / up -d
commands need no prefix.)

---

## 3. Confirming the onboarding

Read the gateway-issued token and query registration state:

```bash
docker exec openmodel-scheduler cat /data/gateway-token.json
```

```bash
curl -s -H "Authorization: Bearer $(docker exec openmodel-scheduler cat /data/gateway-token.json | python3 -c 'import json,sys;print(json.load(sys.stdin)["auth_token"])')" \
  http://127.0.0.1:9090/registration
```

Key fields:

- gateway_link.verdict — serving: the gateway is polling this machine,
  onboarding succeeded; waiting: not started; unreachable: the gateway cannot
  reach your public ports; silent: polling stopped.
- model_admission.models[] — one entry per claimed model:
  - status verified — passed, real traffic can route here
  - status pending — being examined. Probe requests are unbilled, earn
    nothing, and are deliberately indistinguishable from user traffic
  - status failing — not passing; 3 consecutive fails escalate (with one
    independent confirmation re-test before any punishment). floor is the
    score line this model is held to.

Re-trigger examination after fixing something:

```bash
curl -s -X POST -H "Authorization: Bearer <token-from-above>" \
  -H "Content-Type: application/json" \
  -d '{"model":"/models/Qwen--Qwen3-4B-Instruct-2507"}' \
  http://127.0.0.1:9090/registration/reverify
```

One request per model per 10 minutes; a reset clears accumulated evidence too.

---

## 4. Troubleshooting

**gateway self-registration requires a real Lotus client (mode != dev)**
The config is missing mode: prod (the default is dev, which uses a mock Lotus
and is rejected). The template includes the line — do not delete it.

**https gateway_url configured but no CA available ... TLS dials will fail closed**
~/openmodel/data/gateway-ca.crt is missing or mispathed. Without a CA, TLS
fails outright — it never downgrades to plaintext.

**could not query the local Lotus wallet (lotus.api_token missing or lacking permission?)**
.env not created, LOTUS_API_TOKEN unset, token lacks permission (sign or
higher required), or the register-only command lost its --env-file. See step 3.

**neither the worker key ... nor the owner key ... is in the local Lotus wallet**
The local Lotus wallet has no key for this miner (the wallet query itself
succeeded). See prerequisite 1.1.

**failed to load config: open /etc/sidecar/xxx.yaml: no such file or directory**
SIDECAR_CONFIG in .env does not match the actual config filename (typical
after specifying it only once on the command line). Set
SIDECAR_CONFIG=sidecar-mainnet-alpha.yaml in .env and restart.

**bind: address already in use**
38443 / 39443 are taken. Free the ports, or change the local ports under
mtls_front (and update the public mappings; public_* carries the mapped
public ports).

**verdict: unreachable**
The gateway cannot reach your public ports. Test from another machine (not the
worker itself):

```bash
nc -z -w 8 <public-ip> <mapped-inference-port>
```

succeeded means the port is reachable. Do not judge with curl over https:
these ports speak mutual TLS, and a client without a certificate always gets
000 — indistinguishable from a dead port. To dig further:

```bash
timeout 15 openssl s_client -connect <public-ip>:<mapped-inference-port> </dev/null 2>&1 | head -5
```

CONNECTED in the output means the link is fine; the certificate-verification
errors after it are expected (this machine does not trust the private CA).

**First start takes very long to become ready**
Normal: the boot model downloads from Filecoin mainnet on first start (default
4B: 10-60 minutes depending on bandwidth); the worker is invisible to the
gateway meanwhile. Watch progress:

```bash
docker logs -f openmodel-inference 2>&1 | grep -E "download|prefetch|engine started"
```

engine started means ready; a later prefetch ... done means the other claimed
models are staged too.

**ValueError: ... KV cache is needed, which is larger than the available KV cache memory**
max_model_len exceeds this machine's VRAM. Lower it to the reported
estimated maximum model length (4096 recommended for 4B on 10 GB cards), or
use a larger card.

**A model stays pending or failing**
Check the weight directory name and quantisation against the step-4 table;
compare your score with the model's floor in /registration. Common causes:
wrong quantisation, max_model_len too small (answers truncated), out of VRAM.

When reporting a problem, attach the /registration output and the scheduler
container logs.

---

## Appendix: mainnet alpha quick reference

| Item | Value |
|---|---|
| Gateway | https://36.189.235.195:18020 |
| gateway_id | openmodel-mainnet |
| Settlement contract (Filecoin mainnet) | 0x465d979675d401295C529e15dC9187c9b92ed4d1 |
| Settlement tokens | FIL, USDFC |
| Admission floor | power ≥ 1 TiB, miner balance ≥ 1 FIL |
| CA fingerprint (SHA256) | 49:1A:4F:DF:F7:6C:A2:56:7C:D3:15:0F:71:CE:C6:B9:C3:18:63:75:09:9E:B4:8C:1A:5E:2C:51:C7:4A:1F:73 |
| Worker cert validity | 7 days, auto-renewed at half-life |
| Registration rate limit | 6/minute |
