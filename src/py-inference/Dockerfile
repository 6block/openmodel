# vLLM 0.22.1 (CUDA 13 / torch 2.11) — requires the DEVEL base (ships nvcc) because
# flashinfer JIT-compiles sampling/attention kernels at runtime; a *-runtime base has
# no nvcc and the engine fails to init. Also requires a HOST DRIVER >= 580 (CUDA 13.0);
# on driver 570 (CUDA 12.8) this image builds but the engine cannot initialize CUDA.
FROM nvidia/cuda:13.0.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget ca-certificates software-properties-common gcc \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    python3.13 python3.13-venv python3.13-dev espeak-ng \
    && rm -rf /var/lib/apt/lists/*

# Set python3.13 as default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.13 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.13 1

# Install pip
RUN python -m ensurepip --upgrade && python -m pip install --upgrade pip

WORKDIR /app

# Install production dependencies (text / vision: vLLM)
COPY requirements-docker.txt .
RUN python -m pip install --no-cache-dir -r requirements-docker.txt

# Image generation (engine: diffusion). --no-deps: torch / transformers / safetensors
# come from the vLLM install above and must not be re-resolved.
RUN python -m pip install --no-cache-dir --no-deps diffusers==0.39.0 accelerate==1.10.1

# Speech recognition (engine: asr) — faster-whisper on CTranslate2. PyAV 19 removed an
# argument faster-whisper 1.2.1 still passes, hence the upper bound.
RUN python -m pip install --no-cache-dir faster-whisper==1.2.1 soundfile==0.14.0 \
    && python -m pip install --no-cache-dir --only-binary=av "av>=14,<19"
# CTranslate2 4.x links libcublas.so.12 while this base is CUDA 13 (.so.13 only).
# Installed to a separate prefix and added to the search path only for that library;
# torch / vLLM keep their own CUDA 13 libraries.
RUN python -m pip install --no-cache-dir --target /opt/cu12 nvidia-cublas-cu12
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64:/opt/cu12/nvidia/cublas/lib

# Speech synthesis (engine: tts) — Kokoro. kokoro/misaki declare Python < 3.13 but run
# on 3.13; their spaCy chain is installed from binary wheels only (no source builds).
RUN python -m pip install --no-cache-dir --only-binary=spacy,thinc,blis,cymem,preshed,murmurhash,srsly \
      "spacy>=3.8.7" "thinc>=8.3.4" "blis>=1.3" \
    && python -m pip install --no-cache-dir --ignore-requires-python \
      --only-binary=spacy,thinc,blis,cymem,preshed,murmurhash,srsly \
      "kokoro==0.9.4" "misaki[en,zh]==0.9.4"
# Kokoro's English G2P loads the spaCy model en_core_web_sm and, when it is missing,
# downloads it from inside a request (and exits the process if that fails). Ship it.
RUN python -m pip install --no-cache-dir \
    https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl

COPY src/ src/

# Regenerate Python protobuf stubs from proto source
COPY proto/sidecar.proto /tmp/sidecar.proto
RUN python -m grpc_tools.protoc \
    --python_out=src/scheduler_client \
    --pyi_out=src/scheduler_client \
    --grpc_python_out=src/scheduler_client \
    -I /tmp /tmp/sidecar.proto \
    && sed -i 's/^import sidecar_pb2/from . import sidecar_pb2/' src/scheduler_client/sidecar_pb2_grpc.py

COPY entrypoint.sh /app/
RUN chmod +x /app/entrypoint.sh

# vLLM model loading can take minutes — generous start-period
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
  CMD python -c "import httpx; r=httpx.get('http://localhost:8000/health'); exit(0 if r.status_code==200 else 1)"

ENTRYPOINT ["/app/entrypoint.sh"]
