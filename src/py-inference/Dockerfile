FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget ca-certificates software-properties-common gcc \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    python3.13 python3.13-venv python3.13-dev \
    && rm -rf /var/lib/apt/lists/*

# Set python3.13 as default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.13 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.13 1

# Install pip
RUN python -m ensurepip --upgrade && python -m pip install --upgrade pip

WORKDIR /app

# Install production dependencies
COPY requirements-docker.txt .
RUN python -m pip install --no-cache-dir -r requirements-docker.txt

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
