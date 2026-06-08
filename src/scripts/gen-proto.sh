#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PROTO_DIR="$PROJECT_ROOT/proto"

echo "Generating Go gRPC stubs..."
GO_OUT_DIR="$PROJECT_ROOT/go-scheduler/proto/sidecar"
mkdir -p "$GO_OUT_DIR"
protoc \
  --go_out="$GO_OUT_DIR" \
  --go_opt=paths=source_relative \
  --go-grpc_out="$GO_OUT_DIR" \
  --go-grpc_opt=paths=source_relative \
  -I "$PROTO_DIR" \
  "$PROTO_DIR/sidecar.proto"

echo "Generating Python gRPC stubs..."
python3 -m grpc_tools.protoc \
  --python_out="$PROJECT_ROOT/py-inference/src/scheduler_client" \
  --pyi_out="$PROJECT_ROOT/py-inference/src/scheduler_client" \
  --grpc_python_out="$PROJECT_ROOT/py-inference/src/scheduler_client" \
  -I "$PROTO_DIR" \
  "$PROTO_DIR/sidecar.proto"

# Fix Python imports: change absolute to relative imports
# Use a portable sed approach (works on both macOS and Linux)
PY_GRPC="$PROJECT_ROOT/py-inference/src/scheduler_client/sidecar_pb2_grpc.py"
if [[ "$OSTYPE" == "darwin"* ]]; then
  sed -i '' 's/^import sidecar_pb2/from . import sidecar_pb2/' "$PY_GRPC"
else
  sed -i 's/^import sidecar_pb2/from . import sidecar_pb2/' "$PY_GRPC"
fi

echo "Proto generation complete."
