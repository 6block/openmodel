FROM golang:1.25-alpine AS builder

WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download

COPY . .
# Build from the COMMITTED protobuf stubs (proto/sidecar/*.pb.go, kept in sync
# via scripts/gen-proto.sh). We intentionally do NOT regenerate them in-build:
# the previous `protoc-gen-go-grpc@latest` is unpinned and drifted to v1.6.2,
# which requires Go >= 1.25 and breaks this Go-1.24 build. Committed stubs are
# the source of truth (and are exercised by `go test`), so the build is now
# reproducible and free of floating tool versions.
RUN CGO_ENABLED=0 GOOS=linux go build -o /scheduler cmd/scheduler/main.go

FROM alpine:3.19
RUN apk --no-cache add ca-certificates
COPY --from=builder /scheduler /app/scheduler
COPY entrypoint.sh /app/
RUN chmod +x /app/entrypoint.sh

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD wget -qO- http://localhost:9090/health || exit 1

ENTRYPOINT ["/app/entrypoint.sh"]
