FROM golang:1.24-alpine AS builder

RUN apk --no-cache add protobuf protobuf-dev
RUN go install google.golang.org/protobuf/cmd/protoc-gen-go@latest \
    && go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest

WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download

COPY . .
# Regenerate protobuf Go stubs (ensures binary descriptor matches go_package)
RUN protoc \
    --go_out=proto/sidecar --go_opt=paths=source_relative \
    --go-grpc_out=proto/sidecar --go-grpc_opt=paths=source_relative \
    -I proto proto/sidecar.proto
RUN go mod tidy && CGO_ENABLED=0 GOOS=linux go build -o /scheduler cmd/scheduler/main.go

FROM alpine:3.19
RUN apk --no-cache add ca-certificates
COPY --from=builder /scheduler /app/scheduler
COPY entrypoint.sh /app/
RUN chmod +x /app/entrypoint.sh

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD wget -qO- http://localhost:9090/health || exit 1

ENTRYPOINT ["/app/entrypoint.sh"]
