#!/bin/bash
exec python -m src.main --config /etc/sidecar/${SIDECAR_CONFIG:-sidecar-prod-test.yaml} "$@"
