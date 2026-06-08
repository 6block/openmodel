#!/bin/sh
exec /app/scheduler -config /etc/sidecar/${SIDECAR_CONFIG:-sidecar-prod-test.yaml} "$@"
