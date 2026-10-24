#!/usr/bin/env bash
# Real 16-process test; do not reuse previous experiment directories.
set -euo pipefail
WS="${FEDPARA_WS:-ws://127.0.0.1:8080/ws}"
OUT="${1:-swarm-runs/16x8-$(date +%Y%m%d-%H%M%S)}"
python -m patchlib.cli swarm-sim \
  --peers 16 --max-neighbors 8 --faults 2 --seed 20261022 \
  --max-events-per-peer "${FEDPARA_MAX_EVENTS:-256}" \
  --startup-timeout 360 --settle-timeout 300 \
  --ws "$WS" --output "$OUT"
