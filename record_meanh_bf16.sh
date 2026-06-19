#!/usr/bin/env bash
# meanH overlay data for bf16: 50 prompts x N canvases against the bf16 server
# (port 8002). Run ./launch_bf16_probe.sh first. Override SEEDS for >1 canvas/prompt:
#   SEEDS="0 1 2 3" ./record_meanh_bf16.sh
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
MODEL=diffgemma-bf16 PORT=${PORT:-8002} \
  LIVE=/app/diffgemma-probing/runs/bf16.jsonl \
  OUT=/app/diffgemma-probing/runs/meanh50/bf16.jsonl \
  SEEDS="${SEEDS:-0 1 2 3 4}" \
  exec "$HERE/record_meanh.sh"
