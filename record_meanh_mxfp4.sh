#!/usr/bin/env bash
# meanH overlay data for native mxfp4: 50 prompts x N canvases against the mxfp4
# server (port 8003). Run ./launch_mxfp4_probe.sh first. Override SEEDS for >1
# canvas/prompt:  SEEDS="0 1 2 3" ./record_meanh_mxfp4.sh
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
MODEL=diffgemma-mxfp4 PORT=${PORT:-8003} \
  LIVE=/app/diffgemma-probing/runs/mxfp4.jsonl \
  OUT=/app/diffgemma-probing/runs/meanh50/mxfp4.jsonl \
  SEEDS="${SEEDS:-0 1 2 3 4}" \
  exec "$HERE/record_meanh.sh"
