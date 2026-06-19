#!/usr/bin/env bash
# meanH overlay data for the per-token mixed-precision method (committed->fp4,
# uncommitted->bf16): 50 prompts x N canvases against the mixed-precision server
# (port 8006). Run ./launch_mxfp4_mixedprec.sh first (MIXED_PREC=committed).
# Override SEEDS for >1 canvas/prompt:  SEEDS="0 1 2 3" ./record_meanh_mixedprec.sh
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
MODEL=diffgemma-mxfp4 PORT=${PORT:-8006} \
  LIVE=/app/diffgemma-probing/runs/mixedprec.jsonl \
  OUT=/app/diffgemma-probing/runs/meanh50/mixedprec.jsonl \
  SEEDS="${SEEDS:-0 1 2 3 4}" \
  exec "$HERE/record_meanh.sh"
