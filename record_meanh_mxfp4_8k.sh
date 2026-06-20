#!/usr/bin/env bash
# meanH overlay data for native mxfp4 at 8k canvases: 50 prompts x 5 seeds = 250
# generations of 8000 tokens, recorded at BOTH temperature 0 and 0.6, against
# the mxfp4 server (port 8003). Run ./launch_mxfp4_probe.sh first.
#
# Outputs two snapshots so the temps can be overlaid/compared:
#   runs/meanh50/mxfp4_8k_temp0.jsonl
#   runs/meanh50/mxfp4_8k_temp0.6.jsonl
#
# Override SEEDS (canvases/prompt), MAXTOK, or TEMPS:
#   SEEDS="0 1 2" TEMPS="0 0.6 1.0" ./record_meanh_mxfp4_8k.sh
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)

GPU=${GPU:-6}
export HIP_VISIBLE_DEVICES="$GPU"
PORT=${PORT:-$((8000 + GPU))}   # one server per GPU: port = 8000 + GPU index

for TEMP in ${TEMPS:-0 0.6}; do
  echo "=== mxfp4 8k @ temperature ${TEMP} (GPU ${GPU} / port ${PORT}) ==="
  MODEL=diffgemma-mxfp4 PORT="$PORT" \
    LIVE=/app/diffgemma-probing/runs/mxfp4.jsonl \
    OUT=/app/diffgemma-probing/runs/meanh50/mxfp4_8k_temp${TEMP}.jsonl \
    SEEDS="${SEEDS:-$(seq 0 99)}" \
    MAXTOK="${MAXTOK:-8000}" \
    TEMP="${TEMP}" \
    "$HERE/record_meanh.sh"
done
