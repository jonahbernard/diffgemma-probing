#!/usr/bin/env bash
# Duplicate of record_meanh_mxfp4_8k.sh pinned to temperature 0.6 only, targeting
# a SECOND mxfp4 server on port 8013 (launch it with HIP_VISIBLE_DEVICES=7).
# Lets you run temp 0 (GPU 6 / port 8003) and temp 0.6 (GPU 7 / port 8013) at once.
#
# 50 prompts x 100 seeds = 5000 generations of 8000 tokens.
# Output: runs/meanh50/mxfp4_8k_gpu7_temp0.6.jsonl
#
# Prereq: launch the 2nd mxfp4 probe server bound to GPU 7 / port 8013, writing
#         its live probe jsonl to runs/mxfp4_gpu7.jsonl.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)

GPU=${GPU:-7}
export HIP_VISIBLE_DEVICES="$GPU"
PORT=${PORT:-$((8000 + GPU))}   # one server per GPU: port = 8000 + GPU index

for TEMP in ${TEMPS:-0.6}; do
  echo "=== mxfp4 8k @ temperature ${TEMP} (GPU ${GPU} / port ${PORT}) ==="
  MODEL=diffgemma-mxfp4 PORT="$PORT" \
    LIVE=/app/diffgemma-probing/runs/mxfp4_gpu7.jsonl \
    OUT=/app/diffgemma-probing/runs/meanh50/mxfp4_8k_gpu7_temp${TEMP}.jsonl \
    SEEDS="${SEEDS:-$(seq 0 99)}" \
    MAXTOK="${MAXTOK:-8000}" \
    TEMP="${TEMP}" \
    "$HERE/record_meanh.sh"
done
