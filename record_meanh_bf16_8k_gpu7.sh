#!/usr/bin/env bash
# Duplicate of record_meanh_bf16_8k.sh pinned to temperature 0.6 only, targeting
# a SECOND bf16 server on port 8012 (launch it with HIP_VISIBLE_DEVICES=7).
# Lets you run temp 0 (GPU 6 / port 8002) and temp 0.6 (GPU 7 / port 8012) at once.
#
# 50 prompts x 300 seeds = 15000 generations of 8000 tokens.
# Seeds default to 100..399 (fresh, non-overlapping with the prior 0..99 run).
# Output: runs/meanh50/bf16_8k_gpu7_temp0.6.jsonl
#
# Prereq: launch the 2nd bf16 probe server bound to GPU 7 / port 8012, writing
#         its live probe jsonl to runs/bf16_gpu7.jsonl.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)

GPU=${GPU:-0}
export HIP_VISIBLE_DEVICES="$GPU"
PORT=${PORT:-$((8000 + GPU))}   # one server per GPU: port = 8000 + GPU index

for TEMP in ${TEMPS:-0.6}; do
  echo "=== bf16 8k @ temperature ${TEMP} (GPU ${GPU} / port ${PORT}) ==="
  MODEL=diffgemma-bf16 PORT="$PORT" \
    LIVE=/app/diffgemma-probing/runs/bf16_gpu7.jsonl \
    OUT=/app/diffgemma-probing/runs/meanh50/bf16_8k_gpu7_temp${TEMP}.jsonl \
    SEEDS="${SEEDS:-$(seq 100 399)}" \
    MAXTOK="${MAXTOK:-8000}" \
    TEMP="${TEMP}" \
    "$HERE/record_meanh.sh"
done
