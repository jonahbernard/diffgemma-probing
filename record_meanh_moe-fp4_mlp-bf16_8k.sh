#!/usr/bin/env bash
# meanH overlay data for the MoE-fp4 + MLP-bf16 config at 8k canvases: 50 prompts
# x 300 seeds = 15000 generations of 8000 tokens, recorded at BOTH temperature 0 and
# 0.6, against the moe-fp4/mlp-bf16 server (port 8007). Run
# ./launch_moe-fp4_mlp-bf16_probe.sh first.
#
# Seeds default to 100..399 (fresh, non-overlapping with the prior 0..99 run) so
# this collection is additive to the existing canvases.
#
# Outputs two snapshots so the temps can be overlaid/compared:
#   runs/meanh50/moe-fp4-mlp-bf16_8k_temp0.jsonl
#   runs/meanh50/moe-fp4-mlp-bf16_8k_temp0.6.jsonl
#
# Override SEEDS (canvases/prompt), MAXTOK, or TEMPS:
#   SEEDS="0 1 2" TEMPS="0 0.6 1.0" ./record_meanh_moe-fp4_mlp-bf16_8k.sh
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)

GPU=${GPU:-5}
export HIP_VISIBLE_DEVICES="$GPU"
PORT=${PORT:-$((8000 + GPU))}   # one server per GPU: port = 8000 + GPU index

for TEMP in ${TEMPS:-0}; do
  echo "=== moe-fp4/mlp-bf16 8k @ temperature ${TEMP} (GPU ${GPU} / port ${PORT}) ==="
  MODEL=diffgemma-moe-fp4-mlp-bf16 PORT="$PORT" \
    LIVE=/app/diffgemma-probing/runs/moe-fp4-mlp-bf16.jsonl \
    OUT=/app/diffgemma-probing/runs/meanh50/moe-fp4-mlp-bf16_8k_temp${TEMP}.jsonl \
    SEEDS="${SEEDS:-$(seq 100 399)}" \
    MAXTOK="${MAXTOK:-8000}" \
    TEMP="${TEMP}" \
    "$HERE/record_meanh.sh"
done
