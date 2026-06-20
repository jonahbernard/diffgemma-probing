#!/usr/bin/env bash
# meanH overlay data for the MoE-fp4 + MLP-bf16 config: 50 prompts x N canvases
# against the moe-fp4/mlp-bf16 server (port 8004). Run
# ./launch_moe-fp4_mlp-bf16_probe.sh first. Override SEEDS for >1 canvas/prompt:
#   SEEDS="0 1 2 3" ./record_meanh_moe-fp4_mlp-bf16.sh
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
MODEL=diffgemma-moe-fp4-mlp-bf16 PORT=${PORT:-8007} \
  LIVE=/app/diffgemma-probing/runs/moe-fp4-mlp-bf16.jsonl \
  OUT=/app/diffgemma-probing/runs/meanh50/moe-fp4-mlp-bf16.jsonl \
  SEEDS="${SEEDS:-0 1 2 3 4}" \
  exec "$HERE/record_meanh.sh"
