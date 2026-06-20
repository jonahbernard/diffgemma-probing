#!/usr/bin/env bash
# Second moe-fp4/mlp-bf16 probe server, pinned to GPU 7 (port 8007 via 8000+GPU).
# Lets you run a GPU-6 and a GPU-7 server side by side (e.g. temp 0 vs temp 0.6).
# Thin wrapper over launch_moe-fp4_mlp-bf16_probe.sh: only GPU + probe-dump path
# differ, so the served-model-name, model card, --enforce-eager, and all serve
# flags stay identical.
#
# Pairs with record_meanh_moe-fp4_mlp-bf16_8k_gpu7.sh, which reads
# runs/moe-fp4-mlp-bf16_gpu7.jsonl.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)

export HIP_VISIBLE_DEVICES=3
export VLLM_DIFFGEMMA_PROBE=/app/diffgemma-probing/runs/moe-fp4-mlp-bf16_gpu7.jsonl
exec "$HERE/launch_moe-fp4_mlp-bf16_probe.sh"
