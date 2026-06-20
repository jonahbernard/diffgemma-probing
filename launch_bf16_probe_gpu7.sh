#!/usr/bin/env bash
# Second bf16 probe server, pinned to GPU 7 (port 8007 via 8000+GPU). Lets you
# run a GPU-6 and a GPU-7 bf16 server side by side (e.g. temp 0 vs temp 0.6).
# Thin wrapper over launch_bf16_probe.sh: only GPU + probe-dump path differ, so
# the served-model-name, model card, and all serve flags stay identical.
#
# Pairs with record_meanh_bf16_8k_gpu7.sh, which reads runs/bf16_gpu7.jsonl.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)

export HIP_VISIBLE_DEVICES=1
export VLLM_DIFFGEMMA_PROBE=/app/diffgemma-probing/runs/bf16_gpu7.jsonl
exec "$HERE/launch_bf16_probe.sh"
