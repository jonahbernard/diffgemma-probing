#!/usr/bin/env bash
# Second mxfp4 probe server, pinned to GPU 7 (port 8007 via 8000+GPU). Lets you
# run a GPU-6 and a GPU-7 mxfp4 server side by side (e.g. temp 0 vs temp 0.6).
# Thin wrapper over launch_mxfp4_probe.sh: only GPU + probe-dump path differ, so
# the served-model-name, model card, and all serve flags stay identical.
#
# Pairs with record_meanh_mxfp4_8k_gpu7.sh, which reads runs/mxfp4_gpu7.jsonl.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)

export HIP_VISIBLE_DEVICES=7
export VLLM_DIFFGEMMA_PROBE=/app/diffgemma-probing/runs/mxfp4_gpu7.jsonl
exec "$HERE/launch_mxfp4_probe.sh"
