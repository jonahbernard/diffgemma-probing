#!/usr/bin/env bash
# MXFP4 (W4A4) DiffusionGemma server WITH the pre-GEMM activation probe enabled.
# Mirrors launch_mxfp4_probe.sh, plus:
#   - VLLM_DIFFGEMMA_ACT_PROBE points at a JSONL dump of per-GEMM activation
#     statistics (one line per layer x site x forward, plus per-forward ctx)
#   - VLLM_DIFFGEMMA_ACT_PROBE_TAG labels every record "mxfp4" for A/B diffing
#   - --enforce-eager is REQUIRED: activation capture is inline Python and is
#     bypassed under FULL CUDA-graph replay.
# The confidence probe is enabled too (combinable); comment it out if you only
# want activations. Both probes are no-ops when their env is unset.
set -uo pipefail

PROBE_DIR=${PROBE_DIR:-/app/diffgemma-probing/runs}
mkdir -p "$PROBE_DIR"
export VLLM_DIFFGEMMA_PROBE=${VLLM_DIFFGEMMA_PROBE:-$PROBE_DIR/mxfp4.jsonl}
export VLLM_DIFFGEMMA_PROBE_TAG=mxfp4
export VLLM_DIFFGEMMA_ACT_PROBE=${VLLM_DIFFGEMMA_ACT_PROBE:-$PROBE_DIR/mxfp4_act.jsonl}
export VLLM_DIFFGEMMA_ACT_PROBE_TAG=mxfp4
# Optional: block size for block-absmax stats (mxfp4 block) and outlier-k.
export VLLM_DIFFGEMMA_ACT_PROBE_BLOCK=${VLLM_DIFFGEMMA_ACT_PROBE_BLOCK:-32}
export VLLM_DIFFGEMMA_ACT_PROBE_OUTLIER_K=${VLLM_DIFFGEMMA_ACT_PROBE_OUTLIER_K:-6}

export PYTHONPATH=/app/aiter-jonah
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_MOE=1
export GPU_ARCHS=gfx950
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-1}
export VLLM_LOGGING_LEVEL=INFO
cd /app/vllm-jonah
exec vllm serve /app/models/diffusiongemma-26B-A4B-it-mxfp4-v4 \
  --served-model-name diffgemma-mxfp4 \
  --trust-remote-code \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 256000 \
  --max-num-seqs 4 \
  --enable-auto-tool-choice \
  --tool-call-parser gemma4 \
  --reasoning-parser gemma4 \
  --chat-template examples/tool_chat_template_gemma4.jinja \
  --attention-backend TRITON_ATTN \
  --moe-backend aiter \
  --quantization-config '{"moe": {"activation": "mxfp4"}}' \
  --enforce-eager \
  --port ${PORT:-8005}
