#!/usr/bin/env bash
# MXFP4 (W4A4) DiffusionGemma server WITH the confidence probe enabled.
# Mirrors /app/launch_diffgemma_native.sh exactly, plus the probe env vars.
# Records are tagged "mxfp4" so analyze_confidence.py can diff them against the
# bf16 run on matching (step, token-position).
set -uo pipefail

PROBE_DIR=${PROBE_DIR:-/app/diffgemma-probing/runs}
mkdir -p "$PROBE_DIR"
export VLLM_DIFFGEMMA_PROBE=${VLLM_DIFFGEMMA_PROBE:-$PROBE_DIR/mxfp4.jsonl}
export VLLM_DIFFGEMMA_PROBE_TAG=mxfp4

export PYTHONPATH=/app/aiter-jonah
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_MOE=1
export GPU_ARCHS=gfx950
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-3}
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
  --port ${PORT:-8003}
