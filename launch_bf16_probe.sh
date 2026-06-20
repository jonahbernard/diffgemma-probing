#!/usr/bin/env bash
# BF16 DiffusionGemma server WITH the confidence probe enabled.
# Mirrors /app/launch_diffgemma_bf16.sh exactly, plus:
#   - VLLM_DIFFGEMMA_PROBE points at a JSONL dump (one line per decode req/step)
#   - VLLM_DIFFGEMMA_PROBE_TAG labels every record "bf16" for later A/B diffing
# The probe is read once at sampler init and is a no-op when the env is unset,
# so this is the ONLY thing that differs from the normal bf16 launch.
set -uo pipefail

PROBE_DIR=${PROBE_DIR:-/app/diffgemma-probing/runs}
mkdir -p "$PROBE_DIR"
export VLLM_DIFFGEMMA_PROBE=${VLLM_DIFFGEMMA_PROBE:-$PROBE_DIR/bf16.jsonl}
export VLLM_DIFFGEMMA_PROBE_TAG=bf16

export PYTHONPATH=/app/aiter-jonah
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_MOE=1
export GPU_ARCHS=gfx950
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0}
export VLLM_LOGGING_LEVEL=INFO
# One server per GPU: port = 8000 + GPU index (GPU 6 -> 8006, GPU 7 -> 8007).
GPU=${HIP_VISIBLE_DEVICES}
PORT=${PORT:-$((8000 + GPU))}
cd /app/vllm-jonah
exec vllm serve /app/models/diffusiongemma-26B-A4B-it \
  --served-model-name diffgemma-bf16 \
  --trust-remote-code \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 9248 \
  --max-num-seqs 4 \
  --enable-auto-tool-choice \
  --tool-call-parser gemma4 \
  --reasoning-parser gemma4 \
  --chat-template examples/tool_chat_template_gemma4.jinja \
  --attention-backend TRITON_ATTN \
  --port "$PORT"
