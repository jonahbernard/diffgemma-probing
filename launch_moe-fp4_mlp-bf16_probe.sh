#!/usr/bin/env bash
# DiffusionGemma server for the MoE-fp4 + MLP-bf16 config WITH the confidence
# probe enabled (drives meanH). This is the diffgemma-moe-fp4-mlp-bf16 branch's
# headline config: the MoE expert GEMMs run on the native AITER W4A4 (mxfp4)
# path from the quantized card, while the dense (non-MoE) MLP runs in TRUE bf16,
# side-loaded from the un-quantized card (VLLM_DIFFGEMMA_BF16_CARD) lazily on the
# first forward. That lazy load is why --enforce-eager is REQUIRED here (no
# CUDA-graph capture to race with).
#
# Mirrors /app/diffgemma-probing/launch_mxfp4_probe.sh, plus --enforce-eager and
# a distinct probe tag/port so its meanH dump doesn't collide with the plain
# mxfp4 or mixedprec runs. Records are tagged "moe-fp4-mlp-bf16".
#
# Run this, wait for the server to come up, then ./record_meanh_moe-fp4_mlp-bf16.sh
set -uo pipefail

PROBE_DIR=${PROBE_DIR:-/app/diffgemma-probing/runs}
mkdir -p "$PROBE_DIR"
export VLLM_DIFFGEMMA_PROBE=${VLLM_DIFFGEMMA_PROBE:-$PROBE_DIR/moe-fp4-mlp-bf16.jsonl}
export VLLM_DIFFGEMMA_PROBE_TAG=moe-fp4-mlp-bf16

# Un-quantized card the true bf16 dense-MLP weights are side-loaded from. The
# branch defaults to this path already; set here for clarity / overridability.
export VLLM_DIFFGEMMA_BF16_CARD=${VLLM_DIFFGEMMA_BF16_CARD:-/app/models/diffusiongemma-26B-A4B-it}

export PYTHONPATH=/app/aiter-jonah
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_MOE=1
export GPU_ARCHS=gfx950
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-5}
export VLLM_LOGGING_LEVEL=INFO
# One server per GPU: port = 8000 + GPU index (GPU 6 -> 8006, GPU 7 -> 8007).
GPU=${HIP_VISIBLE_DEVICES}
PORT=${PORT:-$((8000 + GPU))}
cd /app/vllm-jonah
exec vllm serve /app/models/diffusiongemma-26B-A4B-it-mxfp4-v4 \
  --served-model-name diffgemma-moe-fp4-mlp-bf16 \
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
  --moe-backend aiter \
  --quantization-config '{"moe": {"activation": "mxfp4"}}' \
  --enforce-eager \
  --port "$PORT"
