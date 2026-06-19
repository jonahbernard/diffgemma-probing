#!/usr/bin/env bash
# MXFP4 DiffusionGemma server running the PER-TOKEN MIXED-PRECISION experiment.
# Same native AITER W4A4 path as launch_mxfp4_probe.sh, but with the per-token
# precision controller enabled (VLLM_DIFFGEMMA_MIXED_PREC): each canvas position,
# each MoE layer, each step runs EITHER the native W4A4 kernel (committed
# positions: fp4 weights + fp4 activations) OR the native W4A16 kernel
# (not-yet-committed positions: same fp4 weights + bf16 activations). The two
# real AITER kernels are blended per token at each MoE layer's output.
#
# Both probes are enabled too, so this produces the SAME data shape as the plain
# mxfp4 run (confidence dump + per-GEMM activation dump) for apples-to-apples
# comparison. --enforce-eager is REQUIRED: the blend + activation capture are
# inline Python and are bypassed under FULL CUDA-graph replay.
#
# Modes (VLLM_DIFFGEMMA_MIXED_PREC):
#   committed  committed -> W4A4 (fp4 acts), uncommitted -> W4A16 (bf16 acts)
#   invert     committed -> W4A16, uncommitted -> W4A4 (control)
# This is a Python-only change in the editable (-e) vllm-jonah install, so just
# (re)launch this script after edits -- no rebuild needed.
set -uo pipefail

PROBE_DIR=${PROBE_DIR:-/app/diffgemma-probing/runs}
mkdir -p "$PROBE_DIR"

# Per-token mixed-precision controller. Override with MIXED_PREC=invert ./...
export VLLM_DIFFGEMMA_MIXED_PREC=${MIXED_PREC:-committed}

# Confidence probe (drives meanH), tagged "mixedprec" so dumps don't collide with
# the plain mxfp4 run and analysis can tell them apart.
export VLLM_DIFFGEMMA_PROBE=${VLLM_DIFFGEMMA_PROBE:-$PROBE_DIR/mixedprec.jsonl}
export VLLM_DIFFGEMMA_PROBE_TAG=mixedprec

# Activation probe is opt-in (it writes multi-GB dumps): pass ACT_PROBE=1 to
# enable. Off by default so plain meanH runs don't collect activation data.
if [ "${ACT_PROBE:-0}" != "0" ]; then
  export VLLM_DIFFGEMMA_ACT_PROBE=${VLLM_DIFFGEMMA_ACT_PROBE:-$PROBE_DIR/mixedprec_act.jsonl}
  export VLLM_DIFFGEMMA_ACT_PROBE_TAG=mixedprec
  export VLLM_DIFFGEMMA_ACT_PROBE_BLOCK=${VLLM_DIFFGEMMA_ACT_PROBE_BLOCK:-32}
  export VLLM_DIFFGEMMA_ACT_PROBE_OUTLIER_K=${VLLM_DIFFGEMMA_ACT_PROBE_OUTLIER_K:-6}
fi

export PYTHONPATH=/app/aiter-jonah
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_MOE=1
export GPU_ARCHS=gfx950
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-7}
export VLLM_LOGGING_LEVEL=INFO
cd /app/vllm-jonah
exec vllm serve /app/models/diffusiongemma-26B-A4B-it-mxfp4-v4 \
  --served-model-name diffgemma-mxfp4 \
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
  --port ${PORT:-8006}
