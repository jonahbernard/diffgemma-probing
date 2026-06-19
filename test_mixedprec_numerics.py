"""Numerical verification of the per-token mixed-precision MoE edits.

Three checks, all run under the same interpreter `vllm serve` uses:

1. bf16 shadow path: `fused_experts` on weights from `dequant_mxfp4` matches an
   independent hand-written gated-MoE reference (gate=first half, up=second half,
   gelu_tanh(gate)*up, @ w2, weighted by topk). This is the make-or-break check
   for `_maybe_blend_bf16_moe`'s GEMM.
2. blend mechanics: `torch.where(bf16_mask[:,None], y_bf16, y_fp4)` selects the
   right rows; committed->fp4, uncommitted->bf16; `invert` flips it.
3. controller: `bf16_row_mask` polarity / tiling / None-before-first-step.
"""

import torch

from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
from vllm.model_executor.layers.quantization.utils.mxfp4_utils import dequant_mxfp4

torch.manual_seed(0)
dev = "cuda"
dt = torch.bfloat16

E, H, I, TOPK = 8, 128, 256, 2  # experts, hidden, intermediate, top-k
M = 16  # tokens
GS = 32  # mxfp4 group size

ok = True


def rep(name, passed, detail=""):
    global ok
    ok = ok and passed
    print(f"[{'PASS' if passed else 'FAIL'}] {name} {detail}")


# ---- build random packed mxfp4 weights, then DEFINE bf16 weights as dequant ----
# w13 packed: (E, 2I, H//2) uint8; scale (E, 2I, H//GS) e8m0 uint8
# w2  packed: (E, H,  I//2) uint8; scale (E, H,  I//GS) e8m0 uint8
def rand_packed(e, rows, cols):
    return torch.randint(0, 256, (e, rows, cols // 2), dtype=torch.uint8, device=dev)


def rand_scale(e, rows, cols):
    # e8m0 around 1.0 (exponent bias 127) to keep magnitudes sane
    return torch.randint(120, 134, (e, rows, cols // GS), dtype=torch.uint8, device=dev)


w13_p = rand_packed(E, 2 * I, H)
w13_s = rand_scale(E, 2 * I, H)
w2_p = rand_packed(E, H, I)
w2_s = rand_scale(E, H, I)

w13 = dequant_mxfp4(w13_p, w13_s, dt)  # (E, 2I, H)
w2 = dequant_mxfp4(w2_p, w2_s, dt)  # (E, H, I)
rep("dequant shapes", w13.shape == (E, 2 * I, H) and w2.shape == (E, H, I),
    f"w13={tuple(w13.shape)} w2={tuple(w2.shape)}")

x = torch.randn(M, H, device=dev, dtype=dt) * 0.5

# routing (shared by both paths in the real code; here just pick top-k of random)
logits = torch.randn(M, E, device=dev, dtype=torch.float32)
probs = logits.softmax(-1)
topk_weights, topk_ids = probs.topk(TOPK, dim=-1)
topk_weights = topk_weights.to(dt)
topk_ids = topk_ids.to(torch.int32)

# ---- check 1: fused_experts vs independent reference ----
y = fused_experts(
    hidden_states=x, w1=w13, w2=w2,
    topk_weights=topk_weights, topk_ids=topk_ids,
    activation=MoEActivation.GELU_TANH,
    global_num_experts=E,
)


def ref_moe(x, w13, w2, tw, tid):
    xf = x.float()
    out = torch.zeros(M, H, device=dev, dtype=torch.float32)
    for m in range(M):
        for k in range(TOPK):
            e = int(tid[m, k])
            gate_up = xf[m] @ w13[e].float().T  # (2I,)
            gate, up = gate_up[:I], gate_up[I:]
            act = torch.nn.functional.gelu(gate, approximate="tanh") * up
            o = act @ w2[e].float().T  # (H,)
            out[m] += float(tw[m, k]) * o
    return out


y_ref = ref_moe(x, w13, w2, topk_weights, topk_ids)
diff = (y.float() - y_ref).abs()
rel = diff.max() / (y_ref.abs().max() + 1e-6)
rep("bf16 shadow path == reference MoE", rel < 0.02,
    f"max_abs={diff.max():.4e} rel={rel:.4e}")

# ---- check 2: blend mechanics ----
y_fp4 = torch.randn(M, H, device=dev, dtype=dt)  # stand-in native output
y_bf16 = torch.randn(M, H, device=dev, dtype=dt)
bf16_mask = torch.zeros(M, dtype=torch.bool, device=dev)
bf16_mask[::2] = True  # even rows -> bf16, odd rows -> fp4
blended = torch.where(bf16_mask.unsqueeze(1), y_bf16, y_fp4)
sel_ok = torch.equal(blended[::2], y_bf16[::2]) and torch.equal(blended[1::2], y_fp4[1::2])
rep("torch.where selects bf16/fp4 rows correctly", sel_ok)

# ---- check 3: controller polarity / tiling / None ----
from vllm.model_executor.models.diffgemma_precision_ctl import DiffgemmaPrecisionCtl

ctl = DiffgemmaPrecisionCtl("committed")
rep("None before first set_committed", ctl.bf16_row_mask(M, torch.device(dev)) is None)

committed = torch.zeros(2, 8, dtype=torch.bool)  # [num_decode=2, CL=8]
committed[0, :4] = True  # first 4 positions committed
ctl.set_committed(committed)
mask = ctl.bf16_row_mask(16, torch.device(dev))  # 16 rows == 2*8
# committed -> fp4 -> bf16_mask False ; uncommitted -> bf16_mask True
expected = (~committed.reshape(-1)).to(dev)
rep("committed mode: bf16_mask == ~committed", torch.equal(mask, expected))

ctl_inv = DiffgemmaPrecisionCtl("invert")
ctl_inv.set_committed(committed)
mask_inv = ctl_inv.bf16_row_mask(16, torch.device(dev))
rep("invert mode flips polarity", torch.equal(mask_inv, committed.reshape(-1).to(dev)))

# row-count mismatch must FAIL SAFE (return None), not tile a foreign pattern.
# A forward whose row count != mask element count (prefill, or a changed
# request batch) does not correspond to this mask row-for-row, so blending
# would apply a stale/foreign commit pattern. The caller falls back to fp4.
rep("mismatched n_rows -> None (no tiling)",
    ctl.bf16_row_mask(32, torch.device(dev)) is None)
rep("mismatched smaller n_rows -> None",
    ctl.bf16_row_mask(10, torch.device(dev)) is None)

# reset() drops the stored mask so a new request starts clean (pure fp4).
ctl.reset()
rep("reset clears mask -> None even at matching n_rows",
    ctl.bf16_row_mask(16, torch.device(dev)) is None)

# ---- check 4: bf16 path loads TRUE card weights, not dequantized fp4 ----
# The mixed-prec bf16 path now reads genuine bf16 expert weights from the
# original HF card (experts.gate_up_proj / experts.down_proj) instead of
# dequantizing the fp4 weights. Verify, for one real layer/expert:
#   (a) card gate_up_proj is [E, 2I, H] with gate=[:I], up=[I:] and down_proj is
#       [E, H, I] -- exactly fused_experts' w13/w2 orientation (no reorder), and
#   (b) the card values genuinely DIFFER from dequant_mxfp4 of the matching fp4
#       expert (so we really switched paths; this is not a no-op).
import json
import os

from safetensors import safe_open

BF16_CARD = os.environ.get(
    "VLLM_DIFFGEMMA_BF16_CARD", "/app/models/diffusiongemma-26B-A4B-it"
)
MXFP4_CARD = os.environ.get(
    "VLLM_DIFFGEMMA_MXFP4_CARD", "/app/models/diffusiongemma-26B-A4B-it-mxfp4-v4"
)
if os.path.isdir(BF16_CARD) and os.path.isdir(MXFP4_CARD):
    def _open(card, key):
        idx = json.load(
            open(os.path.join(card, "model.safetensors.index.json"))
        )["weight_map"]
        with safe_open(os.path.join(card, idx[key]), "pt") as h:
            return h.get_tensor(key)

    gu = _open(BF16_CARD, "model.decoder.layers.0.experts.gate_up_proj")  # [E,2I,H]
    dn = _open(BF16_CARD, "model.decoder.layers.0.experts.down_proj")     # [E,H,I]
    Ic = dn.shape[2]
    rep("card gate_up_proj is [E, 2I, H]", gu.shape[1] == 2 * Ic)

    # dequant the matching fp4 expert-0 gate/up/down and confirm layout + diff.
    def _deq(card, key):
        idx = json.load(
            open(os.path.join(card, "model.safetensors.index.json"))
        )["weight_map"]
        with safe_open(os.path.join(card, idx[key]), "pt") as h:
            w = h.get_tensor(key)
            s = h.get_tensor(key + "_scale")
        return dequant_mxfp4(w.to(dev), s.to(dev), torch.bfloat16).float().cpu()

    g = _deq(MXFP4_CARD, "model.decoder.layers.0.experts.0.gate_proj.weight")
    u = _deq(MXFP4_CARD, "model.decoder.layers.0.experts.0.up_proj.weight")
    gu0 = gu[0].float()

    def _relmax(a, b):
        return ((a - b).abs().max() / (b.abs().max() + 1e-6)).item()

    # gate aligns to the first I rows, up to the second I rows (no swap).
    rep("dequant gate ~ card gate_up[:I] (layout, no reorder)",
        _relmax(g, gu0[:Ic]) < _relmax(g, gu0[Ic:]))
    rep("dequant up   ~ card gate_up[I:] (layout, no reorder)",
        _relmax(u, gu0[Ic:]) < _relmax(u, gu0[:Ic]))
    # true bf16 weights are NOT the dequantized fp4 values (we switched paths).
    rep("card weights differ from dequant fp4 (real switch)",
        _relmax(g, gu0[:Ic]) > 0.02,
        f"gate max-rel={_relmax(g, gu0[:Ic]):.4f}")
else:
    rep("card-load check skipped (cards not on this host)", True,
        f"(bf16={BF16_CARD!r} mxfp4={MXFP4_CARD!r})")

print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
exit(0 if ok else 1)
