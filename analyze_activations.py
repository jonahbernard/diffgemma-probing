#!/usr/bin/env python3
"""Visualize the pre-GEMM activation-statistics dumps (mxfp4 vs bf16).

Reads the two JSONL files produced by the activation probe
(vllm/model_executor/models/diffgemma_act_probe.py) and renders a cross-linked
set of static HTML pages full of charts, so you can eyeball where activation
outliers live and how big they are -- per GEMM site, per layer, per denoising
step -- and how the MXFP4 (W4A4) path differs from BF16.

Each probe record is one (forward, site, layer) activation `[T, H]` summarized
into: global scalars (absmax/std/rms/frac_large), full per-token vectors
(tok_l2/tok_absmax/tok_kurtosis, length T = canvas positions), full per-channel
vectors (ch_absmax/ch_rms/ch_mean, length H), and the per-block absmax
distribution (blk_absmax_max/p99/p50/mean) that fp4 scaling keys off of. A
companion "ctx" record per forward maps forward_id -> denoising step, which is
how we put the right step label on every chart.

The files are large (hundreds of MB) because the per-channel vectors are stored
in full. We therefore stream them twice: pass 1 reads every record but keeps
only scalars (computing per-vector maxima on the fly, then discarding the
vectors); pass 2 re-streams and retains the full vectors for a handful of
representative layer slices, which is what the per-channel / per-token spectrum
charts plot.

Output (default runs/sky/act_report/):
  index.html                 overview, dims, nav to everything
  heat_<metric>.html         layer x step heatmap, per site, mxfp4 / bf16 / diff
  channels_<site>.html       full per-channel spectra at representative slices
  tokens_<site>.html         full per-token profiles at representative slices
  blocks.html                fp4 block-absmax inflation (outlier cost) per site
  outliers.html              ranked outlier-channel / outlier-token tables

Usage:
  ./analyze_activations.py
  ./analyze_activations.py runs/sky/bf16_act.jsonl runs/sky/mxfp4_act.jsonl
  ./analyze_activations.py --out runs/sky/act_report --slices 5
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
import os
from collections import defaultdict

import numpy as np

# Canonical GEMM-site order for stable layout across pages.
SITE_ORDER = [
    "qkv_proj",
    "o_proj",
    "mlp.gate_up",
    "mlp.down",
    "router",
    "moe.predispatch",
    "ple.gate",
    "ple.proj",
]

# (filename-suffix, metric key in the scalar dict, human title, color scale,
# explanation). "log" uses a log color/y scale (guarded against <=0); "lin" is
# a plain linear scale. Each cell of a heatmap is ONE activation tensor [T, H]
# feeding one GEMM (T = canvas-position tokens this forward, H = that GEMM's
# input width), reduced to the single number described below.
HEAT_METRICS = [
    ("absmax", "absmax", "global |activation| max", "log",
     "The single largest absolute value anywhere in the activation tensor "
     "feeding this GEMM (max over all T tokens x H channels). This is the "
     "biggest number fp4 has to represent. One cell = one (layer, step). "
     "Brighter = a bigger spike; if mxfp4's cell is brighter than bf16's at "
     "the same layer/step, quantization is dealing with a larger extreme."),
    ("std", "std", "global std", "log",
     "Standard deviation over every element of the activation tensor -- the "
     "typical spread of values. Compare against 'global |activation| max': if "
     "absmax is huge but std is small, a few outliers dominate an otherwise "
     "small tensor (the bad case for fp4)."),
    ("rms", "rms", "global RMS", "log",
     "Root-mean-square over every element -- the overall 'energy'/scale of the "
     "activation. Similar to std but includes the mean. Useful as the baseline "
     "magnitude that the outliers (absmax) stick out above."),
    ("frac_large", "frac_large", "fraction |a| > k*std (outlier mass)", "lin",
     "Fraction of elements whose magnitude exceeds k*std (k=6 by default) -- "
     "i.e. what share of the tensor counts as outliers. 0.001 = 0.1% of values "
     "are extreme. Small but non-zero is the classic outlier signature: rare "
     "huge values riding on top of a tight distribution."),
    ("blk_max", "blk_max", "max per-block absmax (fp4 worst block)", "log",
     "fp4 quantizes in contiguous blocks of 32 channels, giving each block one "
     "shared scale set by that block's largest magnitude (its absmax). This is "
     "the LARGEST such block-absmax in the tensor -- the worst-quantized block, "
     "where one outlier forces a coarse scale on its 31 neighbors."),
    ("blk_p50", "blk_p50", "median per-block absmax", "log",
     "The MEDIAN block-absmax across all blocks in the tensor -- what a typical "
     "block's fp4 scale is set by. Contrast with 'max per-block absmax': the "
     "gap between them is how much the worst block is inflated vs normal."),
    ("blk_ratio", "blk_ratio", "block absmax inflation (max / p50)", "log",
     "max-block-absmax divided by median-block-absmax. This is the headline fp4 "
     "damage metric: how many times larger the worst block's scale is than a "
     "typical block. 1 = uniform (fp4 is happy). 50 = the worst block is "
     "quantized 50x coarser than typical -- outliers are wrecking precision."),
    ("tok_absmax_max", "tok_absmax_max", "worst-token absmax", "log",
     "Each token (canvas position) has a max-magnitude across its H channels; "
     "this is the largest of those over all tokens -- i.e. the single most "
     "extreme token in this activation. High = an 'outlier token' (one canvas "
     "position whose activation blows up). See the per-token pages to find "
     "WHICH position."),
    ("tok_kurt_max", "tok_kurt_max", "worst-token kurtosis", "log",
     "Kurtosis measures peakedness: a Gaussian is 3, higher means a few "
     "channels dominate that token's vector. This is the highest per-token "
     "kurtosis in the tensor. High = some token's energy is concentrated in a "
     "handful of channels (spiky), which is exactly what fp4 blocks handle "
     "poorly."),
    ("ch_absmax_max", "ch_absmax_max", "worst-channel absmax", "log",
     "Each channel (hidden dim) has a max-magnitude over all tokens; this is "
     "the largest of those over all channels -- the single most extreme "
     "channel. High = an 'outlier channel' (one hidden dim that is persistently "
     "large). See the per-channel pages to find WHICH channel index."),
]

TAGS = ("bf16", "mxfp4")
TAG_COLOR = {"bf16": "#2a7", "mxfp4": "#c44"}


def iter_records(path: str):
    """Yield parsed JSON objects from a probe JSONL, skipping junk.

    Snapshot files made with ``tail -c`` can begin mid-line, and the server
    holds the file open so a trailing partial line is possible; both are skipped
    rather than fatal.
    """
    with open(path, errors="replace") as f:
        for line in f:
            line = line.strip().strip("\x00")
            if not line or line[0] != "{":
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _safe_max(xs) -> float:
    return float(max(xs)) if xs else float("nan")


def pass1(path: str):
    """First stream: scalar metrics for every act record + forward->step map.

    Returns:
      scal: {site: {(layer, step): metrics}} with the last forward for a given
            step winning (handles request resets within one snapshot).
      info: dict with layers/steps/sites sets, per-site hidden dim, counts,
            and the global absmax per run for the summary page.
    """
    fwd2step: dict[int, int] = {}
    # Buffer by raw forward first; we only know each forward's step once its ctx
    # line is seen (ctx is emitted after the forward's act lines).
    raw: dict[str, dict[tuple[int, int], dict]] = defaultdict(dict)
    hidden_by_site: dict[str, int] = {}
    n_act = 0
    n_ctx = 0
    n_prefill = 0
    global_absmax = 0.0

    for r in iter_records(path):
        kind = r.get("kind")
        if kind == "ctx":
            n_ctx += 1
            step_by_slot = r.get("step_by_slot") or {}
            if step_by_slot:
                # First slot's step (single-request probing runs).
                step = int(next(iter(step_by_slot.values())))
                fwd2step[int(r["forward_id"])] = step
            continue
        if kind != "act":
            continue
        n_act += 1
        site = r["site"]
        layer = int(r["layer"])
        fwd = int(r["forward_id"])
        hidden_by_site[site] = int(r["hidden"])
        absmax = float(r.get("absmax", float("nan")))
        if absmax == absmax:  # not NaN
            global_absmax = max(global_absmax, absmax)
        blk_max = float(r.get("blk_absmax_max", float("nan")))
        blk_p50 = float(r.get("blk_absmax_p50", float("nan")))
        ratio = blk_max / blk_p50 if blk_p50 and blk_p50 == blk_p50 else float("nan")
        raw[site][(layer, fwd)] = {
            "absmax": absmax,
            "std": float(r.get("std", float("nan"))),
            "rms": float(r.get("rms", float("nan"))),
            "mean": float(r.get("mean", float("nan"))),
            "frac_large": float(r.get("frac_large", float("nan"))),
            "blk_max": blk_max,
            "blk_p99": float(r.get("blk_absmax_p99", float("nan"))),
            "blk_p50": blk_p50,
            "blk_mean": float(r.get("blk_absmax_mean", float("nan"))),
            "blk_ratio": ratio,
            "n_tokens": int(r.get("n_tokens", 0)),
            "hidden": int(r.get("hidden", 0)),
            # Derived from the full vectors, kept so we never store the vectors.
            "tok_absmax_max": _safe_max(r.get("tok_absmax", [])),
            "tok_l2_max": _safe_max(r.get("tok_l2", [])),
            "tok_kurt_max": _safe_max(r.get("tok_kurtosis", [])),
            "ch_absmax_max": _safe_max(r.get("ch_absmax", [])),
            "ch_rms_max": _safe_max(r.get("ch_rms", [])),
        }

    # Collapse (layer, forward) -> (layer, step), last forward wins per step.
    scal: dict[str, dict[tuple[int, int], dict]] = defaultdict(dict)
    layers: set[int] = set()
    steps: set[int] = set()
    for site, by_lf in raw.items():
        for (layer, fwd) in sorted(by_lf):
            if fwd not in fwd2step:
                n_prefill += 1
                continue
            step = fwd2step[fwd]
            scal[site][(layer, step)] = by_lf[(layer, fwd)]
            layers.add(layer)
            steps.add(step)

    info = {
        "layers": sorted(layers),
        "steps": sorted(steps),
        "sites": [s for s in SITE_ORDER if s in scal]
        + [s for s in scal if s not in SITE_ORDER],
        "hidden_by_site": hidden_by_site,
        "fwd2step": fwd2step,
        "n_act": n_act,
        "n_ctx": n_ctx,
        "n_prefill": n_prefill,
        "global_absmax": global_absmax,
    }
    return scal, info


def pass2(path: str, fwd2step: dict[int, int], keep_layers: set[int]):
    """Second stream: keep full per-channel / per-token vectors for `keep_layers`.

    Returns {site: {(layer, step): {ch_absmax, ch_rms, ch_mean, tok_absmax,
    tok_l2, tok_kurtosis}}}, last forward per step winning.
    """
    raw: dict[str, dict[tuple[int, int], dict]] = defaultdict(dict)
    for r in iter_records(path):
        if r.get("kind") != "act":
            continue
        layer = int(r["layer"])
        if layer not in keep_layers:
            continue
        fwd = int(r["forward_id"])
        if fwd not in fwd2step:
            continue
        raw[r["site"]][(layer, fwd)] = {
            "ch_absmax": np.asarray(r.get("ch_absmax", []), dtype=np.float64),
            "ch_rms": np.asarray(r.get("ch_rms", []), dtype=np.float64),
            "ch_mean": np.asarray(r.get("ch_mean", []), dtype=np.float64),
            "tok_absmax": np.asarray(r.get("tok_absmax", []), dtype=np.float64),
            "tok_l2": np.asarray(r.get("tok_l2", []), dtype=np.float64),
            "tok_kurtosis": np.asarray(r.get("tok_kurtosis", []), dtype=np.float64),
        }
    vec: dict[str, dict[tuple[int, int], dict]] = defaultdict(dict)
    for site, by_lf in raw.items():
        for (layer, fwd) in sorted(by_lf):
            vec[site][(layer, fwd2step[fwd])] = by_lf[(layer, fwd)]
    return vec


def pass_paper(path: str, fwd2step: dict[int, int], want_step: int):
    """Stream once, keeping per-channel vectors for EVERY layer at one step.

    The paper-style charts (SmoothQuant channel x layer surface, AWQ salient-
    channel spectrum, SpinQuant kurtosis-per-layer) need every layer's full
    per-channel vector but only at a single representative denoising step, so we
    hold far less than ``pass2`` would across all steps.

    Returns {site: {layer: {ch_absmax, ch_mean, ch_rms, kurt_mean, n_tokens}}}.
    """
    raw: dict[str, dict[tuple[int, int], dict]] = defaultdict(dict)
    for r in iter_records(path):
        if r.get("kind") != "act":
            continue
        fwd = int(r["forward_id"])
        if fwd2step.get(fwd) != want_step:
            continue
        layer = int(r["layer"])
        tk = r.get("tok_kurtosis", [])
        raw[r["site"]][(layer, fwd)] = {
            "ch_absmax": np.asarray(r.get("ch_absmax", []), dtype=np.float64),
            "ch_mean": np.asarray(r.get("ch_mean", []), dtype=np.float64),
            "ch_rms": np.asarray(r.get("ch_rms", []), dtype=np.float64),
            "kurt_mean": float(np.mean(tk)) if tk else float("nan"),
            "n_tokens": int(r.get("n_tokens", 0)),
        }
    out: dict[str, dict[int, dict]] = defaultdict(dict)
    for site, by_lf in raw.items():
        for (layer, _fwd) in sorted(by_lf):
            out[site][layer] = by_lf[(layer, _fwd)]
    return out


def _downsample_max(v: np.ndarray, n: int) -> np.ndarray:
    """Max-pool a 1-D vector down to ~n bins, preserving spikes (outliers)."""
    if v.size <= n:
        return v
    nb = n
    trim = (v.size // nb) * nb
    if trim == 0:
        return v
    head = v[:trim].reshape(nb, -1).max(axis=1)
    if trim < v.size:
        head = np.concatenate([head, [v[trim:].max()]])
    return head


def pick_slices(values: list[int], k: int) -> list[int]:
    """Pick up to k evenly spaced entries (always including first and last)."""
    if not values:
        return []
    if len(values) <= k:
        return list(values)
    idx = np.linspace(0, len(values) - 1, k).round().astype(int)
    out = []
    for i in idx:
        v = values[int(i)]
        if v not in out:
            out.append(v)
    return out


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def _import_plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def fig_to_b64(fig) -> str:
    import matplotlib.pyplot as plt

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=96, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def img_tag(b64: str) -> str:
    return f'<img src="data:image/png;base64,{b64}">'


NAV_NOTE = (
    "magenta/red = mxfp4 (W4A4), green = bf16. Outliers inflate fp4 block "
    "scales, so big per-block absmax = more quantization error."
)


def page(out_dir: str, fname: str, title: str, nav: str, body: str,
         explain: str = "") -> None:
    explain_html = (
        f'<div class="explain"><b>What am I looking at?</b><br>{explain}</div>'
        if explain else ""
    )
    doc = f"""<!doctype html><html><head><meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>
 body {{ font-family: system-ui, sans-serif; margin: 16px; background:#111;
        color:#ddd; }}
 h1 {{ font-size: 20px; }}
 h2 {{ font-size: 16px; margin-top: 28px; border-bottom:1px solid #333;
       padding-bottom:4px; }}
 a {{ color:#6cf; }}
 .nav {{ position:sticky; top:0; background:#111; padding:8px 0;
         border-bottom:1px solid #333; margin-bottom:12px; font-size:13px;
         line-height:1.8; z-index:10; }}
 .note {{ color:#999; font-size:12px; margin:6px 0 14px; }}
 .explain {{ background:#16202b; border:1px solid #2c4257;
             border-left:4px solid #6cf; border-radius:4px; padding:10px 14px;
             margin:10px 0 16px; font-size:13.5px; line-height:1.55;
             max-width:1000px; color:#cfe3f2; }}
 .explain b {{ color:#9fd0ff; }}
 .explain code {{ color:#fc6; }}
 img {{ max-width:100%; background:#fff; border:1px solid #333; margin:6px 0; }}
 table {{ border-collapse:collapse; font-size:12px; margin:8px 0; }}
 td, th {{ border:1px solid #333; padding:3px 8px; text-align:right; }}
 th {{ background:#1c1c1c; }}
 code {{ color:#fc6; }}
</style></head><body>
<div class="nav">{nav}</div>
<h1>{html.escape(title)}</h1>
<div class="note">{html.escape(NAV_NOTE)}</div>
{explain_html}
{body}
</body></html>"""
    with open(os.path.join(out_dir, fname), "w") as f:
        f.write(doc)


def build_nav(pages: list[tuple[str, str]]) -> str:
    return " &nbsp;|&nbsp; ".join(
        f'<a href="{fn}">{html.escape(t)}</a>' for fn, t in pages
    )


def matrix_for(scal_site: dict, metric: str, layers: list[int],
               steps: list[int]) -> np.ndarray:
    """Build a layer x step matrix for one metric (NaN where missing)."""
    m = np.full((len(layers), len(steps)), np.nan)
    li = {l: i for i, l in enumerate(layers)}
    si = {s: i for i, s in enumerate(steps)}
    for (layer, step), rec in scal_site.items():
        if layer in li and step in si:
            m[li[layer], si[step]] = rec.get(metric, np.nan)
    return m


def heat_page(out_dir, fname, title, nav, scal_by_tag, info_by_tag,
              metric, scale, explain=""):
    """One metric: rows=sites, cols=[mxfp4, bf16, diff] heatmaps over layer x step."""
    plt = _import_plt()
    from matplotlib.colors import LogNorm, TwoSlopeNorm

    # Union layout so both runs share axes.
    layers = sorted(set(info_by_tag["bf16"]["layers"])
                    | set(info_by_tag["mxfp4"]["layers"]))
    steps = sorted(set(info_by_tag["bf16"]["steps"])
                   | set(info_by_tag["mxfp4"]["steps"]))
    sites = [s for s in SITE_ORDER
             if s in scal_by_tag["bf16"] or s in scal_by_tag["mxfp4"]]
    if not sites or not steps:
        page(out_dir, fname, title, nav, "<p>(no data)</p>", explain)
        return

    nrows = len(sites)
    fig, axes = plt.subplots(nrows, 3, figsize=(15, 2.6 * nrows + 1),
                             squeeze=False)
    mats = {}
    vmax = 0.0
    vmin = np.inf
    for tag in TAGS:
        for site in sites:
            mm = matrix_for(scal_by_tag[tag].get(site, {}), metric, layers, steps)
            mats[(tag, site)] = mm
            if np.isfinite(mm).any():
                vmax = max(vmax, np.nanmax(mm))
                pos = mm[np.isfinite(mm) & (mm > 0)]
                if pos.size:
                    vmin = min(vmin, float(pos.min()))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    if not np.isfinite(vmin):
        vmin = vmax / 1e3 if vmax > 0 else 1e-6

    if scale == "log":
        norm = LogNorm(vmin=max(vmin, vmax / 1e6), vmax=vmax)
        seq_kw = dict(norm=norm, cmap="inferno")
    else:
        norm = None
        seq_kw = dict(vmin=0, vmax=vmax, cmap="inferno")

    def ticks(ax):
        if len(steps) > 12:
            sel = np.linspace(0, len(steps) - 1, 12).round().astype(int)
        else:
            sel = range(len(steps))
        ax.set_xticks(list(sel))
        ax.set_xticklabels([str(steps[i]) for i in sel], fontsize=6)
        if len(layers) > 16:
            lsel = np.linspace(0, len(layers) - 1, 16).round().astype(int)
        else:
            lsel = range(len(layers))
        ax.set_yticks(list(lsel))
        ax.set_yticklabels([str(layers[i]) for i in lsel], fontsize=6)

    for ri, site in enumerate(sites):
        mfp = mats[("mxfp4", site)]
        mbf = mats[("bf16", site)]
        for ci, (lab, mm, kw) in enumerate([
            ("mxfp4", mfp, seq_kw),
            ("bf16", mbf, seq_kw),
            ("diff", mfp - mbf, None),
        ]):
            ax = axes[ri][ci]
            if ci == 2:
                d = mfp - mbf
                lim = np.nanmax(np.abs(d)) if np.isfinite(d).any() else 1.0
                lim = lim or 1.0
                im = ax.imshow(d, aspect="auto", origin="lower",
                               cmap="coolwarm",
                               norm=TwoSlopeNorm(0, -lim, lim))
            else:
                im = ax.imshow(mm, aspect="auto", origin="lower", **kw)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            ax.set_title(f"{site} - {lab}", fontsize=8)
            ticks(ax)
            if ci == 0:
                ax.set_ylabel("layer", fontsize=7)
            if ri == nrows - 1:
                ax.set_xlabel("denoising step", fontsize=7)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    body = (f"<p class='note'>x = denoising step, y = decoder layer. "
            f"diff = mxfp4 - bf16 (red = mxfp4 larger).</p>{img_tag(fig_to_b64(fig))}")
    page(out_dir, fname, title, nav, body, explain)


def channels_page(out_dir, fname, site, nav, vec_by_tag, slice_layers,
                  rep_step, steps_by_tag):
    """Per-channel spectra for one site at representative layer slices."""
    explain = (
        f"Each curve is the per-channel absmax of the activation feeding the "
        f"<code>{html.escape(site)}</code> GEMM at ONE layer/step: for every one "
        "of the H hidden channels (x-axis), the largest magnitude that channel "
        "reaches across all tokens. <b>Left plot</b> = value at each channel "
        "index, so a tall spike is an 'outlier channel' (a specific hidden dim "
        "that is persistently large). <b>Right plot</b> = the exact same numbers "
        "sorted high-to-low on a log y-axis, which shows how few channels carry "
        "the extremes and how far above the rest they sit. <b>green = bf16, "
        "red = mxfp4</b>; where red towers over green, quantization is facing a "
        "bigger outlier. A few representative layers are shown (one row each)."
    )
    plt = _import_plt()
    rows = [l for l in slice_layers]
    if not rows:
        page(out_dir, fname, f"channels: {site}", nav, "<p>(no data)</p>",
             explain)
        return
    fig, axes = plt.subplots(len(rows), 2, figsize=(14, 3.1 * len(rows)),
                             squeeze=False)
    for ri, layer in enumerate(rows):
        ax_raw, ax_sorted = axes[ri][0], axes[ri][1]
        for tag in TAGS:
            site_vec = vec_by_tag[tag].get(site, {})
            step = rep_step if (layer, rep_step) in site_vec else None
            if step is None:
                avail = sorted(s for (l, s) in site_vec if l == layer)
                if not avail:
                    continue
                step = avail[len(avail) // 2]
            v = site_vec[(layer, step)]["ch_absmax"]
            if v.size == 0:
                continue
            c = TAG_COLOR[tag]
            ax_raw.plot(v, color=c, lw=0.5, alpha=0.85,
                        label=f"{tag} (step {step})")
            sv = np.sort(v)[::-1]
            ax_sorted.plot(sv, color=c, lw=1.0, label=f"{tag} (step {step})")
        ax_raw.set_title(f"layer {layer}: ch_absmax vs channel index",
                         fontsize=9)
        ax_raw.set_xlabel("channel index", fontsize=7)
        ax_raw.set_ylabel("|a| max", fontsize=7)
        ax_raw.legend(fontsize=6)
        ax_sorted.set_title(f"layer {layer}: ch_absmax sorted (outlier spectrum)",
                            fontsize=9)
        ax_sorted.set_yscale("log")
        ax_sorted.set_xlabel("channel rank", fontsize=7)
        ax_sorted.legend(fontsize=6)
    fig.suptitle(f"outlier channels: {site}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    body = ("<p class='note'>Left: per-channel absmax across the hidden dim "
            "(spikes = outlier channels). Right: same values sorted descending "
            "on a log scale (how few channels dominate, and by how much).</p>"
            f"{img_tag(fig_to_b64(fig))}")
    page(out_dir, fname, f"channels: {site}", nav, body, explain)


def tokens_page(out_dir, fname, site, nav, vec_by_tag, slice_layers, rep_step):
    """Per-token profiles for one site at representative layer slices."""
    explain = (
        f"Each curve profiles the tokens (canvas positions, x-axis) of the "
        f"activation feeding the <code>{html.escape(site)}</code> GEMM at ONE "
        "layer/step. <b>Left plot</b> = per-token absmax: for each position, the "
        "largest magnitude across its H channels -- a tall spike is an 'outlier "
        "token' (one canvas position whose activation blows up). <b>Right plot</b> "
        "= per-token kurtosis (log y): how peaked that token's channel vector is "
        "(Gaussian ~ 3; higher means a few channels dominate that token). "
        "<b>green = bf16, red = mxfp4.</b> The x-axis position is the literal "
        "canvas slot, so you can line a spike up with the token pages in the "
        "confidence report. A few representative layers are shown (one row each)."
    )
    plt = _import_plt()
    rows = [l for l in slice_layers]
    if not rows:
        page(out_dir, fname, f"tokens: {site}", nav, "<p>(no data)</p>", explain)
        return
    fig, axes = plt.subplots(len(rows), 2, figsize=(14, 3.1 * len(rows)),
                             squeeze=False)
    for ri, layer in enumerate(rows):
        ax_a, ax_k = axes[ri][0], axes[ri][1]
        for tag in TAGS:
            site_vec = vec_by_tag[tag].get(site, {})
            step = rep_step if (layer, rep_step) in site_vec else None
            if step is None:
                avail = sorted(s for (l, s) in site_vec if l == layer)
                if not avail:
                    continue
                step = avail[len(avail) // 2]
            d = site_vec[(layer, step)]
            c = TAG_COLOR[tag]
            if d["tok_absmax"].size:
                ax_a.plot(d["tok_absmax"], color=c, lw=0.8, marker=".",
                          ms=2, label=f"{tag} (step {step})")
            if d["tok_kurtosis"].size:
                ax_k.plot(d["tok_kurtosis"], color=c, lw=0.8, marker=".",
                          ms=2, label=f"{tag} (step {step})")
        ax_a.set_title(f"layer {layer}: tok_absmax vs canvas position",
                       fontsize=9)
        ax_a.set_xlabel("canvas position", fontsize=7)
        ax_a.set_ylabel("|a| max", fontsize=7)
        ax_a.legend(fontsize=6)
        ax_k.set_title(f"layer {layer}: tok_kurtosis vs canvas position",
                       fontsize=9)
        ax_k.set_yscale("log")
        ax_k.set_xlabel("canvas position", fontsize=7)
        ax_k.legend(fontsize=6)
    fig.suptitle(f"outlier tokens: {site}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    body = ("<p class='note'>Left: per-token absmax across canvas positions "
            "(spikes = outlier tokens). Right: per-token kurtosis (peakedness; "
            "high = a few channels dominate that token).</p>"
            f"{img_tag(fig_to_b64(fig))}")
    page(out_dir, fname, f"tokens: {site}", nav, body, explain)


def blocks_page(out_dir, fname, nav, scal_by_tag, info_by_tag):
    """fp4 block-absmax inflation per site: max vs p50 over steps, all layers."""
    explain = (
        "fp4 splits each activation row into contiguous blocks of 32 channels "
        "and gives each block ONE shared scale, set by that block's largest "
        "magnitude (its absmax). One panel per GEMM site. Each line tracks, at "
        "each denoising step (x-axis), the worst block's absmax averaged over "
        "all 30 layers (log y-axis). <b>Higher = a bigger scale forced on a "
        "block, so its 31 other (smaller) values get quantized more coarsely</b> "
        "-- i.e. more fp4 error. <b>green = bf16, red = mxfp4.</b> Watch how the "
        "red line moves across denoising steps and how the sites differ: the "
        "site/step where red is highest is where fp4 hurts most."
    )
    plt = _import_plt()
    sites = [s for s in SITE_ORDER
             if s in scal_by_tag["bf16"] or s in scal_by_tag["mxfp4"]]
    if not sites:
        page(out_dir, fname, "block / fp4 cost", nav, "<p>(no data)</p>",
             explain)
        return
    fig, axes = plt.subplots(len(sites), 1, figsize=(12, 2.6 * len(sites)),
                             squeeze=False)
    for ri, site in enumerate(sites):
        ax = axes[ri][0]
        for tag in TAGS:
            sd = scal_by_tag[tag].get(site, {})
            # Aggregate over layers: mean per step of (blk_max) and (blk_ratio).
            by_step_max = defaultdict(list)
            by_step_ratio = defaultdict(list)
            for (layer, step), rec in sd.items():
                if rec.get("blk_max") == rec.get("blk_max"):
                    by_step_max[step].append(rec["blk_max"])
                if rec.get("blk_ratio") == rec.get("blk_ratio"):
                    by_step_ratio[step].append(rec["blk_ratio"])
            steps = sorted(by_step_max)
            if not steps:
                continue
            mx = [np.mean(by_step_max[s]) for s in steps]
            c = TAG_COLOR[tag]
            ax.plot(steps, mx, color=c, lw=1.2, label=f"{tag} mean blk_max")
        ax.set_title(f"{site}: mean per-block absmax (max block) over steps",
                     fontsize=9)
        ax.set_yscale("log")
        ax.set_xlabel("denoising step", fontsize=7)
        ax.set_ylabel("blk absmax", fontsize=7)
        ax.legend(fontsize=6)
    fig.suptitle("fp4 block-absmax (outlier cost) per site", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    body = ("<p class='note'>Higher per-block absmax = larger fp4 scale for that "
            "block = coarser quantization of its other (small) values. Averaged "
            "over layers at each denoising step.</p>"
            f"{img_tag(fig_to_b64(fig))}")
    page(out_dir, fname, "block / fp4 cost", nav, body, explain)


def paper_surface_page(out_dir, fname, nav, paper_by_tag, step_by_tag):
    """SmoothQuant Fig.4 analog: |activation| surface over channel x layer.

    The classic SmoothQuant plot is |a| over (token, channel) for one tensor and
    shows outliers form ridges along the CHANNEL axis. We only store per-channel
    marginals (max over tokens), not the full matrix, so the faithful analog is
    a (channel x layer) surface of per-channel absmax for one GEMM site: the same
    'a few channels are huge everywhere' story, here persisting across depth.
    """
    explain = (
        "<b>SmoothQuant Fig. 4 analog.</b> The famous SmoothQuant plot is a 3D "
        "surface of |activation| over (token, channel) showing outliers form "
        "<b>ridges along the channel axis</b> -- a few hidden channels are huge "
        "for essentially every token. We record per-channel marginals (max over "
        "tokens), so here the surface is <b>channel index (x) x decoder layer "
        "(y)</b>, color = that channel's max |activation| at one representative "
        "denoising step. A bright vertical streak = an outlier channel that stays "
        "large across depth -- exactly the channel-wise structure SmoothQuant "
        "exploits. One row of panels per GEMM site; <b>left = bf16, right = "
        "mxfp4</b>. Channels are max-pooled to ~512 columns so spikes survive."
    )
    plt = _import_plt()
    from matplotlib.colors import LogNorm

    sites = [s for s in SITE_ORDER
             if s in paper_by_tag["bf16"] or s in paper_by_tag["mxfp4"]]
    if not sites:
        page(out_dir, fname, "paper: channel x layer surface", nav,
             "<p>(no data)</p>", explain)
        return
    fig, axes = plt.subplots(len(sites), 2, figsize=(14, 2.4 * len(sites) + 1),
                             squeeze=False)
    NCOL = 512
    for ri, site in enumerate(sites):
        # Shared color scale across both models for this site.
        vmax = 0.0
        vmin = np.inf
        grids = {}
        for tag in TAGS:
            ld = paper_by_tag[tag].get(site, {})
            layers = sorted(ld)
            if not layers:
                continue
            rows = [_downsample_max(ld[l]["ch_absmax"], NCOL) for l in layers]
            w = max(r.size for r in rows)
            g = np.full((len(layers), w), np.nan)
            for i, r in enumerate(rows):
                g[i, :r.size] = r
            grids[tag] = (g, layers)
            if np.isfinite(g).any():
                vmax = max(vmax, np.nanmax(g))
                pos = g[np.isfinite(g) & (g > 0)]
                if pos.size:
                    vmin = min(vmin, float(pos.min()))
        if vmax <= 0:
            vmax = 1.0
        if not np.isfinite(vmin):
            vmin = vmax / 1e3
        norm = LogNorm(vmin=max(vmin, vmax / 1e6), vmax=vmax)
        for ci, tag in enumerate(TAGS):
            ax = axes[ri][ci]
            if tag not in grids:
                ax.set_axis_off()
                continue
            g, layers = grids[tag]
            im = ax.imshow(g, aspect="auto", origin="lower", cmap="inferno",
                           norm=norm)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            st = step_by_tag.get(tag, "?")
            ax.set_title(f"{site} - {tag} (step {st})", fontsize=8)
            ax.set_xlabel("channel (max-pooled)", fontsize=7)
            ax.set_ylabel("layer", fontsize=7)
            if len(layers) > 16:
                lsel = np.linspace(0, len(layers) - 1, 16).round().astype(int)
            else:
                lsel = range(len(layers))
            ax.set_yticks(list(lsel))
            ax.set_yticklabels([str(layers[i]) for i in lsel], fontsize=6)
    fig.suptitle("per-channel |activation| over channel x layer "
                 "(SmoothQuant-style)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    body = ("<p class='note'>Bright vertical streaks = outlier channels that "
            "persist across layers. bf16 (left) vs mxfp4 (right), one "
            "denoising step.</p>" + img_tag(fig_to_b64(fig)))
    page(out_dir, fname, "paper: channel x layer surface", nav, body, explain)


def paper_spectrum_page(out_dir, fname, nav, paper_by_tag, step_by_tag):
    """AWQ / LLM.int8() analog: sorted per-channel magnitude + heavy-tail hist.

    Aggregates each channel's magnitude across ALL layers (max over layers of the
    per-channel absmax) for one site, then shows (left) the sorted descending
    spectrum on log-y -- the canonical 'a tiny set of channels tower over the
    rest' chart -- and (right) the histogram of per-channel magnitudes on log-y,
    the heavy-tail view (QuaRot/SpinQuant before/after style, bf16 vs mxfp4).
    """
    explain = (
        "<b>AWQ / LLM.int8() / QuaRot analog.</b> For one GEMM site we take each "
        "channel's magnitude (per-channel absmax) and reduce over all layers, "
        "then: <b>Left</b> = that magnitude <b>sorted high-to-low</b> on a log "
        "y-axis -- the canonical 'salient channel' spectrum showing how a tiny "
        "fraction of channels tower 10-100x over the rest (AWQ picks exactly "
        "these; LLM.int8() routes them to higher precision). <b>Right</b> = the "
        "<b>histogram</b> of per-channel magnitudes (log y) -- the heavy-tail "
        "view that QuaRot/SpinQuant show shrinking after rotation. <b>green = "
        "bf16, red = mxfp4</b>; a longer/heavier red tail means mxfp4 is "
        "carrying larger activation extremes into fp4. One row per GEMM site."
    )
    plt = _import_plt()
    sites = [s for s in SITE_ORDER
             if s in paper_by_tag["bf16"] or s in paper_by_tag["mxfp4"]]
    if not sites:
        page(out_dir, fname, "paper: channel spectrum", nav, "<p>(no data)</p>",
             explain)
        return
    fig, axes = plt.subplots(len(sites), 2, figsize=(14, 2.8 * len(sites) + 1),
                             squeeze=False)
    for ri, site in enumerate(sites):
        ax_sp, ax_hist = axes[ri][0], axes[ri][1]
        for tag in TAGS:
            ld = paper_by_tag[tag].get(site, {})
            if not ld:
                continue
            # Per-channel magnitude reduced over layers (max).
            stack = [ld[l]["ch_absmax"] for l in sorted(ld)
                     if ld[l]["ch_absmax"].size]
            if not stack:
                continue
            w = max(v.size for v in stack)
            agg = np.zeros(w)
            for v in stack:
                agg[:v.size] = np.maximum(agg[:v.size], v)
            c = TAG_COLOR[tag]
            sv = np.sort(agg)[::-1]
            ax_sp.plot(sv, color=c, lw=1.2, label=tag)
            pos = agg[agg > 0]
            if pos.size:
                ax_hist.hist(pos, bins=80, color=c, alpha=0.5, label=tag,
                             log=True)
        ax_sp.set_title(f"{site}: per-channel magnitude, sorted", fontsize=9)
        ax_sp.set_yscale("log")
        ax_sp.set_xlabel("channel rank", fontsize=7)
        ax_sp.set_ylabel("max |a| over layers", fontsize=7)
        ax_sp.legend(fontsize=6)
        ax_hist.set_title(f"{site}: histogram of per-channel magnitude",
                          fontsize=9)
        ax_hist.set_xlabel("|a|", fontsize=7)
        ax_hist.set_ylabel("channel count (log)", fontsize=7)
        ax_hist.legend(fontsize=6)
    fig.suptitle("salient-channel spectrum & heavy tail (AWQ / LLM.int8 style)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    body = ("<p class='note'>Left: sorted per-channel magnitude (few channels "
            "dominate). Right: heavy-tailed distribution of channel magnitudes. "
            "green = bf16, red = mxfp4.</p>" + img_tag(fig_to_b64(fig)))
    page(out_dir, fname, "paper: channel spectrum", nav, body, explain)


def paper_kurtosis_page(out_dir, fname, nav, paper_by_tag, step_by_tag):
    """SpinQuant Fig.3 analog: activation kurtosis per layer, per site.

    SpinQuant plots per-layer kurtosis dropping toward 3 (Gaussian) after
    rotation. We don't rotate, but bf16 vs mxfp4 is the natural A/B: higher
    kurtosis = more outlier-dominated (spiky) activations.
    """
    explain = (
        "<b>SpinQuant Fig. 3 analog.</b> Kurtosis measures how outlier-dominated "
        "a distribution is: a Gaussian has kurtosis 3, higher means a few values "
        "stick far out. SpinQuant shows per-layer activation kurtosis collapsing "
        "toward 3 after rotation removes outliers. Here each line is one GEMM "
        "site, plotting <b>mean per-token kurtosis (y, log) vs decoder layer "
        "(x)</b> at one denoising step. <b>Solid = bf16, dashed = mxfp4.</b> "
        "Tall values flag layers whose activations are spiky (hard for fp4); a "
        "flat line near 3 would mean near-Gaussian (easy). This is the "
        "outlier-severity-by-depth view in a single chart."
    )
    plt = _import_plt()
    sites = [s for s in SITE_ORDER
             if s in paper_by_tag["bf16"] or s in paper_by_tag["mxfp4"]]
    if not sites:
        page(out_dir, fname, "paper: kurtosis per layer", nav,
             "<p>(no data)</p>", explain)
        return
    fig, ax = plt.subplots(figsize=(13, 6))
    cmap = plt.get_cmap("tab10")
    for si, site in enumerate(sites):
        col = cmap(si % 10)
        for tag in TAGS:
            ld = paper_by_tag[tag].get(site, {})
            layers = sorted(l for l in ld if ld[l]["kurt_mean"] == ld[l]["kurt_mean"])
            if not layers:
                continue
            ys = [ld[l]["kurt_mean"] for l in layers]
            ax.plot(layers, ys, color=col,
                    ls="-" if tag == "bf16" else "--",
                    lw=1.4, marker="." if tag == "bf16" else "x", ms=4,
                    label=f"{site} {tag}")
    ax.axhline(3, color="#888", lw=0.8, ls=":", label="Gaussian (3)")
    ax.set_yscale("log")
    ax.set_xlabel("decoder layer")
    ax.set_ylabel("mean per-token kurtosis (log)")
    ax.set_title("activation kurtosis per layer (SpinQuant-style); "
                 "solid=bf16 dashed=mxfp4", fontsize=11)
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    body = ("<p class='note'>Higher = spikier (outlier-dominated) activations. "
            "Solid = bf16, dashed = mxfp4. Dotted line = Gaussian kurtosis "
            "(3).</p>" + img_tag(fig_to_b64(fig)))
    page(out_dir, fname, "paper: kurtosis per layer", nav, body, explain)


def paper_range_page(out_dir, fname, nav, paper_by_tag, step_by_tag):
    """Outlier-Suppression+ analog: per-channel value range (asymmetric bands).

    OS+ shows per-channel min/max bands that are wide and ASYMMETRIC for outlier
    channels, motivating a per-channel shift. We have ch_mean and ch_absmax per
    channel; we draw the [mean-absmax_env, mean+...] band proxy via ch_mean as
    the center and +/- ch_absmax extent, sorted by extent, for one mid layer.
    """
    explain = (
        "<b>Outlier-Suppression+ analog.</b> OS+ shows that outlier channels "
        "have <b>wide, asymmetric value ranges</b> (e.g. one channel spans -97 "
        "to +43 while most span +/-3), which motivates shifting each channel to "
        "recenter it before quantizing. For one representative mid-network layer "
        "and GEMM site, each x-position is a channel (sorted by spread); the "
        "<b>shaded band</b> is roughly that channel's value range, centered on "
        "its mean (<code>ch_mean</code>) with extent set by its absmax. A few "
        "channels with bands far wider than the bulk -- and not centered on zero "
        "-- are the OS+ targets. <b>green = bf16, red = mxfp4.</b>"
    )
    plt = _import_plt()
    sites = [s for s in SITE_ORDER
             if s in paper_by_tag["bf16"] or s in paper_by_tag["mxfp4"]]
    if not sites:
        page(out_dir, fname, "paper: per-channel range", nav,
             "<p>(no data)</p>", explain)
        return
    fig, axes = plt.subplots(len(sites), 2, figsize=(14, 2.6 * len(sites) + 1),
                             squeeze=False)
    for ri, site in enumerate(sites):
        for ci, tag in enumerate(TAGS):
            ax = axes[ri][ci]
            ld = paper_by_tag[tag].get(site, {})
            if not ld:
                ax.set_axis_off()
                continue
            layers = sorted(ld)
            layer = layers[len(layers) // 2]
            d = ld[layer]
            absmax = d["ch_absmax"]
            mean = d["ch_mean"]
            if absmax.size == 0:
                ax.set_axis_off()
                continue
            n = min(absmax.size, mean.size)
            absmax, mean = absmax[:n], mean[:n]
            order = np.argsort(absmax)[::-1]
            absmax, mean = absmax[order], mean[order]
            x = np.arange(n)
            c = TAG_COLOR[tag]
            ax.fill_between(x, mean - absmax, mean + absmax, color=c, alpha=0.35,
                            lw=0)
            ax.plot(x, mean, color=c, lw=0.6)
            ax.axhline(0, color="#888", lw=0.6, ls=":")
            ax.set_title(f"{site} - {tag} (layer {layer})", fontsize=8)
            ax.set_xlabel("channel (sorted by spread)", fontsize=7)
            ax.set_ylabel("value range", fontsize=7)
    fig.suptitle("per-channel value ranges (Outlier-Suppression+ style)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    body = ("<p class='note'>Each band ~ one channel's value range "
            "(mean +/- absmax), sorted by spread. Wide off-center bands = "
            "outlier channels OS+ targets. green = bf16, red = mxfp4.</p>"
            + img_tag(fig_to_b64(fig)))
    page(out_dir, fname, "paper: per-channel range", nav, body, explain)


def outliers_page(out_dir, fname, nav, scal_by_tag, info_by_tag, topn=20):
    """Ranked tables: worst (site, layer, step) by channel/token/block metrics."""
    def rank(tag, metric):
        rows = []
        for site, sd in scal_by_tag[tag].items():
            for (layer, step), rec in sd.items():
                v = rec.get(metric)
                if v is not None and v == v:
                    rows.append((v, site, layer, step))
        rows.sort(reverse=True)
        return rows[:topn]

    def table(tag, metric, label):
        rows = rank(tag, metric)
        h = (f"<h2>{html.escape(tag)} - top {topn} by {html.escape(label)}</h2>"
             "<table><tr><th>rank</th><th>value</th><th>site</th>"
             "<th>layer</th><th>step</th></tr>")
        for i, (v, site, layer, step) in enumerate(rows, 1):
            h += (f"<tr><td>{i}</td><td>{v:.4g}</td>"
                  f"<td style='text-align:left'>{html.escape(site)}</td>"
                  f"<td>{layer}</td><td>{step}</td></tr>")
        return h + "</table>"

    body = ""
    for metric, label in [
        ("ch_absmax_max", "worst-channel absmax"),
        ("tok_absmax_max", "worst-token absmax"),
        ("tok_kurt_max", "worst-token kurtosis"),
        ("blk_max", "max per-block absmax"),
        ("blk_ratio", "block absmax inflation (max/p50)"),
        ("frac_large", "outlier mass (frac |a|>k*std)"),
    ]:
        body += f"<div style='display:flex; gap:24px; flex-wrap:wrap'>"
        for tag in TAGS:
            body += f"<div>{table(tag, metric, label)}</div>"
        body += "</div>"
    explain = (
        "Leaderboards of the most extreme single activations in the whole run. "
        "Each pair of tables ranks the top-20 (layer, GEMM site, denoising step) "
        "combinations for one metric -- bf16 on the left, mxfp4 on the right -- "
        "so you can jump straight to where outliers are worst instead of hunting "
        "through heatmaps. <b>value</b> is that metric's number; <b>site/layer/"
        "step</b> tell you exactly which activation to inspect on the per-channel "
        "or per-token pages. Metrics: worst-channel absmax (outlier channels), "
        "worst-token absmax &amp; kurtosis (outlier tokens), max per-block absmax "
        "and its inflation ratio (fp4 cost), and outlier mass (frac |a|>k*std)."
    )
    page(out_dir, fname, "outlier rankings", nav, body, explain)


def index_page(out_dir, nav, scal_by_tag, info_by_tag, pages):
    body = "<h2>runs</h2><table><tr><th>run</th><th>act records</th>" \
           "<th>ctx records</th><th>prefill (no step)</th>" \
           "<th>layers</th><th>steps</th><th>global absmax</th></tr>"
    for tag in TAGS:
        i = info_by_tag[tag]
        body += (f"<tr><td style='text-align:left'>{tag}</td>"
                 f"<td>{i['n_act']}</td><td>{i['n_ctx']}</td>"
                 f"<td>{i['n_prefill']}</td>"
                 f"<td>{len(i['layers'])}</td><td>{len(i['steps'])}</td>"
                 f"<td>{i['global_absmax']:.4g}</td></tr>")
    body += "</table>"

    body += "<h2>sites & input dims (hidden H per GEMM)</h2>"
    body += "<table><tr><th>site</th><th>bf16 H</th><th>mxfp4 H</th></tr>"
    sites = [s for s in SITE_ORDER
             if s in scal_by_tag["bf16"] or s in scal_by_tag["mxfp4"]]
    for s in sites:
        hb = info_by_tag["bf16"]["hidden_by_site"].get(s, "-")
        hm = info_by_tag["mxfp4"]["hidden_by_site"].get(s, "-")
        body += (f"<tr><td style='text-align:left'>{html.escape(s)}</td>"
                 f"<td>{hb}</td><td>{hm}</td></tr>")
    body += "</table>"

    body += "<h2>all pages</h2><ul>"
    for fn, t in pages:
        if fn == "index.html":
            continue
        body += f'<li><a href="{fn}">{html.escape(t)}</a></li>'
    body += "</ul>"
    explain = (
        "This report dissects the activations that feed each GEMM (matrix "
        "multiply) inside DiffusionGemma, comparing the BF16 model against its "
        "MXFP4 (4-bit weight + 4-bit activation) twin on the same prompt. fp4 "
        "represents numbers in small blocks that share one scale, so a single "
        "huge value (an 'outlier') forces a coarse scale on its whole block and "
        "loses precision -- these pages are about finding those outliers: which "
        "<b>channels</b> (hidden dims), which <b>tokens</b> (canvas positions), "
        "at which <b>layer</b> and <b>denoising step</b>, and how much bigger "
        "they are under mxfp4. The 'runs' table below shows how many records each "
        "model produced; note the two models took a different number of "
        "denoising steps for the same prompt. The <b>paper:*</b> pages reproduce "
        "the canonical activation charts from the quantization literature "
        "(SmoothQuant channel x layer surface, AWQ/LLM.int8 salient-channel "
        "spectrum + heavy-tail histogram, SpinQuant kurtosis-per-layer, "
        "Outlier-Suppression+ per-channel ranges), using bf16-vs-mxfp4 in place "
        "of their before/after-rotation panels. Start with the <b>paper:*</b> "
        "pages or the <b>heatmaps</b> for the big picture, then drill into the "
        "<b>channels</b>/<b>tokens</b> pages, and use <b>outlier rankings</b> to "
        "jump to the worst spots. "
        + NAV_NOTE
    )
    page(out_dir, "index.html", "activation outlier report", nav, body, explain)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bf16", nargs="?",
                    default="runs/sky/bf16_act.jsonl",
                    help="bf16 activation JSONL")
    ap.add_argument("mxfp4", nargs="?",
                    default="runs/sky/mxfp4_act.jsonl",
                    help="mxfp4 activation JSONL")
    ap.add_argument("--out", default="runs/sky/act_report",
                    help="output directory for the HTML pages")
    ap.add_argument("--slices", type=int, default=5,
                    help="number of representative layer slices for the full "
                    "per-channel / per-token spectrum charts")
    args = ap.parse_args()

    paths = {"bf16": args.bf16, "mxfp4": args.mxfp4}
    os.makedirs(args.out, exist_ok=True)

    scal_by_tag = {}
    info_by_tag = {}
    for tag in TAGS:
        p = paths[tag]
        if not os.path.exists(p):
            print(f"  WARNING: {tag} file not found: {p} (skipping that run)")
            scal_by_tag[tag] = {}
            info_by_tag[tag] = {
                "layers": [], "steps": [], "sites": [], "hidden_by_site": {},
                "fwd2step": {}, "n_act": 0, "n_ctx": 0, "n_prefill": 0,
                "global_absmax": 0.0,
            }
            continue
        print(f"  pass 1 ({tag}): scanning {p} ...")
        scal_by_tag[tag], info_by_tag[tag] = pass1(p)
        print(f"    {info_by_tag[tag]['n_act']} act records, "
              f"{len(info_by_tag[tag]['layers'])} layers, "
              f"{len(info_by_tag[tag]['steps'])} steps")

    all_layers = sorted(set(info_by_tag["bf16"]["layers"])
                        | set(info_by_tag["mxfp4"]["layers"]))
    slice_layers = pick_slices(all_layers, args.slices)
    print(f"  representative layer slices: {slice_layers}")

    vec_by_tag = {}
    for tag in TAGS:
        p = paths[tag]
        if not os.path.exists(p):
            vec_by_tag[tag] = {}
            continue
        print(f"  pass 2 ({tag}): loading full vectors for layers "
              f"{slice_layers} ...")
        vec_by_tag[tag] = pass2(p, info_by_tag[tag]["fwd2step"],
                                set(slice_layers))

    all_steps = sorted(set(info_by_tag["bf16"]["steps"])
                       | set(info_by_tag["mxfp4"]["steps"]))
    rep_step = all_steps[len(all_steps) // 2] if all_steps else 0

    # Paper-style charts need every layer at one step. The two models took a
    # different number of denoising steps, so pick each run's own mid step.
    paper_by_tag = {}
    step_by_tag = {}
    for tag in TAGS:
        p = paths[tag]
        steps = info_by_tag[tag]["steps"]
        if not os.path.exists(p) or not steps:
            paper_by_tag[tag] = {}
            step_by_tag[tag] = "?"
            continue
        st = steps[len(steps) // 2]
        step_by_tag[tag] = st
        print(f"  paper pass ({tag}): all layers at step {st} ...")
        paper_by_tag[tag] = pass_paper(p, info_by_tag[tag]["fwd2step"], st)

    # Page manifest (filename, title) for the shared nav bar.
    pages: list[tuple[str, str]] = [("index.html", "index")]
    pages.append(("paper_surface.html", "paper: channel x layer surface"))
    pages.append(("paper_spectrum.html", "paper: channel spectrum"))
    pages.append(("paper_kurtosis.html", "paper: kurtosis per layer"))
    pages.append(("paper_range.html", "paper: per-channel range"))
    for suffix, _key, title, _scale, _expl in HEAT_METRICS:
        pages.append((f"heat_{suffix}.html", f"heat: {title}"))
    sites = [s for s in SITE_ORDER
             if s in scal_by_tag["bf16"] or s in scal_by_tag["mxfp4"]]
    for s in sites:
        safe = s.replace(".", "_")
        pages.append((f"channels_{safe}.html", f"channels: {s}"))
    for s in sites:
        safe = s.replace(".", "_")
        pages.append((f"tokens_{safe}.html", f"tokens: {s}"))
    pages.append(("blocks.html", "block / fp4 cost"))
    pages.append(("outliers.html", "outlier rankings"))
    nav = build_nav(pages)

    print("  rendering pages ...")
    index_page(args.out, nav, scal_by_tag, info_by_tag, pages)
    paper_surface_page(args.out, "paper_surface.html", nav, paper_by_tag,
                       step_by_tag)
    print("    paper_surface.html")
    paper_spectrum_page(args.out, "paper_spectrum.html", nav, paper_by_tag,
                        step_by_tag)
    print("    paper_spectrum.html")
    paper_kurtosis_page(args.out, "paper_kurtosis.html", nav, paper_by_tag,
                        step_by_tag)
    print("    paper_kurtosis.html")
    paper_range_page(args.out, "paper_range.html", nav, paper_by_tag,
                     step_by_tag)
    print("    paper_range.html")
    for suffix, key, title, scale, expl in HEAT_METRICS:
        heat_page(args.out, f"heat_{suffix}.html", f"heatmap: {title}", nav,
                  scal_by_tag, info_by_tag, key, scale, expl)
        print(f"    heat_{suffix}.html")
    for s in sites:
        safe = s.replace(".", "_")
        channels_page(args.out, f"channels_{safe}.html", s, nav, vec_by_tag,
                      slice_layers, rep_step, all_steps)
        print(f"    channels_{safe}.html")
    for s in sites:
        safe = s.replace(".", "_")
        tokens_page(args.out, f"tokens_{safe}.html", s, nav, vec_by_tag,
                    slice_layers, rep_step)
        print(f"    tokens_{safe}.html")
    blocks_page(args.out, "blocks.html", nav, scal_by_tag, info_by_tag)
    outliers_page(args.out, "outliers.html", nav, scal_by_tag, info_by_tag)

    print(f"\n  done -> {os.path.join(args.out, 'index.html')}")


if __name__ == "__main__":
    main()
