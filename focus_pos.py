#!/usr/bin/env python3
"""Drill into ONE canvas position's activation excursion across adjacent steps.

Motivating case: in the mxfp4 sky run, position 21 is token " Blue" with prob
1.0 at denoising step 31, flips to an Arabic token (" الأز") at step 32, then
recovers to " Blue" at step 33. This is a one-step excursion at a maximally
confident position -- the signature of a transient activation outlier rather
than a slow trajectory drift. This script isolates that position's activation
statistics, per GEMM site, per layer, across the three steps, so we can see at
WHICH layer/site the step-32 activation departs from its step-31/33 neighbors
(where the quantization error is injected) and which channels carry it (the
fp4-block-scale lever for quantizing with less error).

The activation probe stores, per (forward_id, site, layer):
  - per-token vectors tok_l2 / tok_absmax / tok_kurtosis  (index = position)
  - per-channel vectors ch_absmax / ch_rms / ch_mean      (max/agg over tokens)
  - per-block fp4 stats blk_absmax_max / p99 / p50 / mean
A companion "ctx" record maps forward_id -> denoising step.

We pin one position, pull tok_*[pos] for the target steps to find the
divergence layer, then show the channel-outlier / fp4-block context at that
layer so the position-level flip is tied back to concrete channels.

Usage:
  ./focus_pos.py                          # defaults: mxfp4, pos 21, steps 31-33
  ./focus_pos.py --pos 21 --steps 31 32 33
  ./focus_pos.py --act runs/sky/mxfp4_act.jsonl --conf runs/sky/mxfp4.jsonl
"""

from __future__ import annotations

import argparse
import ast
import base64
import html
import io
import json
import os
from collections import defaultdict

import numpy as np

SITE_ORDER = [
    "qkv_proj",
    "o_proj",
    "mlp.gate_up",
    "mlp.down",
    "router",
    "moe.predispatch",
]

# per-position token metrics we can isolate to a single canvas position
TOK_METRICS = [
    ("tok_l2", "token L2 norm",
     "L2 norm of this position's activation vector feeding the GEMM. The "
     "overall magnitude of the token. A step-32 spike here vs step 31/33 means "
     "the whole activation for this position blew up at this layer."),
    ("tok_absmax", "token absmax",
     "Largest single |value| in this position's activation vector. This is the "
     "number fp4 must represent for this token; a spike forces a coarse block "
     "scale and is the direct quantization-error source."),
    ("tok_kurtosis", "token kurtosis",
     "Peakedness of this position's activation (Gaussian = 3). High = the "
     "token's energy is concentrated in a few channels -- spiky vectors are "
     "exactly what fp4's 32-wide blocks quantize worst."),
]


def iter_records(path: str):
    with open(path, errors="replace") as f:
        for line in f:
            line = line.strip().strip("\x00")
            if not line or line[0] != "{":
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _as_dict(v):
    if isinstance(v, dict):
        return v
    if isinstance(v, str):
        try:
            return ast.literal_eval(v)
        except (ValueError, SyntaxError):
            return {}
    return {}


def build_fwd2step(path: str) -> dict[int, int]:
    """forward_id -> denoising step, from the ctx records."""
    fwd2step: dict[int, int] = {}
    for o in iter_records(path):
        if o.get("kind") != "ctx":
            continue
        sbs = _as_dict(o.get("step_by_slot"))
        for _slot, step in sbs.items():
            try:
                fwd2step[int(o["forward_id"])] = int(step)
            except (TypeError, ValueError):
                pass
    return fwd2step


def load_focus(path: str, want_fwds: set[int]):
    """Collect every act record for the target forwards.

    Returns recs[(fid, site, layer)] = full record (vectors kept).
    """
    recs: dict[tuple[int, str, int], dict] = {}
    for o in iter_records(path):
        if o.get("kind") != "act":
            continue
        try:
            fid = int(o["forward_id"])
        except (TypeError, ValueError):
            continue
        if fid not in want_fwds:
            continue
        recs[(fid, o["site"], int(o["layer"]))] = o
    return recs


def conf_for_pos(path: str, pos: int, steps: list[int], fwd2step=None):
    """Decoded-token / prob context for the position at each step."""
    by_step = {}
    for o in iter_records(path):
        if o.get("kind") not in (None, "conf") and "argmax_token" not in o:
            continue
        st = o.get("step")
        if st in steps and "argmax_token" in o:
            by_step[st] = {
                "argmax": o["argmax_token"][pos],
                "max_prob": o["max_prob"][pos],
                "entropy": o["token_entropy"][pos],
                "topk_token": o["topk_token"][pos],
                "topk_prob": o["topk_prob"][pos],
            }
    return by_step


def try_decode(tokens, model_path):
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_path)
        return {t: tok.decode([t]) for t in tokens}
    except Exception:
        return {}


# ---------------------------------------------------------------- plotting

def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def fig_b64(fig) -> str:
    import matplotlib.pyplot as plt
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=96, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def img(b64: str) -> str:
    return f'<img src="data:image/png;base64,{b64}">'


STEP_COLORS = {
    "before": "#5fd35f",   # green - step before the flip
    "flip": "#ff3b6b",     # red   - the flip step
    "after": "#6ca6ff",    # blue  - recovery step
}


def site_profile_fig(recs, fwd2step, fid_by_step, steps, site, pos, metric):
    """tok_<metric>[pos] vs layer, one line per step, for one site."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(9, 3.2))
    role = {steps[0]: "before", steps[1]: "flip"}
    if len(steps) > 2:
        role[steps[2]] = "after"
    plotted = False
    for st in steps:
        fid = fid_by_step[st]
        layers = sorted(l for (f, s, l) in recs if f == fid and s == site)
        if not layers:
            continue
        ys = []
        for l in layers:
            r = recs[(fid, site, l)]
            v = r.get(metric)
            ys.append(v[pos] if v and pos < len(v) else np.nan)
        col = STEP_COLORS.get(role.get(st, "flip"), "#aaa")
        lw = 2.6 if role.get(st) == "flip" else 1.6
        ax.plot(layers, ys, "-o", ms=3, lw=lw, color=col,
                label=f"step {st} ({role.get(st,'')})")
        plotted = True
    if not plotted:
        plt.close(fig)
        return None
    ax.set_xlabel("layer")
    ax.set_ylabel(metric)
    ax.set_title(f"{site} — {metric}[pos {pos}]")
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=8, framealpha=0.3)
    fig.patch.set_facecolor("#111")
    ax.set_facecolor("#181818")
    for sp in ax.spines.values():
        sp.set_color("#444")
    ax.tick_params(colors="#aaa")
    ax.xaxis.label.set_color("#ccc")
    ax.yaxis.label.set_color("#ccc")
    ax.title.set_color("#eee")
    return fig_b64(fig)


def divergence_table(recs, fid_by_step, steps, site, pos, metric):
    """Per-layer ratio of flip-step value to the mean of neighbor steps.

    Returns list of (layer, before, flip, after, ratio) sorted by ratio desc.
    Ratio = flip / mean(before, after); >1 means the flip step spiked here.
    """
    before_s, flip_s = steps[0], steps[1]
    after_s = steps[2] if len(steps) > 2 else None
    rows = []
    layers = sorted({l for (f, s, l) in recs
                     if s == site and f == fid_by_step[flip_s]})
    for l in layers:
        def val(st):
            r = recs.get((fid_by_step[st], site, l))
            if not r:
                return np.nan
            v = r.get(metric)
            return v[pos] if v and pos < len(v) else np.nan
        b, fl = val(before_s), val(flip_s)
        a = val(after_s) if after_s is not None else np.nan
        neigh = np.nanmean([b, a])
        ratio = fl / neigh if neigh and neigh == neigh and neigh != 0 else np.nan
        rows.append((l, b, fl, a, ratio))
    rows.sort(key=lambda r: (-(r[4] if r[4] == r[4] else -1)))
    return rows


def channel_context_fig(recs, fid_by_step, steps, site, layer):
    """ch_absmax spectrum at the divergence layer, flip vs neighbor steps.

    Channel vectors are aggregated over all tokens, so this is context (which
    channels are outliers at this layer/step), not pos-isolated. The fp4 story:
    a channel whose absmax dwarfs the block median forces a coarse shared scale.
    """
    plt = _plt()
    fig, ax = plt.subplots(figsize=(9, 3.0))
    role = {steps[0]: "before", steps[1]: "flip"}
    if len(steps) > 2:
        role[steps[2]] = "after"
    any_plot = False
    for st in steps:
        r = recs.get((fid_by_step[st], site, layer))
        if not r or not r.get("ch_absmax"):
            continue
        y = np.asarray(r["ch_absmax"], dtype=float)
        col = STEP_COLORS.get(role.get(st, "flip"), "#aaa")
        lw = 1.8 if role.get(st) == "flip" else 1.0
        al = 0.95 if role.get(st) == "flip" else 0.6
        ax.plot(np.arange(y.size), y, lw=lw, color=col, alpha=al,
                label=f"step {st}")
        any_plot = True
    if not any_plot:
        plt.close(fig)
        return None
    ax.set_xlabel("channel")
    ax.set_ylabel("ch_absmax")
    ax.set_title(f"{site} layer {layer} — per-channel absmax")
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=8, framealpha=0.3)
    fig.patch.set_facecolor("#111")
    ax.set_facecolor("#181818")
    for sp in ax.spines.values():
        sp.set_color("#444")
    ax.tick_params(colors="#aaa")
    ax.xaxis.label.set_color("#ccc")
    ax.yaxis.label.set_color("#ccc")
    ax.title.set_color("#eee")
    return fig_b64(fig)


def top_channels(recs, fid_by_step, steps, site, layer, topn=12):
    """Channels where flip-step ch_absmax most exceeds neighbor steps."""
    before_s, flip_s = steps[0], steps[1]
    after_s = steps[2] if len(steps) > 2 else None

    def vec(st):
        r = recs.get((fid_by_step[st], site, layer))
        return np.asarray(r["ch_absmax"], dtype=float) if r and r.get("ch_absmax") else None
    fl = vec(flip_s)
    if fl is None:
        return [], None
    b = vec(before_s)
    a = vec(after_s) if after_s is not None else None
    neigh = np.nanmean(np.stack([x for x in (b, a) if x is not None]), axis=0) \
        if (b is not None or a is not None) else np.ones_like(fl)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = fl / neigh
    blk_p50 = None
    rflip = recs.get((fid_by_step[flip_s], site, layer))
    if rflip:
        blk_p50 = rflip.get("blk_absmax_p50")
    idx = np.argsort(-np.nan_to_num(ratio))[:topn]
    rows = []
    for c in idx:
        rows.append((int(c), float(fl[c]),
                     float(b[c]) if b is not None else float("nan"),
                     float(a[c]) if a is not None else float("nan"),
                     float(ratio[c])))
    return rows, blk_p50


# ---------------------------------------------------------------- html

def fmt(x, p=3):
    if x is None or (isinstance(x, float) and x != x):
        return "—"
    return f"{x:.{p}f}"


def write_page(out_path, title, body):
    doc = f"""<!doctype html><html><head><meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>
 body {{ font-family: system-ui, sans-serif; margin: 16px; background:#111;
        color:#ddd; }}
 h1 {{ font-size: 20px; }}
 h2 {{ font-size: 16px; margin-top: 30px; border-bottom:1px solid #333;
       padding-bottom:4px; }}
 h3 {{ font-size: 14px; color:#bbb; margin-top:18px; }}
 a {{ color:#6cf; }}
 img {{ max-width: 100%; background:#111; border:1px solid #222;
        border-radius:4px; }}
 table {{ border-collapse: collapse; margin: 8px 0; font-size: 12px; }}
 th, td {{ border:1px solid #333; padding:3px 8px; text-align:right; }}
 th {{ background:#1c1c1c; color:#ccc; }}
 td.k {{ text-align:left; color:#9cf; }}
 .hi {{ color:#ff7b9c; font-weight:600; }}
 .explain {{ background:#16181c; border-left:3px solid #6cf; padding:8px 12px;
            margin:8px 0; font-size:13px; color:#bcd; }}
 .grid2 {{ display:grid; grid-template-columns:1fr 1fr; gap:12px; }}
 .card {{ background:#141414; border:1px solid #262626; border-radius:6px;
          padding:8px; }}
 code {{ color:#9f9; }}
</style></head><body>
{body}
</body></html>"""
    with open(out_path, "w") as f:
        f.write(doc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--act", default="runs/sky/mxfp4_act.jsonl")
    ap.add_argument("--conf", default="runs/sky/mxfp4.jsonl")
    ap.add_argument("--pos", type=int, default=21)
    ap.add_argument("--steps", type=int, nargs="+", default=[31, 32, 33])
    ap.add_argument("--model", default="/app/models/diffusiongemma-26B-A4B-it-mxfp4-v4")
    ap.add_argument("--out", default="runs/sky/focus_pos21.html")
    ap.add_argument("--topn", type=int, default=12)
    args = ap.parse_args()

    pos, steps = args.pos, args.steps
    print(f"[focus] position {pos}, steps {steps}")

    fwd2step = build_fwd2step(args.act)
    step2fwd = {}
    for fid, st in fwd2step.items():
        step2fwd.setdefault(st, fid)
    missing = [s for s in steps if s not in step2fwd]
    if missing:
        raise SystemExit(f"steps {missing} have no forward_id in {args.act}; "
                         f"available steps {sorted(step2fwd)[:3]}..{sorted(step2fwd)[-3:]}")
    fid_by_step = {s: step2fwd[s] for s in steps}
    print(f"[focus] step->forward_id {fid_by_step}")

    want_fwds = set(fid_by_step.values())
    recs = load_focus(args.act, want_fwds)
    sites = [s for s in SITE_ORDER
             if any(k[1] == s for k in recs)] or \
            sorted({k[1] for k in recs})
    print(f"[focus] loaded {len(recs)} act records, sites {sites}")

    conf = conf_for_pos(args.conf, pos, steps)
    all_toks = sorted({t for st in conf for t in
                       ([conf[st]["argmax"]] + conf[st]["topk_token"])})
    decoded = try_decode(all_toks, args.model)

    # ---- header / token context
    body = [f"<h1>Position {pos} activation excursion — steps {steps}</h1>"]
    body.append(
        '<div class="explain"><b>What am I looking at?</b><br>'
        f"Canvas position {pos} in the mxfp4 run is decoded below. The middle "
        "step is where the chosen token flips; the flanking steps are the "
        "stable before/after. Each chart pins this one position and walks it "
        "down the network, so where the red (flip) line departs from the "
        "green/blue lines is the layer that injected the divergence. "
        "Per-channel charts then show which channels carry the spike — the "
        "fp4 block-scale lever.</div>")

    if conf:
        body.append("<h2>Decoded token at this position</h2>")
        body.append("<table><tr><th>step</th><th>role</th><th class='k'>token</th>"
                    "<th>prob</th><th>entropy</th><th class='k'>top-5</th></tr>")
        roles = {steps[0]: "before", steps[1]: "FLIP"}
        if len(steps) > 2:
            roles[steps[2]] = "after"
        for st in steps:
            c = conf.get(st)
            if not c:
                continue
            tk = decoded.get(c["argmax"], str(c["argmax"]))
            top = " ".join(
                f"{html.escape(repr(decoded.get(t, t)))}={fmt(p,2)}"
                for t, p in zip(c["topk_token"], c["topk_prob"]))
            cls = " class='hi'" if roles.get(st) == "FLIP" else ""
            body.append(
                f"<tr{cls}><td>{st}</td><td>{roles.get(st,'')}</td>"
                f"<td class='k'>{html.escape(repr(tk))}</td>"
                f"<td>{fmt(c['max_prob'],3)}</td><td>{fmt(c['entropy'],3)}</td>"
                f"<td class='k'>{top}</td></tr>")
        body.append("</table>")

    # ---- per-site profiles + divergence ranking
    # find, per site, the layer with the biggest flip/neighbor ratio on
    # tok_absmax (the fp4-relevant metric) to use for the channel context.
    div_layer_by_site = {}
    for site in sites:
        body.append(f"<h2>{site}</h2>")
        body.append('<div class="grid2">')
        for metric, label, expl in TOK_METRICS:
            b64 = site_profile_fig(recs, fwd2step, fid_by_step, steps,
                                   site, pos, metric)
            if b64:
                body.append(f'<div class="card">{img(b64)}'
                            f'<div class="explain">{expl}</div></div>')
        body.append('</div>')

        rows = divergence_table(recs, fid_by_step, steps, site, pos,
                                "tok_absmax")
        rows = [r for r in rows if r[4] == r[4]]
        if rows:
            div_layer_by_site[site] = rows[0][0]
            body.append("<h3>layers ranked by step-%d tok_absmax spike "
                        "(flip / mean neighbors)</h3>" % steps[1])
            body.append("<table><tr><th>layer</th>"
                        f"<th>step {steps[0]}</th><th>step {steps[1]}</th>"
                        + (f"<th>step {steps[2]}</th>" if len(steps) > 2 else "")
                        + "<th>ratio</th></tr>")
            for (l, b, fl, a, ratio) in rows[:8]:
                cls = " class='hi'" if ratio and ratio > 1.3 else ""
                body.append(
                    f"<tr{cls}><td>{l}</td><td>{fmt(b)}</td><td>{fmt(fl)}</td>"
                    + (f"<td>{fmt(a)}</td>" if len(steps) > 2 else "")
                    + f"<td>{fmt(ratio,2)}×</td></tr>")
            body.append("</table>")

    # ---- channel context at each site's divergence layer
    body.append("<h2>Channel context at the divergence layer</h2>")
    body.append('<div class="explain">For each site, the layer where the '
                "flip-step tok_absmax spiked most is shown. The spectrum "
                "overlays per-channel absmax for the three steps; the table "
                "lists channels whose flip-step absmax most exceeds the "
                "neighbor steps. Compare each channel's absmax to the block "
                "median (blk_p50): a channel ≫ blk_p50 forces a coarse fp4 "
                "scale on its 32-channel block — the concrete place to spend "
                "precision.</div>")
    for site in sites:
        layer = div_layer_by_site.get(site)
        if layer is None:
            continue
        body.append(f"<h3>{site} — layer {layer}</h3>")
        b64 = channel_context_fig(recs, fid_by_step, steps, site, layer)
        if b64:
            body.append(img(b64))
        chrows, blk_p50 = top_channels(recs, fid_by_step, steps, site, layer,
                                       args.topn)
        if chrows:
            body.append(f"<p>block median absmax (blk_p50) = "
                        f"<code>{fmt(blk_p50,3)}</code></p>")
            body.append("<table><tr><th>channel</th>"
                        f"<th>step {steps[0]}</th><th>step {steps[1]}</th>"
                        + (f"<th>step {steps[2]}</th>" if len(steps) > 2 else "")
                        + "<th>flip/neigh</th><th>flip/blk_p50</th></tr>")
            for (c, fl, b, a, ratio) in chrows:
                vs_blk = fl / blk_p50 if blk_p50 else float("nan")
                cls = " class='hi'" if vs_blk == vs_blk and vs_blk > 4 else ""
                body.append(
                    f"<tr{cls}><td>{c}</td><td>{fmt(b)}</td><td>{fmt(fl)}</td>"
                    + (f"<td>{fmt(a)}</td>" if len(steps) > 2 else "")
                    + f"<td>{fmt(ratio,2)}×</td><td>{fmt(vs_blk,1)}×</td></tr>")
            body.append("</table>")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    write_page(args.out, f"position {pos} excursion", "\n".join(body))
    print(f"[focus] wrote {args.out}")


if __name__ == "__main__":
    main()
