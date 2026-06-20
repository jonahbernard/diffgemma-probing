#!/usr/bin/env python3
"""Per-canvas-POSITION meanH overlay across prompts, all variants on one chart.

For each variant (bf16/moe x temp0/temp0.6) we average mean_entropy per step
across all PROMPTS but keep canvas *position* separate: position p is the p-th
canvas generated for each prompt. The driver records nothing but req_id, so
position is inferred from emission order -- the record driver is synchronous
(one blocking curl per generation) and loops prompt-outer / seed-inner, so the
i-th canvas in the file has prompt = i // nseed and position = i % nseed.

So each variant contributes up to `nseed` lines (one per position), each line =
mean over the (<=50) prompts of that position's per-step entropy. With 4 variants
and ~100 positions that is ~400 lines: we color by variant hue and shade by
position (light = early position, dark = late) so any position/seed drift is
visible. Lines are thin/semi-transparent with no markers.

nseed defaults to round(canvas_count / 50) per file (prompts50.txt has 50 lines),
overridable with --nseed.

Usage:
  ./per_position_overlay.py \
     "bf16 t0=runs/meanh50/bf16_8k_temp0.jsonl" \
     "bf16 t0.6=runs/meanh50/bf16_8k_gpu7_temp0.6.jsonl" \
     "moe t0=runs/meanh50/moe-fp4-mlp-bf16_8k_temp0.jsonl" \
     "moe t0.6=runs/meanh50/moe-fp4-mlp-bf16_8k_gpu7_temp0.6.jsonl" \
     -o runs/meanh50/per_position.html
"""

from __future__ import annotations

import argparse
import json
import os

from meanh_overlay import _default_label, load_by_canvas

NPROMPT = 50  # prompts50.txt

# One base hue per variant (H in HSL); position maps to lightness within the hue.
VARIANT_HUES = [145, 0, 220, 38]  # green, red, blue, orange


def positions_from_order(canvases: dict, nseed: int):
    """Group canvases by position = (insertion index) % nseed.

    load_by_canvas preserves first-seen order, which equals emission/send order.
    Returns {position: [trajectory dict, ...]} (one entry per prompt at that pos).
    """
    by_pos: dict[int, list] = {}
    for i, traj in enumerate(canvases.values()):
        by_pos.setdefault(i % nseed, []).append(traj)
    return by_pos


def mean_per_step(trajs: list[dict[int, float]]):
    """Average entropy per step across a list of trajectories -> [[step,v],...]."""
    by_step: dict[int, list[float]] = {}
    for traj in trajs:
        for step, val in traj.items():
            by_step.setdefault(step, []).append(val)
    return [[step, round(sum(v) / len(v), 6)]
            for step in sorted(by_step) for v in [by_step[step]]]


TEMPLATE = """<!doctype html>
<html><head><meta charset="utf-8"><title>per-position meanH overlay</title>
<style>
  body {{ font-family: monospace; margin: 16px; background:#fafafa; color:#222; }}
  h3 {{ margin:6px 0; }}
  .legend {{ font-size:12px; color:#555; margin-bottom:8px; }}
  .sw {{ display:inline-block; width:11px; height:11px; vertical-align:middle;
         margin:0 3px 0 10px; border:1px solid #888; }}
  #readout {{ font-size:12px; color:#222; height:16px; margin-top:4px; }}
  label.legend {{ margin-left:12px; }}
  svg {{ background:#fff; border:1px solid #ddd; }}
  .grid {{ stroke:#eee; stroke-width:1; }}
  .axis {{ stroke:#888; stroke-width:1; }}
  .thr {{ stroke:#e6a000; stroke-width:1; stroke-dasharray:4 3; }}
  .ax-lbl {{ font-size:11px; fill:#666; }}
</style></head><body>
  <h3>mean token entropy per step, by canvas position within prompt ({npos} positions x {nvar} variants, {nprompt} prompts avg)</h3>
  <div class="legend">
    {legend_html}
    <span class="sw" style="background:#e6a000"></span>commit threshold ({conf_disp})
    <label class="legend"><input type="checkbox" id="logy" checked> log y-axis</label>
    <label class="legend">show variant
      <select id="varsel"><option value="-1" selected>all</option>{var_opts}</select>
    </label>
  </div>
  <div id="chart"></div>
  <div id="readout">light = early position, dark = late position</div>
<script>
// VARIANTS: [{{name, hue, lines:[[ [step,v],... ] per position ]}}]
const VARIANTS = {variants_json};
const CONF = {conf};
const W = 1100, H = 560, PADL = 64, PADR = 20, PADT = 20, PADB = 48;
const PW = W - PADL - PADR, PH = H - PADT - PADB;
let logY = true, varFilter = -1;

let xMin = Infinity, xMax = -Infinity, yMaxRaw = CONF, yMinPos = Infinity;
for (const V of VARIANTS) for (const ln of V.lines) for (const d of ln) {{
  if (d[0] < xMin) xMin = d[0]; if (d[0] > xMax) xMax = d[0];
  if (d[1] > yMaxRaw) yMaxRaw = d[1];
  if (d[1] > 0 && d[1] < yMinPos) yMinPos = d[1];
}}
if (CONF > 0 && CONF < yMinPos) yMinPos = CONF;
if (!isFinite(yMinPos)) yMinPos = 1e-6;
if (!isFinite(xMin)) {{ xMin = 0; xMax = 1; }}

const xScale = s => PADL + (xMax === xMin ? 0 : (s - xMin) / (xMax - xMin) * PW);
function yScale(v) {{
  if (logY) {{
    const lo = Math.log10(yMinPos), hi = Math.log10(yMaxRaw || 1);
    const vv = Math.log10(Math.max(v, yMinPos));
    return PADT + (hi === lo ? 0 : (1 - (vv - lo) / (hi - lo)) * PH);
  }}
  return PADT + (1 - v / (yMaxRaw || 1)) * PH;
}}
function linePath(ln) {{
  return ln.map((d, i) =>
    (i ? "L" : "M") + xScale(d[0]).toFixed(1) + " " + yScale(d[1]).toFixed(1)
  ).join(" ");
}}
// HSL line color: variant hue, lightness ramps from light(early) to dark(late).
function posColor(hue, pos, npos) {{
  const t = npos <= 1 ? 0 : pos / (npos - 1);
  const L = 72 - 47 * t;   // 72% -> 25%
  const S = 70;
  return "hsl(" + hue + "," + S + "%," + L + "%)";
}}
function ticksY() {{
  const out = [];
  if (logY) {{
    let e = Math.floor(Math.log10(yMinPos));
    const top = Math.ceil(Math.log10(yMaxRaw || 1));
    for (; e <= top; e++) out.push(Math.pow(10, e));
  }} else {{
    for (let i = 0; i <= 5; i++) out.push(yMaxRaw * i / 5);
  }}
  return out;
}}
function ticksX() {{
  const out = [], span = xMax - xMin, sg = Math.ceil(span / 12) || 1;
  for (let s = xMin; s <= xMax; s += sg) out.push(s);
  if (out[out.length - 1] !== xMax) out.push(xMax);
  return out;
}}
const SVGNS = "http://www.w3.org/2000/svg";
function el(name, attrs) {{
  const n = document.createElementNS(SVGNS, name);
  for (const k in attrs) n.setAttribute(k, attrs[k]);
  return n;
}}
let svg;
function draw() {{
  const chart = document.getElementById('chart'); chart.innerHTML = "";
  svg = el("svg", {{width: W, height: H, viewBox: "0 0 " + W + " " + H}});
  for (const t of ticksY()) {{
    const y = yScale(t);
    svg.appendChild(el("line", {{class:"grid", x1:PADL, y1:y, x2:W-PADR, y2:y}}));
    const l = el("text", {{class:"ax-lbl", x:PADL-6, y:y+3, "text-anchor":"end"}});
    l.textContent = (logY ? "1e"+Math.round(Math.log10(t)) :
      (t >= 0.01 || t === 0 ? t.toFixed(3) : t.toExponential(0)));
    svg.appendChild(l);
  }}
  for (const t of ticksX()) {{
    const x = xScale(t);
    svg.appendChild(el("line", {{class:"grid", x1:x, y1:PADT, x2:x, y2:H-PADB}}));
    const l = el("text", {{class:"ax-lbl", x:x, y:H-PADB+16, "text-anchor":"middle"}});
    l.textContent = t; svg.appendChild(l);
  }}
  svg.appendChild(el("line", {{class:"axis", x1:PADL, y1:PADT, x2:PADL, y2:H-PADB}}));
  svg.appendChild(el("line", {{class:"axis", x1:PADL, y1:H-PADB, x2:W-PADR, y2:H-PADB}}));
  const xt = el("text", {{class:"ax-lbl", x:PADL+PW/2, y:H-6, "text-anchor":"middle"}});
  xt.textContent = "denoising step"; svg.appendChild(xt);
  const yt = el("text", {{class:"ax-lbl", x:14, y:PADT+PH/2, "text-anchor":"middle",
    transform:"rotate(-90 14 " + (PADT+PH/2) + ")"}});
  yt.textContent = "mean token entropy" + (logY ? " (log)" : ""); svg.appendChild(yt);
  if (!(logY && CONF <= 0)) {{
    const yT = yScale(CONF);
    svg.appendChild(el("line", {{class:"thr", x1:PADL, y1:yT, x2:W-PADR, y2:yT}}));
  }}
  VARIANTS.forEach((V, vi) => {{
    if (varFilter >= 0 && varFilter !== vi) return;
    const npos = V.lines.length;
    V.lines.forEach((ln, pos) => {{
      if (ln.length < 2) return;
      svg.appendChild(el("path", {{d:linePath(ln), fill:"none",
        stroke:posColor(V.hue, pos, npos), "stroke-width":1, "stroke-opacity":0.5}}));
    }});
  }});
  chart.appendChild(svg);
}}
document.getElementById('logy').addEventListener('change', e => {{
  logY = e.target.checked; draw();
}});
document.getElementById('varsel').addEventListener('change', e => {{
  varFilter = parseInt(e.target.value, 10); draw();
}});
draw();
</script></body></html>
"""


def parse_inputs(inputs, labels):
    paths, names = [], []
    for tok in inputs:
        if "=" in tok and not os.path.exists(tok):
            label, path = tok.split("=", 1)
            names.append(label.strip() or _default_label(path))
            paths.append(path)
        else:
            paths.append(tok)
            names.append(_default_label(tok))
    if labels:
        for i, p in enumerate([s.strip() for s in labels.split(",")][:len(names)]):
            if p:
                names[i] = p
    return paths, names


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("inputs", nargs="+", help="meanH JSONLs, 'label=path' or 'path'")
    ap.add_argument("-o", "--out", default=None, help="output HTML")
    ap.add_argument("--nseed", type=int, default=None,
                    help="canvas positions per prompt (default: round(count/50))")
    ap.add_argument("-n", "--limit", type=int, default=None,
                    help="use only the first N canvases (request order) of each "
                         "file. Default: all.")
    ap.add_argument("--nprompt", type=int, default=NPROMPT,
                    help="prompts per variant (default 50)")
    ap.add_argument("--labels", default=None, help="comma legend labels")
    args = ap.parse_args()

    paths, names = parse_inputs(args.inputs, args.labels)

    variants = []
    conf = None
    npos_global = 0
    for i, (name, path) in enumerate(zip(names, paths)):
        canvases, c = load_by_canvas(path)
        if c is not None and conf is None:
            conf = c
        if args.limit:
            canvases = dict(list(canvases.items())[:args.limit])
        count = len(canvases)
        nseed = args.nseed or max(1, round(count / args.nprompt))
        by_pos = positions_from_order(canvases, nseed)
        lines = [mean_per_step(by_pos[p]) for p in sorted(by_pos)]
        npos_global = max(npos_global, len(lines))
        hue = VARIANT_HUES[i % len(VARIANT_HUES)]
        variants.append({"name": name, "hue": hue, "lines": lines})
        print(f"{name}: {count} canvases, nseed={nseed} -> {len(lines)} positions, "
              f"max {max((len(by_pos[p]) for p in by_pos), default=0)} prompts/pos")

    if conf is None:
        conf = 0.0
    # Legend: a swatch at mid-lightness per variant hue.
    legend_html = "\n    ".join(
        f'<span class="sw" style="background:hsl({v["hue"]},70%,48%)"></span>{v["name"]}'
        for v in variants)
    var_opts = "".join(
        f'<option value="{i}">{v["name"]}</option>'
        for i, v in enumerate(variants))

    out = args.out or os.path.join(
        os.path.dirname(os.path.abspath(paths[0])), "per_position.html")
    with open(out, "w") as f:
        f.write(TEMPLATE.format(
            variants_json=json.dumps(variants),
            conf=json.dumps(conf),
            conf_disp=conf,
            npos=npos_global,
            nvar=len(variants),
            nprompt=args.nprompt,
            legend_html=legend_html,
            var_opts=var_opts,
        ))
    print(f"per-position overlay -> {out}")


if __name__ == "__main__":
    main()
