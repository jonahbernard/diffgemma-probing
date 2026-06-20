#!/usr/bin/env python3
"""Fraction of canvases below the commit threshold vs denoising step.

The meanH median/mean charts answer "how high is entropy"; this answers the
question that actually matters for diffusion commit behavior: "what fraction of
canvases have committed (dropped to/below the confidence threshold) by step N?"
It's a survival/CDF view -- robust to skew and directly comparable across methods
and temperatures (e.g. does moe-fp4 quantization make canvases commit later?).

Definition of "committed at step N" for a canvas:
  the canvas's mean_entropy is <= threshold at the largest recorded step <= N.
A canvas's trajectory is the first monotonic run of steps (same parsing as
meanh_overlay.load_by_canvas). Once a canvas's entropy is at/below threshold it is
counted as committed for that step and all later steps, so each curve is
monotonically non-decreasing from 0 toward 1.

Reuses load_by_canvas from meanh_overlay.py -- aggregation only, no re-collection.

Usage:
  ./commit_fraction.py bf16=runs/meanh50/bf16.jsonl \
      mxfp4=runs/meanh50/mxfp4.jsonl -o runs/meanh50/commit_fraction.html
"""

from __future__ import annotations

import argparse
import json
import os

from meanh_overlay import PALETTE, _default_label, load_by_canvas


def commit_fraction(canvases: dict[str, dict[int, float]], thr: float):
    """Fraction of canvases committed (entropy <= thr) at each step.

    For every step from 1..max, count canvases whose entropy at their largest
    recorded step <= that step is <= thr, divided by total canvas count. Returns
    [[step, fraction], ...]. Curve is monotonic non-decreasing because commit is
    treated as absorbing (a canvas that converged stays committed).
    """
    n = len(canvases)
    if n == 0:
        return []
    # For each canvas, the earliest step at which it is at/below threshold (its
    # commit step). None if it never drops to threshold within its trajectory.
    commit_step: list[int] = []
    max_step = 0
    for traj in canvases.values():
        steps = sorted(traj)
        if steps:
            max_step = max(max_step, steps[-1])
        cs = None
        for s in steps:
            if traj[s] <= thr:
                cs = s
                break
        if cs is not None:
            commit_step.append(cs)
    out = []
    for step in range(1, max_step + 1):
        committed = sum(1 for cs in commit_step if cs <= step)
        out.append([step, round(committed / n, 5)])
    return out


TEMPLATE = """<!doctype html>
<html><head><meta charset="utf-8"><title>diffgemma commit fraction</title>
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
  <h3>fraction of canvases committed (meanH &le; {conf_disp}) per denoising step ({ncanvas} canvases/method)</h3>
  <div class="legend">
    {legend_html}
    <label class="legend"><input type="checkbox" id="half" checked> 50% guide</label>
  </div>
  <div id="chart"></div>
  <div id="readout">&nbsp;</div>
<script>
// each: {{frac:[[step,f]...], color:"#..", name:".."}}
const SERIES = {series_json};
const W = 980, H = 500, PADL = 64, PADR = 20, PADT = 20, PADB = 44;
const PW = W - PADL - PADR, PH = H - PADT - PADB;

let showHalf = true;
let xMin = Infinity, xMax = -Infinity;
for (const s of SERIES) {{
  for (const d of s.frac) {{ if (d[0] < xMin) xMin = d[0]; if (d[0] > xMax) xMax = d[0]; }}
}}
if (!isFinite(xMin)) {{ xMin = 0; xMax = 1; }}
// y axis is a fraction, fixed 0..1.
const yMax = 1;

const xScale = s => PADL + (xMax === xMin ? 0 : (s - xMin) / (xMax - xMin) * PW);
const yScale = v => PADT + (1 - v / yMax) * PH;
function linePath(series) {{
  return series.map((d, i) =>
    (i ? "L" : "M") + xScale(d[0]).toFixed(1) + " " + yScale(d[1]).toFixed(1)
  ).join(" ");
}}
function ticksY() {{
  const out = [];
  for (let i = 0; i <= 5; i++) out.push(i / 5);
  return out;
}}
function ticksX() {{
  const out = [], span = xMax - xMin, stepGuess = Math.ceil(span / 10) || 1;
  for (let s = xMin; s <= xMax; s += stepGuess) out.push(s);
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
  const chart = document.getElementById('chart');
  chart.innerHTML = "";
  svg = el("svg", {{width: W, height: H, viewBox: "0 0 " + W + " " + H}});
  for (const t of ticksY()) {{
    const y = yScale(t);
    svg.appendChild(el("line", {{class:"grid", x1:PADL, y1:y, x2:W-PADR, y2:y}}));
    const lbl = el("text", {{class:"ax-lbl", x:PADL-6, y:y+3, "text-anchor":"end"}});
    lbl.textContent = t.toFixed(1);
    svg.appendChild(lbl);
  }}
  for (const t of ticksX()) {{
    const x = xScale(t);
    svg.appendChild(el("line", {{class:"grid", x1:x, y1:PADT, x2:x, y2:H-PADB}}));
    const lbl = el("text", {{class:"ax-lbl", x:x, y:H-PADB+16, "text-anchor":"middle"}});
    lbl.textContent = t;
    svg.appendChild(lbl);
  }}
  svg.appendChild(el("line", {{class:"axis", x1:PADL, y1:PADT, x2:PADL, y2:H-PADB}}));
  svg.appendChild(el("line", {{class:"axis", x1:PADL, y1:H-PADB, x2:W-PADR, y2:H-PADB}}));
  const xt = el("text", {{class:"ax-lbl", x:PADL+PW/2, y:H-6, "text-anchor":"middle"}});
  xt.textContent = "denoising step"; svg.appendChild(xt);
  const yt = el("text", {{class:"ax-lbl", x:14, y:PADT+PH/2,
    "text-anchor":"middle", transform:"rotate(-90 14 " + (PADT+PH/2) + ")"}});
  yt.textContent = "fraction committed"; svg.appendChild(yt);
  if (showHalf) {{
    const yH = yScale(0.5);
    svg.appendChild(el("line", {{class:"thr", x1:PADL, y1:yH, x2:W-PADR, y2:yH}}));
  }}
  for (const s of SERIES) {{
    svg.appendChild(el("path", {{d:linePath(s.frac), fill:"none", stroke:s.color,
      "stroke-width":2}}));
    for (const d of s.frac) {{
      svg.appendChild(el("circle", {{cx:xScale(d[0]), cy:yScale(d[1]), r:2,
        fill:s.color}}));
    }}
  }}
  const guide = el("line", {{class:"axis", x1:0, y1:PADT, x2:0, y2:H-PADB,
    stroke:"#bbb", "stroke-dasharray":"2 2", visibility:"hidden"}});
  svg.appendChild(guide);
  const hit = el("rect", {{x:PADL, y:PADT, width:PW, height:PH, fill:"transparent"}});
  svg.appendChild(hit);
  const ro = document.getElementById('readout');
  function nearest(series, sx) {{
    let best = null, bd = Infinity;
    for (const d of series) {{ const dx = Math.abs(xScale(d[0]) - sx);
      if (dx < bd) {{ bd = dx; best = d; }} }}
    return best;
  }}
  hit.addEventListener('mousemove', e => {{
    const r = svg.getBoundingClientRect();
    const sx = (e.clientX - r.left) * (W / r.width);
    guide.setAttribute('x1', sx); guide.setAttribute('x2', sx);
    guide.setAttribute('visibility', 'visible');
    let parts = [];
    let step = "-";
    for (const s of SERIES) {{
      const p = nearest(s.frac, sx);
      if (p) step = p[0];
      parts.push("<span style='color:" + s.color + "'>" + s.name + " " +
        (p ? (p[1] * 100).toFixed(1) + "%" : "-") + "</span>");
    }}
    ro.innerHTML = "step " + step + " &nbsp; " + parts.join(" &nbsp; ");
  }});
  hit.addEventListener('mouseleave', () => {{
    guide.setAttribute('visibility', 'hidden'); ro.innerHTML = "&nbsp;";
  }});
  chart.appendChild(svg);
}}
document.getElementById('half').addEventListener('change', e => {{
  showHalf = e.target.checked; draw();
}});
draw();
</script></body></html>
"""


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Fraction of canvases below commit threshold vs step.")
    ap.add_argument("inputs", nargs="+",
                    help="one or more meanH JSONLs, each 'label=path' or 'path'.")
    ap.add_argument("-o", "--out", default=None,
                    help="output HTML (default commit_fraction.html next to the "
                         "first input)")
    ap.add_argument("-n", "--limit", type=int, default=None,
                    help="use only the first N canvases of each method. Default: "
                         "auto = min canvas count across methods. 0 = all.")
    ap.add_argument("-t", "--threshold", type=float, default=None,
                    help="commit threshold override. Default: confidence_threshold "
                         "from the probe records.")
    ap.add_argument("--labels", default=None,
                    help="comma-separated legend labels overriding per-input "
                         "labels, in input order")
    args = ap.parse_args()

    paths, names = [], []
    for tok in args.inputs:
        if "=" in tok and not os.path.exists(tok):
            label, path = tok.split("=", 1)
            names.append(label.strip() or _default_label(path))
            paths.append(path)
        else:
            paths.append(tok)
            names.append(_default_label(tok))

    if args.labels:
        parts = [s.strip() for s in args.labels.split(",")]
        for i, p in enumerate(parts[:len(names)]):
            if p:
                names[i] = p

    specs = [(names[i], paths[i], PALETTE[i % len(PALETTE)])
             for i in range(len(paths))]
    loaded = []
    conf = None
    for name, path, color in specs:
        canvases, c = load_by_canvas(path)
        if c is not None and conf is None:
            conf = c
        loaded.append((name, color, canvases))

    thr = args.threshold if args.threshold is not None else conf
    if thr is None:
        thr = 0.005
        print(f"no confidence_threshold in records; defaulting to {thr}")

    counts = [len(c) for _, _, c in loaded]
    if args.limit is None:
        limit = min(counts)
    elif args.limit == 0:
        limit = None
    else:
        limit = args.limit
    if limit is not None and min(counts) != max(counts) and args.limit != 0:
        print(f"matching to first {limit} canvases/method (raw counts: {counts})")

    series = []
    ncanvas = 0
    for name, color, canvases in loaded:
        used = canvases
        if limit is not None:
            used = dict(list(canvases.items())[:limit])
        ncanvas = max(ncanvas, len(used))
        frac = commit_fraction(used, thr)
        final = frac[-1][1] if frac else 0.0
        series.append({"name": name, "color": color, "frac": frac})
        print(f"{name}: {len(used)} canvases, {len(frac)} steps -> "
              f"{final*100:.1f}% committed by step {frac[-1][0] if frac else '-'}")

    legend_html = "\n    ".join(
        f'<span class="sw" style="background:{s["color"]}"></span>{s["name"]}'
        for s in series)
    out = args.out or os.path.join(
        os.path.dirname(os.path.abspath(paths[0])), "commit_fraction.html")
    with open(out, "w") as f:
        f.write(TEMPLATE.format(
            series_json=json.dumps(series),
            conf_disp=thr,
            ncanvas=ncanvas,
            legend_html=legend_html,
        ))
    print(f"commit fraction -> {out}")


if __name__ == "__main__":
    main()
