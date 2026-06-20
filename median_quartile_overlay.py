#!/usr/bin/env python3
"""Overlay median-meanH-vs-step with quantile bands across many canvases.

A robust alternative to meanh_overlay.py. Mean +/- 1 sigma is misleading for this
data: per-step entropy is right-skewed and bounded at 0, so the mean is dragged up
by a few stubborn canvases and the lower sigma band clips at 0. Here we draw the
MEDIAN line with a p25-p75 (interquartile) band and an outer p10-p90 band, which
never clip nonsensically and show the true spread.

Reuses load_by_canvas from meanh_overlay.py (identical canvas/step-reset parsing)
so this is purely an aggregation change -- no re-collection needed.

Usage:
  ./median_quartile_overlay.py bf16=runs/meanh50/bf16.jsonl \
      mxfp4=runs/meanh50/mxfp4.jsonl -o runs/meanh50/median_quartile.html
"""

from __future__ import annotations

import argparse
import json
import os

from meanh_overlay import PALETTE, _default_label, load_by_canvas


def quantile(sorted_vals: list[float], q: float) -> float:
    """Linear-interpolated quantile of an already-sorted list. q in [0,1]."""
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    if n == 1:
        return sorted_vals[0]
    pos = q * (n - 1)
    lo = int(pos)
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def aggregate_quantiles(canvases: dict[str, dict[int, float]]):
    """Per step, median + quartile/decile bands across canvases.

    Returns (med, q25, q75, q10, q90) each as [[step, value], ...].
    """
    by_step: dict[int, list[float]] = {}
    for traj in canvases.values():
        for step, val in traj.items():
            by_step.setdefault(step, []).append(val)
    med, q25, q75, q10, q90 = [], [], [], [], []
    for step in sorted(by_step):
        vals = sorted(by_step[step])
        med.append([step, round(quantile(vals, 0.50), 5)])
        q25.append([step, round(quantile(vals, 0.25), 5)])
        q75.append([step, round(quantile(vals, 0.75), 5)])
        q10.append([step, round(quantile(vals, 0.10), 5)])
        q90.append([step, round(quantile(vals, 0.90), 5)])
    return med, q25, q75, q10, q90


TEMPLATE = """<!doctype html>
<html><head><meta charset="utf-8"><title>diffgemma median/quartile overlay</title>
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
  <h3>median token entropy per denoising step ({ncanvas} canvases/method)</h3>
  <div class="legend">
    {legend_html}
    <span class="sw" style="background:#e6a000"></span>commit threshold ({conf_disp})
    <label class="legend"><input type="checkbox" id="logy"> log y-axis</label>
    <label class="legend"><input type="checkbox" id="iqr" checked> p25-p75 band</label>
    <label class="legend"><input type="checkbox" id="deciles"> p10-p90 band</label>
  </div>
  <div id="chart"></div>
  <div id="readout">&nbsp;</div>
<script>
// each: {{med:[[step,v]...], q25, q75, q10, q90, color:"#..", name:".."}}
const SERIES = {series_json};
const CONF = {conf};
const W = 980, H = 500, PADL = 64, PADR = 20, PADT = 20, PADB = 44;
const PW = W - PADL - PADR, PH = H - PADT - PADB;

let logY = false, showIQR = true, showDeciles = false;
let xMin = Infinity, xMax = -Infinity, yMaxRaw = CONF, yMinPos = Infinity;
const noteV = v => {{
  if (v > yMaxRaw) yMaxRaw = v;
  if (v > 0 && v < yMinPos) yMinPos = v;
}};
const noteX = x => {{ if (x < xMin) xMin = x; if (x > xMax) xMax = x; }};
for (const s of SERIES) {{
  for (const d of s.q90) {{ noteX(d[0]); noteV(d[1]); }}
  for (const d of s.q10) {{ noteV(d[1]); }}
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
function linePath(series) {{
  return series.map((d, i) =>
    (i ? "L" : "M") + xScale(d[0]).toFixed(1) + " " + yScale(d[1]).toFixed(1)
  ).join(" ");
}}
function bandPath(lo, hi) {{
  const up = hi.map((d, i) =>
    (i ? "L" : "M") + xScale(d[0]).toFixed(1) + " " + yScale(d[1]).toFixed(1));
  const dn = lo.slice().reverse().map(d =>
    "L" + xScale(d[0]).toFixed(1) + " " + yScale(d[1]).toFixed(1));
  return up.join(" ") + " " + dn.join(" ") + " Z";
}}
function ticksY() {{
  const out = [];
  if (logY) {{
    let e = Math.floor(Math.log10(yMinPos));
    const top = Math.ceil(Math.log10(yMaxRaw || 1));
    for (; e <= top; e++) out.push(Math.pow(10, e));
  }} else {{
    const n = 5;
    for (let i = 0; i <= n; i++) out.push(yMaxRaw * i / n);
  }}
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
function hexA(hex, a) {{
  let h = hex.replace("#", "");
  if (h.length === 3) h = h[0]+h[0] + h[1]+h[1] + h[2]+h[2];
  const r = parseInt(h.slice(0, 2), 16), g = parseInt(h.slice(2, 4), 16),
        b = parseInt(h.slice(4, 6), 16);
  return "rgba(" + r + "," + g + "," + b + "," + a + ")";
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
    lbl.textContent = t >= 0.01 || t === 0 ? t.toFixed(3) : t.toExponential(0);
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
  yt.textContent = "token entropy (median)"; svg.appendChild(yt);
  if (!(logY && CONF <= 0)) {{
    const yT = yScale(CONF);
    svg.appendChild(el("line", {{class:"thr", x1:PADL, y1:yT, x2:W-PADR, y2:yT}}));
  }}
  if (showDeciles) {{
    for (const s of SERIES) {{
      svg.appendChild(el("path", {{d:bandPath(s.q10, s.q90), fill:hexA(s.color, 0.08),
        stroke:"none"}}));
    }}
  }}
  if (showIQR) {{
    for (const s of SERIES) {{
      svg.appendChild(el("path", {{d:bandPath(s.q25, s.q75), fill:hexA(s.color, 0.15),
        stroke:"none"}}));
    }}
  }}
  for (const s of SERIES) {{
    svg.appendChild(el("path", {{d:linePath(s.med), fill:"none", stroke:s.color,
      "stroke-width":2}}));
    for (const d of s.med) {{
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
      const p = nearest(s.med, sx);
      if (p) step = p[0];
      parts.push("<span style='color:" + s.color + "'>" + s.name + " med=" +
        (p ? p[1] : "-") + "</span>");
    }}
    ro.innerHTML = "step " + step + " &nbsp; " + parts.join(" &nbsp; ");
  }});
  hit.addEventListener('mouseleave', () => {{
    guide.setAttribute('visibility', 'hidden'); ro.innerHTML = "&nbsp;";
  }});
  chart.appendChild(svg);
}}
document.getElementById('logy').addEventListener('change', e => {{
  logY = e.target.checked; draw();
}});
document.getElementById('iqr').addEventListener('change', e => {{
  showIQR = e.target.checked; draw();
}});
document.getElementById('deciles').addEventListener('change', e => {{
  showDeciles = e.target.checked; draw();
}});
draw();
</script></body></html>
"""


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Overlay median + quantile bands of meanH-vs-step.")
    ap.add_argument("inputs", nargs="+",
                    help="one or more meanH JSONLs, each 'label=path' or 'path'.")
    ap.add_argument("-o", "--out", default=None,
                    help="output HTML (default median_quartile.html next to the "
                         "first input)")
    ap.add_argument("-n", "--limit", type=int, default=None,
                    help="use only the first N canvases of each method. Default: "
                         "auto = min canvas count across methods. 0 = all.")
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
        med, q25, q75, q10, q90 = aggregate_quantiles(used)
        series.append({"name": name, "color": color, "med": med,
                       "q25": q25, "q75": q75, "q10": q10, "q90": q90})
        print(f"{name}: {len(used)} canvases, {len(med)} steps -> "
              f"max step {med[-1][0] if med else '-'}")

    if conf is None:
        conf = 0.0
    legend_html = "\n    ".join(
        f'<span class="sw" style="background:{s["color"]}"></span>{s["name"]}'
        for s in series)
    out = args.out or os.path.join(
        os.path.dirname(os.path.abspath(paths[0])), "median_quartile.html")
    with open(out, "w") as f:
        f.write(TEMPLATE.format(
            series_json=json.dumps(series),
            conf=json.dumps(conf),
            conf_disp=conf,
            ncanvas=ncanvas,
            legend_html=legend_html,
        ))
    print(f"median/quartile overlay -> {out}")


if __name__ == "__main__":
    main()
