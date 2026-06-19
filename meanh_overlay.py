#!/usr/bin/env python3
"""Overlay meanH-vs-step for bf16, mxfp4, and mixed-precision across many canvases.

Each method's JSONL holds confidence records from MANY generations (50 prompts x
several seeds). Records are grouped by req_id (one req_id == one canvas/denoising
trajectory); for each method we average mean_entropy per step across all canvases
and draw the mean line with a shaded +/-1 sigma band. Self-contained HTML: inline
SVG drawn by JS, log-y toggle, hover readout. No external libraries.

Usage:
  ./meanh_overlay.py runs/meanh50/bf16.jsonl runs/meanh50/mxfp4.jsonl \
      runs/meanh50/mixedprec.jsonl -o runs/meanh50/meanh_overlay.html
"""

from __future__ import annotations

import argparse
import json
import math
import os


def load_by_canvas(path: str) -> dict[str, dict[int, float]]:
    """Load a probe JSONL into {req_id: {step: mean_entropy}}.

    Each req_id is one canvas (one generation). A request that is re-run reuses a
    fresh req_id, so distinct trajectories never collide.

    Records arrive in emission order. A canvas's denoising trajectory is the
    initial run of strictly increasing steps; after it converges the probe emits
    a final commit pass that RESETS the step counter (a record whose step is <=
    the previous one, e.g. ``step=0 committing=True`` then ``step=1``). Those
    trailing records are a separate phase, not part of the convergence curve, and
    they collide on (req_id, step) with the real early steps. We therefore keep
    only the first monotonic run per canvas and drop everything from the first
    step reset onward, so the trailing ``step=1`` cannot clobber the real one.
    """
    canvases: dict[str, dict[int, float]] = {}
    last_step: dict[str, int] = {}
    done: set[str] = set()
    conf = None
    with open(path) as f:
        for line in f:
            line = line.strip().strip("\x00")
            if not line.startswith("{"):
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            rid = rec.get("req_id", "?")
            if rid.startswith("_warmup"):
                continue
            if conf is None:
                conf = rec.get("confidence_threshold")
            if rid in done:
                continue
            step = rec["step"]
            if rid in last_step and step <= last_step[rid]:
                # Step counter reset: the convergence run is over; ignore the
                # trailing commit-pass records for this canvas.
                done.add(rid)
                continue
            last_step[rid] = step
            canvases.setdefault(rid, {})[step] = float(rec.get("mean_entropy", 0.0))
    return canvases, conf


def per_canvas_lines(canvases: dict[str, dict[int, float]]):
    """Each canvas as its own step-sorted polyline ``[[step, val], ...]``.

    Returns a list of polylines (one per canvas) for the spaghetti view, in the
    same canvas order as the input dict so colors/order stay stable.
    """
    lines = []
    for traj in canvases.values():
        lines.append([[s, round(traj[s], 5)] for s in sorted(traj)])
    return lines


def aggregate(canvases: dict[str, dict[int, float]]):
    """Per step, mean and stddev of mean_entropy across canvases.

    Returns (mean_series, lo_series, hi_series) as [[step, value], ...], where
    lo/hi are mean -/+ 1 stddev (clamped at 0).
    """
    by_step: dict[int, list[float]] = {}
    for traj in canvases.values():
        for step, val in traj.items():
            by_step.setdefault(step, []).append(val)
    mean_s, lo_s, hi_s = [], [], []
    for step in sorted(by_step):
        vals = by_step[step]
        mu = sum(vals) / len(vals)
        if len(vals) > 1:
            var = sum((v - mu) ** 2 for v in vals) / (len(vals) - 1)
            sd = math.sqrt(var)
        else:
            sd = 0.0
        mean_s.append([step, round(mu, 5)])
        lo_s.append([step, round(max(0.0, mu - sd), 5)])
        hi_s.append([step, round(mu + sd, 5)])
    return mean_s, lo_s, hi_s


TEMPLATE = """<!doctype html>
<html><head><meta charset="utf-8"><title>diffgemma meanH overlay</title>
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
  <h3>mean token entropy per denoising step ({ncanvas} canvases/method)</h3>
  <div class="legend">
    <span class="sw" style="background:#2a7"></span>bf16
    <span class="sw" style="background:#c44"></span>mxfp4
    <span class="sw" style="background:#36c"></span>mixed-prec
    <span class="sw" style="background:#e6a000"></span>commit threshold ({conf_disp})
    <label class="legend"><input type="checkbox" id="logy"> log y-axis</label>
    <label class="legend"><input type="checkbox" id="band" checked> +/-1 sigma band</label>
    <label class="legend"><input type="checkbox" id="percanvas"> per-canvas lines</label>
  </div>
  <div id="chart"></div>
  <div id="readout">&nbsp;</div>
<script>
// each: {{mean:[[step,v]...], lo:[...], hi:[...], color:"#..", name:".."}}
const SERIES = {series_json};
const CONF = {conf};
const W = 980, H = 500, PADL = 64, PADR = 20, PADT = 20, PADB = 44;
const PW = W - PADL - PADR, PH = H - PADT - PADB;

let logY = false, showBand = true, perCanvas = false;
const allSteps = [], allVals = [];
for (const s of SERIES) {{
  for (const d of s.hi) {{ allSteps.push(d[0]); allVals.push(d[1]); }}
  for (const d of s.lo) {{ allVals.push(d[1]); }}
  for (const ln of (s.lines || [])) {{
    for (const d of ln) {{ allSteps.push(d[0]); allVals.push(d[1]); }}
  }}
}}
const xMin = Math.min(...allSteps), xMax = Math.max(...allSteps);
const yMaxRaw = Math.max(...allVals, CONF);
const posVals = allVals.concat([CONF]).filter(v => v > 0);
const yMinPos = posVals.length ? Math.min(...posVals) : 1e-6;

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
  const r = parseInt(hex.slice(1, 3), 16), g = parseInt(hex.slice(3, 5), 16),
        b = parseInt(hex.slice(5, 7), 16);
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
  yt.textContent = "mean token entropy"; svg.appendChild(yt);
  if (!(logY && CONF <= 0)) {{
    const yT = yScale(CONF);
    svg.appendChild(el("line", {{class:"thr", x1:PADL, y1:yT, x2:W-PADR, y2:yT}}));
  }}
  if (showBand && !perCanvas) {{
    for (const s of SERIES) {{
      svg.appendChild(el("path", {{d:bandPath(s.lo, s.hi), fill:hexA(s.color, 0.13),
        stroke:"none"}}));
    }}
  }}
  if (perCanvas) {{
    for (const s of SERIES) {{
      for (const ln of (s.lines || [])) {{
        if (ln.length < 2) continue;
        svg.appendChild(el("path", {{d:linePath(ln), fill:"none",
          stroke:hexA(s.color, 0.45), "stroke-width":1}}));
      }}
    }}
  }}
  for (const s of SERIES) {{
    svg.appendChild(el("path", {{d:linePath(s.mean), fill:"none", stroke:s.color,
      "stroke-width":2}}));
    for (const d of s.mean) {{
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
      const p = nearest(s.mean, sx);
      if (p) step = p[0];
      parts.push("<span style='color:" + s.color + "'>" + s.name + " meanH=" +
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
document.getElementById('band').addEventListener('change', e => {{
  showBand = e.target.checked; draw();
}});
document.getElementById('percanvas').addEventListener('change', e => {{
  perCanvas = e.target.checked; draw();
}});
draw();
</script></body></html>
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("bf16", help="bf16 meanH JSONL")
    ap.add_argument("mxfp4", help="mxfp4 meanH JSONL")
    ap.add_argument("mixedprec", help="mixed-precision meanH JSONL")
    ap.add_argument("-o", "--out", default=None,
                    help="output HTML (default meanh_overlay.html next to bf16)")
    ap.add_argument("-n", "--limit", type=int, default=None,
                    help="use only the first N canvases of each method (matched "
                         "prompt order). Default: auto = min canvas count across "
                         "methods. Pass 0 to use all canvases per method.")
    args = ap.parse_args()

    specs = [
        ("bf16", args.bf16, "#2a7"),
        ("mxfp4", args.mxfp4, "#c44"),
        ("mixed-prec", args.mixedprec, "#36c"),
    ]
    loaded = []
    conf = None
    for name, path, color in specs:
        canvases, c = load_by_canvas(path)
        if c is not None and conf is None:
            conf = c
        loaded.append((name, color, canvases))

    # Match the canvas set across methods so the comparison is apples-to-apples:
    # the driver loops prompt-outer/seed-inner, so the first N req_ids of each
    # method correspond to the same N prompts. limit<None auto-picks min count.
    counts = [len(c) for _, _, c in loaded]
    if args.limit is None:
        limit = min(counts)
    elif args.limit == 0:
        limit = None
    else:
        limit = args.limit
    if limit is not None and min(counts) != max(counts) and args.limit != 0:
        print(f"matching to first {limit} canvases/method "
              f"(raw counts: {counts})")

    series = []
    ncanvas = 0
    for name, color, canvases in loaded:
        used = canvases
        if limit is not None:
            used = dict(list(canvases.items())[:limit])
        ncanvas = max(ncanvas, len(used))
        mean_s, lo_s, hi_s = aggregate(used)
        series.append({"name": name, "color": color,
                       "mean": mean_s, "lo": lo_s, "hi": hi_s,
                       "lines": per_canvas_lines(used)})
        print(f"{name}: {len(used)} canvases, "
              f"{len(mean_s)} steps -> max step {mean_s[-1][0] if mean_s else '-'}")

    if conf is None:
        conf = 0.0
    out = args.out or os.path.join(
        os.path.dirname(os.path.abspath(args.bf16)), "meanh_overlay.html")
    with open(out, "w") as f:
        f.write(TEMPLATE.format(
            series_json=json.dumps(series),
            conf=json.dumps(conf),
            conf_disp=conf,
            ncanvas=ncanvas,
        ))
    print(f"meanH overlay -> {out}")


if __name__ == "__main__":
    main()
