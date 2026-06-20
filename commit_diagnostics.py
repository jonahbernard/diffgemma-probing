#!/usr/bin/env python3
"""Diagnostics for the bf16-vs-moe-fp4 commit-fraction gap.

The commit-fraction chart shows bf16 committing ~2x faster than moe-fp4/mlp-bf16,
yet the median/mean meanH charts look nearly identical. The reason: the commit
threshold (0.005) lives in the deep low-entropy tail, which a 0..4 linear median
axis crushes to a flat line near zero. These four charts magnify that tail and
view the gap from multiple angles. Single parse pass over the inputs; emits four
self-contained HTML files.

  1. ecdf      per-canvas entropy ECDF at chosen steps (log-x). The vertical line
               at the threshold: each curve's height there IS its commit fraction;
               its crossing of y=0.5 is the median. Reconciles the two charts.
  2. sweep     commit fraction vs step at several thresholds. Shows whether the 2x
               is deep-tail-specific or holds at every threshold.
  3. commitpmf histogram + CDF of the first step each canvas crosses threshold.
               Per-canvas timing-gap view.
  4. tailq     p10/p25/p50 entropy per step on log-y. The tail the median hides.

Usage:
  ./commit_diagnostics.py "bf16 t0=runs/meanh50/bf16_8k_temp0.jsonl" ... \
      --outdir runs/meanh50 [--steps 5,10,15,20,25,30] \
      [--thresholds 0.001,0.005,0.01,0.05,0.1]
"""

from __future__ import annotations

import argparse
import json
import os

from meanh_overlay import PALETTE, _default_label, load_by_canvas
from median_quartile_overlay import quantile

DEFAULT_STEPS = [5, 10, 15, 20, 25, 30]
DEFAULT_THRESHOLDS = [0.001, 0.005, 0.01, 0.05, 0.1]
ECDF_Q = [i / 100 for i in range(101)]  # 0.00 .. 1.00


def by_step_values(canvases):
    """{step: sorted list of per-canvas entropy at that step}."""
    bs: dict[int, list[float]] = {}
    for traj in canvases.values():
        for step, val in traj.items():
            bs.setdefault(step, []).append(val)
    for step in bs:
        bs[step].sort()
    return bs


def ecdf_at_steps(bs_sorted, steps):
    """For each requested step, the quantile function as [[value, q], ...].

    Plotted as x=value, y=q this is the ECDF: y is the fraction of canvases with
    entropy <= value. Skips steps absent for this series.
    """
    out = {}
    for step in steps:
        vals = bs_sorted.get(step)
        if not vals:
            continue
        out[str(step)] = [[round(quantile(vals, q), 6), q] for q in ECDF_Q]
    return out


def commit_steps(canvases, thr):
    """List of first step each canvas crosses (<=) thr; plus never-committed count."""
    first, never = [], 0
    for traj in canvases.values():
        cs = None
        for s in sorted(traj):
            if traj[s] <= thr:
                cs = s
                break
        if cs is None:
            never += 1
        else:
            first.append(cs)
    return first, never


def commit_fraction_curve(canvases, thr):
    """[[step, fraction committed by step], ...], commit treated as absorbing."""
    n = len(canvases)
    if n == 0:
        return []
    first, _ = commit_steps(canvases, thr)
    max_step = max((max(t) for t in (traj.keys() for traj in canvases.values())),
                   default=0)
    return [[step, round(sum(1 for cs in first if cs <= step) / n, 5)]
            for step in range(1, max_step + 1)]


def tail_quantiles(bs_sorted):
    """p10/p25/p50 per step as ([[step,v]...], ...)."""
    p10, p25, p50 = [], [], []
    for step in sorted(bs_sorted):
        vals = bs_sorted[step]
        p10.append([step, round(quantile(vals, 0.10), 6)])
        p25.append([step, round(quantile(vals, 0.25), 6)])
        p50.append([step, round(quantile(vals, 0.50), 6)])
    return p10, p25, p50


def commit_pmf(first, max_step):
    """Histogram [[step,count]...] of first-crossing steps over 1..max_step."""
    counts = {}
    for cs in first:
        counts[cs] = counts.get(cs, 0) + 1
    return [[s, counts.get(s, 0)] for s in range(1, max_step + 1)]


# ---------------------------------------------------------------------------
# Shared HTML scaffold. Each chart supplies BODY (controls + script).
# ---------------------------------------------------------------------------
HEAD = """<!doctype html>
<html><head><meta charset="utf-8"><title>{title}</title>
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
  <h3>{heading}</h3>
  <div class="legend">{legend_html}{controls}</div>
  <div id="chart"></div>
  <div id="readout">&nbsp;</div>
<script>
const SERIES = {series_json};
const CFG = {cfg_json};
const W = 980, H = 520, PADL = 64, PADR = 20, PADT = 20, PADB = 48;
const PW = W - PADL - PADR, PH = H - PADT - PADB;
const SVGNS = "http://www.w3.org/2000/svg";
function el(name, attrs) {{
  const n = document.createElementNS(SVGNS, name);
  for (const k in attrs) n.setAttribute(k, attrs[k]);
  return n;
}}
function hexA(hex, a) {{
  let h = hex.replace("#", "");
  if (h.length === 3) h = h[0]+h[0]+h[1]+h[1]+h[2]+h[2];
  const r = parseInt(h.slice(0,2),16), g = parseInt(h.slice(2,4),16),
        b = parseInt(h.slice(4,6),16);
  return "rgba("+r+","+g+","+b+","+a+")";
}}
"""

TAIL = """
</script></body></html>
"""


def legend(series):
    return "\n    ".join(
        f'<span class="sw" style="background:{s["color"]}"></span>{s["name"]}'
        for s in series)


# ---------------------------------------------------------------------------
# 1. ECDF by step (log-x). x=entropy, y=fraction of canvases <= x.
# ---------------------------------------------------------------------------
ECDF_JS = """
let stepSel = CFG.steps[Math.min(CFG.defaultIdx, CFG.steps.length-1)];
const FLOOR = CFG.floor, THR = CFG.thr;
function curveFor(s, step) {{ return (s.ecdf[step] || []); }}
function xLog(v) {{
  const lo = Math.log10(FLOOR), hi = Math.log10(CFG.xmax);
  const vv = Math.log10(Math.max(v, FLOOR));
  return PADL + (hi===lo?0:(vv-lo)/(hi-lo))*PW;
}}
const yScale = q => PADT + (1-q)*PH;
let svg;
function draw() {{
  const chart = document.getElementById('chart'); chart.innerHTML="";
  svg = el("svg",{{width:W,height:H,viewBox:"0 0 "+W+" "+H}});
  for (let i=0;i<=5;i++) {{ const q=i/5, y=yScale(q);
    svg.appendChild(el("line",{{class:"grid",x1:PADL,y1:y,x2:W-PADR,y2:y}}));
    const l=el("text",{{class:"ax-lbl",x:PADL-6,y:y+3,"text-anchor":"end"}});
    l.textContent=q.toFixed(1); svg.appendChild(l); }}
  let e=Math.floor(Math.log10(FLOOR)); const top=Math.ceil(Math.log10(CFG.xmax));
  for (;e<=top;e++) {{ const x=xLog(Math.pow(10,e));
    svg.appendChild(el("line",{{class:"grid",x1:x,y1:PADT,x2:x,y2:H-PADB}}));
    const l=el("text",{{class:"ax-lbl",x:x,y:H-PADB+16,"text-anchor":"middle"}});
    l.textContent="1e"+e; svg.appendChild(l); }}
  svg.appendChild(el("line",{{class:"axis",x1:PADL,y1:PADT,x2:PADL,y2:H-PADB}}));
  svg.appendChild(el("line",{{class:"axis",x1:PADL,y1:H-PADB,x2:W-PADR,y2:H-PADB}}));
  const xt=el("text",{{class:"ax-lbl",x:PADL+PW/2,y:H-6,"text-anchor":"middle"}});
  xt.textContent="token entropy (log)"; svg.appendChild(xt);
  const yt=el("text",{{class:"ax-lbl",x:14,y:PADT+PH/2,"text-anchor":"middle",
    transform:"rotate(-90 14 "+(PADT+PH/2)+")"}});
  yt.textContent="fraction of canvases <= x"; svg.appendChild(yt);
  // threshold vertical + median horizontal guides
  const xT=xLog(THR);
  svg.appendChild(el("line",{{class:"thr",x1:xT,y1:PADT,x2:xT,y2:H-PADB}}));
  const tl=el("text",{{class:"ax-lbl",x:xT+3,y:PADT+12,fill:"#e6a000"}});
  tl.textContent="thr "+THR; svg.appendChild(tl);
  const yM=yScale(0.5);
  svg.appendChild(el("line",{{class:"thr",x1:PADL,y1:yM,x2:W-PADR,y2:yM}}));
  for (const s of SERIES) {{
    const c=curveFor(s,stepSel); if(!c.length) continue;
    const d=c.map((p,i)=>(i?"L":"M")+xLog(p[0]).toFixed(1)+" "+yScale(p[1]).toFixed(1)).join(" ");
    svg.appendChild(el("path",{{d:d,fill:"none",stroke:s.color,"stroke-width":2}}));
  }}
  // readout: commit fraction (y at THR) per series
  const ro=document.getElementById('readout'); let parts=[];
  for (const s of SERIES) {{ const c=curveFor(s,stepSel); if(!c.length) continue;
    let frac=0; for (const p of c) {{ if(p[0]<=THR) frac=p[1]; }}
    parts.push("<span style='color:"+s.color+"'>"+s.name+" "+(frac*100).toFixed(1)+"% <= thr</span>"); }}
  ro.innerHTML="step "+stepSel+" &nbsp; "+parts.join(" &nbsp; ");
  chart.appendChild(svg);
}}
const sel=document.getElementById('stepsel');
sel.addEventListener('change',e=>{{ stepSel=e.target.value; draw(); }});
draw();
"""


def render_ecdf(series, steps, thr, outpath):
    # x range from smallest positive quantile value to global max
    xmax = thr
    floor = thr
    for s in series:
        for arr in s["ecdf"].values():
            for v, _q in arr:
                if v > xmax:
                    xmax = v
                if 0 < v < floor:
                    floor = v
    floor = max(floor, 1e-6)
    xmax = max(xmax, thr * 10)
    avail = [str(x) for x in steps if any(str(x) in s["ecdf"] for s in series)]
    default_idx = min(3, len(avail) - 1) if avail else 0
    opts = "".join(f'<option value="{v}">step {v}</option>' for v in avail)
    controls = (f'<label class="legend">step '
                f'<select id="stepsel">{opts}</select></label>')
    cfg = {"steps": avail, "defaultIdx": default_idx,
           "floor": floor, "xmax": xmax, "thr": thr}
    html = (HEAD.format(
                title="ECDF of per-canvas entropy by step",
                heading=("per-canvas entropy ECDF at a fixed step "
                         "(height at the threshold line = commit fraction; "
                         "crossing y=0.5 = median)"),
                legend_html=legend(series), controls=controls,
                series_json=json.dumps(series), cfg_json=json.dumps(cfg))
            + ECDF_JS.replace("{{", "{").replace("}}", "}") + TAIL)
    # set the dropdown default selection
    sel_val = avail[default_idx] if avail else ""
    html = html.replace(f'<option value="{sel_val}">step {sel_val}</option>',
                        f'<option value="{sel_val}" selected>step {sel_val}</option>',
                        1)
    with open(outpath, "w") as f:
        f.write(html)


# ---------------------------------------------------------------------------
# 2. Threshold sweep: commit fraction vs step, pick threshold.
# ---------------------------------------------------------------------------
SWEEP_JS = """
let thrSel = CFG.thresholds[CFG.defaultIdx];
let xMin=Infinity,xMax=-Infinity;
for (const s of SERIES) for (const t in s.curves)
  for (const d of s.curves[t]) {{ if(d[0]<xMin)xMin=d[0]; if(d[0]>xMax)xMax=d[0]; }}
if(!isFinite(xMin)){{xMin=0;xMax=1;}}
const xScale=s=>PADL+(xMax===xMin?0:(s-xMin)/(xMax-xMin))*PW;
const yScale=v=>PADT+(1-v)*PH;
let svg;
function draw() {{
  const chart=document.getElementById('chart'); chart.innerHTML="";
  svg=el("svg",{{width:W,height:H,viewBox:"0 0 "+W+" "+H}});
  for (let i=0;i<=5;i++) {{ const y=yScale(i/5);
    svg.appendChild(el("line",{{class:"grid",x1:PADL,y1:y,x2:W-PADR,y2:y}}));
    const l=el("text",{{class:"ax-lbl",x:PADL-6,y:y+3,"text-anchor":"end"}});
    l.textContent=(i/5).toFixed(1); svg.appendChild(l); }}
  const span=xMax-xMin, sg=Math.ceil(span/10)||1;
  for (let t=xMin;t<=xMax;t+=sg) {{ const x=xScale(t);
    svg.appendChild(el("line",{{class:"grid",x1:x,y1:PADT,x2:x,y2:H-PADB}}));
    const l=el("text",{{class:"ax-lbl",x:x,y:H-PADB+16,"text-anchor":"middle"}});
    l.textContent=t; svg.appendChild(l); }}
  svg.appendChild(el("line",{{class:"axis",x1:PADL,y1:PADT,x2:PADL,y2:H-PADB}}));
  svg.appendChild(el("line",{{class:"axis",x1:PADL,y1:H-PADB,x2:W-PADR,y2:H-PADB}}));
  const xt=el("text",{{class:"ax-lbl",x:PADL+PW/2,y:H-6,"text-anchor":"middle"}});
  xt.textContent="denoising step"; svg.appendChild(xt);
  const yt=el("text",{{class:"ax-lbl",x:14,y:PADT+PH/2,"text-anchor":"middle",
    transform:"rotate(-90 14 "+(PADT+PH/2)+")"}});
  yt.textContent="fraction committed"; svg.appendChild(yt);
  const yH=yScale(0.5);
  svg.appendChild(el("line",{{class:"thr",x1:PADL,y1:yH,x2:W-PADR,y2:yH}}));
  for (const s of SERIES) {{ const c=s.curves[thrSel]||[];
    const d=c.map((p,i)=>(i?"L":"M")+xScale(p[0]).toFixed(1)+" "+yScale(p[1]).toFixed(1)).join(" ");
    svg.appendChild(el("path",{{d:d,fill:"none",stroke:s.color,"stroke-width":2}})); }}
  document.getElementById('readout').innerHTML="threshold = "+thrSel;
  chart.appendChild(svg);
}}
document.getElementById('thrsel').addEventListener('change',e=>{{thrSel=e.target.value;draw();}});
draw();
"""


def render_sweep(series, thresholds, outpath):
    default_idx = thresholds.index(0.005) if 0.005 in thresholds else 0
    opts = "".join(
        f'<option value="{t}"{" selected" if i==default_idx else ""}>thr {t}</option>'
        for i, t in enumerate(thresholds))
    controls = (f'<label class="legend">threshold '
                f'<select id="thrsel">{opts}</select></label>')
    cfg = {"thresholds": [str(t) for t in thresholds], "defaultIdx": default_idx}
    # series curves keyed by str(threshold); defaultIdx is an integer index into
    # cfg.thresholds (the JS does CFG.thresholds[CFG.defaultIdx]).
    html = (HEAD.format(
                title="Commit fraction vs step, threshold sweep",
                heading="fraction of canvases committed vs step, by commit threshold",
                legend_html=legend(series), controls=controls,
                series_json=json.dumps(series), cfg_json=json.dumps(cfg))
            + SWEEP_JS.replace("{{", "{").replace("}}", "}") + TAIL)
    with open(outpath, "w") as f:
        f.write(html)


# ---------------------------------------------------------------------------
# 3. Commit-step PMF (bars) + CDF (lines).
# ---------------------------------------------------------------------------
PMF_JS = """
let xMin=1,xMax=1;
for (const s of SERIES) for (const d of s.pmf) if(d[0]>xMax)xMax=d[0];
let yMax=0;
for (const s of SERIES) for (const d of s.pmf) {{ const f=d[1]/s.n; if(f>yMax)yMax=f; }}
yMax=yMax*1.1||1;
const xScale=s=>PADL+(xMax<=1?0:(s-xMin)/(xMax-xMin))*PW;
const yScale=v=>PADT+(1-v/yMax)*PH;
const yCdf=v=>PADT+(1-v)*PH;
let svg;
function draw() {{
  const chart=document.getElementById('chart'); chart.innerHTML="";
  svg=el("svg",{{width:W,height:H,viewBox:"0 0 "+W+" "+H}});
  for (let i=0;i<=5;i++) {{ const y=PADT+(i/5)*PH;
    svg.appendChild(el("line",{{class:"grid",x1:PADL,y1:y,x2:W-PADR,y2:y}})); }}
  const span=xMax-xMin, sg=Math.ceil(span/12)||1;
  for (let t=xMin;t<=xMax;t+=sg) {{ const x=xScale(t);
    const l=el("text",{{class:"ax-lbl",x:x,y:H-PADB+16,"text-anchor":"middle"}});
    l.textContent=t; svg.appendChild(l); }}
  // left axis = pmf fraction
  for (let i=0;i<=5;i++) {{ const v=yMax*i/5, y=yScale(v);
    const l=el("text",{{class:"ax-lbl",x:PADL-6,y:y+3,"text-anchor":"end"}});
    l.textContent=v.toFixed(3); svg.appendChild(l); }}
  // right axis = cdf 0..1
  for (let i=0;i<=5;i++) {{ const q=i/5,y=yCdf(q);
    const l=el("text",{{class:"ax-lbl",x:W-PADR+4,y:y+3,"text-anchor":"start"}});
    l.textContent=q.toFixed(1); svg.appendChild(l); }}
  svg.appendChild(el("line",{{class:"axis",x1:PADL,y1:PADT,x2:PADL,y2:H-PADB}}));
  svg.appendChild(el("line",{{class:"axis",x1:PADL,y1:H-PADB,x2:W-PADR,y2:H-PADB}}));
  svg.appendChild(el("line",{{class:"axis",x1:W-PADR,y1:PADT,x2:W-PADR,y2:H-PADB}}));
  const xt=el("text",{{class:"ax-lbl",x:PADL+PW/2,y:H-6,"text-anchor":"middle"}});
  xt.textContent="first step crossing threshold"; svg.appendChild(xt);
  const yt=el("text",{{class:"ax-lbl",x:14,y:PADT+PH/2,"text-anchor":"middle",
    transform:"rotate(-90 14 "+(PADT+PH/2)+")"}});
  yt.textContent="fraction of canvases (bars)"; svg.appendChild(yt);
  const nser=SERIES.length, bw=(xScale(xMin+1)-xScale(xMin))/(nser+1)||4;
  SERIES.forEach((s,si)=>{{
    for (const d of s.pmf) {{ const f=d[1]/s.n; if(!f) continue;
      const x=xScale(d[0])+ (si-(nser-1)/2)*bw, y=yScale(f);
      svg.appendChild(el("rect",{{x:x-bw/2,y:y,width:Math.max(1,bw-0.5),
        height:(H-PADB)-y,fill:hexA(s.color,0.55)}})); }}
  }});
  for (const s of SERIES) {{ // cdf line
    let acc=0; const pts=[];
    for (const d of s.pmf) {{ acc+=d[1]/s.n; pts.push([d[0],acc]); }}
    const dd=pts.map((p,i)=>(i?"L":"M")+xScale(p[0]).toFixed(1)+" "+yCdf(p[1]).toFixed(1)).join(" ");
    svg.appendChild(el("path",{{d:dd,fill:"none",stroke:s.color,"stroke-width":2}})); }}
  const ro=document.getElementById('readout'); let parts=[];
  for (const s of SERIES) parts.push("<span style='color:"+s.color+"'>"+s.name+
    " never-commit "+(s.never/s.n*100).toFixed(1)+"%</span>");
  ro.innerHTML=parts.join(" &nbsp; ");
  chart.appendChild(svg);
}}
draw();
"""


def render_pmf(series, outpath):
    html = (HEAD.format(
                title="Commit-step distribution",
                heading=("distribution of the first step each canvas crosses the "
                         "threshold (bars=PMF, lines=CDF)"),
                legend_html=legend(series), controls="",
                series_json=json.dumps(series), cfg_json=json.dumps({}))
            + PMF_JS.replace("{{", "{").replace("}}", "}") + TAIL)
    with open(outpath, "w") as f:
        f.write(html)


# ---------------------------------------------------------------------------
# 4. Tail quantiles on log-y.
# ---------------------------------------------------------------------------
TAILQ_JS = """
const THR=CFG.thr;
let xMin=Infinity,xMax=-Infinity,yMin=Infinity,yMax=THR;
for (const s of SERIES) for (const key of ["p50","p25","p10"])
  for (const d of s[key]) {{ if(d[0]<xMin)xMin=d[0]; if(d[0]>xMax)xMax=d[0];
    if(d[1]>yMax)yMax=d[1]; if(d[1]>0&&d[1]<yMin)yMin=d[1]; }}
if(THR>0&&THR<yMin)yMin=THR; if(!isFinite(yMin))yMin=1e-6;
if(!isFinite(xMin)){{xMin=0;xMax=1;}}
const xScale=s=>PADL+(xMax===xMin?0:(s-xMin)/(xMax-xMin))*PW;
function yScale(v) {{ const lo=Math.log10(yMin),hi=Math.log10(yMax);
  const vv=Math.log10(Math.max(v,yMin));
  return PADT+(hi===lo?0:(1-(vv-lo)/(hi-lo))*PH); }}
const DASH={{p50:"",p25:"5 3",p10:"2 3"}};
let svg;
function draw() {{
  const chart=document.getElementById('chart'); chart.innerHTML="";
  svg=el("svg",{{width:W,height:H,viewBox:"0 0 "+W+" "+H}});
  let e=Math.floor(Math.log10(yMin)); const top=Math.ceil(Math.log10(yMax));
  for (;e<=top;e++) {{ const y=yScale(Math.pow(10,e));
    svg.appendChild(el("line",{{class:"grid",x1:PADL,y1:y,x2:W-PADR,y2:y}}));
    const l=el("text",{{class:"ax-lbl",x:PADL-6,y:y+3,"text-anchor":"end"}});
    l.textContent="1e"+e; svg.appendChild(l); }}
  const span=xMax-xMin, sg=Math.ceil(span/10)||1;
  for (let t=xMin;t<=xMax;t+=sg) {{ const x=xScale(t);
    svg.appendChild(el("line",{{class:"grid",x1:x,y1:PADT,x2:x,y2:H-PADB}}));
    const l=el("text",{{class:"ax-lbl",x:x,y:H-PADB+16,"text-anchor":"middle"}});
    l.textContent=t; svg.appendChild(l); }}
  svg.appendChild(el("line",{{class:"axis",x1:PADL,y1:PADT,x2:PADL,y2:H-PADB}}));
  svg.appendChild(el("line",{{class:"axis",x1:PADL,y1:H-PADB,x2:W-PADR,y2:H-PADB}}));
  const xt=el("text",{{class:"ax-lbl",x:PADL+PW/2,y:H-6,"text-anchor":"middle"}});
  xt.textContent="denoising step"; svg.appendChild(xt);
  const yt=el("text",{{class:"ax-lbl",x:14,y:PADT+PH/2,"text-anchor":"middle",
    transform:"rotate(-90 14 "+(PADT+PH/2)+")"}});
  yt.textContent="token entropy (log)"; svg.appendChild(yt);
  const yT=yScale(THR);
  svg.appendChild(el("line",{{class:"thr",x1:PADL,y1:yT,x2:W-PADR,y2:yT}}));
  for (const s of SERIES) for (const key of ["p10","p25","p50"]) {{
    const c=s[key]; if(!c.length) continue;
    const d=c.map((p,i)=>(i?"L":"M")+xScale(p[0]).toFixed(1)+" "+yScale(p[1]).toFixed(1)).join(" ");
    svg.appendChild(el("path",{{d:d,fill:"none",stroke:s.color,"stroke-width":key==="p50"?2:1,
      "stroke-dasharray":DASH[key]}})); }}
  document.getElementById('readout').innerHTML=
    "solid=median &nbsp; dashed=p25 &nbsp; dotted=p10 &nbsp; (lower = more committed)";
  chart.appendChild(svg);
}}
draw();
"""


def render_tailq(series, thr, outpath):
    cfg = {"thr": thr}
    html = (HEAD.format(
                title="Low-tail entropy quantiles (log-y)",
                heading="p10 / p25 / p50 token entropy per step on a log axis",
                legend_html=legend(series), controls="",
                series_json=json.dumps(series), cfg_json=json.dumps(cfg))
            + TAILQ_JS.replace("{{", "{").replace("}}", "}") + TAIL)
    with open(outpath, "w") as f:
        f.write(html)


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
    ap.add_argument("--outdir", default=None, help="output dir (default: dir of first input)")
    ap.add_argument("-n", "--limit", type=int, default=None,
                    help="first N canvases/method. Default auto=min count. 0=all.")
    ap.add_argument("-t", "--threshold", type=float, default=None,
                    help="commit threshold. Default: from records.")
    ap.add_argument("--steps", default=None, help="comma steps for ECDF (default 5,10,15,20,25,30)")
    ap.add_argument("--thresholds", default=None,
                    help="comma thresholds for sweep (default 0.001,0.005,0.01,0.05,0.1)")
    ap.add_argument("--labels", default=None, help="comma legend labels")
    ap.add_argument("--only", default=None,
                    help="comma subset of charts: ecdf,sweep,commitpmf,tailq")
    args = ap.parse_args()

    paths, names = parse_inputs(args.inputs, args.labels)
    steps = ([int(x) for x in args.steps.split(",")] if args.steps else DEFAULT_STEPS)
    thresholds = ([float(x) for x in args.thresholds.split(",")]
                  if args.thresholds else DEFAULT_THRESHOLDS)
    want = set(args.only.split(",")) if args.only else {
        "ecdf", "sweep", "commitpmf", "tailq"}

    loaded, conf = [], None
    for i, (name, path) in enumerate(zip(names, paths)):
        canvases, c = load_by_canvas(path)
        if c is not None and conf is None:
            conf = c
        loaded.append((name, PALETTE[i % len(PALETTE)], canvases))
        print(f"parsed {name}: {len(canvases)} canvases")

    thr = args.threshold if args.threshold is not None else (conf if conf is not None else 0.005)
    if 0.005 not in thresholds and thr == 0.005:
        thresholds = sorted(set(thresholds + [0.005]))

    counts = [len(c) for _, _, c in loaded]
    if args.limit is None:
        limit = min(counts)
    elif args.limit == 0:
        limit = None
    else:
        limit = args.limit
    if limit is not None and min(counts) != max(counts):
        print(f"matching to first {limit} canvases/method (raw counts: {counts})")

    # Build per-series payloads for each chart from the SAME canvas subset.
    ecdf_series, sweep_series, pmf_series, tailq_series = [], [], [], []
    global_max_step = 0
    pmf_tmp = []
    for name, color, canvases in loaded:
        used = canvases if limit is None else dict(list(canvases.items())[:limit])
        bs = by_step_values(used)
        mx = max(bs) if bs else 0
        global_max_step = max(global_max_step, mx)
        if "ecdf" in want:
            ecdf_series.append({"name": name, "color": color,
                                "ecdf": ecdf_at_steps(bs, steps)})
        if "sweep" in want:
            curves = {str(t): commit_fraction_curve(used, t) for t in thresholds}
            sweep_series.append({"name": name, "color": color, "curves": curves})
        if "tailq" in want:
            p10, p25, p50 = tail_quantiles(bs)
            tailq_series.append({"name": name, "color": color,
                                 "p10": p10, "p25": p25, "p50": p50})
        if "commitpmf" in want:
            first, never = commit_steps(used, thr)
            pmf_tmp.append((name, color, first, never, len(used), mx))

    outdir = args.outdir or os.path.dirname(os.path.abspath(paths[0]))
    os.makedirs(outdir, exist_ok=True)

    if "ecdf" in want:
        out = os.path.join(outdir, "diag_ecdf.html")
        render_ecdf(ecdf_series, steps, thr, out)
        print(f"ECDF by step -> {out}")
    if "sweep" in want:
        out = os.path.join(outdir, "diag_threshold_sweep.html")
        render_sweep(sweep_series, thresholds, out)
        print(f"threshold sweep -> {out}")
    if "commitpmf" in want:
        for name, color, first, never, n, mx in pmf_tmp:
            pmf_series.append({"name": name, "color": color,
                               "pmf": commit_pmf(first, global_max_step),
                               "never": never, "n": n})
            print(f"  {name}: {n} canvases, {never} never-commit "
                  f"({never/n*100:.1f}%)")
        out = os.path.join(outdir, "diag_commit_step.html")
        render_pmf(pmf_series, out)
        print(f"commit-step distribution -> {out}")
    if "tailq" in want:
        out = os.path.join(outdir, "diag_tail_quantiles.html")
        render_tailq(tailq_series, thr, out)
        print(f"tail quantiles -> {out}")


if __name__ == "__main__":
    main()
