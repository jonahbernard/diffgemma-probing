#!/usr/bin/env python3
"""Finish-step distribution sliced by canvas POSITION within the prompt.

Same four views as finish_step_dist.py (histogram / buckets / box / CDF), but
instead of pooling every canvas of a variant, this slices by canvas *position*:
position p is the p-th canvas generated for each prompt. The question it answers:
do the finish_step_dist.html curves look the same if you only look at the first
canvas per prompt, then only the second, and so on?

Position is inferred from emission order exactly as per_position_overlay.py does:
load_by_canvas preserves first-seen (== send) order, the driver loops
prompt-outer / seed-inner, so the i-th canvas has prompt = i // nseed and
position = i % nseed. nseed defaults to round(canvas_count / nprompt) per file.

Variants have different nseed (bf16 ~300, moe ~100), so positions are capped to
the smallest nseed across the inputs, keeping all variants comparable at every
position. One self-contained HTML: a position dropdown redraws all four charts.

Usage:
  ./finish_step_dist_per_position.py \
    "bf16 t0=runs/meanh50/bf16_8k_temp0.jsonl" \
    "bf16 t0.6=runs/meanh50/bf16_8k_gpu7_temp0.6.jsonl" \
    "moe t0=runs/meanh50/moe-fp4-mlp-bf16_8k_temp0.jsonl" \
    "moe t0.6=runs/meanh50/moe-fp4-mlp-bf16_8k_gpu7_temp0.6.jsonl" \
    -o runs/meanh50/finish_step_dist_per_position.html
"""

from __future__ import annotations

import argparse
import json
import os

from meanh_overlay import PALETTE, _default_label, load_by_canvas
from median_quartile_overlay import quantile
from commit_diagnostics import commit_steps
from finish_step_dist import histogram, cdf, buckets, box_stats, DEFAULT_BUCKETS

NPROMPT = 50  # prompts50.txt


def positions_from_order(canvases: dict, nseed: int):
    """{position: [trajectory dict, ...]} grouped by (insertion index) % nseed."""
    by_pos: dict[int, list] = {}
    for i, traj in enumerate(canvases.values()):
        by_pos.setdefault(i % nseed, []).append(traj)
    return by_pos


TEMPLATE = """<!doctype html>
<html><head><meta charset="utf-8"><title>finish-step distribution per position</title>
<style>
  body {{ font-family: monospace; margin: 16px; background:#fafafa; color:#222; }}
  h3 {{ margin:14px 0 4px; }} h4 {{ margin:10px 0 2px; font-weight:normal; color:#555; }}
  .legend {{ font-size:12px; color:#555; margin-bottom:6px; }}
  .sw {{ display:inline-block; width:11px; height:11px; vertical-align:middle;
         margin:0 3px 0 10px; border:1px solid #888; }}
  svg {{ background:#fff; border:1px solid #ddd; display:block; margin-bottom:8px; }}
  .grid {{ stroke:#eee; stroke-width:1; }}
  .axis {{ stroke:#888; stroke-width:1; }}
  .ax-lbl {{ font-size:11px; fill:#666; }}
  table {{ border-collapse:collapse; font-size:12px; margin:4px 0 14px; }}
  td,th {{ border:1px solid #ddd; padding:2px 8px; text-align:right; }}
  th {{ background:#f0f0f0; }}
  select {{ font-family:monospace; font-size:13px; }}
</style></head><body>
  <h3>finish-step distribution by canvas position &mdash; commit = meanH &le; {conf_disp}</h3>
  <div class="legend">{legend_html}</div>
  <div class="legend">
    canvas position within prompt
    <select id="possel">{pos_opts}</select>
    &nbsp; <span id="poscount"></span>
  </div>
  <div id="summary"></div>
  <h4>1. histogram &mdash; fraction of canvases finishing ON each step</h4>
  <div id="hist"></div>
  <h4>2. finish-step buckets &mdash; grouped bars (early &rarr; late)</h4>
  <div id="bucket"></div>
  <h4>3. box plot &mdash; finish-step spread per variant (box=IQR, whiskers=p10/p90, dot=mean)</h4>
  <div id="box"></div>
  <h4>4. CDF &mdash; fraction finished BY step</h4>
  <div id="cdf"></div>
  <div id="cdf-readout" style="font-size:12px;color:#222;height:16px;margin:-4px 0 8px;">&nbsp;</div>
<script>
// POSITIONS[p] = [ {{name,color,ncommit,never,hist,cdf,buckets,box}} per variant ]
const POSITIONS = {positions_json};
const META = {meta_json};
let VARIANTS = POSITIONS[0];
const W = 1000, PADL = 56, PADR = 24, PADT = 16, PADB = 40;
const SVGNS = "http://www.w3.org/2000/svg";
function el(n, a) {{ const e = document.createElementNS(SVGNS, n);
  for (const k in a) e.setAttribute(k, a[k]); return e; }}
function svgIn(id, h) {{ const s = el("svg", {{width:W, height:h,
  viewBox:"0 0 "+W+" "+h}}); document.getElementById(id).appendChild(s); return s; }}

const maxStep = META.maxStep;
function xStep(s, PW) {{ return PADL + (maxStep<=1?0:(s-1)/(maxStep-1))*PW; }}

function summary() {{
  let html = "<table><tr><th>variant</th><th>n committed</th><th>never-commit</th>"
    +"<th>mean</th><th>median</th><th>p10</th><th>p90</th><th>min</th><th>max</th></tr>";
  for (const v of VARIANTS) {{ const b=v.box; const tot=v.ncommit+v.never;
    html += "<tr><td style='text-align:left'><span class='sw' style='background:"
      +v.color+"'></span>"+v.name+"</td><td>"+v.ncommit+"</td><td>"+v.never
      +" ("+(tot?(v.never/tot*100).toFixed(1):"0.0")+"%)</td>"
      +(b?("<td>"+b.mean+"</td><td>"+b.med+"</td><td>"+b.p10+"</td><td>"+b.p90
      +"</td><td>"+b.min+"</td><td>"+b.max+"</td>"):"<td colspan=6>-</td>")+"</tr>";
  }}
  document.getElementById('summary').innerHTML = html + "</table>";
}}

function axes(svg, H, PW, yMax, yfmt, xlabel, ylabel) {{
  for (let i=0;i<=5;i++) {{ const y=PADT+(i/5)*(H-PADT-PADB), v=yMax*(1-i/5);
    svg.appendChild(el("line",{{class:"grid",x1:PADL,y1:y,x2:W-PADR,y2:y}}));
    const l=el("text",{{class:"ax-lbl",x:PADL-6,y:y+3,"text-anchor":"end"}});
    l.textContent=yfmt(v); svg.appendChild(l); }}
  const sg=Math.ceil((maxStep-1)/14)||1;
  for (let s=1;s<=maxStep;s+=sg) {{ const x=xStep(s,PW);
    svg.appendChild(el("line",{{class:"grid",x1:x,y1:PADT,x2:x,y2:H-PADB}}));
    const l=el("text",{{class:"ax-lbl",x:x,y:H-PADB+14,"text-anchor":"middle"}});
    l.textContent=s; svg.appendChild(l); }}
  svg.appendChild(el("line",{{class:"axis",x1:PADL,y1:PADT,x2:PADL,y2:H-PADB}}));
  svg.appendChild(el("line",{{class:"axis",x1:PADL,y1:H-PADB,x2:W-PADR,y2:H-PADB}}));
  const xt=el("text",{{class:"ax-lbl",x:PADL+PW/2,y:H-4,"text-anchor":"middle"}});
  xt.textContent=xlabel; svg.appendChild(xt);
  const yt=el("text",{{class:"ax-lbl",x:13,y:PADT+(H-PADT-PADB)/2,"text-anchor":"middle",
    transform:"rotate(-90 13 "+(PADT+(H-PADT-PADB)/2)+")"}});
  yt.textContent=ylabel; svg.appendChild(yt);
}}

function drawHist() {{
  const H=300, PW=W-PADL-PADR; const svg=svgIn('hist',H);
  let yMax=0; for (const v of VARIANTS) for (const d of v.hist) if(d[1]>yMax)yMax=d[1];
  yMax=yMax*1.1||1;
  axes(svg,H,PW,yMax,x=>(x*100).toFixed(1)+"%","finish step","fraction");
  const yS=v=>PADT+(1-v/yMax)*(H-PADT-PADB);
  for (const v of VARIANTS) {{ if(!v.hist.length) continue;
    const d=v.hist.map((p,i)=>(i?"L":"M")+xStep(p[0],PW).toFixed(1)+" "+yS(p[1]).toFixed(1)).join(" ");
    svg.appendChild(el("path",{{d:d,fill:"none",stroke:v.color,"stroke-width":2}})); }}
}}

function drawBucket() {{
  const H=280, PW=W-PADL-PADR; const svg=svgIn('bucket',H);
  const labels=META.bucketLabels, nb=labels.length, nv=VARIANTS.length;
  let yMax=0; for (const v of VARIANTS) for (const f of v.buckets) if(f>yMax)yMax=f;
  yMax=yMax*1.1||1;
  const yS=v=>PADT+(1-v/yMax)*(H-PADT-PADB);
  for (let i=0;i<=5;i++) {{ const y=PADT+(i/5)*(H-PADT-PADB), val=yMax*(1-i/5);
    svg.appendChild(el("line",{{class:"grid",x1:PADL,y1:y,x2:W-PADR,y2:y}}));
    const l=el("text",{{class:"ax-lbl",x:PADL-6,y:y+3,"text-anchor":"end"}});
    l.textContent=(val*100).toFixed(0)+"%"; svg.appendChild(l); }}
  svg.appendChild(el("line",{{class:"axis",x1:PADL,y1:PADT,x2:PADL,y2:H-PADB}}));
  svg.appendChild(el("line",{{class:"axis",x1:PADL,y1:H-PADB,x2:W-PADR,y2:H-PADB}}));
  const gw=PW/nb, bw=gw*0.8/nv;
  for (let bi=0;bi<nb;bi++) {{
    const gx=PADL+bi*gw+gw*0.1;
    for (let vi=0;vi<nv;vi++) {{ const f=VARIANTS[vi].buckets[bi], x=gx+vi*bw;
      svg.appendChild(el("rect",{{x:x,y:yS(f),width:bw*0.92,height:(H-PADB)-yS(f),
        fill:VARIANTS[vi].color}})); }}
    const l=el("text",{{class:"ax-lbl",x:PADL+bi*gw+gw/2,y:H-PADB+14,"text-anchor":"middle"}});
    l.textContent="step "+labels[bi]; svg.appendChild(l);
  }}
  const yt=el("text",{{class:"ax-lbl",x:13,y:PADT+(H-PADT-PADB)/2,"text-anchor":"middle",
    transform:"rotate(-90 13 "+(PADT+(H-PADT-PADB)/2)+")"}});
  yt.textContent="fraction of canvases"; svg.appendChild(yt);
}}

function drawBox() {{
  const nv=VARIANTS.length, H=40+nv*46, PW=W-PADL-PADR; const svg=svgIn('box',H);
  const xS=s=>PADL+(maxStep<=1?0:(s-1)/(maxStep-1))*PW;
  const sg=Math.ceil((maxStep-1)/14)||1;
  for (let s=1;s<=maxStep;s+=sg) {{ const x=xS(s);
    svg.appendChild(el("line",{{class:"grid",x1:x,y1:PADT,x2:x,y2:H-PADB}}));
    const l=el("text",{{class:"ax-lbl",x:x,y:H-PADB+14,"text-anchor":"middle"}});
    l.textContent=s; svg.appendChild(l); }}
  svg.appendChild(el("line",{{class:"axis",x1:PADL,y1:H-PADB,x2:W-PADR,y2:H-PADB}}));
  VARIANTS.forEach((v,i)=>{{ const b=v.box; if(!b) return;
    const cy=PADT+22+i*46, h=22;
    svg.appendChild(el("line",{{x1:xS(b.p10),y1:cy,x2:xS(b.p90),y2:cy,
      stroke:v.color,"stroke-width":1}}));
    svg.appendChild(el("line",{{x1:xS(b.p10),y1:cy-5,x2:xS(b.p10),y2:cy+5,stroke:v.color}}));
    svg.appendChild(el("line",{{x1:xS(b.p90),y1:cy-5,x2:xS(b.p90),y2:cy+5,stroke:v.color}}));
    svg.appendChild(el("rect",{{x:xS(b.q1),y:cy-h/2,width:Math.max(1,xS(b.q3)-xS(b.q1)),
      height:h,fill:v.color,"fill-opacity":0.25,stroke:v.color}}));
    svg.appendChild(el("line",{{x1:xS(b.med),y1:cy-h/2,x2:xS(b.med),y2:cy+h/2,
      stroke:v.color,"stroke-width":2}}));
    svg.appendChild(el("circle",{{cx:xS(b.mean),cy:cy,r:3,fill:"#222"}}));
    const lbl=el("text",{{class:"ax-lbl",x:PADL+4,y:cy-h/2-3,fill:v.color}});
    lbl.textContent=v.name+"  med "+b.med+"  IQR "+b.q1+"-"+b.q3; svg.appendChild(lbl);
  }});
  const xt=el("text",{{class:"ax-lbl",x:PADL+PW/2,y:H-4,"text-anchor":"middle"}});
  xt.textContent="finish step"; svg.appendChild(xt);
}}

function drawCdf() {{
  const H=300, PW=W-PADL-PADR; const svg=svgIn('cdf',H);
  axes(svg,H,PW,1,x=>x.toFixed(1),"denoising step","fraction finished");
  const yS=v=>PADT+(1-v)*(H-PADT-PADB);
  for (const v of VARIANTS) {{ if(!v.cdf.length) continue;
    const d=v.cdf.map((p,i)=>(i?"L":"M")+xStep(p[0],PW).toFixed(1)+" "+yS(p[1]).toFixed(1)).join(" ");
    svg.appendChild(el("path",{{d:d,fill:"none",stroke:v.color,"stroke-width":2}})); }}
  // hover: vertical guide + per-variant readout at the nearest step.
  const guide=el("line",{{x1:0,y1:PADT,x2:0,y2:H-PADB,stroke:"#bbb",
    "stroke-dasharray":"2 2",visibility:"hidden"}});
  svg.appendChild(guide);
  const dots=[];
  const hit=el("rect",{{x:PADL,y:PADT,width:PW,height:H-PADT-PADB,fill:"transparent"}});
  svg.appendChild(hit);
  const ro=document.getElementById('cdf-readout');
  hit.addEventListener('mousemove', e => {{
    const r=svg.getBoundingClientRect();
    const sx=(e.clientX-r.left)*(W/r.width);
    let step=Math.round(1+(maxStep-1)*(sx-PADL)/PW);
    if(step<1)step=1; if(step>maxStep)step=maxStep;
    const gx=xStep(step,PW);
    guide.setAttribute('x1',gx); guide.setAttribute('x2',gx);
    guide.setAttribute('visibility','visible');
    dots.forEach(d=>d.remove()); dots.length=0;
    let parts=[];
    for (const v of VARIANTS) {{ if(!v.cdf.length) continue;
      const p=v.cdf[step-1]; if(!p) continue;
      const dot=el("circle",{{cx:gx,cy:yS(p[1]),r:3,fill:v.color}});
      svg.appendChild(dot); dots.push(dot);
      parts.push("<span style='color:"+v.color+"'>"+v.name+" "
        +(p[1]*100).toFixed(1)+"%</span>"); }}
    ro.innerHTML="step "+step+" &nbsp; "+parts.join(" &nbsp; ");
  }});
  hit.addEventListener('mouseleave', () => {{
    guide.setAttribute('visibility','hidden');
    dots.forEach(d=>d.remove()); dots.length=0;
    ro.innerHTML="&nbsp;";
  }});
}}

function redraw() {{
  for (const id of ['hist','bucket','box','cdf']) document.getElementById(id).innerHTML="";
  summary(); drawHist(); drawBucket(); drawBox(); drawCdf();
  const tot=VARIANTS.reduce((a,v)=>a+v.ncommit+v.never,0);
  document.getElementById('poscount').textContent=
    "("+tot+" canvases across "+VARIANTS.length+" variants at this position)";
}}

const sel=document.getElementById('possel');
sel.addEventListener('change', e => {{ VARIANTS=POSITIONS[parseInt(e.target.value,10)]; redraw(); }});
redraw();
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
    ap.add_argument("-t", "--threshold", type=float, default=None,
                    help="commit threshold. Default: from records.")
    ap.add_argument("--buckets", default=None,
                    help="comma bucket upper-bounds (default 10,15,20)")
    ap.add_argument("--nseed", type=int, default=None,
                    help="canvas positions per prompt (default round(count/nprompt) "
                         "per file)")
    ap.add_argument("--nprompt", type=int, default=NPROMPT,
                    help="prompts per variant (default 50)")
    ap.add_argument("--maxpos", type=int, default=None,
                    help="cap number of positions shown (default: min nseed across "
                         "inputs)")
    ap.add_argument("--labels", default=None, help="comma legend labels")
    args = ap.parse_args()

    paths, names = parse_inputs(args.inputs, args.labels)
    bounds = ([int(x) for x in args.buckets.split(",")] if args.buckets
              else DEFAULT_BUCKETS)

    # Load + group each variant by canvas position.
    loaded, conf, nseeds = [], None, []
    for i, (name, path) in enumerate(zip(names, paths)):
        canvases, c = load_by_canvas(path)
        if c is not None and conf is None:
            conf = c
        nseed = args.nseed or max(1, round(len(canvases) / args.nprompt))
        nseeds.append(nseed)
        by_pos = positions_from_order(canvases, nseed)
        loaded.append((name, PALETTE[i % len(PALETTE)], by_pos))
        print(f"parsed {name}: {len(canvases)} canvases, nseed={nseed} "
              f"-> {len(by_pos)} positions")

    thr = args.threshold if args.threshold is not None else (
        conf if conf is not None else 0.005)

    # Cap positions to the common range so every variant is comparable everywhere.
    npos = min(nseeds)
    if args.maxpos is not None:
        npos = min(npos, args.maxpos)
    print(f"positions 0..{npos-1} (capped to min nseed across inputs)")

    # First pass: global max finish step across every position+variant, so the
    # x-axis is stable when flipping positions in the dropdown.
    max_step = 0
    per_pos_first = []  # [pos][variant] = (name, color, first, never)
    for pos in range(npos):
        row = []
        for name, color, by_pos in loaded:
            trajs = by_pos.get(pos, [])
            cans = {i: t for i, t in enumerate(trajs)}
            first, never = commit_steps(cans, thr)
            if first:
                max_step = max(max_step, max(first))
            row.append((name, color, first, never))
        per_pos_first.append(row)

    # Second pass: build chart payloads per position.
    labels = buckets([], bounds)[0]
    positions = []
    for pos in range(npos):
        variants = []
        for name, color, first, never in per_pos_first[pos]:
            _, bfracs = buckets(first, bounds)
            variants.append({
                "name": name, "color": color,
                "ncommit": len(first), "never": never,
                "hist": histogram(first, max_step),
                "cdf": cdf(first, max_step),
                "buckets": bfracs,
                "box": box_stats(first),
            })
        positions.append(variants)

    meta = {"maxStep": max_step, "bucketLabels": labels, "npos": npos}
    legend_html = "\n    ".join(
        f'<span class="sw" style="background:{c}"></span>{n}'
        for n, c, _ in [(nm, co, bp) for nm, co, bp in loaded])
    pos_opts = "".join(f'<option value="{p}">position {p}</option>'
                       for p in range(npos))
    out = args.out or os.path.join(
        os.path.dirname(os.path.abspath(paths[0])),
        "finish_step_dist_per_position.html")
    with open(out, "w") as f:
        f.write(TEMPLATE.format(
            positions_json=json.dumps(positions),
            meta_json=json.dumps(meta),
            conf_disp=thr,
            legend_html=legend_html,
            pos_opts=pos_opts,
        ))
    print(f"finish-step distribution per position -> {out}")


if __name__ == "__main__":
    main()
