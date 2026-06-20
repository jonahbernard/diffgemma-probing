#!/usr/bin/env python3
"""Finish-step distribution per variant: where do canvases commit, and how do the
four variants differ in the early vs late tail?

"Finish step" of a canvas = the first denoising step at which its mean_entropy
crosses (<=) the commit threshold. Canvases that never cross are counted
separately (never-commit) and excluded from the distributions.

Four stacked views in one HTML, all from a single parse:
  1. Histogram overlay  -- fraction of canvases finishing on each step, one curve
                           per variant. The direct answer: is there an early bump
                           in bf16 that moe lacks, or just a stretched late tail?
  2. Bucketed bars      -- finish steps grouped into early/mid/late bands, grouped
                           bars per variant. Headline "X% finish early" numbers.
  3. Box plot           -- median/quartile/whisker of finish step per variant.
                           Distinguishes uniform shift (box moves) from tail-only
                           (whisker stretches).
  4. CDF overlay        -- fraction finished BY step S. Where curves separate tells
                           you which regime (early vs late) drives the difference.

Usage:
  ./finish_step_dist.py \
    "bf16 t0=runs/bf16.jsonl" "bf16 t0.6=runs/bf16_gpu7.jsonl" \
    "moe t0=runs/moe-fp4-mlp-bf16.jsonl" "moe t0.6=runs/moe-fp4-mlp-bf16_gpu7.jsonl" \
    --limit 10000 -o runs/meanh50/finish_step_dist.html
"""

from __future__ import annotations

import argparse
import json
import os

from meanh_overlay import PALETTE, _default_label, load_by_canvas
from median_quartile_overlay import quantile
from commit_diagnostics import commit_steps

DEFAULT_BUCKETS = [10, 15, 20]  # boundaries -> bands <=10, 11-15, 16-20, 21+


def histogram(first, max_step):
    """[[step, fraction], ...] over 1..max_step. Fraction of committed canvases."""
    n = len(first)
    if n == 0:
        return []
    counts = {}
    for s in first:
        counts[s] = counts.get(s, 0) + 1
    return [[s, round(counts.get(s, 0) / n, 6)] for s in range(1, max_step + 1)]


def cdf(first, max_step):
    """[[step, fraction finished by step], ...] over committed canvases."""
    n = len(first)
    if n == 0:
        return []
    counts = {}
    for s in first:
        counts[s] = counts.get(s, 0) + 1
    out, acc = [], 0
    for s in range(1, max_step + 1):
        acc += counts.get(s, 0)
        out.append([s, round(acc / n, 6)])
    return out


def buckets(first, bounds):
    """Fraction of committed canvases in each band defined by bounds.

    bounds=[10,15,20] -> bands: <=10, 11-15, 16-20, 21+. Returns (labels, fracs).
    """
    n = len(first)
    labels = []
    lo = 1
    edges = bounds + [None]
    for hi in edges:
        if hi is None:
            labels.append(f"{lo}+")
        elif lo == hi:
            labels.append(f"{lo}")
        else:
            labels.append(f"{lo}-{hi}")
        lo = (hi + 1) if hi is not None else lo
    if n == 0:
        return labels, [0.0] * len(labels)
    fracs = [0] * len(labels)
    for s in first:
        idx = len(bounds)
        for i, b in enumerate(bounds):
            if s <= b:
                idx = i
                break
        fracs[idx] += 1
    return labels, [round(f / n, 6) for f in fracs]


def box_stats(first):
    """Five-number summary + p10/p90 for whiskers, of finish steps."""
    if not first:
        return None
    v = sorted(first)
    return {
        "min": v[0], "max": v[-1],
        "p10": round(quantile(v, 0.10), 3),
        "q1": round(quantile(v, 0.25), 3),
        "med": round(quantile(v, 0.50), 3),
        "q3": round(quantile(v, 0.75), 3),
        "p90": round(quantile(v, 0.90), 3),
        "mean": round(sum(v) / len(v), 3),
    }


TEMPLATE = """<!doctype html>
<html><head><meta charset="utf-8"><title>finish-step distribution</title>
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
</style></head><body>
  <h3>finish-step distribution &mdash; {ncanvas} canvases/variant (commit = meanH &le; {conf_disp})</h3>
  <div class="legend">{legend_html}</div>
  <div id="summary"></div>
  <h4>1. histogram &mdash; fraction of canvases finishing ON each step</h4>
  <div id="hist"></div>
  <h4>2. finish-step buckets &mdash; grouped bars (early &rarr; late)</h4>
  <div id="bucket"></div>
  <h4>3. box plot &mdash; finish-step spread per variant (box=IQR, whiskers=p10/p90, dot=mean)</h4>
  <div id="box"></div>
  <h4>4. CDF &mdash; fraction finished BY step</h4>
  <div id="cdf"></div>
<script>
const VARIANTS = {variants_json};
const META = {meta_json};
const W = 1000, PADL = 56, PADR = 24, PADT = 16, PADB = 40;
const SVGNS = "http://www.w3.org/2000/svg";
function el(n, a) {{ const e = document.createElementNS(SVGNS, n);
  for (const k in a) e.setAttribute(k, a[k]); return e; }}
function svgIn(id, h) {{ const s = el("svg", {{width:W, height:h,
  viewBox:"0 0 "+W+" "+h}}); document.getElementById(id).appendChild(s); return s; }}

const maxStep = META.maxStep;
function xStep(s, PW) {{ return PADL + (maxStep<=1?0:(s-1)/(maxStep-1))*PW; }}

// ---- summary table ----
(function() {{
  let html = "<table><tr><th>variant</th><th>n committed</th><th>never-commit</th>"
    +"<th>mean</th><th>median</th><th>p10</th><th>p90</th><th>min</th><th>max</th></tr>";
  for (const v of VARIANTS) {{ const b=v.box;
    html += "<tr><td style='text-align:left'><span class='sw' style='background:"
      +v.color+"'></span>"+v.name+"</td><td>"+v.ncommit+"</td><td>"+v.never
      +" ("+(v.never/(v.ncommit+v.never)*100).toFixed(1)+"%)</td>"
      +(b?("<td>"+b.mean+"</td><td>"+b.med+"</td><td>"+b.p10+"</td><td>"+b.p90
      +"</td><td>"+b.min+"</td><td>"+b.max+"</td>"):"<td colspan=6>-</td>")+"</tr>";
  }}
  document.getElementById('summary').innerHTML = html + "</table>";
}})();

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

// ---- 1. histogram overlay ----
(function() {{
  const H=300, PW=W-PADL-PADR; const svg=svgIn('hist',H);
  let yMax=0; for (const v of VARIANTS) for (const d of v.hist) if(d[1]>yMax)yMax=d[1];
  yMax=yMax*1.1||1;
  axes(svg,H,PW,yMax,x=>(x*100).toFixed(1)+"%","finish step","fraction");
  const yS=v=>PADT+(1-v/yMax)*(H-PADT-PADB);
  for (const v of VARIANTS) {{ if(!v.hist.length) continue;
    const d=v.hist.map((p,i)=>(i?"L":"M")+xStep(p[0],PW).toFixed(1)+" "+yS(p[1]).toFixed(1)).join(" ");
    svg.appendChild(el("path",{{d:d,fill:"none",stroke:v.color,"stroke-width":2}})); }}
}})();

// ---- 2. bucketed grouped bars ----
(function() {{
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
}})();

// ---- 3. box plot (horizontal, one row per variant) ----
(function() {{
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
    // whisker p10..p90
    svg.appendChild(el("line",{{x1:xS(b.p10),y1:cy,x2:xS(b.p90),y2:cy,
      stroke:v.color,"stroke-width":1}}));
    svg.appendChild(el("line",{{x1:xS(b.p10),y1:cy-5,x2:xS(b.p10),y2:cy+5,stroke:v.color}}));
    svg.appendChild(el("line",{{x1:xS(b.p90),y1:cy-5,x2:xS(b.p90),y2:cy+5,stroke:v.color}}));
    // IQR box
    svg.appendChild(el("rect",{{x:xS(b.q1),y:cy-h/2,width:Math.max(1,xS(b.q3)-xS(b.q1)),
      height:h,fill:v.color,"fill-opacity":0.25,stroke:v.color}}));
    // median
    svg.appendChild(el("line",{{x1:xS(b.med),y1:cy-h/2,x2:xS(b.med),y2:cy+h/2,
      stroke:v.color,"stroke-width":2}}));
    // mean dot
    svg.appendChild(el("circle",{{cx:xS(b.mean),cy:cy,r:3,fill:"#222"}}));
    const lbl=el("text",{{class:"ax-lbl",x:PADL+4,y:cy-h/2-3,fill:v.color}});
    lbl.textContent=v.name+"  med "+b.med+"  IQR "+b.q1+"-"+b.q3; svg.appendChild(lbl);
  }});
  const xt=el("text",{{class:"ax-lbl",x:PADL+PW/2,y:H-4,"text-anchor":"middle"}});
  xt.textContent="finish step"; svg.appendChild(xt);
}})();

// ---- 4. CDF overlay ----
(function() {{
  const H=300, PW=W-PADL-PADR; const svg=svgIn('cdf',H);
  axes(svg,H,PW,1,x=>x.toFixed(1),"denoising step","fraction finished");
  const yS=v=>PADT+(1-v)*(H-PADT-PADB);
  for (const v of VARIANTS) {{ if(!v.cdf.length) continue;
    const d=v.cdf.map((p,i)=>(i?"L":"M")+xStep(p[0],PW).toFixed(1)+" "+yS(p[1]).toFixed(1)).join(" ");
    svg.appendChild(el("path",{{d:d,fill:"none",stroke:v.color,"stroke-width":2}})); }}
}})();
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
    ap.add_argument("-n", "--limit", type=int, default=None,
                    help="first N canvases (request order) per file. Default all.")
    ap.add_argument("-t", "--threshold", type=float, default=None,
                    help="commit threshold. Default: from records.")
    ap.add_argument("--buckets", default=None,
                    help="comma bucket upper-bounds (default 10,15,20)")
    ap.add_argument("--labels", default=None, help="comma legend labels")
    args = ap.parse_args()

    paths, names = parse_inputs(args.inputs, args.labels)
    bounds = ([int(x) for x in args.buckets.split(",")] if args.buckets
              else DEFAULT_BUCKETS)

    loaded, conf = [], None
    for i, (name, path) in enumerate(zip(names, paths)):
        canvases, c = load_by_canvas(path)
        if c is not None and conf is None:
            conf = c
        if args.limit:
            canvases = dict(list(canvases.items())[:args.limit])
        loaded.append((name, PALETTE[i % len(PALETTE)], canvases))
        print(f"parsed {name}: {len(canvases)} canvases")

    thr = args.threshold if args.threshold is not None else (
        conf if conf is not None else 0.005)

    # First pass: commit steps per variant + global max finish step.
    firsts = []
    max_step = 0
    for name, color, canvases in loaded:
        first, never = commit_steps(canvases, thr)
        if first:
            max_step = max(max_step, max(first))
        firsts.append((name, color, first, never))

    variants = []
    for name, color, first, never in firsts:
        labels, bfracs = buckets(first, bounds)
        variants.append({
            "name": name, "color": color,
            "ncommit": len(first), "never": never,
            "hist": histogram(first, max_step),
            "cdf": cdf(first, max_step),
            "buckets": bfracs,
            "box": box_stats(first),
        })
        b = box_stats(first)
        print(f"  {name}: {len(first)} committed, {never} never "
              f"({never/(len(first)+never)*100:.1f}%), "
              f"median finish {b['med'] if b else '-'}, "
              f"buckets {dict(zip(labels, bfracs))}")

    meta = {"maxStep": max_step, "bucketLabels": labels}
    ncanvas = max((v["ncommit"] + v["never"] for v in variants), default=0)
    legend_html = "\n    ".join(
        f'<span class="sw" style="background:{v["color"]}"></span>{v["name"]}'
        for v in variants)
    out = args.out or os.path.join(
        os.path.dirname(os.path.abspath(paths[0])), "finish_step_dist.html")
    with open(out, "w") as f:
        f.write(TEMPLATE.format(
            variants_json=json.dumps(variants),
            meta_json=json.dumps(meta),
            conf_disp=thr,
            ncanvas=ncanvas,
            legend_html=legend_html,
        ))
    print(f"finish-step distribution -> {out}")


if __name__ == "__main__":
    main()
