#!/usr/bin/env python3
"""Interactive scrubber for the decoded canvas across denoising steps.

Same paired-row view as animate_tokens.py (mxfp4 on top, bf16 below, reading
order), but emits a single self-contained HTML file you scrub by hand: a slider
plus Left/Right arrow keys (Home/End jump to first/last). No autoplay, no server
— open the file in any browser.

Uses data already captured by the probe (argmax_token + max_prob); only the
tokenizer is needed to turn ids into glyphs.

Usage:
  ./interactive_tokens.py runs/sky/bf16.jsonl runs/sky/mxfp4.jsonl
  ./interactive_tokens.py --cols 16 --max-pos 64 runs/sky/bf16.jsonl runs/sky/mxfp4.jsonl

Writes token_scrubber.html next to the bf16 input.
"""

from __future__ import annotations

import argparse
import html
import json
import os

DEFAULT_MODEL = "/app/models/diffusiongemma-26B-A4B-it"


def load(path: str, slot: int | None) -> dict[int, dict]:
    """Load a probe JSONL into {step: record} for one request slot."""
    by_step: dict[int, dict] = {}
    with open(path) as f:
        for line in f:
            line = line.strip().strip("\x00")
            if not line or not line.startswith("{"):
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if slot is not None and rec["slot"] != slot:
                continue
            by_step[rec["step"]] = rec
    return by_step


def read_entropy_bound(model_dir: str) -> float | None:
    """Pull entropy_bound from a model dir's generation_config.json, if present.

    The sampler's per-step entropy budget is a config constant (not in the probe
    dumps), so we read it straight from the model the probe ran against."""
    cfg_path = os.path.join(model_dir, "generation_config.json")
    try:
        with open(cfg_path) as f:
            cfg = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    return (cfg.get("sampler_config") or {}).get("entropy_bound")


def clean(text: str) -> str:
    """Make a token string printable inside a small grid cell."""
    text = text.replace("▁", "_")
    text = text.replace("\n", "\\n").replace("\t", "\\t")
    return "∅" if text == "" else text


def compute_stability(run):
    """For each recorded step, how many consecutive prior steps had the identical
    argmax canvas (0 = changed from previous step, or first step). This mirrors
    the sampler's stability check that gates convergence."""
    steps = sorted(run)
    stable = {}
    for idx, s in enumerate(steps):
        count = 0
        cur = run[s]["argmax_token"]
        j = idx - 1
        while j >= 0 and run[steps[j]]["argmax_token"] == cur:
            count += 1
            j -= 1
        stable[s] = count
    return stable


def build_frames(run, steps, n_pos, glyph, stability):
    """For each step, hold the last step<=current and return per-cell
    {glyph, max_prob, entropy} plus per-step stats (mean entropy, confidence
    threshold, consecutive-unchanged count)."""
    frames = []
    avail_steps = sorted(run)
    for step in steps:
        prior = [s for s in avail_steps if s <= step]
        if not prior:
            frames.append({"committed": True, "mean_e": 0.0, "conf": 0.0,
                           "stable": 0,
                           "cells": [{"t": "", "p": 0.0, "e": 0.0}] * n_pos})
            continue
        held = max(prior)
        r = run[held]
        toks = (r["argmax_token"] + [None] * n_pos)[:n_pos]
        probs = (r["max_prob"] + [0.0] * n_pos)[:n_pos]
        ents = (r.get("token_entropy", []) + [0.0] * n_pos)[:n_pos]
        cells = [{"t": glyph.get(t, "") if t is not None else "",
                  "p": round(float(p), 4), "e": round(float(e), 4)}
                 for t, p, e in zip(toks, probs, ents)]
        committed = not any(s >= step for s in avail_steps)
        frames.append({
            "committed": committed,
            "mean_e": round(float(r.get("mean_entropy", 0.0)), 5),
            "conf": round(float(r.get("confidence_threshold", 0.0)), 5),
            "stable": stability.get(held, 0),
            "cells": cells,
        })
    return frames


HTML_TEMPLATE = """<!doctype html>
<html><head><meta charset="utf-8"><title>diffgemma token scrubber</title>
<style>
  body {{ font-family: monospace; margin: 16px; background:#fafafa; color:#222; }}
  #bar {{ position: sticky; top:0; background:#fafafa; padding:8px 0;
          border-bottom:1px solid #ddd; }}
  #slider {{ width: 60%; vertical-align: middle; }}
  .pos {{ display:inline-block; width:84px; vertical-align:top; margin:0 1px 6px 0;
          padding:1px; box-sizing:border-box; outline:2px solid transparent; }}
  .pos.diff {{ outline:2px solid #e600e6; }}
  .cell {{ height:22px; line-height:22px; font-size:11px; text-align:center;
           overflow:hidden; white-space:nowrap; border:1px solid; box-sizing:border-box; }}
  .mx {{ border-color:#c44; }}
  .bf {{ border-color:#2a7; }}
  .rowlab {{ display:inline-block; width:40px; font-size:11px; }}
  .legend {{ font-size:12px; color:#555; }}
  h3 {{ margin:6px 0; }}
  #stats {{ font-size:12px; margin:4px 0; }}
  #stats b {{ font-weight:bold; }}
  .ok {{ color:#2a7; }} .no {{ color:#c44; }}
</style></head><body>
<div id="bar">
  <h3 id="title"></h3>
  step <span id="stepnum"></span> / {maxstep}
  <input id="slider" type="range" min="0" max="{nframes_1}" value="0" step="1">
  <label class="legend"><input type="checkbox" id="showent"> color by entropy (low=green)</label>
  <div id="stats"></div>
  <span class="legend">← / → arrows step · Home/End jump · top=mxfp4 (red) bottom=bf16 (green) · cell color = <span id="metriclbl">top-token prob</span> (red→green) · magenta outline = mxfp4≠bf16 · hover a cell for per-token entropy vs bound</span>
  <span class="legend"> · timelines: <a href="token_timeline_mxfp4.html">mxfp4 →</a> <a href="token_timeline_bf16.html">bf16 →</a> · <a href="meanh_chart.html">meanH chart →</a> · <a href="focus_pos21.html">pos-21 flip drill-down →</a></span>
</div>
<div id="grid"></div>
<script>
const STEPS = {steps_json};
const BF = {bf_json};
const MX = {mx_json};
const COLS = {cols};
const NPOS = {npos};
const EMAX = {emax};   // max token_entropy across all frames, for normalizing
const EBOUND = {ebound};  // sampler per-step entropy budget (config constant)

let showEnt = false;   // toggle: false=max_prob, true=token_entropy

// red -> yellow -> green by a confidence score in [0,1] (1=green=confident)
function colr(score) {{
  const r = score < 0.5 ? 255 : Math.round(255*(1-(score-0.5)*2));
  const g = score < 0.5 ? Math.round(255*(score*2)) : 200;
  return `rgb(${{r}},${{g}},90)`;
}}

// Map a cell to a [0,1] confidence score and a tooltip, per active metric.
// Entropy is inverted+normalized so low entropy (confident) reads green.
// Tooltip always carries both prob and entropy, plus the entropy bound.
function cellScore(c) {{
  const bound = EBOUND === null ? "?" : EBOUND;
  const tip = "p=" + c.p + " · H=" + c.e + " · bound=" + bound;
  if (showEnt) {{
    const s = EMAX > 0 ? 1 - (c.e / EMAX) : 1;
    return {{score: s, tip: tip}};
  }}
  return {{score: c.p, tip: tip}};
}}

const grid = document.getElementById('grid');
// Build static DOM once: one .pos per canvas position (wrapping via flow),
// each holding an mxfp4 cell over a bf16 cell. Row labels at row starts.
let cellsMX = [], cellsBF = [], posEls = [];
for (let p=0; p<NPOS; p++) {{
  if (p % COLS === 0) {{
    const lab = document.createElement('div');
    lab.className='rowlab';
    lab.innerHTML = '<div class="cell" style="border:none">mxfp4</div>'
                  + '<div class="cell" style="border:none">bf16</div>';
    grid.appendChild(lab);
  }}
  const d = document.createElement('div'); d.className='pos';
  const m = document.createElement('div'); m.className='cell mx';
  const b = document.createElement('div'); b.className='cell bf';
  d.appendChild(m); d.appendChild(b); grid.appendChild(d);
  cellsMX.push(m); cellsBF.push(b); posEls.push(d);
  if (p % COLS === COLS-1) grid.appendChild(document.createElement('br'));
}}

function paint(cells, frame) {{
  for (let i=0; i<NPOS; i++) {{
    const c = frame.cells[i];
    cells[i].textContent = c.t;
    if (c.t === "") {{ cells[i].style.background = "#eee"; cells[i].title = ""; }}
    else {{ const s = cellScore(c);
            cells[i].style.background = colr(s.score); cells[i].title = s.tip; }}
  }}
}}

const slider = document.getElementById('slider');
function render(i) {{
  i = Math.max(0, Math.min(STEPS.length-1, i));
  slider.value = i;
  document.getElementById('stepnum').textContent = STEPS[i];
  paint(cellsMX, MX[i]); paint(cellsBF, BF[i]);
  // Outline positions where both paths committed a token and they disagree.
  let ndiff = 0;
  for (let p=0; p<NPOS; p++) {{
    const mt = MX[i].cells[p].t, bt = BF[i].cells[p].t;
    const differ = mt !== "" && bt !== "" && mt !== bt;
    posEls[p].classList.toggle('diff', differ);
    if (differ) ndiff++;
  }}
  const mxd = MX[i].committed ? "mxfp4 (committed)" : "mxfp4";
  const bfd = BF[i].committed ? "bf16 (committed)" : "bf16";
  document.getElementById('title').textContent =
     "denoising step " + STEPS[i] + "   [" + mxd + " top / " + bfd + " bottom]"
     + "   " + ndiff + " positions differ";
  document.getElementById('stats').innerHTML =
     statLine("mxfp4", MX[i]) + " &nbsp;&nbsp; " + statLine("bf16", BF[i])
     + " &nbsp;&nbsp; entropy bound/token: <b>" + (EBOUND === null ? "?" : EBOUND) + "</b>";
}}

// Per-step summary for one run: mean entropy vs the commit threshold (with a
// confident/not flag), and how many consecutive prior steps were unchanged.
function statLine(label, f) {{
  const conf = f.mean_e < f.conf;
  const cls = conf ? "ok" : "no";
  return label + ": meanH=<b>" + f.mean_e + "</b> "
    + "<span class='" + cls + "'>" + (conf ? "&lt;" : "&ge;")
    + " thr=" + f.conf + (conf ? " ✓confident" : "") + "</span> "
    + "· unchanged <b>" + f.stable + "</b> step(s)";
}}

slider.addEventListener('input', e => render(+e.target.value));
document.getElementById('showent').addEventListener('change', e => {{
  showEnt = e.target.checked;
  document.getElementById('metriclbl').textContent =
    showEnt ? "token entropy" : "top-token prob";
  render(+slider.value);
}});
window.addEventListener('keydown', e => {{
  if (e.key === 'ArrowRight') {{ render(+slider.value+1); e.preventDefault(); }}
  else if (e.key === 'ArrowLeft') {{ render(+slider.value-1); e.preventDefault(); }}
  else if (e.key === 'Home') {{ render(0); e.preventDefault(); }}
  else if (e.key === 'End') {{ render(STEPS.length-1); e.preventDefault(); }}
}});
render(0);
</script></body></html>
"""


# Timeline page: one run's decode as a flowing one-line sentence, stacked one row
# per denoising step (earliest at top, final at bottom). Each token is a span
# tinted by its top-token probability; tokens whose top id differs from the OTHER
# run at that step get a magenta underline so divergence is legible down the
# stack. Generated once per run (mxfp4 and bf16), each linking to the other.
TIMELINE_TEMPLATE = """<!doctype html>
<html><head><meta charset="utf-8"><title>diffgemma {self_label} timeline</title>
<style>
  body {{ font-family: monospace; margin: 16px; background:#fafafa; color:#222; }}
  #bar {{ position: sticky; top:0; background:#fafafa; padding:8px 0;
          border-bottom:1px solid #ddd; }}
  h3 {{ margin:6px 0; }}
  .legend {{ font-size:12px; color:#555; }}
  .row {{ white-space:nowrap; font-size:13px; line-height:20px;
          border-bottom:1px solid #eee; padding:1px 0; }}
  .row:hover {{ background:#f0f0f0; }}
  .stp {{ display:inline-block; width:46px; color:#888; font-size:11px;
          text-align:right; margin-right:8px; }}
  .committed .stp {{ color:#2a7; font-weight:bold; }}
  .rowstat {{ display:inline-block; width:320px; margin-right:10px; font-size:11px;
              color:#777; }}
  .rowstat .ok {{ color:#2a7; }} .rowstat .no {{ color:#c44; }}
  .tok {{ padding:0 0.5px; }}
  .tok.diff {{ border-bottom:2px solid #e600e6; }}
  body.nocolor .tok {{ background:transparent !important; }}
  body.nomagenta .tok.diff {{ border-bottom:none; }}
  /* "not yet final": current top token differs from this position's token at the
     last step. Only shown when the toggle is on; orange dashed top edge so it
     reads distinctly from the magenta cross-run divergence underline. */
  body.showfinal .tok.nonfinal {{ border-top:2px dashed #f80; }}
</style></head><body>
<div id="bar">
  <h3>{self_label} decode — one row per denoising step (top = earliest, bottom = final)</h3>
  <label class="legend"><input type="checkbox" id="nocolor"> hide confidence coloring</label>
  <label class="legend"><input type="checkbox" id="showent"> color by entropy (low=green)</label>
  <label class="legend"><input type="checkbox" id="nomagenta"> hide magenta (cross-run diff)</label>
  <label class="legend"><input type="checkbox" id="showfinal"> mark not-yet-final tokens</label>
  <span class="legend">token color = <span id="metriclbl">top-token prob</span> (red→green) · magenta underline = {self_label}≠{other_label} at that step · orange dashed top = differs from this position's final token · committed step in green · per-row stats at left (meanH vs commit threshold · unchanged-step count) · entropy bound/token = <b>{ebound_disp}</b> · hover a token for its entropy vs bound · <a href="{other_page}">{other_label} timeline →</a> · <a href="token_scrubber.html">scrubber</a></span>
</div>
<div id="stack"></div>
<script>
const STEPS = {steps_json};
const SELF = {self_json};
const OTHER = {other_json};
const OTHER_LABEL = {other_label_json};
const NPOS = {npos};
const EMAX = {emax};   // max token_entropy across all frames, for normalizing
const EBOUND = {ebound};  // sampler per-step entropy budget (config constant)

let showEnt = false;   // toggle: false=max_prob, true=token_entropy

function colr(score) {{
  const r = score < 0.5 ? 255 : Math.round(255*(1-(score-0.5)*2));
  const g = score < 0.5 ? Math.round(255*(score*2)) : 200;
  return `rgb(${{r}},${{g}},90)`;
}}

// Map a cell to a [0,1] confidence score and a tooltip, per active metric.
// Entropy is inverted+normalized so low entropy (confident) reads green.
// Tooltip always carries both prob and entropy, plus the entropy bound.
function cellScore(c) {{
  const bound = EBOUND === null ? "?" : EBOUND;
  const tip = "p=" + c.p + " · H=" + c.e + " · bound=" + bound;
  if (showEnt) {{
    const s = EMAX > 0 ? 1 - (c.e / EMAX) : 1;
    return {{score: s, tip: tip}};
  }}
  return {{score: c.p, tip: tip}};
}}

// Each position's final token = its value at the last step (the committed state).
const FINAL = SELF[STEPS.length-1].cells.map(c => c.t);

const stack = document.getElementById('stack');
// Keep each rendered span with its source cell + position so toggling the
// active metric can repaint background/tooltip without rebuilding the DOM.
const painted = [];
for (let i=0; i<STEPS.length; i++) {{
  const row = document.createElement('div');
  row.className = 'row' + (SELF[i].committed ? ' committed' : '');
  const lab = document.createElement('span');
  lab.className = 'stp'; lab.textContent = STEPS[i];
  row.appendChild(lab);
  // Per-step stats at the left, before the tokens: mean entropy vs commit
  // threshold (confident flag) and consecutive-unchanged-step count.
  const f = SELF[i];
  const conf = f.mean_e < f.conf;
  const stat = document.createElement('span');
  stat.className = 'rowstat';
  stat.innerHTML = "meanH=" + f.mean_e + " <span class='" + (conf ? "ok" : "no")
    + "'>" + (conf ? "&lt;" : "&ge;") + " thr=" + f.conf
    + (conf ? " ✓" : "") + "</span> · unchanged " + f.stable;
  row.appendChild(stat);
  for (let p=0; p<NPOS; p++) {{
    const c = SELF[i].cells[p];
    if (c.t === "") continue;          // position not yet decoded this step
    const s = document.createElement('span');
    s.className = 'tok';
    const ot = OTHER[i].cells[p].t;
    if (ot !== "" && ot !== c.t) s.classList.add('diff');
    // Differs from where this position ends up at the final step.
    if (FINAL[p] !== "" && FINAL[p] !== c.t) s.classList.add('nonfinal');
    s.textContent = c.t;
    row.appendChild(s);
    painted.push({{el: s, c: c, p: p, ot: ot}});
  }}
  stack.appendChild(row);
}}

function repaint() {{
  for (const {{el, c, p, ot}} of painted) {{
    const sc = cellScore(c);
    el.style.background = colr(sc.score);
    el.title = "pos " + p + " · " + sc.tip
      + (el.classList.contains('diff') ? " · " + OTHER_LABEL + "=" + ot : "")
      + (el.classList.contains('nonfinal') ? " · final=" + FINAL[p] : "");
  }}
}}
repaint();

function bindToggle(id, cls) {{
  document.getElementById(id).addEventListener('change', e => {{
    document.body.classList.toggle(cls, e.target.checked);
  }});
}}
bindToggle('nocolor', 'nocolor');
bindToggle('nomagenta', 'nomagenta');
bindToggle('showfinal', 'showfinal');
document.getElementById('showent').addEventListener('change', e => {{
  showEnt = e.target.checked;
  document.getElementById('metriclbl').textContent =
    showEnt ? "token entropy" : "top-token prob";
  repaint();
}});
</script></body></html>
"""


def write_timeline(path, self_label, other_label, other_page, steps,
                   self_frames, other_frames, n_pos, emax, entropy_bound):
    with open(path, "w") as f:
        f.write(TIMELINE_TEMPLATE.format(
            self_label=self_label,
            other_label=other_label,
            other_label_json=json.dumps(other_label),
            other_page=other_page,
            steps_json=json.dumps(steps),
            self_json=json.dumps(self_frames),
            other_json=json.dumps(other_frames),
            npos=n_pos,
            emax=emax,
            ebound=json.dumps(entropy_bound),
            ebound_disp=("?" if entropy_bound is None else entropy_bound),
        ))


# Standalone line chart of per-step mean entropy, bf16 vs mxfp4. Self-contained:
# an inline SVG drawn by JS from the two series, with the commit threshold as a
# horizontal reference line and a hover readout. No external libraries.
MEANH_TEMPLATE = """<!doctype html>
<html><head><meta charset="utf-8"><title>diffgemma meanH over steps</title>
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
  <h3>mean token entropy per denoising step</h3>
  <div class="legend">
    <span class="sw" style="background:#c44"></span>mxfp4
    <span class="sw" style="background:#2a7"></span>bf16
    <span class="sw" style="background:#e6a000"></span>commit threshold ({conf_disp})
    <label class="legend"><input type="checkbox" id="logy"> log y-axis</label>
    · <a href="token_scrubber.html">scrubber</a>
  </div>
  <div id="chart"></div>
  <div id="readout">&nbsp;</div>
<script>
const BF = {bf_series};   // [[step, meanH], ...]
const MX = {mx_series};
const CONF = {conf};      // commit threshold (mean_entropy < CONF => confident)
const W = 900, H = 460, PADL = 64, PADR = 20, PADT = 20, PADB = 44;
const PW = W - PADL - PADR, PH = H - PADT - PADB;

let logY = false;
const allSteps = BF.concat(MX).map(d => d[0]);
const allVals = BF.concat(MX).map(d => d[1]);
const xMin = Math.min(...allSteps), xMax = Math.max(...allSteps);
const yMaxRaw = Math.max(...allVals, CONF);
// Smallest positive value (for log-scale floor); fall back to a tiny epsilon.
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

function path(series) {{
  return series.map((d, i) =>
    (i ? "L" : "M") + xScale(d[0]).toFixed(1) + " " + yScale(d[1]).toFixed(1)
  ).join(" ");
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

let svg, dotsLayer;
function draw() {{
  const chart = document.getElementById('chart');
  chart.innerHTML = "";
  svg = el("svg", {{width: W, height: H, viewBox: "0 0 " + W + " " + H}});

  // y grid + labels
  for (const t of ticksY()) {{
    const y = yScale(t);
    svg.appendChild(el("line", {{class:"grid", x1:PADL, y1:y, x2:W-PADR, y2:y}}));
    const lbl = el("text", {{class:"ax-lbl", x:PADL-6, y:y+3, "text-anchor":"end"}});
    lbl.textContent = t >= 0.01 || t === 0 ? t.toFixed(3) : t.toExponential(0);
    svg.appendChild(lbl);
  }}
  // x grid + labels
  for (const t of ticksX()) {{
    const x = xScale(t);
    svg.appendChild(el("line", {{class:"grid", x1:x, y1:PADT, x2:x, y2:H-PADB}}));
    const lbl = el("text", {{class:"ax-lbl", x:x, y:H-PADB+16, "text-anchor":"middle"}});
    lbl.textContent = t;
    svg.appendChild(lbl);
  }}
  // axes
  svg.appendChild(el("line", {{class:"axis", x1:PADL, y1:PADT, x2:PADL, y2:H-PADB}}));
  svg.appendChild(el("line", {{class:"axis", x1:PADL, y1:H-PADB, x2:W-PADR, y2:H-PADB}}));
  // axis titles
  const xt = el("text", {{class:"ax-lbl", x:PADL+PW/2, y:H-6, "text-anchor":"middle"}});
  xt.textContent = "denoising step"; svg.appendChild(xt);
  const yt = el("text", {{class:"ax-lbl", x:14, y:PADT+PH/2,
    "text-anchor":"middle", transform:"rotate(-90 14 " + (PADT+PH/2) + ")"}});
  yt.textContent = "mean token entropy"; svg.appendChild(yt);

  // commit threshold reference line (skipped on log axis if <= 0)
  if (!(logY && CONF <= 0)) {{
    const yT = yScale(CONF);
    svg.appendChild(el("line", {{class:"thr", x1:PADL, y1:yT, x2:W-PADR, y2:yT}}));
  }}

  // series lines
  svg.appendChild(el("path", {{d:path(BF), fill:"none", stroke:"#2a7", "stroke-width":2}}));
  svg.appendChild(el("path", {{d:path(MX), fill:"none", stroke:"#c44", "stroke-width":2}}));

  // dot markers
  for (const [series, color] of [[BF, "#2a7"], [MX, "#c44"]]) {{
    for (const d of series) {{
      svg.appendChild(el("circle", {{cx:xScale(d[0]), cy:yScale(d[1]), r:2.5,
        fill:color}}));
    }}
  }}

  // hover guide + nearest-point readout
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
    const b = nearest(BF, sx), m = nearest(MX, sx);
    ro.innerHTML =
      "step " + (m ? m[0] : "-") +
      " &nbsp; <span style='color:#c44'>mxfp4 meanH=" + (m ? m[1] : "-") + "</span>" +
      " &nbsp; <span style='color:#2a7'>bf16 meanH=" + (b ? b[1] : "-") + "</span>";
  }});
  hit.addEventListener('mouseleave', () => {{
    guide.setAttribute('visibility', 'hidden'); ro.innerHTML = "&nbsp;";
  }});

  chart.appendChild(svg);
}}

document.getElementById('logy').addEventListener('change', e => {{
  logY = e.target.checked; draw();
}});
draw();
</script></body></html>
"""


def write_meanh_chart(path, bf, mx, conf):
    bf_series = [[s, round(float(bf[s].get("mean_entropy", 0.0)), 5)]
                 for s in sorted(bf)]
    mx_series = [[s, round(float(mx[s].get("mean_entropy", 0.0)), 5)]
                 for s in sorted(mx)]
    with open(path, "w") as f:
        f.write(MEANH_TEMPLATE.format(
            bf_series=json.dumps(bf_series),
            mx_series=json.dumps(mx_series),
            conf=json.dumps(conf),
            conf_disp=conf,
        ))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("bf16", help="bf16 probe JSONL")
    ap.add_argument("mxfp4", help="mxfp4 probe JSONL")
    ap.add_argument("--slot", type=int, default=None,
                    help="restrict to one request slot")
    ap.add_argument("--cols", type=int, default=16,
                    help="positions per row; default 16")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help=f"tokenizer dir/name (default {DEFAULT_MODEL})")
    ap.add_argument("--max-pos", type=int, default=None,
                    help="cap positions shown (default all n_valid)")
    ap.add_argument("--entropy-bound", type=float, default=None,
                    help="per-step entropy budget; default read from the model's "
                         "generation_config.json")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    entropy_bound = args.entropy_bound
    if entropy_bound is None:
        entropy_bound = read_entropy_bound(args.model)

    bf = load(args.bf16, args.slot)
    mx = load(args.mxfp4, args.slot)
    steps = sorted(set(bf) | set(mx))
    if not steps:
        print("no steps to render")
        return

    n_valid = max(
        max((r["n_valid"] for r in bf.values()), default=0),
        max((r["n_valid"] for r in mx.values()), default=0),
    )
    n_pos = min(n_valid, args.max_pos) if args.max_pos else n_valid

    ids = set()
    for run in (bf, mx):
        for r in run.values():
            ids.update(r["argmax_token"][:n_pos])
    glyph = {i: html.escape(clean(tok.decode([i]))) for i in ids}

    bf_stable = compute_stability(bf)
    mx_stable = compute_stability(mx)
    bf_frames = build_frames(bf, steps, n_pos, glyph, bf_stable)
    mx_frames = build_frames(mx, steps, n_pos, glyph, mx_stable)

    # Max entropy across both runs / all steps, to normalize the entropy coloring.
    emax = max(
        (c["e"] for frames in (bf_frames, mx_frames)
         for fr in frames for c in fr["cells"]),
        default=0.0,
    )

    out = os.path.join(os.path.dirname(os.path.abspath(args.bf16)),
                       "token_scrubber.html")
    with open(out, "w") as f:
        f.write(HTML_TEMPLATE.format(
            maxstep=steps[-1],
            nframes_1=len(steps) - 1,
            steps_json=json.dumps(steps),
            bf_json=json.dumps(bf_frames),
            mx_json=json.dumps(mx_frames),
            cols=args.cols,
            npos=n_pos,
            emax=emax,
            ebound=json.dumps(entropy_bound),
        ))
    print(f"scrubber -> {out} ({len(steps)} steps, {n_pos} positions)")

    outdir = os.path.dirname(os.path.abspath(args.bf16))
    tl_mx = os.path.join(outdir, "token_timeline_mxfp4.html")
    tl_bf = os.path.join(outdir, "token_timeline_bf16.html")
    write_timeline(tl_mx, "mxfp4", "bf16", "token_timeline_bf16.html",
                   steps, mx_frames, bf_frames, n_pos, emax, entropy_bound)
    write_timeline(tl_bf, "bf16", "mxfp4", "token_timeline_mxfp4.html",
                   steps, bf_frames, mx_frames, n_pos, emax, entropy_bound)
    print(f"timeline -> {tl_mx} (mxfp4 sentence, one row per step)")
    print(f"timeline -> {tl_bf} (bf16 sentence, one row per step)")

    # Commit threshold for the reference line: read from any record (it is a
    # per-step constant), falling back to 0.0 if absent.
    conf = next((r.get("confidence_threshold", 0.0)
                 for run in (bf, mx) for r in run.values()), 0.0)
    chart = os.path.join(outdir, "meanh_chart.html")
    write_meanh_chart(chart, bf, mx, conf)
    print(f"meanH chart -> {chart} (bf16 vs mxfp4 mean entropy per step)")

    print("open the scrubber in a browser; use the slider or ←/→ arrow keys, "
          "or follow the timeline links")


if __name__ == "__main__":
    main()
