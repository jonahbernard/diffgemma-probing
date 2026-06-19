#!/usr/bin/env python3
"""Animate the decoded canvas: one box per position showing the most-confident
token at each denoising step, bf16 vs mxfp4 side by side.

Uses data already captured by the probe — no hook changes needed:
  - argmax_token : most-confident token id per position per step
  - max_prob     : that token's probability (used to color each box)
The tokenizer turns ids into readable glyphs for display.

Canvas positions are laid out in reading order, wrapping into rows. Each
position is a stacked PAIR of cells: mxfp4 on top, bf16 directly below it, so
the two paths line up position-for-position. Cell text = decoded token; cell
color = confidence (red = low max_prob, green = high). Watching the rows fill in
shows where mxfp4 commits to different / less-confident tokens than bf16, step
for step.

Positions where the two paths disagree on the top token (and both have already
committed one) are ringed with a thick magenta box spanning the pair, and the
title carries a running count of how many positions currently differ — so the
eye is drawn straight to the divergences at each timestep.

Usage:
  ./animate_tokens.py runs/sky/bf16.jsonl runs/sky/mxfp4.jsonl
  ./animate_tokens.py --cols 16 --fps 2 runs/sky/bf16.jsonl runs/sky/mxfp4.jsonl
  ./animate_tokens.py --model /app/models/diffusiongemma-26B-A4B-it ...

Writes token_anim.gif next to the bf16 input.
"""

from __future__ import annotations

import argparse
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


def clean(text: str) -> str:
    """Make a token string printable inside a small grid cell."""
    # Gemma/SentencePiece uses U+2581 ("▁") for a leading space; show it as a
    # visible underscore so word boundaries are legible.
    text = text.replace("▁", "_")
    text = text.replace("\n", "\\n").replace("\t", "\\t")
    if text == "":
        return "∅"
    return text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("bf16", help="bf16 probe JSONL")
    ap.add_argument("mxfp4", help="mxfp4 probe JSONL")
    ap.add_argument("--slot", type=int, default=None,
                    help="restrict to one request slot")
    ap.add_argument("--cols", type=int, default=16,
                    help="grid columns (positions per row); default 16")
    ap.add_argument("--fps", type=int, default=2, help="animation frames/sec")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help=f"tokenizer dir/name (default {DEFAULT_MODEL})")
    ap.add_argument("--max-pos", type=int, default=None,
                    help="cap positions shown (e.g. 64 to focus on the answer "
                    "span); default shows all n_valid")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    bf = load(args.bf16, args.slot)
    mx = load(args.mxfp4, args.slot)
    steps = sorted(set(bf) | set(mx))
    if not steps:
        print("no steps to animate")
        return

    n_valid = max(
        max((r["n_valid"] for r in bf.values()), default=0),
        max((r["n_valid"] for r in mx.values()), default=0),
    )
    n_pos = min(n_valid, args.max_pos) if args.max_pos else n_valid
    cols = args.cols
    rows = (n_pos + cols - 1) // cols

    # Pre-decode every unique token id once so per-frame rendering is cheap.
    ids = set()
    for run in (bf, mx):
        for r in run.values():
            ids.update(r["argmax_token"][:n_pos])
    glyph = {i: clean(tok.decode([i])) for i in ids}

    def frame_for(run: dict[int, dict], step: int):
        """Return (token_ids, max_probs) holding the last step <= current."""
        avail = [s for s in run if s <= step]
        if not avail:
            return [None] * n_pos, [0.0] * n_pos
        r = run[max(avail)]
        toks = (r["argmax_token"] + [None] * n_pos)[:n_pos]
        probs = (r["max_prob"] + [0.0] * n_pos)[:n_pos]
        return toks, probs

    # Single axis, reading-order layout. Each wrap-row of `cols` positions
    # occupies two cell-rows: mxfp4 on top, bf16 below. A thin gap separates
    # successive wrap-rows so the paired bands stay visually grouped.
    cell_w, cell_h = 0.92, 0.92
    pair_h = 2.0          # two stacked cells per position
    row_gap = 0.6         # blank space between wrap-rows
    band = pair_h + row_gap

    fig_w = 2 + cols * 0.9
    fig_h = 1 + rows * band * 0.5
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    suptitle = fig.suptitle("")
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows * band)
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")

    cmap = plt.get_cmap("RdYlGn")  # red (low prob) -> green (high prob)

    def build():
        """Rectangles+text for every position, mxfp4 cell over bf16 cell.

        Each position also gets a `diff` rectangle spanning the full pair, drawn
        with no fill and a thick magenta edge. It stays invisible until the two
        paths disagree on the top token at that position, at which point it lights
        up to ring the mismatched pair."""
        rects_mx, texts_mx, rects_bf, texts_bf, rects_diff = [], [], [], [], []
        for p in range(n_pos):
            r, c = divmod(p, cols)
            y_top = r * band                  # mxfp4 (top) cell origin
            y_bot = r * band + 1.0            # bf16 (bottom) cell origin
            x = c + (1 - cell_w) / 2
            rm = plt.Rectangle((x, y_top + (1 - cell_h) / 2), cell_w, cell_h,
                               facecolor="#eee", edgecolor="#c44", linewidth=0.5)
            rb = plt.Rectangle((x, y_bot + (1 - cell_h) / 2), cell_w, cell_h,
                               facecolor="#eee", edgecolor="#2a7", linewidth=0.5)
            # Spans both cells; sits on top so its border frames the pair.
            rd = plt.Rectangle((x - 0.04, y_top + (1 - cell_h) / 2 - 0.04),
                               cell_w + 0.08, cell_h + 1.08,
                               fill=False, edgecolor="#e600e6", linewidth=2.0,
                               visible=False, zorder=5)
            ax.add_patch(rm)
            ax.add_patch(rb)
            ax.add_patch(rd)
            tm = ax.text(c + 0.5, y_top + 0.5, "", ha="center", va="center",
                         fontsize=6, family="monospace")
            tb = ax.text(c + 0.5, y_bot + 0.5, "", ha="center", va="center",
                         fontsize=6, family="monospace")
            rects_mx.append(rm); texts_mx.append(tm)
            rects_bf.append(rb); texts_bf.append(tb)
            rects_diff.append(rd)
        return rects_mx, texts_mx, rects_bf, texts_bf, rects_diff

    rects_mx, texts_mx, rects_bf, texts_bf, rects_diff = build()
    # Row labels (mxfp4 top / bf16 bottom) on the first wrap-row for orientation.
    ax.text(-0.15, 0.5, "mxfp4", ha="right", va="center", fontsize=8,
            color="#c44", family="monospace")
    ax.text(-0.15, 1.5, "bf16", ha="right", va="center", fontsize=8,
            color="#2a7", family="monospace")

    def paint(rects, texts, toks, probs):
        for rect, t, tid, p in zip(rects, texts, toks, probs):
            if tid is None:
                rect.set_facecolor("#eee")
                t.set_text("")
            else:
                rect.set_facecolor(cmap(p))
                t.set_text(glyph.get(tid, str(tid)))

    def update(step: int):
        bf_toks, bf_probs = frame_for(bf, step)
        mx_toks, mx_probs = frame_for(mx, step)
        paint(rects_bf, texts_bf, bf_toks, bf_probs)
        paint(rects_mx, texts_mx, mx_toks, mx_probs)
        # Ring every position where both paths have committed a token and the two
        # top tokens disagree. Positions still blank on either side don't count.
        ndiff = 0
        for rd, tb, tm in zip(rects_diff, bf_toks, mx_toks):
            differ = tb is not None and tm is not None and tb != tm
            rd.set_visible(differ)
            ndiff += differ
        bf_done = "bf16 (committed)" if not any(s >= step for s in bf) else "bf16"
        mx_done = "mxfp4 (committed)" if not any(s >= step for s in mx) else "mxfp4"
        suptitle.set_text(
            f"denoising step {step}   [{mx_done} top / {bf_done} bottom]   "
            f"(cell color = top-token prob: red=low → green=high)   "
            f"magenta ring = mxfp4≠bf16 ({ndiff} positions differ)"
        )
        return []

    anim = FuncAnimation(fig, update, frames=steps, blit=False)
    out = os.path.join(os.path.dirname(os.path.abspath(args.bf16)),
                       "token_anim.gif")
    anim.save(out, writer=PillowWriter(fps=args.fps))
    plt.close(fig)
    print(f"animation -> {out} ({len(steps)} frames @ {args.fps} fps, "
          f"{n_pos} positions in {rows}x{cols} grid)")


if __name__ == "__main__":
    main()
