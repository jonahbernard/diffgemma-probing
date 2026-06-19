#!/usr/bin/env python3
"""Diff per-token, per-step confidence between a bf16 and an mxfp4 probe run.

Reads the two JSONL dumps produced by the server-side probe
(vllm/model_executor/models/diffgemma_probe.py) and reports how the MXFP4 (W4A4)
denoising trajectory diverges from bf16:

  - per denoising-step: mean entropy (confidence proxy) for each path and the
    gap between them, plus how many steps each path took to commit
  - per token position: where in the canvas the confidence gap concentrates
  - argmax disagreement rate per step (did quantization flip the chosen token?)

This answers the first probe question directly: are mxfp4 confidence scores
systematically worse, and does that explain the ~2x denoising-step blowup?

Usage:
  ./analyze_confidence.py runs/sky/bf16.jsonl runs/sky/mxfp4.jsonl
  ./analyze_confidence.py --slot 0 runs/sky/bf16.jsonl runs/sky/mxfp4.jsonl
  ./analyze_confidence.py --animate runs/sky/bf16.jsonl runs/sky/mxfp4.jsonl
      (writes the GIF next to the bf16 input: runs/sky/entropy_anim.gif)
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict


def load(path: str, slot: int | None) -> dict[int, dict]:
    """Load a probe JSONL into {step: record}, keeping the first req's slot.

    Each denoising step for a given request is one record. We key by step so the
    two runs line up; if multiple slots are present, filter to one with --slot.
    """
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
            # Last write for a given step wins (commit step overwrites nothing;
            # steps are unique per request trajectory).
            by_step[rec["step"]] = rec
    return by_step


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def animate(bf: dict[int, dict], mx: dict[int, dict], out_path: str) -> None:
    """Render a side-by-side animated GIF of per-position entropy over steps.

    Left panel = bf16, right panel = mxfp4. Both panels advance in lockstep
    over the union of step indices; whichever run has already committed holds
    its final frame so the two stay aligned to the same step number. Bars are
    per canvas position; shared y-axis makes the confidence gap visually obvious.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    steps = sorted(set(bf) | set(mx))
    if not steps:
        print("  (no steps to animate)")
        return

    # Canvas width and a shared y-limit across both runs and all steps so bar
    # heights are comparable frame-to-frame and panel-to-panel.
    width = max(
        max((r["n_valid"] for r in bf.values()), default=0),
        max((r["n_valid"] for r in mx.values()), default=0),
    )
    ymax = max(
        max((max(r["token_entropy"], default=0) for r in bf.values()), default=0),
        max((max(r["token_entropy"], default=0) for r in mx.values()), default=0),
    )
    ymax = ymax * 1.05 or 1.0

    def frame_for(run: dict[int, dict], step: int) -> list[float]:
        # Hold the last available step <= current so a committed run freezes
        # on its final distribution instead of vanishing.
        avail = [s for s in run if s <= step]
        if not avail:
            return [0.0] * width
        ent = run[max(avail)]["token_entropy"]
        return list(ent) + [0.0] * (width - len(ent))

    fig, (ax_bf, ax_mx) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    x = range(width)
    bars_bf = ax_bf.bar(x, [0] * width, color="#2a7", width=0.9)
    bars_mx = ax_mx.bar(x, [0] * width, color="#c44", width=0.9)
    for ax, title in ((ax_bf, "bf16"), (ax_mx, "mxfp4")):
        ax.set_ylim(0, ymax)
        ax.set_xlim(-0.5, width - 0.5)
        ax.set_xlabel("canvas position")
        ax.set_title(title)
    ax_bf.set_ylabel("token entropy (lower = more confident)")
    suptitle = fig.suptitle("")

    def update(step: int):
        for bar, h in zip(bars_bf, frame_for(bf, step)):
            bar.set_height(h)
        for bar, h in zip(bars_mx, frame_for(mx, step)):
            bar.set_height(h)
        bf_done = "" if any(s >= step for s in bf) else " (committed)"
        mx_done = "" if any(s >= step for s in mx) else " (committed)"
        ax_bf.set_title(f"bf16{bf_done}")
        ax_mx.set_title(f"mxfp4{mx_done}")
        suptitle.set_text(f"denoising step {step}")
        return [*bars_bf, *bars_mx, suptitle]

    anim = FuncAnimation(fig, update, frames=steps, blit=False)
    fps = 2
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"\n  animation -> {out_path} ({len(steps)} frames @ {fps} fps)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("bf16", help="bf16 probe JSONL")
    ap.add_argument("mxfp4", help="mxfp4 probe JSONL")
    ap.add_argument(
        "--slot",
        type=int,
        default=None,
        help="restrict to one request slot (use if the run had >1 concurrent req)",
    )
    ap.add_argument(
        "--animate",
        action="store_true",
        help="render a side-by-side animated GIF of per-position entropy over "
        "denoising steps, written next to the bf16 input as entropy_anim.gif",
    )
    args = ap.parse_args()

    bf = load(args.bf16, args.slot)
    mx = load(args.mxfp4, args.slot)

    if args.animate:
        out = os.path.join(os.path.dirname(os.path.abspath(args.bf16)),
                           "entropy_anim.gif")
        animate(bf, mx, out)

    bf_steps = sorted(bf)
    mx_steps = sorted(mx)
    print("=== denoising step counts ===")
    print(f"  bf16 : {len(bf_steps)} steps recorded, max step = "
          f"{bf_steps[-1] if bf_steps else 'n/a'}")
    print(f"  mxfp4: {len(mx_steps)} steps recorded, max step = "
          f"{mx_steps[-1] if mx_steps else 'n/a'}")
    if bf_steps and mx_steps and mx_steps[-1]:
        ratio = (mx_steps[-1] + 1) / (bf_steps[-1] + 1)
        print(f"  mxfp4/bf16 step ratio ~ {ratio:.2f}x")

    print("\n=== per-step mean entropy (confidence proxy; lower = more confident) ===")
    print(f"  {'step':>4}  {'bf16':>8}  {'mxfp4':>8}  {'gap':>8}  {'argmax_disagree':>15}")
    common = sorted(set(bf) & set(mx))
    pos_gap_accum: dict[int, list[float]] = defaultdict(list)
    for s in common:
        b, m = bf[s], mx[s]
        be, me = b["mean_entropy"], m["mean_entropy"]
        # argmax disagreement over overlapping positions
        n = min(b["n_valid"], m["n_valid"])
        disagree = sum(
            1 for i in range(n) if b["argmax_token"][i] != m["argmax_token"][i]
        )
        dis_rate = disagree / n if n else float("nan")
        print(f"  {s:>4}  {be:>8.4f}  {me:>8.4f}  {me - be:>8.4f}  {dis_rate:>15.3f}")
        # accumulate per-position entropy gap
        for i in range(n):
            pos_gap_accum[i].append(m["token_entropy"][i] - b["token_entropy"][i])

    if not common:
        print("  (no overlapping steps — check --slot or that both runs ran the "
              "same prompt at temperature 0)")
        return

    print("\n=== per-token-position mean entropy gap (mxfp4 - bf16), avg over steps ===")
    print("  positions with the largest gap are where quantization most erodes "
          "confidence")
    ranked = sorted(
        ((pos, mean(gaps)) for pos, gaps in pos_gap_accum.items()),
        key=lambda kv: kv[1],
        reverse=True,
    )
    print(f"  {'pos':>4}  {'mean_gap':>10}")
    for pos, g in ranked[:16]:
        print(f"  {pos:>4}  {g:>10.4f}")

    overall = mean([g for gaps in pos_gap_accum.values() for g in gaps])
    print(f"\n  overall mean entropy gap (mxfp4 - bf16): {overall:+.4f}")
    print("  positive => mxfp4 is less confident on average (squashed outliers "
          "are a likely cause)")


if __name__ == "__main__":
    main()
