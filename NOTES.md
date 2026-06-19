# DiffusionGemma MXFP4 confidence probing

Private probing harness for diagnosing why MXFP4 (W4A4) DiffusionGemma does
~2x the denoising steps of bf16. **Not** part of the vllm fork — keep it out of
that repo. (A git repo will be created here later.)

## First probe question

> Are the per-token confidence scores at each denoising step worse in the mxfp4
> path than in bf16, and does that explain the extra denoising steps?

The sampler commits a request once its **mean per-token entropy** drops below
`confidence_threshold` and the canvas is stable (see
`_compiled_sample_step` in `vllm/model_executor/models/diffusion_gemma.py`).
If mxfp4 keeps entropy higher per step, convergence is delayed → more steps.
That is exactly what this harness measures.

## How it works

1. A server-side probe (`vllm/model_executor/models/diffgemma_probe.py`, in the
   fork) is **env-gated** and a no-op unless `VLLM_DIFFGEMMA_PROBE` is set.
2. When enabled it dumps one JSONL record per decode request per denoising step:
   per-position `token_entropy`, `max_prob`, `argmax_token`, plus the
   `mean_entropy`/`confident` signals the sampler actually thresholds on. It
   reads the same temperature-scaled logits the sampler uses, outside the
   compiled region, so it never perturbs decoding.
3. `analyze_confidence.py` diffs the bf16 and mxfp4 dumps step-for-step and
   per token position.

## Workflow

Each step is a zero-arg script. bf16 server = HIP 2 / port 8002,
mxfp4 server = HIP 3 / port 8003. Same hardcoded prompt, temperature 0, so the
only difference between the runs is quantization.

```bash
./launch_bf16_probe.sh     # terminal 1
./launch_mxfp4_probe.sh    # terminal 2

./record_bf16.sh           # prompt bf16 + snapshot
./record_mxfp4.sh          # prompt mxfp4 + snapshot

./analyze_confidence.py runs/sky/bf16.jsonl runs/sky/mxfp4.jsonl
```

To probe a different prompt, edit the `PROMPT=` line in the two `record_*.sh`.

## Reading the output

- **step counts** — confirms the ~2x blowup quantitatively.
- **per-step mean entropy + gap** — if the gap is consistently positive, mxfp4
  is less confident every step; if it *grows* with step index, error is
  compounding along the denoising trajectory (the diffusion-specific failure
  mode, worse than autoregressive).
- **argmax disagreement** — how often quantization flips the chosen token.
- **per-position entropy gap** — which canvas positions lose the most
  confidence; concentration points toward outlier-channel squashing.

## Files

- `launch_bf16_probe.sh` / `launch_mxfp4_probe.sh` — servers with probe enabled.
- `record_bf16.sh` / `record_mxfp4.sh` — zero-arg: prompt + snapshot into `runs/sky/`.
- `analyze_confidence.py` — diff the two runs.
- `runs/` — recorded dumps + completions.

## Probe env vars

- `VLLM_DIFFGEMMA_PROBE` — output JSONL path. Unset = probe disabled.
- `VLLM_DIFFGEMMA_PROBE_TAG` — label written into each record (`bf16`/`mxfp4`).
