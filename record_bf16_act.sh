#!/usr/bin/env bash
# Zero-arg: prompt the bf16 ACT-PROBE server and snapshot its dumps.
# Run ./launch_bf16_actprobe.sh first (bf16 act-probe server on port 8004).
# That launch enables BOTH probes, so we snapshot the confidence dump AND the
# per-GEMM activation dump.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)

PROMPT="Explain in two sentences why the sky is blue."
CONF=/app/diffgemma-probing/runs/bf16.jsonl
ACT=/app/diffgemma-probing/runs/bf16_act.jsonl
OUT=/app/diffgemma-probing/runs/sky
mkdir -p "$OUT"

# The server holds these files open, so truncating them from here would leave a
# sparse null-byte hole (the server keeps writing at its old offset). Instead
# remember the current end of each and snapshot only bytes appended by this
# prompt.
START_CONF=$(wc -c < "$CONF" 2>/dev/null || echo 0)
START_ACT=$(wc -c < "$ACT" 2>/dev/null || echo 0)

REQ=$(PROMPT="$PROMPT" python3 -c '
import json, os
print(json.dumps({
    "model": "diffgemma-bf16",
    "messages": [{"role": "user", "content": os.environ["PROMPT"]}],
    "max_tokens": 256, "temperature": 0, "seed": 0,
}))')

curl -s "http://localhost:8004/v1/chat/completions" \
  -H 'Content-Type: application/json' -d "$REQ" \
  | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d["choices"][0]["message"].get("content") or "(empty)")' \
  | tee "$OUT/bf16.completion.txt"

tail -c +$((START_CONF + 1)) "$CONF" > "$OUT/bf16.jsonl"
echo "recorded $(grep -c . "$OUT/bf16.jsonl") conf-records -> $OUT/bf16.jsonl"

tail -c +$((START_ACT + 1)) "$ACT" > "$OUT/bf16_act.jsonl"
echo "recorded $(grep -c . "$OUT/bf16_act.jsonl") act-records -> $OUT/bf16_act.jsonl"
