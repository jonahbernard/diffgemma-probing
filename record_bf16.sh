#!/usr/bin/env bash
# Zero-arg: prompt the bf16 server and snapshot its confidence dump.
# Run ./launch_bf16_probe.sh first (bf16 server on port 8002).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)

PROMPT="Explain in two sentences why the sky is blue."
LIVE=/app/diffgemma-probing/runs/bf16.jsonl
OUT=/app/diffgemma-probing/runs/sky
mkdir -p "$OUT"

# The server holds this file open, so truncating it from here would leave a
# sparse null-byte hole (the server keeps writing at its old offset). Instead
# remember the current end and snapshot only bytes appended by this prompt.
START=$(wc -c < "$LIVE" 2>/dev/null || echo 0)

REQ=$(PROMPT="$PROMPT" python3 -c '
import json, os
print(json.dumps({
    "model": "diffgemma-bf16",
    "messages": [{"role": "user", "content": os.environ["PROMPT"]}],
    "max_tokens": 256, "temperature": 0, "seed": 0,
}))')

curl -s "http://localhost:8002/v1/chat/completions" \
  -H 'Content-Type: application/json' -d "$REQ" \
  | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d["choices"][0]["message"].get("content") or "(empty)")' \
  | tee "$OUT/bf16.completion.txt"

tail -c +$((START + 1)) "$LIVE" > "$OUT/bf16.jsonl"
echo "recorded $(grep -c . "$OUT/bf16.jsonl") step-records -> $OUT/bf16.jsonl"
