#!/usr/bin/env bash
# Zero-arg: prompt the MIXED-PRECISION mxfp4 server (same sky prompt) and snapshot
# its dumps into a NEW directory so you get all the same data as runs/sky/ but for
# the per-token mixed-precision method. Run ./launch_mxfp4_mixedprec.sh first
# (server on port 8006, both probes enabled).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)

PROMPT="Explain in two sentences why the sky is blue."
CONF=/app/diffgemma-probing/runs/mixedprec.jsonl
ACT=/app/diffgemma-probing/runs/mixedprec_act.jsonl
OUT=/app/diffgemma-probing/runs/sky_mixedprec
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
    "model": "diffgemma-mxfp4",
    "messages": [{"role": "user", "content": os.environ["PROMPT"]}],
    "max_tokens": 256, "temperature": 0, "seed": 0,
}))')

curl -s "http://localhost:8006/v1/chat/completions" \
  -H 'Content-Type: application/json' -d "$REQ" \
  | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d["choices"][0]["message"].get("content") or "(empty)")' \
  | tee "$OUT/mixedprec.completion.txt"

# Snapshot under the SAME base filenames the analysis tools expect (mxfp4.jsonl /
# mxfp4_act.jsonl) so existing scripts can point at this dir unchanged.
tail -c +$((START_CONF + 1)) "$CONF" > "$OUT/mxfp4.jsonl"
echo "recorded $(grep -c . "$OUT/mxfp4.jsonl") conf-records -> $OUT/mxfp4.jsonl"

tail -c +$((START_ACT + 1)) "$ACT" > "$OUT/mxfp4_act.jsonl"
echo "recorded $(grep -c . "$OUT/mxfp4_act.jsonl") act-records -> $OUT/mxfp4_act.jsonl"
