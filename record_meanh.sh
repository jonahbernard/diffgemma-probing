#!/usr/bin/env bash
# Shared driver: prompt ONE diffgemma server with every prompt in a prompt file
# (optionally across several seeds = canvases) and snapshot all confidence records
# appended during the run into a single JSONL. Each generation is separable by its
# req_id, so the meanH overlay chart can aggregate per-step mean entropy across all
# canvases. This only drives the confidence probe (cheap); the activation probe is
# untouched / irrelevant here.
#
# Usage (normally via the record_meanh_{bf16,mxfp4,mixedprec}.sh wrappers):
#   MODEL=diffgemma-bf16 PORT=8002 LIVE=runs/bf16.jsonl OUT=runs/meanh50/bf16.jsonl \
#     ./record_meanh.sh
# Env knobs:
#   PROMPTS  prompt file, one prompt per line   (default prompts50.txt)
#   SEEDS    space-separated seeds = canvases    (default "0")
#   MAXTOK   max_tokens per request             (default 256)
#   TEMP     sampling temperature               (default 0)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)

: "${MODEL:?set MODEL (served-model-name)}"
: "${PORT:?set PORT}"
: "${LIVE:?set LIVE (the live probe jsonl the server writes)}"
: "${OUT:?set OUT (snapshot destination)}"
PROMPTS=${PROMPTS:-$HERE/prompts50.txt}
SEEDS=${SEEDS:-0}
MAXTOK=${MAXTOK:-256}
TEMP=${TEMP:-0}

mkdir -p "$(dirname "$OUT")"

# Ensure LIVE (and its dir) exist before we read its size. The `< "$LIVE"`
# redirection below fails outright on a missing file -- the `|| echo 0` can't
# catch a redirection error -- so create an empty one if the server hasn't yet.
mkdir -p "$(dirname "$LIVE")"
[ -e "$LIVE" ] || : > "$LIVE"

# The server holds LIVE open; truncating it here would punch a sparse hole (the
# server keeps writing at its old offset). Snapshot only bytes appended by this run.
START=$(wc -c < "$LIVE" 2>/dev/null || echo 0)

n=0
while IFS= read -r PROMPT || [ -n "$PROMPT" ]; do
  [ -z "$PROMPT" ] && continue
  for SEED in $SEEDS; do
    REQ=$(PROMPT="$PROMPT" MODEL="$MODEL" SEED="$SEED" MAXTOK="$MAXTOK" TEMP="$TEMP" python3 -c '
import json, os
print(json.dumps({
    "model": os.environ["MODEL"],
    "messages": [{"role": "user", "content": os.environ["PROMPT"]}],
    "max_tokens": int(os.environ["MAXTOK"]),
    "temperature": float(os.environ["TEMP"]), "seed": int(os.environ["SEED"]),
}))')
    curl -s "http://localhost:${PORT}/v1/chat/completions" \
      -H 'Content-Type: application/json' -d "$REQ" >/dev/null
    n=$((n + 1))
    printf '\r  %d generations sent...' "$n" >&2
  done
done < "$PROMPTS"
printf '\n' >&2

tail -c +$((START + 1)) "$LIVE" > "$OUT"
echo "recorded $(grep -c . "$OUT") step-records from $n generations -> $OUT"
