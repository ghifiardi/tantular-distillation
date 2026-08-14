#!/usr/bin/env bash
# Chunked, resumable 260-family generation with tunnel self-healing.
#
# Three failure modes have actually occurred this session: the tunnel dropping
# mid-run, ai19 becoming unreachable, and a preflight whose exit code was
# masked by a pipe. This handles all three.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

PROMPTS="${1:-prompts/expanded.jsonl}"
OUT="${2:-data/expanded/traces.r0.jsonl}"
CHUNK=40

ensure_tunnel() {
  pgrep -f "11435:localhost:11434" >/dev/null && \
    curl -s -m 10 -o /dev/null http://localhost:11435/v1/models && return 0
  pkill -f "11435:localhost:11434" 2>/dev/null
  nohup ssh -i ~/.ssh/tantular_ai19 -p 2219 -o IdentitiesOnly=yes \
    -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=3 \
    -N -L 11435:localhost:11434 ghifi@office.ina17.com >/tmp/ai19-tunnel.log 2>&1 &
  disown; sleep 6
  curl -s -m 10 -o /dev/null http://localhost:11435/v1/models
}

total=$(wc -l < "$PROMPTS" | tr -d ' ')
for attempt in $(seq 1 12); do
  done_n=0; [ -f "$OUT" ] && done_n=$(wc -l < "$OUT" | tr -d ' ')
  [ "$done_n" -ge "$total" ] && { echo "COMPLETE: $done_n/$total"; exit 0; }

  echo "--- attempt $attempt: $done_n/$total done ---"
  if ! ensure_tunnel; then
    echo "    tunnel down and ai19 unreachable; waiting 120s"; sleep 120; continue
  fi
  # Preflight WITHOUT a pipe: a pipeline's exit status is the last command's,
  # which silently let a dead-endpoint run proceed once already.
  if ! ./.venv/bin/python src/preflight.py --teacher muse-glimmer --host ai19-ollama \
        --verify data/calibration/int4/signature.json >/dev/null 2>&1; then
    echo "    PREFLIGHT FAILED — model signature changed or endpoint down; stopping"
    exit 1
  fi
  ./.venv/bin/python src/generate_normalized.py --host ai19-ollama \
    --prompts "$PROMPTS" --out "$OUT" --resume --limit "$CHUNK"
done
echo "gave up after 12 attempts"; exit 1
