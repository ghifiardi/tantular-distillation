#!/usr/bin/env bash
# Generate the FP8 treatment arm, aborting on any condition that would make the
# result uninterpretable — and stopping the rented instance whatever happens.
#
#   ./scripts/run_fp8_arm.sh --vllm-log /tmp/vllm.log \
#       --budget-min 90 --rate 0.79 \
#       --on-finish "runpodctl stop pod $RUNPOD_POD_ID"
#
# Run AFTER scripts/verify_fp8_host.sh has passed and the server is up.
#
# Five gates, in the order that fails cheapest first. Each corresponds to a way
# this study has already gone wrong or could silently produce a number nobody
# can interpret:
#
#   1 hardware      the card measurably supports FP8 (cc 8.9+), from hardware.json
#   2 server        vLLM's own log says it selected fp8 — config claims are not
#                   evidence, and preflight reports the CONFIG's quantization
#   3 signature     the served model is the one the study names
#   4 arm health    no empty or truncated traces
#   5 comparability identical prompts and decoding vs the int4-normalized baseline
#
# The instance is stopped by an EXIT trap, so it stops on success, on abort, on
# the budget timeout, and on Ctrl-C. A rented card left running after a failed
# gate is the expensive failure mode.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

VLLM_LOG=""; BUDGET_MIN=90; RATE=""; ON_FINISH=""
HW_JSON="data/calibration/fp8/hardware.json"
OUT="data/calibration/fp8/traces.jsonl"
BASELINE="data/calibration/int4-normalized/traces.jsonl"
PROMPTS="prompts/calibration.jsonl"
PY=./.venv/bin/python

while [[ $# -gt 0 ]]; do
  case "$1" in
    --vllm-log)   VLLM_LOG="$2"; shift 2;;
    --budget-min) BUDGET_MIN="$2"; shift 2;;
    --rate)       RATE="$2"; shift 2;;
    --on-finish)  ON_FINISH="$2"; shift 2;;
    --hardware)   HW_JSON="$2"; shift 2;;
    *) echo "unknown argument: $1" >&2; exit 2;;
  esac
done

START=$(date +%s)

finish() {
  local code=$?
  local mins=$(( ($(date +%s) - START) / 60 ))
  echo
  echo "elapsed: ${mins} min"
  [[ -n "$RATE" ]] && echo "approx cost: \$$($PY -c "print(f'{$mins/60*$RATE:.2f}')")"
  if [[ -n "$ON_FINISH" ]]; then
    echo "stopping instance: $ON_FINISH"
    eval "$ON_FINISH" || echo "WARNING: stop command failed — STOP THE POD MANUALLY" >&2
  else
    echo "NOTE: no --on-finish given; the instance is still running and still billing."
  fi
  exit $code
}
trap finish EXIT INT TERM

abort() { echo; echo "ABORT: $*" >&2; exit 1; }

# Budget watchdog. Kills the run rather than letting a stalled load bill all
# night; the EXIT trap then stops the instance.
#
# It must kill the CHILD first. Signalling only this shell does not interrupt a
# blocked child — bash defers trap handling until the current command returns —
# so a preflight sitting on a 600s timeout, or a generation against a wedged
# server, would run past the budget and keep billing. That is precisely the
# failure this watchdog exists to prevent, so it terminates the process group's
# children and then this shell.
( sleep $((BUDGET_MIN * 60))
  echo "BUDGET EXCEEDED (${BUDGET_MIN} min) — terminating" >&2
  pkill -TERM -P $$ 2>/dev/null
  sleep 2
  kill -TERM $$ 2>/dev/null ) &
WATCHDOG=$!
trap 'kill $WATCHDOG 2>/dev/null; finish' EXIT INT TERM

# --- gate 1: hardware --------------------------------------------------------
echo "=== gate 1: hardware ==="
[[ -f "$HW_JSON" ]] || abort "no $HW_JSON — run scripts/verify_fp8_host.sh first"
$PY - "$HW_JSON" <<'PY' || exit 1
import json, sys
hw = json.load(open(sys.argv[1]))
cap = str(hw.get("compute_capability", "0"))
major, _, minor = cap.partition(".")
ok = hw.get("fp8_capable") and (int(major or 0), int(minor or 0)) >= (8, 9)
print(f"  {hw.get('gpu_name')}  cc {cap}")
if not ok:
    sys.exit(f"ABORT: hardware.json does not establish FP8 capability (cc {cap})")
print("  OK")
PY

# --- gate 2: the server actually selected fp8 --------------------------------
echo "=== gate 2: vLLM quantization ==="
[[ -n "$VLLM_LOG" ]] || abort "--vllm-log is required: preflight reports the CONFIG's
       quantization, so the server's own log is the only evidence it served fp8"
[[ -f "$VLLM_LOG" ]] || abort "no such log: $VLLM_LOG"
if grep -qiE "cannot use fp8|fp8 is not supported|falling back|quantization.*(awq|gptq|none)" "$VLLM_LOG"; then
  grep -iE "cannot use fp8|fp8 is not supported|falling back|quantization" "$VLLM_LOG" | head -5 >&2
  abort "vLLM log shows a fallback or a non-fp8 method — this would not be an FP8 arm"
fi
grep -qi "fp8" "$VLLM_LOG" || abort "vLLM log never mentions fp8; refusing to assume it"
grep -iE "quantization|fp8" "$VLLM_LOG" | head -3 | sed 's/^/  /'
echo "  OK"

# --- gate 3: signature -------------------------------------------------------
echo "=== gate 3: preflight signature ==="
mkdir -p data/calibration/fp8
$PY src/preflight.py --teacher muse-glimmer --host rented-48gb \
    --record data/calibration/fp8/signature.json || abort "preflight failed"
$PY - <<'PY' || exit 1
import json, sys
sig = json.load(open("data/calibration/fp8/signature.json"))
base = json.load(open("data/calibration/int4/signature.json"))
if sig.get("requested_model") != base.get("requested_model"):
    sys.exit(f"ABORT: model differs from the baseline study "
             f"({sig.get('requested_model')} vs {base.get('requested_model')})")
if sig.get("reported_model") != sig.get("requested_model"):
    sys.exit(f"ABORT: server reported a different model than requested "
             f"({sig.get('reported_model')} vs {sig.get('requested_model')})")
if sig.get("quantization") == base.get("quantization"):
    sys.exit(f"ABORT: this arm reports the same quantization as the baseline "
             f"({sig.get('quantization')}) — that is a comparison of int4 with itself")
print(f"  model {sig['reported_model']}  quantization {sig['quantization']}")
print("  OK")
PY

# --- generate ----------------------------------------------------------------
echo "=== generating 52 synthetic calibration prompts ==="
$PY src/generate_normalized.py --teacher muse-glimmer --host rented-48gb \
    --prompts "$PROMPTS" --out "$OUT" --resume || abort "generation failed"

# --- gate 4: arm health ------------------------------------------------------
echo "=== gate 4: arm health ==="
$PY src/calibrate.py score "$OUT" | sed 's/^/  /'
$PY - "$OUT" <<'PY' || exit 1
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
empty = sum(1 for r in rows if not r.get("completion", "").strip())
trunc = sum(1 for r in rows if r.get("provenance", {}).get("truncated"))
if len(rows) != 52:
    sys.exit(f"ABORT: {len(rows)} traces, expected 52")
if empty or trunc:
    sys.exit(f"ABORT: {empty} empty and {trunc} truncated traces — a broken arm "
             "would read as a quality difference")
print("  52 traces, 0 empty, 0 truncated\n  OK")
PY

# --- gate 5: comparability ---------------------------------------------------
echo "=== gate 5: comparability with the int4-normalized baseline ==="
$PY - "$OUT" "$BASELINE" <<'PY' || exit 1
import json, sys
def load(p): return {t["family"]: t for t in
                     (json.loads(l) for l in open(p) if l.strip())}
new, base = load(sys.argv[1]), load(sys.argv[2])
shared = sorted(set(new) & set(base))
if len(shared) != 52:
    sys.exit(f"ABORT: only {len(shared)} families in common, expected 52")
bad = [f for f in shared
       if new[f]["provenance"].get("prompt_sha256")
       != base[f]["provenance"].get("prompt_sha256")]
if bad:
    sys.exit(f"ABORT: {len(bad)} prompts differ from the baseline — the arms "
             f"answered different questions: {bad[:3]}")
for field in ("template_sha256", "temperature", "seed", "reasoning_strength"):
    a = {t["provenance"].get(field) for t in new.values()}
    b = {t["provenance"].get(field) for t in base.values()}
    if a != b:
        sys.exit(f"ABORT: {field} differs between arms: {a} vs {b}")
print(f"  52 identical prompts; template, temperature, seed and "
      f"reasoning_strength all match\n  OK")
PY

echo
echo "FP8 ARM COMPLETE — all five gates passed."
echo "Next, on any machine:"
echo "  $PY src/calibrate.py compare $BASELINE $OUT --prompts $PROMPTS"
echo "  $PY src/blind_review.py prepare --baseline $BASELINE --treatment $OUT \\"
echo "      --prompts $PROMPTS --out data/calibration/_review --salt <string> --write"
