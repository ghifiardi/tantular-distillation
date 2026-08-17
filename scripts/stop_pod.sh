#!/usr/bin/env bash
# Stop the rented pod, and verify it actually stopped.
#
#   ./scripts/stop_pod.sh --check          # validate only; run BEFORE the arm
#   ./scripts/stop_pod.sh --delay 600      # wait, stop, confirm
#
# Wraps the stop so that everything it depends on is checked in one place:
# runpodctl present, RUNPOD_POD_ID set and plausible, and the pod reaching a
# stopped state afterwards.
#
# WHY A WRAPPER. run_fp8_arm.sh's gate 0 validates the FIRST BINARY of the
# --on-finish command. Given `sleep 600 && runpodctl stop pod $ID` it therefore
# checks `sleep` — the delay hides the very binary that matters, and an unset
# RUNPOD_POD_ID inside a compound command is invisible to it. Passing this
# script instead makes the first binary the thing that knows how to check
# everything else.
#
# --check exists because the validation is otherwise too late: --on-finish runs
# from an EXIT trap, so a missing runpodctl surfaces after the run, with the pod
# still billing. Run --check before starting the arm, and the same code path is
# proven before it is relied on.
set -uo pipefail

DELAY=0
CHECK_ONLY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --check) CHECK_ONLY=1; shift;;
    --delay) DELAY="$2"; shift 2;;
    *) echo "usage: stop_pod.sh [--check] [--delay SECONDS]" >&2; exit 2;;
  esac
done

fail() { echo "STOP-POD ABORT: $*" >&2; exit 1; }

command -v runpodctl >/dev/null 2>&1 \
  || fail "runpodctl is not on PATH. The pod cannot be stopped automatically —
         install it, or stop the pod manually and do not rely on --on-finish."

[[ -n "${RUNPOD_POD_ID:-}" ]] \
  || fail "RUNPOD_POD_ID is unset or empty.
         export RUNPOD_POD_ID=<your-pod-id>   and re-run."

[[ "$RUNPOD_POD_ID" =~ ^[A-Za-z0-9_-]+$ ]] \
  || fail "RUNPOD_POD_ID='$RUNPOD_POD_ID' does not look like a pod id
         (expected letters, digits, - or _). A quoted empty expansion or a
         stray flag would look like this."

echo "stop-pod: runpodctl found; RUNPOD_POD_ID=$RUNPOD_POD_ID"

if (( CHECK_ONLY )); then
  # Prove the id resolves, not merely that it is well-formed. A typo passes
  # every check above and is only visible when the API is asked.
  if runpodctl get pod "$RUNPOD_POD_ID" >/dev/null 2>&1; then
    echo "stop-pod: pod resolves — OK"
  else
    fail "runpodctl cannot see pod '$RUNPOD_POD_ID'. Wrong id, or wrong account
         credentials. Fix this now: at trap time it is too late and the pod
         keeps billing."
  fi
  exit 0
fi

if (( DELAY > 0 )); then
  echo "stop-pod: waiting ${DELAY}s before stopping (retrieval window)"
  sleep "$DELAY"
fi

echo "stop-pod: stopping $RUNPOD_POD_ID"
runpodctl stop pod "$RUNPOD_POD_ID" || echo "WARNING: stop command returned non-zero" >&2

# Verify rather than assume. A stop that silently failed is exactly the case
# where a warning scrolls past and the card bills all night.
for attempt in 1 2 3; do
  sleep 5
  STATUS="$(runpodctl get pod "$RUNPOD_POD_ID" 2>/dev/null || echo '')"
  if grep -qiE "exited|stopped" <<<"$STATUS"; then
    echo "stop-pod: confirmed stopped"
    exit 0
  fi
  echo "stop-pod: not stopped yet (attempt $attempt/3); retrying stop"
  runpodctl stop pod "$RUNPOD_POD_ID" >/dev/null 2>&1
done

echo "STOP-POD FAILED TO CONFIRM: pod '$RUNPOD_POD_ID' is not reporting a
stopped state after 3 attempts. STOP IT MANUALLY IN THE CONSOLE NOW — it is
still billing." >&2
exit 1
