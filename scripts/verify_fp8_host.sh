#!/usr/bin/env bash
# Prove the rented box can actually do FP8. Run ON the rented box, BEFORE serving.
#
#   ./scripts/verify_fp8_host.sh [output.json]
#
# WHY THIS EXISTS, and why preflight.py is not enough:
#
# preflight.py reports `quantization` from configs/hosts/<host>.yaml — it is the
# host config's claim, not a measurement. Point it at an RTX A6000 while using
# the rented-48gb config and it prints "fp8" for a card that physically has no
# FP8 path. The run then completes, the numbers look fine, and the "treatment
# arm" is int4: a comparison of int4 against itself, which cannot fail and means
# nothing.
#
# So this reads the hardware. FP8 tensor cores need compute capability 8.9+
# (Ada / Hopper). Ampere is 8.0-8.6 and has none, whatever any config says.
set -uo pipefail

OUT="${1:-hardware.json}"

fail() { echo "ABORT: $*" >&2; exit 1; }

command -v nvidia-smi >/dev/null || fail "nvidia-smi not found — is this a GPU host?"

NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | xargs)"
CAP="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | xargs)"
COUNT="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l | xargs)"
MEM="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1 | xargs)"

echo "GPU        : $NAME"
echo "compute cap: ${CAP:-unknown}"
echo "count      : $COUNT"
echo "memory     : $MEM"
echo

[[ -n "$CAP" ]] || fail "could not read compute capability; refusing to guess"

# Named refusals for the cards most likely to be picked by mistake. The generic
# capability check below would catch these anyway; naming them makes the reason
# obvious at the point of failure instead of leaving someone to decode "8.6".
case "$NAME" in
  *A6000*)  fail "RTX A6000 is AMPERE (cc 8.6) — no FP8 path. It is the cheap
         option that silently produces an int4 'treatment' arm. Rent an
         RTX 6000 Ada or L40S instead." ;;
  *3090*)   fail "RTX 3090 is Ampere (cc 8.6) — no FP8 path. This is ai19's card;
         generating here would repeat the baseline, not test against it." ;;
  *A100*)   fail "A100 is Ampere (cc 8.0) — no FP8 path." ;;
  *V100*|*T4*) fail "$NAME predates FP8 entirely." ;;
esac

MAJOR="${CAP%%.*}"; MINOR="${CAP##*.}"
if (( MAJOR < 8 )) || { (( MAJOR == 8 )) && (( MINOR < 9 )); }; then
  fail "compute capability $CAP is below 8.9 — no FP8 tensor cores.
         Ada (8.9) or Hopper (9.0+) is required. Config claiming fp8 does not
         change the silicon."
fi

echo "OK — compute capability $CAP supports FP8 (8.9+ required)"

python3 - "$OUT" "$NAME" "$CAP" "$COUNT" "$MEM" <<'PY'
import json, sys
path, name, cap, count, mem = sys.argv[1:6]
json.dump({
    "_what": "Measured hardware of the FP8 treatment host. Recorded because "
             "preflight.py reports quantization from host CONFIG, not from the "
             "device — this file is the evidence the silicon could do it.",
    "gpu_name": name,
    "compute_capability": cap,
    "gpu_count": int(count),
    "memory_total": mem,
    "fp8_capable": True,
    "_checked": "compute capability >= 8.9, and not an Ampere card by name",
    "_still_unverified": "That vLLM actually SERVED at fp8. Confirm from the "
                         "server log with: scripts/verify_fp8_host.sh --serving <logfile>",
}, open(path, "w"), indent=2)
print(f"\nwrote {path}")
PY

cat <<'EOF'

NEXT: this proves the card CAN do FP8, not that vLLM DID.
After starting the server, confirm from its startup log:

    grep -iE "quantization|fp8" <vllm-log> | head

vLLM prints the quantization method it selected. If it says none/awq/gptq, or
warns that it fell back, stop — the arm would not be an FP8 arm.
EOF
