#!/usr/bin/env bash
# Prove a rented box can serve the BASE STUDENT in bf16. Run ON the pod, BEFORE
# serving anything.
#
#   ./scripts/verify_student_host.sh [hardware.json]
#
# This is check 1 of the five in calibration/STUDENT_ENDPOINT_VALIDATION.md.
# It exists because the two previous rentals were both lost to things knowable
# in the first sixty seconds: a card that could not do what the config claimed,
# and a driver too old for the image's CUDA kernels. Neither cost was necessary.
#
# NOTE what is NOT required here: FP8. The student is served bf16 on purpose
# (configs/teachers/office-student-9b.yaml), so Ampere is perfectly acceptable —
# the opposite of scripts/verify_fp8_host.sh, which refuses it.
set -uo pipefail

OUT="${1:-student-hardware.json}"
fail() { echo "ABORT: $*" >&2; exit 1; }

command -v nvidia-smi >/dev/null || fail "nvidia-smi not found — is this a GPU host?"

NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | xargs)"
CAP="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | xargs)"
COUNT="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l | xargs)"
MEM_MIB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | xargs)"
FREE_MIB="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | xargs)"
DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | xargs)"
MAXCUDA="$(nvidia-smi 2>/dev/null | grep -oE 'CUDA Version: [0-9]+\.[0-9]+' | head -1 | awk '{print $3}')"

echo "GPU        : $NAME"
echo "compute cap: ${CAP:-unknown}"
echo "driver     : ${DRIVER:-unknown}   (max CUDA ${MAXCUDA:-unknown})"
echo "count      : $COUNT"
echo "memory     : ${MEM_MIB} MiB total, ${FREE_MIB} MiB free"
echo

[[ -n "$CAP" ]] || fail "could not read compute capability; refusing to guess"
[[ -n "$DRIVER" ]] || fail "could not read the driver version; refusing to guess"

# bf16 needs Ampere or newer. Turing and older have no bfloat16 path and vLLM
# will either refuse or silently fall back to fp16, which is NOT what the
# student config says and would make the baseline a measurement of fp16.
MAJOR="${CAP%%.*}"
(( MAJOR >= 8 )) || fail "compute capability $CAP has no bfloat16 support.
         Ampere (8.0) or newer is required. Serving fp16 instead would make the
         baseline a measurement of a precision we never trained against."

# 9B at bf16 is ~18GB of weights before any KV cache. 24GB is a squeeze that
# leaves almost no context; 40GB+ is the intended shape.
(( FREE_MIB >= 40000 )) || {
  (( FREE_MIB >= 22000 )) && echo "WARNING: only ${FREE_MIB} MiB free. 9B bf16 is
         ~18GB of weights alone; expect to reduce --max-model-len." >&2 || \
  fail "only ${FREE_MIB} MiB free — 9B at bf16 needs ~18GB of weights plus KV cache."
}

# The driver, not the container, decides which CUDA a kernel may use. A recent
# vLLM image built against CUDA 12.8+/13 will not start on an old driver, and
# the error arrives twenty minutes into a paid rental. This is the check that
# was missing from the first rental spec.
DRV_MAJOR="${DRIVER%%.*}"
if (( DRV_MAJOR < 525 )); then
  fail "driver $DRIVER is too old for any current vLLM build (needs 525+, and
         580+ if the image ships CUDA 13 kernels). Pick a pod template with a
         newer driver rather than trying to upgrade it inside the container."
elif (( DRV_MAJOR < 580 )); then
  echo "NOTE: driver $DRIVER (max CUDA ${MAXCUDA:-?}) predates CUDA 13. Use a
      CUDA 12.x vLLM image; a CUDA 13 build will fail to launch." >&2
fi

cat > "$OUT" <<JSON
{
  "gpu": "$NAME",
  "compute_capability": "$CAP",
  "driver": "$DRIVER",
  "max_cuda": "${MAXCUDA:-unknown}",
  "gpu_count": $COUNT,
  "memory_total_mib": $MEM_MIB,
  "memory_free_mib": $FREE_MIB,
  "bf16_capable": true,
  "verdict": "OK for serving the 9B student at bf16"
}
JSON
echo "PASS — this box can serve the student at bf16. Wrote $OUT"
