#!/usr/bin/env bash
# Serve the BASE STUDENT for the before/after gates. Run ON the serving pod.
#
#   ./scripts/serve_student.sh office-student-9b student-serve
#
# Deliberately separate from serve_teacher.sh. That script serves a 30B teacher
# at whatever quantization its host config picks; this one serves the 9B student
# and hard-refuses any quantization at all. The reason is the whole point of the
# baseline: QLoRA trains against a bnb-NF4 view of these weights, vLLM cannot
# serve that, and bf16 is the honest substitute — same weights, same tokenizer,
# no second lossy transform. Serving int4 or fp8 here would make the "before"
# number a measurement of quantization damage that the adapter would then
# appear to have fixed.
set -euo pipefail

MODEL="${1:-office-student-9b}"
HOST="${2:-student-serve}"
# Optional third/fourth arguments serve a trained LoRA ALONGSIDE the base.
#
#   ./scripts/serve_student.sh office-student-9b student-serve \
#       ~/tantular-runs/v1/adapter tantular-office-9b-v1
#
# The adapter gets its OWN model id. That is the whole point: vLLM keeps both in
# one process, and a request naming the base id returns base answers even while
# the LoRA is loaded. run_gates --stage after asks for the adapter id and
# refuses to run if /v1/models does not list it.
ADAPTER_PATH="${3:-}"
ADAPTER_ID="${4:-}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

eval "$(python3 "$ROOT/src/config.py" --shell "$MODEL" "$HOST")"

if [[ "$HOST_QUANTIZATION" != "bf16" ]]; then
  echo "REFUSING: host '$HOST' asks for quantization '$HOST_QUANTIZATION'." >&2
  echo "The student baseline is served bf16 or not at all." >&2
  exit 2
fi

LORA_ARGS=()
if [[ -n "$ADAPTER_PATH" ]]; then
  [[ -n "$ADAPTER_ID" ]] || { echo "an adapter path needs an adapter id (arg 4)" >&2; exit 2; }
  [[ -f "$ADAPTER_PATH/adapter_config.json" ]] || {
    echo "REFUSING: $ADAPTER_PATH has no adapter_config.json — not a LoRA adapter." >&2
    exit 2; }
  if [[ "$ADAPTER_ID" == "$TEACHER_REPO" ]]; then
    echo "REFUSING: the adapter id must differ from the base id ($TEACHER_REPO)." >&2
    echo "Serving both under one id makes the after gates unable to address the adapter." >&2
    exit 2
  fi
  # --max-lora-rank must be >= the r in train/qlora_9b.yaml (32) or vLLM
  # refuses the adapter at load time, after the base is already resident.
  LORA_ARGS=(--enable-lora --max-lora-rank 32
             --lora-modules "$ADAPTER_ID=$ADAPTER_PATH")
  echo "  adapter: $ADAPTER_ID -> $ADAPTER_PATH"
fi

echo "Serving STUDENT $MODEL on $HOST"
echo "  repo   : $TEACHER_REPO"
echo "  dtype  : bfloat16   tp: $HOST_TENSOR_PARALLEL_SIZE"
echo "  url    : http://0.0.0.0:$TEACHER_PORT/v1"

# Registered under both the short name and the repo path, for the same reason
# serve_teacher.sh does it: preflight and generate_normalized ask for the repo
# path, and vLLM answers only to --served-model-name. run_gates' identity check
# accepts either, but it compares against the config's base_model, so the repo
# path must be one of the names actually served.
exec vllm serve "$TEACHER_REPO" \
  --served-model-name "$TEACHER_REPO" "$MODEL" \
  --dtype bfloat16 \
  "${LORA_ARGS[@]+"${LORA_ARGS[@]}"}" \
  --port "$TEACHER_PORT" \
  --tensor-parallel-size "$HOST_TENSOR_PARALLEL_SIZE" \
  --gpu-memory-utilization "$HOST_GPU_MEMORY_UTILIZATION" \
  --max-model-len "$HOST_MAX_MODEL_LEN"
