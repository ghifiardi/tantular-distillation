#!/usr/bin/env bash
# Serve one teacher on whichever host you happen to be on.
#
#   ./scripts/serve_teacher.sh muse-glimmer ai19
#   ./scripts/serve_teacher.sh nemotron rented-48gb
#
# Host config supplies capacity (parallelism, quantization, memory budget);
# teacher config supplies identity (HF repo, port, served name). Nothing
# downstream needs to know which host it is — bridge_client.py only ever
# sees http://<host>:<port>/v1.
#
# Serve ONE teacher at a time. Two 30B models do not co-resident in 48GB at
# FP8, and dropping both to int4 to make them fit degrades every trace the
# student will ever learn from. Generation is offline batch work; sequential
# passes cost wall-clock, not quality.
set -euo pipefail

TEACHER="${1:?usage: serve_teacher.sh <teacher> <host>}"
HOST="${2:?usage: serve_teacher.sh <teacher> <host>}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Prefer the project venv's binaries. The pinned torch/vllm live there, not on
# the system PATH, and a bare `vllm` either is not found (exit 127, which is how
# this surfaced on the smoke pod) or is some other install with different pins —
# which would be worse, because it would run.
VLLM_BIN="$ROOT/.venv/bin/vllm"; [[ -x "$VLLM_BIN" ]] || VLLM_BIN="$(command -v vllm || true)"
PYTHON_BIN="$ROOT/.venv/bin/python"; [[ -x "$PYTHON_BIN" ]] || PYTHON_BIN="$(command -v python3)"
[[ -n "$VLLM_BIN" ]] || { echo "vllm not found in $ROOT/.venv/bin or on PATH." >&2
  echo "Install requirements-train.txt into the project venv." >&2; exit 127; }

TEACHER_CFG="$ROOT/configs/teachers/$TEACHER.yaml"
HOST_CFG="$ROOT/configs/hosts/$HOST.yaml"
[[ -f "$TEACHER_CFG" ]] || { echo "no such teacher: $TEACHER" >&2; exit 1; }
[[ -f "$HOST_CFG" ]] || { echo "no such host: $HOST" >&2; exit 1; }

# Single source of truth for config parsing: reuse the same loader the Python
# side uses, so a serve command and a generate run can never disagree about
# which repo or port is in play.
eval "$("$PYTHON_BIN" "$ROOT/src/config.py" --shell "$TEACHER" "$HOST")"

if [[ "${HOST_RUNTIME:-vllm}" == "mlx" ]]; then
  echo "Host '$HOST' is validation-scale (int4/Metal), not a vLLM host."
  echo "Serve it locally instead, then point generate.py at it with --base-url:"
  echo "  mlx_lm.server --model $TEACHER_REPO --port $TEACHER_PORT"
  echo "Ollama also carries a Metal build, but its 21GB tag is a tight fit in 24GB:"
  echo "  ollama run $TEACHER_SERVED_MODEL_NAME:30b-mlx"
  exit 2
fi

echo "Serving $TEACHER on $HOST"
echo "  repo   : $TEACHER_REPO"
echo "  quant  : $HOST_QUANTIZATION   tp: $HOST_TENSOR_PARALLEL_SIZE"
echo "  url    : http://0.0.0.0:$TEACHER_PORT/v1"

# Serve under BOTH the short name and the repo path. preflight.py asks for
# TEACHER_REPO whenever a base_url is set, and generate_normalized.py always
# asks for TEACHER_REPO — but vLLM only answers to --served-model-name, so a
# single short name makes the endpoint reject the very id the pipeline requests
# ("model 'RedHatAI/...' is not served; available: muse-glimmer"). Registering
# both aliases satisfies every caller without renaming the repo in the teacher
# config, which would corrupt the `repo` field recorded in every trace's
# provenance. Measured 2026-08-18.
#
# The quantization flag is deliberately NOT passed: this checkpoint declares
# compressed-tensors/FP8_BLOCK itself, and forcing "fp8" makes vLLM refuse to
# start. Letting the checkpoint decide also makes the FP8 evidence in the log
# independent of what we asked for.
exec "$VLLM_BIN" serve "$TEACHER_REPO" \
  --served-model-name "$TEACHER_SERVED_MODEL_NAME" "$TEACHER_REPO" \
  --port "$TEACHER_PORT" \
  --tensor-parallel-size "$HOST_TENSOR_PARALLEL_SIZE" \
  --gpu-memory-utilization "$HOST_GPU_MEMORY_UTILIZATION" \
  --max-model-len "$HOST_MAX_MODEL_LEN"
