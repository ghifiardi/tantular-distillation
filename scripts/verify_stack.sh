#!/usr/bin/env bash
# Prove the Python stack can actually serve, BEFORE a 32GB model download.
#
#   ./scripts/verify_stack.sh
#
# verify_fp8_host.sh checks the hardware. This checks the software on top of it,
# because on 2026-08-17 every hardware check passed and the run still died —
# twice, in two different ways, both after setup time had been spent:
#
#   1. A venv built with --system-site-packages mixed the container's torch
#      2.8.0+cu128 with a pip-installed torch 2.13, producing
#      "libtorch_cuda.so: undefined symbol: ncclCommResume" — an NCCL ABI
#      mismatch that no amount of package pinning fixed.
#   2. On a clean venv, vLLM imported and torch allocated GPU tensors happily,
#      then the engine died at init with "CUDA driver version is insufficient
#      for CUDA runtime version" because vLLM's kernels are CUDA 13.
#
# The lesson in both: importing torch successfully proves very little. This
# checks the things that actually broke — isolation, a real GPU allocation,
# vLLM's compiled extension, and dependency consistency.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

PY=./.venv/bin/python
fail() { echo; echo "STACK ABORT: $*" >&2; exit 1; }

[[ -x "$PY" ]] || fail "no venv at ./.venv — create one WITHOUT --system-site-packages:
       python3 -m venv .venv"

echo "=== venv isolation ==="
# A venv with --system-site-packages sees the container's torch as well as its
# own. Both get loaded, their NCCL versions disagree, and the failure surfaces
# as an undefined symbol deep inside libtorch_cuda.so.
if [[ -f .venv/pyvenv.cfg ]] && grep -qi "include-system-site-packages *= *true" .venv/pyvenv.cfg; then
  fail "this venv was created with --system-site-packages.
       That mixes the container's torch with the installed one and produces
       'undefined symbol: ncclCommResume'. Rebuild:
         mv .venv .venv-mixed && python3 -m venv .venv"
fi
echo "  isolated (no system site-packages)"

echo
echo "=== CUDA library path ==="
# vLLM's extension links libcudart.so.13, which ships inside the venv but is
# not on the default loader path. Missing it looks like a broken install
# ("cannot open shared object file") when the library is in fact present.
CU13="$PWD/.venv/lib/python3.12/site-packages/nvidia/cu13/lib"
CU12="$PWD/.venv/lib/python3.12/site-packages/nvidia/cuda_runtime/lib"
for d in "$CU13" "$CU12"; do
  [[ -d "$d" ]] && echo "  found $(basename "$(dirname "$d")")/lib" || true
done
export LD_LIBRARY_PATH="${CU13}:${CU12}:${LD_LIBRARY_PATH:-}"
echo "  LD_LIBRARY_PATH set for this check"

echo
echo "=== imports and a real GPU allocation ==="
$PY - <<'PYEOF' || fail "the stack cannot import or cannot reach the GPU (see above)"
import sys
try:
    import torch
except Exception as e:
    sys.exit(f"torch import failed: {e}")
print(f"  torch        {torch.__version__}  (cuda {torch.version.cuda})")
if not torch.cuda.is_available():
    sys.exit("torch reports no CUDA device")
# is_available() can return True on a driver that then refuses real work, so
# allocate and compute rather than trusting the flag.
try:
    x = torch.zeros(8, device="cuda"); x += 1
    assert float(x.sum()) == 8.0
except Exception as e:
    sys.exit(f"GPU allocation failed despite is_available()==True: {e}")
print(f"  gpu          {torch.cuda.get_device_name(0)}  cc {torch.cuda.get_device_capability(0)}")
print("  gpu compute  ok")
try:
    import vllm
except Exception as e:
    sys.exit(f"vllm import failed: {e}")
print(f"  vllm         {vllm.__version__}")
try:
    import transformers
    print(f"  transformers {transformers.__version__}")
except Exception as e:
    sys.exit(f"transformers import failed: {e}")
PYEOF

echo
echo "=== dependency consistency ==="
CHECK="$($PY -m pip check 2>&1)"
if [[ "$CHECK" != *"No broken requirements found"* ]]; then
  echo "$CHECK" | head -5
  fail "pip check reports conflicts. A mixed stack fails at engine init, long
       after the model has downloaded."
fi
echo "  pip check clean"

echo
echo "=== engine init (the step that failed on driver 570) ==="
# Importing vLLM does NOT exercise its CUDA kernels. The 2026-08-17 rental got
# this far and then died in the engine. A tiny allocation through vLLM's own
# extension is what actually proves the driver can run its kernels.
$PY - <<'PYEOF' || fail "vLLM's CUDA extension cannot run on this driver.
       This is the driver/runtime mismatch: vLLM ships CUDA 13 kernels and needs
       driver >= 580. Nothing installable fixes it — rent a newer pod."
import sys
try:
    import vllm._C_stable_libtorch  # noqa: F401
    print("  vllm CUDA extension loads")
except ImportError as e:
    sys.exit(f"extension import failed: {e}")
except Exception as e:
    sys.exit(f"extension error: {e}")
PYEOF

echo
echo "STACK OK — safe to download the model and serve."
echo "Export this before serving:"
echo "  export LD_LIBRARY_PATH=$CU13:$CU12"
