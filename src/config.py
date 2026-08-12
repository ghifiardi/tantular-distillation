"""Compose a teacher config with a host config.

The whole point of splitting these two files is that a run is portable: the
teacher says *what* to serve, the host says *what the hardware can do*. Move
from ai19 to a rented card by changing one CLI argument, not a config.

Also enforces the two hardware rules that are easy to get wrong and expensive
to discover after a 20-minute model load:
  - NVFP4 builds need Blackwell; Ada (4090 / RTX 6000 Ada / L40S) cannot load them.
  - FP8 needs Ada or Hopper; Ampere (A6000, A100) silently has no FP8 path.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import yaml
except ImportError:  # keep the failure legible rather than a stack trace
    sys.exit("pyyaml is required: pip install -r requirements.txt")

ROOT = Path(__file__).resolve().parent.parent


def _load(kind: str, name: str) -> dict:
    path = ROOT / "configs" / kind / f"{name}.yaml"
    if not path.exists():
        available = sorted(p.stem for p in (ROOT / "configs" / kind).glob("*.yaml"))
        sys.exit(f"no such {kind[:-1]}: {name} (have: {', '.join(available)})")
    return yaml.safe_load(path.read_text()) or {}


def resolve(teacher_name: str, host_name: str) -> dict:
    teacher = _load("teachers", teacher_name)
    host = _load("hosts", host_name)

    quant = host.get("quantization", "fp8")
    repo = (teacher.get("repos") or {}).get(quant)
    if not repo:
        have = ", ".join(sorted((teacher.get("repos") or {}).keys()))
        sys.exit(
            f"teacher '{teacher_name}' has no {quant} build for host '{host_name}' "
            f"(available: {have})"
        )

    # A wrong-architecture build loads for a long time and then dies. Catch it here.
    if "nvfp4" in repo.lower():
        sys.exit(
            f"{repo} is an NVFP4 build and requires Blackwell. Host '{host_name}' "
            "is Ada or older — use the fp8 or bf16 repo instead."
        )

    return {
        "TEACHER_NAME": teacher.get("name", teacher_name),
        "TEACHER_REPO": repo,
        "TEACHER_PORT": str(teacher.get("port", 8001)),
        "TEACHER_SERVED_MODEL_NAME": teacher.get("served_model_name", teacher_name),
        "TEACHER_LICENSE": teacher.get("license", "unknown"),
        "HOST_NAME": host.get("name", host_name),
        "HOST_RUNTIME": host.get("runtime", "vllm"),
        "HOST_QUANTIZATION": quant,
        "HOST_TENSOR_PARALLEL_SIZE": str(host.get("tensor_parallel_size", 1)),
        "HOST_GPU_MEMORY_UTILIZATION": str(host.get("gpu_memory_utilization", 0.90)),
        "HOST_MAX_MODEL_LEN": str(host.get("max_model_len", 32768)),
        "HOST_VALIDATE_ONLY": "1" if host.get("validate_only") else "",
        "HOST_MAX_PROMPTS": str(host.get("max_prompts", 0)),
        "SAMPLING": teacher.get("sampling", {}),
    }


def base_url(resolved: dict, host: str = "localhost") -> str:
    return f"http://{host}:{resolved['TEACHER_PORT']}/v1"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("teacher")
    parser.add_argument("host")
    parser.add_argument(
        "--shell",
        action="store_true",
        help="emit KEY=value lines for eval in serve_teacher.sh",
    )
    args = parser.parse_args()

    resolved = resolve(args.teacher, args.host)
    if args.shell:
        for key, value in resolved.items():
            if key == "SAMPLING":
                continue
            print(f'{key}="{value}"')
    else:
        for key, value in resolved.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
