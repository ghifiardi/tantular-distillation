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
import os
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


def training_guard(host_name: str, hostname: str | None = None) -> dict:
    """Refuse to train on a host that is not allowed to train.

    ai19 backs the openai.ina17.com gateway and a face_ai_service. A QLoRA run
    there competes for VRAM with production traffic, and freeing room means
    stopping Ollama, which takes the gateway down. The operating decision
    (2026-08-19) is that ai19 trains nothing.

    Two independent checks, because either alone is evadable:

      declared   the host config must say `training_allowed: true`. A host that
                 is silent is refused: a training host is a deliberate choice,
                 not a default.
      actual     if the machine we are ACTUALLY running on matches a forbidden
                 host's `hostnames`, the run is refused whatever was declared.
                 Otherwise `--train-host rented-48gb` typed on ai19 would pass.
    """
    import socket
    host = _load("hosts", host_name)
    actual = (hostname or socket.gethostname()).lower()

    for path in sorted((ROOT / "configs" / "hosts").glob("*.yaml")):
        other = yaml.safe_load(path.read_text()) or {}
        if other.get("training_allowed") is not False:
            continue
        for pattern in other.get("hostnames") or []:
            if pattern.lower() in actual:
                sys.exit(
                    f"this machine ({actual}) is host '{path.stem}', which is "
                    f"not allowed to train:\n  {other.get('no_training_reason', 'declared training_allowed: false')}\n"
                    f"Declaring --train-host {host_name} does not change which "
                    "machine the process is on."
                )

    if host.get("training_allowed") is not True:
        sys.exit(
            f"host '{host_name}' is not declared as a training host "
            f"(training_allowed: {host.get('training_allowed')!r}).\n"
            + (f"  {host['no_training_reason']}\n" if host.get("no_training_reason") else "")
            + "Training must be an explicit property of a host, not an assumption."
        )
    return host


def resolve(teacher_name: str, host_name: str) -> dict:
    teacher = _load("teachers", teacher_name)
    host = _load("hosts", host_name)

    if teacher.get("parked"):
        sys.exit(
            f"teacher '{teacher_name}' is parked and not part of the active "
            "pipeline. Muse Glimmer is the sole teacher; un-park deliberately "
            "in its config if you mean to bring it back."
        )

    if host.get("teacher_serving") is False:
        sys.exit(
            f"host '{host_name}' does not serve teachers "
            f"({host.get('role', 'non-serving')} host).\n"
            "Its Ollama already serves this model to the gateway, so "
            "`--host gateway` reaches the same weights on the same cards. "
            "Serving a second copy here would cost VRAM and change nothing."
        )

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
            f"{repo} is an NVFP4 build and requires Blackwell (compute capability "
            f"10.0+). Host '{host_name}' is older — use the fp8 or bf16 repo."
        )

    # FP8 needs Ada or Hopper (>= 8.9). Ampere silently has no FP8 path, and
    # ai19's 3090s are 8.6 — catching this here rather than after a long load.
    if quant == "fp8" and host.get("supports_fp8") is False:
        sys.exit(
            f"host '{host_name}' ({host.get('gpu', '?')}, compute capability "
            f"{host.get('compute_capability', '?')}) cannot do FP8 — that needs 8.9+. "
            "Set quantization: int4 for this host, or use an Ada/Hopper card."
        )

    # A gateway serves someone else's weights, so `repo` is a remote model id
    # rather than something to download. It also charges thinking tokens
    # against max_tokens, so a small budget silently returns empty content —
    # raise it here rather than letting a whole run come back blank.
    sampling = dict(teacher.get("sampling", {}))
    floor = host.get("min_max_tokens")
    if floor and sampling.get("max_tokens", 0) < floor:
        sampling["max_tokens"] = floor

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
        "HOST_DATA_EGRESS": host.get("data_egress", "external"),
        # Runtime endpoint override. student-serve deliberately has no committed
        # address because rental addresses are ephemeral; the runbook has always
        # said to export HOST_BASE_URL, but resolve() previously ignored it.
        # That let /v1/models fall back to localhost while normalized generation
        # aborted with "host has no base_url". An explicit environment value wins
        # over YAML so a tunnel or a new rental address needs no source edit.
        "HOST_BASE_URL": os.environ.get("HOST_BASE_URL", "").strip()
                         or host.get("base_url", ""),
        # The chat path is not always OpenAI's. The Tantular companion exposes
        # /api/chat-completions and translates to Ollama's native /api/chat so
        # thinking can be disabled — the OpenAI-compatible path ignores every
        # thinking control. Measuring the product means measuring ITS path.
        "HOST_CHAT_PATH": host.get("chat_completions_path", "/v1/chat/completions"),
        # Local companions run self-signed TLS. Off only where the config says
        # so, never inferred from the URL.
        "HOST_TLS_VERIFY": "" if host.get("tls_verify") is False else "1",
        "HOST_API_KEY_ENV": host.get("api_key_env", ""),
        "HOST_CONCURRENCY": str(host.get("concurrency", 16)),
        "HOST_REQUEST_TIMEOUT_S": str(host.get("request_timeout_s", 600)),
        "SAMPLING": sampling,
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
