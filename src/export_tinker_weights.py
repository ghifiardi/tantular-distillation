"""Download a trained Tinker checkpoint and convert it to a PEFT adapter.

This is the bridge from ``trained_unvalidated`` to something the repository's
existing vLLM/after-gate path can evaluate.  It does not promote a model.

Default invocation is a local record check only.  ``--execute`` requires a TTY
confirmation before reading TINKER_API_KEY or importing Tinker's weight tools.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.exit("pyyaml is required: pip install -r requirements.txt")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import train_qlora
import train_tinker_sft

CONFIRMATION_PHRASE = "I AUTHORIZE TINKER WEIGHT DOWNLOAD"


def die(message: str, code: int = 2) -> None:
    print(f"\nTINKER EXPORT ABORTED: {message}", file=sys.stderr)
    raise SystemExit(code)


def pinned_file(record: dict, key: str, label: str) -> Path:
    """A path plus digest the record claims; both must still hold."""
    pin = record.get(key)
    if not isinstance(pin, dict):
        die(f"run record does not pin a {label}")
    path = train_tinker_sft.rooted(pin.get("path") or "")
    if not path.is_file():
        die(f"{label} named by the run record is missing: {path}")
    if train_tinker_sft.file_sha256(path) != pin.get("sha256"):
        die(f"{label} changed since the run record was written: {path}")
    return path


def load_record(path: Path, config: dict, config_path: Path) -> dict:
    """Accept only a record a genuine authorized run could have produced.

    Status strings are cheap to type. This re-walks the whole chain the runner
    walked — schema-v3 manifest, its authorization and same-base baseline, the
    promotion manifest, the exact uploaded payload digests, and the raw
    checkpoint evidence — so a hand-written file naming an arbitrary
    tinker:// path cannot be exported as though it were a trained checkpoint.
    """
    if not path.is_file():
        die(f"run record missing: {path}")
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        die(f"run record is not valid JSON: {exc}")
    if record.get("status") != "trained_unvalidated":
        die(
            "run status must be exactly 'trained_unvalidated'; "
            f"got {record.get('status')!r}"
        )
    if record.get("checkpoint_label") != "trained_unvalidated":
        die("run record checkpoint label is not trained_unvalidated")
    if config["output"]["checkpoint_label"] != "trained_unvalidated":
        die("config checkpoint label changed")
    if record.get("config", {}).get("sha256") != train_tinker_sft.file_sha256(
        config_path
    ):
        die("run record was not produced from the current Tinker config")

    manifest_path = pinned_file(record, "run_manifest", "schema-v3 run manifest")
    promotion_path = pinned_file(
        record, "promotion_manifest", "promotion manifest"
    )
    # The same helper the runner used, so the authorization cannot be weaker
    # here than it was at training time.
    freeze = train_tinker_sft.check_schema_v3_freeze(
        manifest_path, config_path, promotion_path, config
    )
    manifest = freeze["manifest"]
    if manifest.get("execution_authorized") is not True:
        die("the pinned run manifest never authorized execution")
    if manifest.get("preview") is not False:
        die("the pinned run manifest is a preview")
    for field in ("training_authorization", "before_baseline"):
        if not isinstance(manifest.get(field), dict):
            die(f"the pinned run manifest has no {field}")

    # The exported adapter must trace to the exact bytes that were uploaded.
    uploaded = ((manifest.get("backend") or {}).get("payload") or {})
    payload = record.get("payload")
    if not isinstance(payload, dict):
        die("run record does not record the uploaded payload")
    for split in ("train", "eval"):
        written = payload.get(split)
        if not isinstance(written, dict):
            die(f"run record has no payload.{split}")
        expected = (uploaded.get(split) or {}).get("upload_sha256")
        if not expected or written.get("upload_sha256") != expected:
            die(
                f"payload.{split} digest does not match the frozen upload:\n"
                f"  manifest {expected}\n"
                f"  record   {written.get('upload_sha256')}"
            )

    for field in ("packages", "cookbook_api_surface", "tokens", "pricing"):
        if not record.get(field):
            die(
                f"run record has no {field}; a genuine runner record carries "
                "the environment and measurements the run was made under"
            )

    # The checkpoint must be the one the raw evidence actually contains.
    evidence = record.get("checkpoint_evidence")
    if not isinstance(evidence, dict):
        die("run record has no checkpoint_evidence")
    final = train_tinker_sft.validate_final_checkpoint(evidence)
    checkpoint = record.get("checkpoint") or {}
    if checkpoint != final:
        die(
            "run record's checkpoint does not match the final row of its own "
            "checkpoint evidence"
        )
    sampler_path = checkpoint.get("sampler_path")
    if not isinstance(sampler_path, str) or not sampler_path.startswith("tinker://"):
        die("run record has no valid Tinker sampler_path")
    return record


def require_confirmation() -> None:
    if not sys.stdin.isatty():
        die("weight download requires an interactive TTY", code=1)
    print("\nThis will contact Tinker and download checkpoint weights.")
    print(f"Type exactly: {CONFIRMATION_PHRASE}")
    if input("> ") != CONFIRMATION_PHRASE:
        die("confirmation phrase did not match", code=1)


def inspect_peft_compatibility(peft: Path, config: dict) -> dict:
    """Fail closed on known Qwen3.5/vLLM adapter incompatibilities."""
    from safetensors.torch import load_file

    peft_config = json.loads(
        (peft / "adapter_config.json").read_text(encoding="utf-8")
    )
    targets = peft_config.get("target_modules")
    if not isinstance(targets, list) or not targets:
        return {"compatible": False, "errors": ["target_modules is absent or empty"]}
    forbidden = sorted(
        target for target in targets
        if str(target).split(".")[-1] in {"in_proj_a", "in_proj_b"}
    )
    tensors = load_file(str(peft / "adapter_model.safetensors"))
    keys = sorted(tensors)
    errors = []
    if peft_config.get("base_model_name_or_path") != config["base_model"]:
        errors.append("adapter names a different base model")
    if forbidden:
        errors.append(
            "target modules unsupported by the qualified vLLM path: "
            + ", ".join(forbidden)
        )
    layer_keys = [key for key in keys if ".layers." in key]
    if not layer_keys:
        errors.append("adapter contains no layer LoRA tensors")
    missing_serving_prefix = [
        key for key in layer_keys
        if "base_model.model.model.language_model.layers." not in key
    ]
    if missing_serving_prefix:
        errors.append(
            f"{len(missing_serving_prefix)} layer tensor(s) lack the measured "
            "Qwen3.5 vLLM language_model prefix"
        )
    b_keys = [key for key in keys if "lora_B" in key]
    if not b_keys:
        errors.append("adapter contains no lora_B tensors")
        nonzero_b = 0
    else:
        nonzero_b = sum(
            int(float(tensors[key].abs().max().item()) > 0.0) for key in b_keys
        )
        if nonzero_b == 0:
            errors.append("every lora_B tensor is zero; adapter is untrained")
    return {
        "compatible": not errors,
        "errors": errors,
        "target_modules": targets,
        "forbidden_target_modules": forbidden,
        "tensor_count": len(keys),
        "lora_b_tensors": len(b_keys),
        "nonzero_lora_b_tensors": nonzero_b,
        "language_model_prefix": not missing_serving_prefix if layer_keys else False,
        "sample_keys": keys[:5],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-record", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    config_path = ROOT / "train" / "tinker_sft_9b.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    record = load_record(args.run_record.expanduser(), config, config_path)
    output = args.output_dir.expanduser()
    raw = output / "tinker-adapter"
    peft = output / "peft-adapter"

    sampler_path = record["checkpoint"]["sampler_path"]
    print("=== TINKER EXPORT PLAN ===")
    print(f"  base model   {config['base_model']}")
    print(f"  sampler path {sampler_path}")
    print(f"  raw adapter  {raw}")
    print(f"  PEFT adapter {peft}")
    print("  resulting status: peft_unvalidated")

    if not args.execute:
        print("\nDRY RUN OK — no credential read, download, conversion, or API call.")
        return

    train_qlora.check_run_dir_outside_git(output)
    if output.exists():
        die(f"output directory already exists: {output}")
    train_tinker_sft.assert_package_versions(config)
    # weights.download / weights.build_lora_adapter are part of the pinned
    # surface; drift must abort here rather than after a billable download.
    train_tinker_sft.assert_cookbook_api()
    require_confirmation()
    if not os.environ.get("TINKER_API_KEY"):
        die("TINKER_API_KEY is not set; no download started", code=1)

    from tinker_cookbook import weights

    output.mkdir(parents=True)
    record_path = output / "EXPORT.json"
    started = {
        "status": "export_started",
        "base_model": config["base_model"],
        "sampler_path": sampler_path,
        "raw_adapter": {"path": str(raw), "sha256": None},
        "peft_adapter": {"path": str(peft), "sha256": None},
        "_not_promotable": "A download-start record is never a model artifact.",
    }
    train_tinker_sft.write_json_atomic(record_path, started)

    # A download is billable and slow. If it or the conversion raises, the
    # evidence of what was already fetched must survive, exactly as the
    # training runner keeps its checkpoint evidence.
    try:
        downloaded = Path(
            weights.download(tinker_path=sampler_path, output_dir=str(raw))
        )
        if downloaded.resolve() != raw.resolve():
            die(f"weight downloader returned unexpected directory: {downloaded}")
        weights.build_lora_adapter(
            base_model=config["base_model"],
            adapter_path=str(raw),
            output_path=str(peft),
        )
        if not (peft / "adapter_config.json").is_file():
            die("PEFT conversion produced no adapter_config.json")
        if not (peft / "adapter_model.safetensors").is_file():
            die("PEFT conversion produced no adapter_model.safetensors")
        compatibility = inspect_peft_compatibility(peft, config)
    except BaseException as exc:
        train_tinker_sft.write_json_atomic(record_path, {
            **started,
            "status": "export_failed",
            "error": {"type": type(exc).__name__, "message": str(exc)},
            "raw_adapter": {
                "path": str(raw),
                "sha256": train_qlora.digest_tree(raw),
            },
            "peft_adapter": {
                "path": str(peft),
                "sha256": train_qlora.digest_tree(peft),
            },
            "_not_promotable": (
                "Export did not produce a statically checked PEFT adapter. "
                "Inspect the recorded error before retrying."
            ),
        })
        if isinstance(exc, (SystemExit, KeyboardInterrupt)):
            raise
        die(
            f"Tinker export failed; durable evidence was written to "
            f"{record_path}: {type(exc).__name__}: {exc}"
        )

    export_record = {
        "status": (
            "peft_unvalidated" if compatibility["compatible"]
            else "peft_incompatible"
        ),
        "source_run_record": {
            "path": str(args.run_record),
            "sha256": train_tinker_sft.file_sha256(args.run_record.expanduser()),
        },
        "base_model": config["base_model"],
        "sampler_path": sampler_path,
        "raw_adapter": {
            "path": str(raw),
            "sha256": train_qlora.digest_tree(raw),
        },
        "peft_adapter": {
            "path": str(peft),
            "sha256": train_qlora.digest_tree(peft),
        },
        "static_vllm_compatibility": compatibility,
        "_not_promotable": (
            "Serve this PEFT adapter under a distinct model id, run the existing "
            "after gates, and compare them with the pinned same-base baseline."
        ),
    }
    train_tinker_sft.write_json_atomic(record_path, export_record)
    if not compatibility["compatible"]:
        die(
            "exported adapter failed static vLLM compatibility checks; "
            f"see {record_path}:\n  "
            + "\n  ".join(compatibility["errors"])
        )
    print("\nEXPORT COMPLETE, NOT VALIDATED, NOT PROMOTABLE.")
    print(f"  PEFT adapter: {peft}")
    print(f"  record:       {record_path}")


if __name__ == "__main__":
    main()
