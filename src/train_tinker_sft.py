"""Failure-closed Tinker SFT runner for the Tantular Office experiment.

Default invocation is local-only and writes nothing:

    ./.venv/bin/python src/train_tinker_sft.py

Render the exact upload artifact without contacting Tinker:

    ./.venv/bin/python src/train_tinker_sft.py \
        --render-only ~/tantular-runs/tinker-v1/payload

Cloud execution exists, but the checked-in preview manifest deliberately
cannot authorize it.  A run additionally needs:

* a live before-gate report for Qwen/Qwen3.5-9B-Base;
* a reviewed train/TRAINING_JUSTIFIED.md;
* a fresh, non-preview schema-v3 freeze pinning both;
* explicit host, egress reference, cost ceiling, --execute, and a TTY phrase.

Tinker and tinker-cookbook are imported only for explicit --verify-renderer or
an authorized --execute path.  Renderer verification is local-only.  No
credential is read until the final interactive confirmation has passed.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import math
import os
import sys
from collections.abc import Mapping
from datetime import date
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.exit("pyyaml is required: pip install -r requirements.txt")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import freeze_training_run
import train_qlora
from tinker_payload import PayloadError, render_files, verify_upload_against_audit

TINKER_MANIFEST_SCHEMA_VERSION = 7   # v7 records provenance_audit + harness
CONFIRMATION_PHRASE = "I AUTHORIZE TANTULAR TINKER SFT V1"


def die(message: str, code: int = 2) -> None:
    print(f"\nTINKER RUNNER ABORTED: {message}", file=sys.stderr)
    raise SystemExit(code)


def rooted(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json_atomic(path: Path, payload: dict) -> None:
    """Persist a run-state transition before proceeding to the next one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)
    try:
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError:
        # Some filesystems do not allow fsync on a directory. The file itself
        # has already been flushed and atomically replaced.
        pass


def load_config(path: Path) -> dict:
    if not path.is_file():
        die(f"config missing: {path}")
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if config.get("backend") != "tinker":
        die(f"{path} does not declare backend: tinker")
    try:
        if config["output"]["checkpoint_label"] != "trained_unvalidated":
            die("output.checkpoint_label must be exactly 'trained_unvalidated'")
        if config["sequence"]["on_overflow"] != "abort":
            die("sequence.on_overflow must be 'abort'; truncation is not allowed")
        if config["egress"]["allowed_source_classes"] != ["synthetic"]:
            die("Tinker initially allows only the exact source class 'synthetic'")
        if config["evaluation"]["required_baseline_model"] != config["base_model"]:
            die("evaluation.required_baseline_model must exactly equal base_model")
        template = rooted(config["serving"]["chat_template"])
        if not template.is_file():
            die(f"serving chat template is missing: {template}")
        model_config_path = (
            ROOT / "configs" / "teachers"
            / f"{config['evaluation']['required_model_config']}.yaml"
        )
        if not model_config_path.is_file():
            die(f"required serving model config is missing: {model_config_path}")
        model_config = yaml.safe_load(
            model_config_path.read_text(encoding="utf-8")
        ) or {}
        if model_config.get("served_model_name") != config["base_model"]:
            die("serving model config does not name the Tinker base_model")
        if model_config.get("chat_template") != config["serving"]["chat_template"]:
            die("serving model config does not use the frozen chat template")
    except (KeyError, TypeError) as exc:
        die(f"Tinker config is incomplete: {exc}")
    return config


def load_evaluation_config(config: dict) -> dict:
    try:
        spec = config["evaluation"]
        path = rooted(spec["config"])
        expected = spec["config_sha256"]
    except (KeyError, TypeError) as exc:
        die(f"evaluation config pin is incomplete: {exc}")
    if not path.is_file():
        die(f"shared evaluation config missing: {path}")
    actual = file_sha256(path)
    if actual != expected:
        die(
            "shared evaluation config is STALE or changed.\n"
            f"  pinned  {expected}\n"
            f"  on disk {actual}"
        )
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def check_schema_v3_freeze(
    manifest_path: Path,
    config_path: Path,
    promotion_path: Path,
    config: dict,
) -> dict:
    common = train_qlora.check_run_freeze(
        manifest_path,
        config_path,
        promotion_path,
        expected_schema_version=TINKER_MANIFEST_SCHEMA_VERSION,
    )
    manifest = common["manifest"]
    expected_backend = freeze_training_run.tinker_backend_snapshot(config_path)
    if manifest.get("backend") != expected_backend:
        die(
            "schema-v3 backend or rendered-payload snapshot is STALE or changed.\n"
            "Regenerate it with src/freeze_training_run.py --backend tinker."
        )

    authorized = manifest.get("execution_authorized") is True
    if authorized and manifest.get("preview") is not False:
        die("an authorized schema-v3 manifest must explicitly set preview=false")
    auth = manifest.get("training_authorization")
    baseline = manifest.get("before_baseline")
    if authorized:
        if not isinstance(auth, dict) or not isinstance(baseline, dict):
            die("manifest says execution_authorized but lacks authorization/baseline")
        auth_path = rooted(auth.get("path", ""))
        if not auth_path.is_file() or file_sha256(auth_path) != auth.get("sha256"):
            die("training authorization is missing, stale, or changed")
        baseline_path = rooted(baseline.get("path", ""))
        if (
            not baseline_path.is_file()
            or file_sha256(baseline_path) != baseline.get("sha256")
        ):
            die("before-baseline report is missing, stale, or changed")
        required_baseline = config["evaluation"].get("required_baseline_model")
        if required_baseline != config["base_model"]:
            die(
                "evaluation.required_baseline_model must exactly equal base_model; "
                f"got {required_baseline!r} vs {config['base_model']!r}"
            )
        live_baseline = freeze_training_run.baseline_snapshot(
            baseline_path, config_path, required_baseline
        )
        if live_baseline != baseline:
            die(
                "before-baseline contents no longer match the validated snapshot "
                "pinned in the schema-v3 manifest"
            )
        required = rooted(config["authorization"]["required_document"])
        if auth_path.resolve() != required.resolve():
            die(
                f"authorization must be the reviewed document at {required}; "
                f"manifest pins {auth_path}"
            )
    return common


def render_payload(config: dict) -> dict:
    try:
        rendered = render_files(
            rooted(config["data"]["train"]),
            rooted(config["data"]["eval"]),
        )
    except (KeyError, TypeError, PayloadError) as exc:
        die(f"payload refused: {exc}")

    max_tokens = int(config["sequence"]["max_tokens"])
    reserve = int(config["sequence"]["template_reserve_tokens"])
    for split in ("train", "eval"):
        item = rendered[split]
        verify_upload_against_audit(item["upload"], item["audit"])
        upper_bound = item["max_content_utf8_bytes"] + reserve
        if upper_bound > max_tokens:
            die(
                f"{split} conservative token upper bound {upper_bound} exceeds "
                f"max_tokens {max_tokens}; refusing silent truncation"
            )
        item["conservative_token_upper_bound"] = upper_bound
    return rendered


def write_payload(rendered: dict, directory: Path) -> dict:
    train_qlora.check_run_dir_outside_git(directory)
    directory.mkdir(parents=True, exist_ok=True)
    written = {}
    for split in ("train", "eval"):
        upload = directory / f"{split}.conversations.jsonl"
        audit = directory / f"{split}.audit.jsonl"
        upload.write_bytes(rendered[split]["upload"])
        audit.write_bytes(rendered[split]["audit"])
        verify_upload_against_audit(upload.read_bytes(), audit.read_bytes())
        written[split] = {
            "upload": str(upload),
            "upload_sha256": file_sha256(upload),
            "audit": str(audit),
            "audit_sha256": file_sha256(audit),
            "rows": rendered[split]["rows"],
        }
    return written


def pricing_summary(config: dict, rendered: dict) -> dict:
    price = config["pricing"]
    observed = date.fromisoformat(str(price["observed_at"]))
    age_days = (date.today() - observed).days
    if age_days < 0:
        die(f"pricing observation is in the future: {observed.isoformat()}")
    max_age = int(price["max_age_days"])
    rate = float(price["train_usd_per_million_tokens"])
    train_rows = int(rendered["train"]["rows"])
    eval_rows = int(rendered["eval"]["rows"])
    epochs = int(config["training"]["num_epochs"])
    max_tokens = int(config["sequence"]["max_tokens"])
    max_steps = int(config["training"]["max_steps"])
    eval_every = int(config["training"]["eval_every"])
    # +1 rather than a bare ceil: cookbook training loops commonly evaluate at
    # step 0 and/or once more after the final step. A ceiling that assumes
    # neither turns the "worst case" into a typical case, and the ceiling the
    # operator approves is compared against this number.
    eval_passes = math.ceil(max_steps / eval_every) + 1 if eval_every > 0 else 0
    worst_train_tokens = train_rows * epochs * max_tokens
    worst_eval_tokens = eval_rows * eval_passes * max_tokens
    worst_tokens = worst_train_tokens + worst_eval_tokens
    return {
        "observed_at": observed.isoformat(),
        "age_days": age_days,
        "max_age_days": max_age,
        "rate_usd_per_million_tokens": rate,
        "eval_passes": eval_passes,
        "worst_case_train_tokens": worst_train_tokens,
        "worst_case_eval_tokens": worst_eval_tokens,
        "worst_case_total_tokens": worst_tokens,
        "worst_case_estimate_usd": round(worst_tokens / 1_000_000 * rate, 4),
        "stale": age_days > max_age,
    }


def assert_package_versions(config: dict) -> dict:
    """Assert the direct pins AND the load-bearing transitive environment.

    tinker and tinker-cookbook alone do not determine behaviour: transformers
    and tokenizers decide the exact token ids the renderer produces, and torch
    and safetensors decide the adapter layout the exporter checks. An
    environment recorded only by its two direct pins can produce different ids
    from the one that was verified while every recorded version still matches.
    """
    versions = {}
    for section, required in (
        ("packages", True),
        ("runtime_packages", False),
    ):
        for package, expected in (config.get(section) or {}).items():
            try:
                actual = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                die(
                    f"{package} is not installed. Create a dedicated client "
                    "environment and install requirements-tinker.lock.",
                    code=3,
                )
            if actual != str(expected):
                die(
                    f"{package} version mismatch: expected {expected}, got "
                    f"{actual}.\n"
                    + (
                        "Reinstall requirements-tinker.lock, or refresh the "
                        "pin and re-freeze after re-verifying the renderer."
                        if not required else
                        "Reinstall requirements-tinker.txt."
                    ),
                    code=3,
                )
            versions[package] = actual
    return versions


# Exactly the keyword arguments execute_training() and the exporter pass. If a
# pinned release renames or drops one, the call fails — and it would fail after
# the billable client exists, which is how `warmup_ratio` was discovered in the
# local trainer (see src/train_qlora.py, build_sft_trainer). Introspect the
# surface locally instead, before any confirmation prompt.
COOKBOOK_API = {
    "tinker_cookbook.supervised.train:Config": (
        "log_path", "model_name", "recipe_name", "renderer_name",
        "dataset_builder", "learning_rate", "lr_schedule", "num_epochs",
        "lora_rank", "save_every", "eval_every", "max_steps",
    ),
    "tinker_cookbook.supervised.data:FromConversationFileBuilder": (
        "common_config", "file_path", "test_size", "shuffle_seed",
    ),
    "tinker_cookbook.supervised.types:ChatDatasetBuilderCommonConfig": (
        "model_name_for_tokenizer", "renderer_name", "max_length",
        "batch_size", "train_on_what",
    ),
    "tinker_cookbook.weights:download": ("tinker_path", "output_dir"),
    "tinker_cookbook.weights:build_lora_adapter": (
        "base_model", "adapter_path", "output_path",
    ),
}


def assert_cookbook_api() -> dict:
    """Prove the pinned cookbook still accepts every argument this code passes."""
    import importlib
    import inspect

    surface = {}
    for target, expected in COOKBOOK_API.items():
        module_name, _, attribute = target.partition(":")
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            die(f"cookbook API drift: cannot import {module_name}: {exc}", code=3)
        obj = getattr(module, attribute, None)
        if obj is None:
            die(f"cookbook API drift: {module_name} has no {attribute}", code=3)
        try:
            signature = inspect.signature(obj)
        except (TypeError, ValueError) as exc:
            die(
                f"cookbook API drift: cannot introspect {target}: "
                f"{type(exc).__name__}: {exc}",
                code=3,
            )
        parameters = signature.parameters
        accepts_kwargs = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
        missing = [name for name in expected if name not in parameters]
        if missing and not accepts_kwargs:
            die(
                f"cookbook API drift: {target} does not accept "
                + ", ".join(missing)
                + ".\nThe pinned release changed under this integration; update "
                "the call sites and re-freeze rather than passing them blindly.",
                code=3,
            )
        surface[target] = sorted(parameters)
    return surface


def normalize_token_ids(value) -> list[int]:
    """Reduce apply_chat_template's several return shapes to a flat id list.

    transformers 5.x returns a BatchEncoding, which subclasses UserDict rather
    than dict — so an `isinstance(value, dict)` test misses it and iterating the
    object yields its KEYS. That made the parity check abort on every row with
    "invalid literal for int() with base 10: 'input_ids'", which is a check that
    can never pass rather than a check that passes.
    """
    if hasattr(value, "input_ids"):
        value = value.input_ids
    elif isinstance(value, Mapping):
        if "input_ids" not in value:
            raise ValueError(
                f"tokenizer returned a mapping without input_ids: {sorted(value)}"
            )
        value = value["input_ids"]
    value = list(value)
    if value and isinstance(value[0], (list, tuple)):
        if len(value) != 1:
            raise ValueError(
                f"expected one tokenized conversation, got {len(value)}"
            )
        value = list(value[0])
    return [int(token) for token in value]


def exact_token_validation(config: dict, rendered: dict) -> dict:
    """Verify loss spans and serving-template parity with the pinned renderer."""
    try:
        from tinker_cookbook.renderers import TrainOnWhat, get_renderer
        from tinker_cookbook.tokenizer_utils import get_tokenizer
    except ImportError as exc:
        die(
            f"cannot import the pinned renderer surface: {exc}. Install "
            "requirements-tinker.txt in the dedicated client environment.",
            code=3,
        )

    # Loading reaches the Hugging Face Hub on a cold cache. A network failure
    # here is a controlled abort, not a traceback: it says nothing about the
    # payload, and under restricted networking it is the expected outcome.
    try:
        tokenizer = get_tokenizer(config["base_model"])
    except Exception as exc:
        die(
            f"cannot load the tokenizer for {config['base_model']}: "
            f"{type(exc).__name__}: {exc}\n"
            "If the tokenizer has already been downloaded once, re-run with "
            "--offline (or HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1) to verify "
            "from the local cache without any network call.",
            code=3,
        )
    try:
        renderer = get_renderer(config["renderer"], tokenizer)
    except Exception as exc:
        die(
            f"cannot construct the {config['renderer']!r} renderer: "
            f"{type(exc).__name__}: {exc}",
            code=3,
        )
    template_path = rooted(config["serving"]["chat_template"])
    chat_template = template_path.read_text(encoding="utf-8")
    max_tokens = int(config["sequence"]["max_tokens"])
    report = {
        "renderer": config["renderer"],
        "serving_chat_template": str(template_path),
        "serving_chat_template_sha256": file_sha256(template_path),
    }
    for split in ("train", "eval"):
        lengths = []
        for index, line in enumerate(
            rendered[split]["upload"].decode("utf-8").splitlines(), start=1
        ):
            conversation = json.loads(line)["messages"]
            model_input, weights = renderer.build_supervised_example(
                conversation,
                train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
            )
            length = int(model_input.length)
            if length > max_tokens:
                die(
                    f"{split} row {index} renders to {length} tokens, above "
                    f"max_tokens {max_tokens}; refusing truncation"
                )
            weight_values = weights.tolist()
            if not any(float(value) > 0 for value in weight_values):
                die(f"{split} row {index} has no loss-bearing assistant tokens")
            try:
                tokens = [int(token) for token in model_input.to_ints()]
            except Exception as exc:
                die(
                    f"{split} row {index}: cannot extract renderer tokens: "
                    f"{type(exc).__name__}: {exc}"
                )
            if len(tokens) != len(weight_values):
                die(
                    f"{split} row {index}: renderer returned {len(tokens)} tokens "
                    f"but {len(weight_values)} loss weights"
                )
            target_tokens = [
                token for token, weight in zip(tokens, weight_values)
                if float(weight) > 0
            ]
            target_text = tokenizer.decode(target_tokens)
            answer = conversation[-1]["content"].strip()
            if answer not in target_text:
                die(
                    f"{split} row {index}: renderer loss span does not contain "
                    "the assistant completion byte-for-byte"
                )
            if "<think>" in target_text or "</think>" in target_text:
                die(f"{split} row {index}: thinking tokens entered the loss span")

            # The product gates call /v1/chat/completions. vLLM must therefore
            # apply a Jinja template that tokenizes to the same generation
            # prompt Tinker trained. A textual resemblance is insufficient.
            prefix = conversation[:-1]
            try:
                tinker_prompt = [
                    int(token)
                    for token in renderer.build_generation_prompt(prefix).to_ints()
                ]
                served_prompt = tokenizer.apply_chat_template(
                    prefix,
                    tokenize=True,
                    add_generation_prompt=True,
                    chat_template=chat_template,
                )
                served_prompt = normalize_token_ids(served_prompt)
            except Exception as exc:
                die(
                    f"{split} row {index}: cannot verify serving-template parity: "
                    f"{type(exc).__name__}: {exc}"
                )
            if tinker_prompt != served_prompt:
                mismatch = next(
                    (
                        i for i, (left, right) in
                        enumerate(zip(tinker_prompt, served_prompt))
                        if left != right
                    ),
                    min(len(tinker_prompt), len(served_prompt)),
                )
                die(
                    f"{split} row {index}: Tinker renderer and vLLM chat "
                    f"template differ at token {mismatch} "
                    f"({len(tinker_prompt)} vs {len(served_prompt)} tokens)"
                )
            lengths.append(length)
        report[split] = {
            "rows": len(lengths),
            "min_tokens": min(lengths),
            "max_tokens": max(lengths),
            "mean_tokens": round(sum(lengths) / len(lengths), 2),
            "total_tokens": sum(lengths),
        }
    return report


def print_token_report(token_report: dict) -> None:
    """Print only the per-split statistics.

    The report also carries scalar provenance fields (renderer, template path
    and digest). Iterating every key and indexing it as a split crashed the
    authorized execute path with a TypeError immediately before the human
    confirmation prompt, so the split names are named explicitly here.
    """
    for split in ("train", "eval"):
        stats = token_report.get(split)
        if not isinstance(stats, dict):
            continue
        print(
            f"  {split:<5} {stats['min_tokens']}..{stats['max_tokens']} tokens, "
            f"{stats['total_tokens']:,} total"
        )


def require_confirmation() -> None:
    if not sys.stdin.isatty():
        die(
            "cloud execution requires an interactive TTY; refusing to accept "
            "confirmation from a pipe or EOF",
            code=1,
        )
    print("\nThis next step creates a billable external Tinker training run.")
    print(f"Type exactly: {CONFIRMATION_PHRASE}")
    if input("> ") != CONFIRMATION_PHRASE:
        die("confirmation phrase did not match; no API client was created", code=1)


def checkpoint_evidence(log_path: Path) -> dict:
    path = log_path / "checkpoints.jsonl"
    if not path.is_file():
        return {"path": str(path), "exists": False, "rows": [], "raw_lines": []}
    try:
        raw_lines = [
            line for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except OSError as exc:
        return {
            "path": str(path),
            "exists": True,
            "rows": [],
            "raw_lines": [],
            "read_error": f"{type(exc).__name__}: {exc}",
        }
    rows, parse_errors = [], []
    for number, line in enumerate(raw_lines, start=1):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            parse_errors.append({"line": number, "error": str(exc), "raw": line})
    return {
        "path": str(path),
        "exists": True,
        "sha256": file_sha256(path),
        "rows": rows,
        "raw_lines": raw_lines,
        "parse_errors": parse_errors,
    }


def validate_final_checkpoint(evidence: dict) -> dict:
    if not evidence.get("exists"):
        die("Tinker training returned without checkpoints.jsonl")
    if evidence.get("read_error"):
        die(f"Tinker checkpoints.jsonl could not be read: {evidence['read_error']}")
    if evidence.get("parse_errors"):
        die("Tinker checkpoints.jsonl contains malformed JSON")
    rows = evidence.get("rows") or []
    if not rows:
        die("Tinker checkpoints.jsonl is empty")
    final = rows[-1]
    if not final.get("final"):
        die("last Tinker checkpoint is not marked final")
    if not final.get("state_path") or not final.get("sampler_path"):
        die("final checkpoint lacks state_path or sampler_path")
    return final


async def execute_training(config: dict, payload_paths: dict, log_path: Path) -> None:
    from tinker_cookbook.renderers import TrainOnWhat
    from tinker_cookbook.supervised import train
    from tinker_cookbook.supervised.data import FromConversationFileBuilder
    from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig

    train_common = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=config["base_model"],
        renderer_name=config["renderer"],
        max_length=int(config["sequence"]["max_tokens"]),
        batch_size=int(config["training"]["train_batch_size"]),
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )
    eval_common = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=config["base_model"],
        renderer_name=config["renderer"],
        max_length=int(config["sequence"]["max_tokens"]),
        batch_size=int(config["training"]["eval_batch_size"]),
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )

    class SeparateConversationFiles:
        def __call__(self):
            train_ds, _ = FromConversationFileBuilder(
                common_config=train_common,
                file_path=payload_paths["train"]["upload"],
                test_size=0,
                shuffle_seed=0,
            )()
            eval_ds, _ = FromConversationFileBuilder(
                common_config=eval_common,
                file_path=payload_paths["eval"]["upload"],
                test_size=0,
                shuffle_seed=0,
            )()
            return train_ds, eval_ds

    tr = config["training"]
    run_config = train.Config(
        log_path=str(log_path),
        model_name=config["base_model"],
        recipe_name="tantular_tinker_sft_v1",
        renderer_name=config["renderer"],
        dataset_builder=SeparateConversationFiles(),
        learning_rate=float(tr["learning_rate"]),
        lr_schedule=tr["lr_schedule"],
        num_epochs=int(tr["num_epochs"]),
        lora_rank=int(config["lora"]["rank"]),
        save_every=int(tr["save_every"]),
        eval_every=int(tr["eval_every"]),
        max_steps=int(tr["max_steps"]),
    )
    await train.main(run_config)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="train/tinker_sft_9b.yaml")
    parser.add_argument(
        "--run-manifest",
        default="train/RUN_MANIFEST.tinker-sft-v1.preview.json",
    )
    parser.add_argument(
        "--promotion-manifest",
        default="train/RUN_MANIFEST.v1-mechanical.json",
    )
    parser.add_argument("--run-dir", default="~/tantular-runs/tinker-v1")
    parser.add_argument("--train-host", default=None)
    parser.add_argument("--egress-approval", default=None)
    parser.add_argument("--max-cost-usd", type=float, default=None)
    parser.add_argument(
        "--render-only",
        type=Path,
        help="write payload and audit sidecars locally, then stop",
    )
    parser.add_argument(
        "--verify-renderer",
        action="store_true",
        help="locally import the pinned Tinker packages and prove tokenizer, "
             "loss-span, and vLLM chat-template parity without an API call",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="verify from the local Hugging Face cache only; no network call. "
             "Requires the tokenizer to have been downloaded once already",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="request cloud execution; still requires pinned authorization, "
             "same-base baseline, cost/egress approval, and interactive confirmation",
    )
    args = parser.parse_args()

    if args.offline:
        # Set before any transformers/huggingface_hub import reads them.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    config_path = rooted(args.config)
    manifest_path = rooted(args.run_manifest)
    promotion_path = rooted(args.promotion_manifest)
    config = load_config(config_path)
    freeze = check_schema_v3_freeze(
        manifest_path, config_path, promotion_path, config
    )
    eval_config = load_evaluation_config(config)
    train_qlora.check_eval_held_out(eval_config)
    rendered = render_payload(config)
    pricing = pricing_summary(config, rendered)

    print("\n=== TINKER PAYLOAD ===")
    for split in ("train", "eval"):
        item = rendered[split]
        print(
            f"  {split:<5} {item['rows']:>3} rows  "
            f"{item['sha256'][:16]}…  source_class=synthetic  "
            f"upper_bound={item['conservative_token_upper_bound']} tokens"
        )
    print("\n=== COST BOUND ===")
    print(
        f"  pricing observed {pricing['observed_at']} "
        f"({pricing['age_days']} day(s) old)"
    )
    print(f"  train bound {pricing['worst_case_train_tokens']:,} tokens")
    print(
        f"  eval bound  {pricing['worst_case_eval_tokens']:,} tokens "
        f"across {pricing['eval_passes']} pass(es)"
    )
    print(
        f"  total bound {pricing['worst_case_total_tokens']:,} tokens "
        f"= ${pricing['worst_case_estimate_usd']:.4f}"
    )
    print("  This is a planning bound, not an API-enforced billing cap.")

    # Renderer verification runs BEFORE the --render-only early return. A
    # requested safety check that silently does not run is worse than one that
    # fails: the operator reads "DRY RUN OK" and believes parity was proven.
    versions = None
    token_report = None
    if args.verify_renderer:
        versions = assert_package_versions(config)
        assert_cookbook_api()
        token_report = exact_token_validation(config, rendered)
        print("\n=== EXACT RENDERER / SERVING PARITY ===")
        print_token_report(token_report)
        print(
            f"  template {token_report['serving_chat_template_sha256'][:16]}… "
            "matches Tinker generation prompts"
        )

    if args.render_only:
        payload_paths = write_payload(rendered, args.render_only.expanduser())
        print(f"\nwrote local payload under {args.render_only.expanduser()}")
        print(json.dumps(payload_paths, indent=2))
        if not args.execute:
            return

    if not args.execute:
        print(
            "\nDRY RUN OK — schema-v3 preview freeze, corpus, held-out gates, "
            "external-only source classification, payload digests, sequence "
            "bound, and pricing bound verified."
        )
        if args.verify_renderer:
            print("No credential was read and no Tinker API call or training occurred.")
        else:
            print("Nothing was written, imported from Tinker, uploaded, or trained.")
        if freeze["manifest"].get("execution_authorized") is not True:
            print(
                "\nEXECUTION BLOCKED — no reviewed training justification and/or "
                "valid live before baseline for Qwen/Qwen3.5-9B-Base is pinned."
            )
        return

    if freeze["manifest"].get("execution_authorized") is not True:
        die(
            "the run manifest is preview-only. Training remains NOT JUSTIFIED "
            "and/or lacks a valid same-base before report.",
            code=1,
        )
    if args.train_host != config["egress"]["host"]:
        die(
            f"--train-host must be {config['egress']['host']!r}; "
            f"got {args.train_host!r}"
        )
    from config import training_guard
    training_guard(args.train_host)
    if not args.egress_approval or not args.egress_approval.strip():
        die("--egress-approval <reference> is required for external training", code=1)
    if pricing["stale"]:
        die(
            f"pricing is {pricing['age_days']} days old, beyond the "
            f"{pricing['max_age_days']}-day limit; refresh and re-freeze",
            code=1,
        )
    if args.max_cost_usd is None or not math.isfinite(args.max_cost_usd):
        die("--max-cost-usd is required for cloud execution", code=1)
    if args.max_cost_usd < pricing["worst_case_estimate_usd"]:
        die(
            f"approved cost ceiling ${args.max_cost_usd:.4f} is below the "
            f"worst-case estimate ${pricing['worst_case_estimate_usd']:.4f}",
            code=1,
        )

    # These imports and tokenizer work are local but intentionally absent from
    # the ordinary dry run.  They validate the exact pinned client surface
    # before the billable API client can exist.
    versions = versions or assert_package_versions(config)
    api_surface = assert_cookbook_api()
    token_report = token_report or exact_token_validation(config, rendered)
    run_dir = Path(args.run_dir).expanduser()
    train_qlora.check_run_dir_outside_git(run_dir)
    occupied = [
        path for path in (
            run_dir / "RUN.request.json",
            run_dir / "RUN.json",
            run_dir / "payload",
            run_dir / "tinker",
        )
        if path.exists()
    ]
    if occupied:
        die(
            "run directory already contains Tinker execution artifacts; "
            "refusing an implicit resume or overwrite:\n  "
            + "\n  ".join(str(path) for path in occupied),
            code=1,
        )
    payload_paths = write_payload(rendered, run_dir / "payload")
    print("\n=== EXACT TOKENIZATION ===")
    print_token_report(token_report)

    require_confirmation()
    # Deliberately read the secret only after the final human decision.
    if not os.environ.get("TINKER_API_KEY"):
        die("TINKER_API_KEY is not set; no API client was created", code=1)

    run_dir.mkdir(parents=True, exist_ok=True)
    request_record = {
        "status": "authorized_not_started",
        "checkpoint_label": "trained_unvalidated",
        "config": {"path": str(config_path), "sha256": file_sha256(config_path)},
        "run_manifest": {
            "path": str(manifest_path),
            "sha256": file_sha256(manifest_path),
        },
        "promotion_manifest": {
            "path": str(promotion_path),
            "sha256": file_sha256(promotion_path),
        },
        "payload": payload_paths,
        "packages": versions,
        "cookbook_api_surface": api_surface,
        "tokens": token_report,
        "pricing": pricing,
        "approved_cost_ceiling_usd": args.max_cost_usd,
        "egress_approval": args.egress_approval,
    }
    write_json_atomic(run_dir / "RUN.request.json", request_record)

    log_path = run_dir / "tinker"
    run_record_path = run_dir / "RUN.json"
    write_json_atomic(run_record_path, {
        **request_record,
        "status": "training_started",
        "checkpoint_evidence": checkpoint_evidence(log_path),
        "_not_promotable": "A training-start record is never a model artifact.",
    })
    try:
        asyncio.run(execute_training(config, payload_paths, log_path))
    except BaseException as exc:
        failed_record = {
            **request_record,
            "status": "training_failed_or_interrupted",
            "error": {"type": type(exc).__name__, "message": str(exc)},
            "checkpoint_evidence": checkpoint_evidence(log_path),
            "_not_promotable": (
                "Training did not produce a locally validated final checkpoint. "
                "Inspect checkpoint_evidence before any recovery attempt."
            ),
        }
        write_json_atomic(run_record_path, failed_record)
        if isinstance(exc, KeyboardInterrupt):
            raise
        die(
            f"Tinker training failed; durable evidence was written to "
            f"{run_record_path}: {type(exc).__name__}: {exc}"
        )

    evidence = checkpoint_evidence(log_path)
    returned_record = {
        **request_record,
        "status": "training_returned_checkpoint_unverified",
        "checkpoint_evidence": evidence,
        "_not_promotable": (
            "The Tinker call returned, but checkpoint shape has not yet been "
            "accepted."
        ),
    }
    write_json_atomic(run_record_path, returned_record)
    try:
        checkpoint = validate_final_checkpoint(evidence)
    except SystemExit:
        # The raw checkpoint file and every parseable row are already durable.
        raise
    final_record = {
        **request_record,
        "status": "trained_unvalidated",
        "checkpoint": checkpoint,
        "checkpoint_evidence": evidence,
        "_not_promotable": (
            "Download sampler weights, verify PEFT/vLLM compatibility, serve "
            "under a distinct adapter id, and pass the existing after gates."
        ),
    }
    write_json_atomic(run_record_path, final_record)
    print("\nTRAINING COMPLETE, NOT VALIDATED, NOT PROMOTABLE.")
    print(f"  sampler_path: {checkpoint['sampler_path']}")
    print(f"  state_path:   {checkpoint['state_path']}")
    print(f"  record:       {run_dir / 'RUN.json'}")


if __name__ == "__main__":
    main()
