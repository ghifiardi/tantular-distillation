"""Freeze the exact corpus, config and gate verdict a training run starts from.

    ./.venv/bin/python src/freeze_training_run.py \
        --corpus data/v3-candidate/traces.r0.jsonl \
        --config train/qlora_9b.yaml \
        --promotion-manifest train/RUN_MANIFEST.v1-mechanical.json \
        --waiver calibration/INT4_WAIVER.md \
        --out train/RUN_MANIFEST.v1.json \
        --frozen-at 2026-08-19T12:00:00+07:00 --write

A training run is a claim about what produced a checkpoint. Six months later the
adapter exists and the question is what it was trained on — which corpus bytes,
which hyperparameters, and under what authorisation. Reconstructing that from
memory produces a story, not a record.

THE GATE IS RUN, NOT ASKED ABOUT. This executes `verify_corpus.py --gate` and
records its real exit code and stderr. It does not accept an exit code as an
argument, because the one thing this file must never do is let a failing gate be
written down as a pass. int4 traces FAIL the gate; a waiver authorises
proceeding despite that failure and does not convert it into a pass. The
manifest therefore records `gate_exit_code: 1` alongside the waiver reference,
which is the honest shape of the decision.

The promotion manifest is part of the freeze, not merely an input used by the
trainer later. Its own digest and the exact promoted train/eval bytes are
recorded so changing either the promotion decision or its outputs makes the
freeze stale.

Refuses to write if a waiver is required (gate failed) and none was supplied,
or if the promotion manifest does not describe this exact source corpus.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import subprocess
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.exit("pyyaml is required: pip install -r requirements.txt")

ROOT = Path(__file__).resolve().parent.parent
# v4/v5 add the provenance_audit block. The bump exists so an older
# manifest cannot be mistaken for one that simply had nothing to report.
SCHEMA_VERSION = 4
TINKER_SCHEMA_VERSION = 5
REQUIRED_INT4_WAIVER = ROOT / "calibration" / "INT4_WAIVER.md"


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            sha.update(chunk)
    return sha.hexdigest()


def load_json(path: Path, label: str) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        sys.exit(f"{label} is not readable JSON: {path}: {e}")
    if not isinstance(payload, dict):
        sys.exit(f"{label} must contain a JSON object: {path}")
    return payload


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def provenance_audit(corpus: Path, frozen_at: str) -> dict:
    """The corpus audit src/distill_plan.py prints, recorded in the freeze.

    Additive evidence: it does NOT change what the corpus gate decides. What it
    adds is the standing reason a checkpoint's claims are limited — quantized
    teacher, synthetic source class, teacher licence freshness — recorded at
    freeze time instead of reconstructed from memory later.

    Fail-closed: an audit that cannot run aborts the freeze rather than writing
    a manifest whose evidence block is quietly absent.
    """
    sys.path.insert(0, str(ROOT / "src"))
    import distill_plan

    try:
        today = datetime.date.fromisoformat(str(frozen_at)[:10])
    except ValueError:
        sys.exit(f"--frozen-at must start with an ISO date; got {frozen_at!r}. "
                 "Licence freshness is measured against it, so it cannot be "
                 "guessed.")
    try:
        report = distill_plan.audit_corpus(corpus, today)
    except SystemExit as exc:
        sys.exit(f"the provenance audit could not run on {corpus} ({exc}). "
                 "Refusing to freeze without it.")
    for key in ("fp8_gate", "source_classes", "license_freshness",
                "identity_verification"):
        if key not in report:
            sys.exit(f"the provenance audit returned no {key!r}; refusing to "
                     "record a partial verdict.")
    # authorizes_training stays. A freeze is read by someone reconstructing what
    # a checkpoint was allowed to claim, and the answer has to be in the block
    # they are reading, not inferred from its absence.
    report["_what"] = (
        "Mechanical limits of the corpus at freeze time. Additive evidence: it "
        "does not decide the gate, and it authorises nothing.")
    report["as_of_date"] = today.isoformat()
    return report


def corpus_gate_errors(corpus: Path) -> list[str]:
    sys.path.insert(0, str(ROOT / "src"))
    import splits as splits_module
    import verify_corpus

    manifest = splits_module.load()
    records = verify_corpus.load_corpus([corpus])
    return verify_corpus.check(records, manifest, gate=True)


def promotion_snapshot(path: Path, corpus: Path) -> dict:
    """Validate and freeze the mechanical promotion decision and its outputs."""
    manifest = load_json(path, "promotion manifest")
    try:
        source = manifest["source_corpus"]
        promoted = manifest["promoted"]
        split_fingerprint = manifest["splits"]["fingerprint"]
    except (KeyError, TypeError):
        sys.exit("promotion manifest is malformed: expected source_corpus, "
                 "promoted, and splits.fingerprint")

    if source.get("sha256") != digest(corpus):
        sys.exit(
            "promotion manifest describes a different source corpus.\n"
            f"  promotion source {source.get('sha256')}\n"
            f"  corpus on disk    {digest(corpus)}"
        )

    snapshot = {}
    for split in ("train", "eval"):
        entry = promoted.get(split)
        if not isinstance(entry, dict):
            sys.exit(f"promotion manifest has no promoted.{split} object")
        promoted_path = resolve(Path(entry.get("path", "")))
        if not promoted_path.is_file():
            sys.exit(f"promoted {split} file is missing: {promoted_path}")
        actual_sha = digest(promoted_path)
        if actual_sha != entry.get("sha256"):
            sys.exit(
                f"promoted {split} changed since promotion.\n"
                f"  promotion manifest {entry.get('sha256')}\n"
                f"  file on disk      {actual_sha}"
            )
        try:
            rows = [json.loads(line) for line in
                    promoted_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()]
        except json.JSONDecodeError as e:
            sys.exit(f"promoted {split} is malformed JSONL: {e}")
        if len(rows) != entry.get("traces"):
            sys.exit(f"promoted {split} has {len(rows)} traces, promotion "
                     f"manifest says {entry.get('traces')}")
        snapshot[split] = {
            "path": entry["path"],
            "sha256": actual_sha,
            "traces": len(rows),
        }

    return {
        "path": str(path),
        "sha256": digest(path),
        "source_corpus_sha256": source["sha256"],
        "split_fingerprint": split_fingerprint,
        "promoted": snapshot,
    }


def tinker_backend_snapshot(config_path: Path) -> dict:
    """Freeze backend-specific fields and the exact derived upload payload."""
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if config.get("backend") != "tinker":
        sys.exit(
            f"--backend tinker requires backend: tinker in {config_path}"
        )
    try:
        data = config["data"]
        train_path = resolve(Path(data["train"]))
        eval_path = resolve(Path(data["eval"]))
        base_model = config["base_model"]
        renderer = config["renderer"]
        rank = config["lora"]["rank"]
        max_tokens = config["sequence"]["max_tokens"]
        packages = config["packages"]
        runtime_packages = config["runtime_packages"]
        pricing = config["pricing"]
        serving = config["serving"]
        chat_template = resolve(Path(serving["chat_template"]))
        checkpoint_label = config["output"]["checkpoint_label"]
    except (KeyError, TypeError) as exc:
        sys.exit(f"Tinker config is incomplete: missing or malformed {exc}")
    if checkpoint_label != "trained_unvalidated":
        sys.exit(
            "Tinker output.checkpoint_label must be exactly "
            "'trained_unvalidated'; training alone is not promotion"
        )
    for path in (train_path, eval_path, chat_template):
        if not path.is_file():
            sys.exit(f"Tinker data file missing: {path}")

    sys.path.insert(0, str(ROOT / "src"))
    from tinker_payload import PayloadError, render_files

    try:
        rendered = render_files(train_path, eval_path)
    except PayloadError as exc:
        sys.exit(f"Tinker payload refused: {exc}")

    payload = {}
    for split in ("train", "eval"):
        item = rendered[split]
        payload[split] = {
            "source_path": str(Path(data[split])),
            "source_sha256": digest(resolve(Path(data[split]))),
            "rows": item["rows"],
            "families": item["families"],
            "source_classes": item["source_classes"],
            "upload_sha256": item["sha256"],
            "audit_sha256": item["audit_sha256"],
            "max_content_utf8_bytes": item["max_content_utf8_bytes"],
        }
    return {
        "name": "tinker",
        "config_schema_version": config.get("config_schema_version"),
        "external_service": True,
        "base_model": base_model,
        "renderer": renderer,
        "serving": {
            "chat_template": serving["chat_template"],
            "chat_template_sha256": digest(chat_template),
        },
        "lora_rank": rank,
        "max_tokens": max_tokens,
        "training": config.get("training"),
        "egress": config.get("egress"),
        "packages": packages,
        "runtime_packages": runtime_packages,
        "pricing": pricing,
        "payload": payload,
        "checkpoint_label": checkpoint_label,
    }


def declared_gate_specs(config_path: Path) -> list[dict]:
    """The gates the digest-pinned shared evaluation config declares.

    Resolved through run_gates.load_gate_config so the freezer and the gate
    runner read one definition, including the inherited stop sequences.
    """
    sys.path.insert(0, str(ROOT / "src"))
    import run_gates

    effective, _shared = run_gates.load_gate_config(config_path)
    return [spec for spec in (effective.get("eval_gates") or [])
            if isinstance(spec, dict) and spec.get("name")]


def baseline_snapshot(path: Path, config_path: Path, base_model: str) -> dict:
    report = load_json(path, "before-baseline report")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    errors = []
    if report.get("stage") != "before":
        errors.append("stage is not 'before'")
    if report.get("measured") is not True:
        errors.append("measured is not true")
    if report.get("config", {}).get("sha256") != digest(config_path):
        errors.append("report was not run from this exact Tinker config")
    model = report.get("model") or {}
    if model.get("expected") != base_model:
        errors.append(
            f"expected model is {model.get('expected')!r}, not {base_model!r}"
        )
    if model.get("teacher") != (config.get("evaluation") or {}).get(
        "required_model_config"
    ):
        errors.append("report used a different serving model config")
    if model.get("host") != (config.get("evaluation") or {}).get(
        "required_host_config"
    ):
        errors.append("report used a different serving host config")
    identity = model.get("identity")
    if not isinstance(identity, dict) or identity.get("expected") != base_model:
        errors.append("live endpoint identity is absent or names another model")
    else:
        sys.path.insert(0, str(ROOT / "src"))
        from model_ids import any_match
        identity_matches = any_match(base_model, identity.get("served", []))
        if not identity_matches:
            errors.append(
                "live endpoint identity does not list the required base model"
            )
    evaluation = config.get("evaluation") or {}
    expected_shared_path = resolve(Path(evaluation.get("config", "")))
    expected_shared = {
        "path": str(expected_shared_path),
        "sha256": evaluation.get("config_sha256"),
    }
    if report.get("shared_evaluation_config") != expected_shared:
        errors.append("report used a different shared evaluation config")
    # GATE COMPLETENESS. A non-empty list is not an evaluation: a report
    # carrying only indonesian_voice satisfies every per-gate check above while
    # leaving the edit contract and build health unmeasured, and it would then
    # be pinned as "the baseline". The declared set comes from the SAME
    # digest-pinned shared config the run will be gated against, so the baseline
    # and the after run cannot cover different ground.
    gates = report.get("gates")
    declared = declared_gate_specs(config_path)
    expected_names = {spec["name"] for spec in declared}
    if not isinstance(gates, list) or not gates:
        errors.append("report contains no gate results")
    elif not expected_names:
        errors.append("the shared evaluation config declares no gates")
    else:
        names = [gate.get("name") for gate in gates]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            errors.append(
                "duplicate gate result(s): " + ", ".join(str(d) for d in duplicates)
            )
        missing = sorted(expected_names - set(names))
        unexpected = sorted(set(names) - expected_names)
        if missing:
            errors.append(
                "report is missing declared gate(s): " + ", ".join(missing)
            )
        if unexpected:
            errors.append(
                "report contains undeclared gate(s): "
                + ", ".join(str(u) for u in unexpected)
            )
        by_name = {spec["name"]: spec for spec in declared}
        for gate in gates:
            spec = by_name.get(gate.get("name"))
            if spec is None:
                continue
            label = gate.get("name")
            if gate.get("model_dependent") != bool(spec.get("model_dependent")):
                errors.append(
                    f"{label}: model_dependent is {gate.get('model_dependent')!r}, "
                    f"config declares {bool(spec.get('model_dependent'))!r}"
                )
            if gate.get("threshold") != spec.get("min_pass_rate"):
                errors.append(
                    f"{label}: threshold {gate.get('threshold')!r} is not the "
                    f"configured {spec.get('min_pass_rate')!r}"
                )
            if spec.get("items") is not None and gate.get("items") != spec["items"]:
                errors.append(
                    f"{label}: measured {gate.get('items')!r} items, config "
                    f"declares {spec['items']!r}"
                )
            if not spec.get("model_dependent"):
                continue
            if gate.get("from_fixtures") is not False:
                errors.append(
                    f"{label}: from_fixtures is {gate.get('from_fixtures')!r}; a "
                    "fixture run never touched the model"
                )
            if gate.get("generated_by_model_id") != base_model:
                errors.append(
                    f"{label}: generated by "
                    f"{gate.get('generated_by_model_id')!r}, not {base_model!r}"
                )
            expected_stops = list(spec.get("stop_sequences") or [])
            if list(gate.get("stop_sequences") or []) != expected_stops:
                errors.append(
                    f"{label}: stop sequences {gate.get('stop_sequences')!r} are "
                    f"not the configured {expected_stops!r}"
                )
    if errors:
        sys.exit(
            "before-baseline report is not valid for Tinker execution:\n  "
            + "\n  ".join(errors)
        )
    return {
        "path": str(path),
        "sha256": digest(path),
        "stage": "before",
        "base_model": base_model,
        "model_identity": identity,
        "shared_evaluation_config": report.get("shared_evaluation_config"),
        # Retained in full: a snapshot that strips items, stop sequences and
        # the generating model id cannot later prove what was measured.
        "gates": [
            {
                "name": gate.get("name"),
                "rate": gate.get("rate"),
                "threshold": gate.get("threshold"),
                "items": gate.get("items"),
                "passed": gate.get("passed"),
                "model_dependent": gate.get("model_dependent"),
                "from_fixtures": gate.get("from_fixtures"),
                "generated_by_model_id": gate.get("generated_by_model_id"),
                "stop_sequences": gate.get("stop_sequences"),
            }
            for gate in sorted(gates, key=lambda g: str(g.get("name")))
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--promotion-manifest", type=Path,
                        default=Path("train/RUN_MANIFEST.v1-mechanical.json"),
                        help="mechanical promotion decision and promoted outputs")
    parser.add_argument("--waiver", type=Path,
                        help="required when the gate fails; recorded by reference")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--frozen-at", required=True,
                        help="ISO timestamp, supplied rather than read from the clock")
    parser.add_argument("--backend", choices=("tinker",),
                        help="emit a backend-specific schema-v3 run freeze")
    parser.add_argument("--baseline-report", type=Path,
                        help="valid live before-gate report for the backend base model")
    parser.add_argument("--authorization", type=Path,
                        help="reviewed document that lifts the standing product block")
    parser.add_argument("--preview", action="store_true",
                        help="record a blocked Tinker preflight without authorizing execution")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    for path in (args.corpus, args.config, args.promotion_manifest):
        if not path.exists():
            sys.exit(f"no such file: {path}")
    if args.waiver and not args.waiver.is_file():
        sys.exit(f"no such waiver: {args.waiver}")

    try:
        traces = [json.loads(l) for l in
                  args.corpus.read_text(encoding="utf-8").splitlines() if l.strip()]
    except json.JSONDecodeError as e:
        sys.exit(f"corpus is malformed JSONL: {e}")
    promotion = promotion_snapshot(args.promotion_manifest, args.corpus)
    pinned = ("split_fingerprint", "split_seed", "template_sha256", "repo",
              "quantization", "host", "reasoning_strength", "temperature", "seed")
    provenance = {}
    for field in pinned:
        seen = sorted({json.dumps(t.get("provenance", {}).get(field)) for t in traces})
        provenance[field] = json.loads(seen[0]) if len(seen) == 1 else \
            [json.loads(s) for s in seen]

    # Run the gate. Its exit code is evidence, not a parameter.
    proc = subprocess.run(
        [sys.executable, str(ROOT / "src" / "verify_corpus.py"), str(args.corpus), "--gate"],
        capture_output=True, text=True)
    gate_out = (proc.stdout + proc.stderr).strip()
    errors = corpus_gate_errors(args.corpus)
    failed = proc.returncode != 0
    if failed != bool(errors):
        sys.exit("internal gate inconsistency: subprocess exit and structured "
                 "violations disagree")

    print(f"corpus     {args.corpus}")
    print(f"gate exit  {proc.returncode}  ({'FAILED' if failed else 'passed'})")
    if failed:
        import verify_corpus
        if not verify_corpus.waiver_covers(errors):
            sys.exit(
                "\nThe gate FAILED for reasons the int4 waiver does not cover:\n  "
                + "\n  ".join(errors)
                + "\nFix the corpus; a quantization waiver cannot authorize an "
                  "unrelated failure."
            )
        if not args.waiver:
            sys.exit("\nThe gate FAILED and no --waiver was given. A failing gate may only be\n"
                     "proceeded past under a recorded waiver. Supply one or fix the corpus;\n"
                     "this tool will not write a manifest that omits the authorisation.")
        if resolve(args.waiver).resolve() != REQUIRED_INT4_WAIVER.resolve():
            sys.exit("the failed int4 gate requires the accepted waiver at "
                     f"{REQUIRED_INT4_WAIVER}; got {resolve(args.waiver)}")

    backend = None
    baseline = None
    authorization = None
    execution_authorized = None
    schema_version = SCHEMA_VERSION
    if args.backend == "tinker":
        schema_version = TINKER_SCHEMA_VERSION
        backend = tinker_backend_snapshot(args.config)
        if args.baseline_report:
            baseline_path = resolve(args.baseline_report)
            if not baseline_path.is_file():
                sys.exit(f"no such baseline report: {baseline_path}")
            baseline = baseline_snapshot(
                baseline_path, args.config, backend["base_model"]
            )
        if args.authorization:
            authorization_path = resolve(args.authorization)
            if not authorization_path.is_file():
                sys.exit(f"no such authorization: {authorization_path}")
            authorization = {
                "path": str(authorization_path),
                "sha256": digest(authorization_path),
            }
        execution_authorized = bool(baseline and authorization)
        if execution_authorized and args.preview:
            sys.exit(
                "--preview cannot be combined with a complete baseline and "
                "authorization; write the authorized manifest without it"
            )
        if not execution_authorized and not args.preview:
            sys.exit(
                "Tinker execution is still blocked: both --baseline-report and "
                "--authorization are required for a non-preview schema-v3 freeze.\n"
                "Pass --preview only to record the current blocked preflight state."
            )

    what = (
        "Preview of exact inputs and controls for a possible training run. "
        "It does not authorize or record a started run."
        if backend is not None and not execution_authorized
        else
        "The exact inputs and authorisation a training run started from. "
        "Written before training, never edited after."
    )
    audit = provenance_audit(args.corpus, args.frozen_at)

    manifest = {
        "schema_version": schema_version,
        "_what": what,
        "frozen_at": args.frozen_at,
        "corpus": {
            "path": str(args.corpus),
            "sha256": digest(args.corpus),
            "traces": len(traces),
            "families": len({t.get("family") for t in traces}),
            "provenance": provenance,
        },
        "training_config": {
            "path": str(args.config),
            "sha256": digest(args.config),
        },
        "promotion_manifest": promotion,
        "gate": {
            "command": f"src/verify_corpus.py {args.corpus} --gate",
            "exit_code": proc.returncode,
            "verdict": "FAILED" if failed else "passed",
            "violations": errors,
            "_not_a_pass": "A non-zero exit is the correct and expected state for an "
                           "int4 corpus. The waiver authorises proceeding DESPITE this "
                           "failure; it does not convert it into a pass, and no FP8 "
                           "claim may be made on the strength of it.",
            "output": gate_out.splitlines()[-6:] if gate_out else [],
        },
        "waiver": {
            "path": str(args.waiver) if args.waiver else None,
            "sha256": digest(args.waiver) if args.waiver else None,
        },
        "provenance_audit": audit,
        "claims_this_run_may_NOT_support": [
            "Any statement about performance on real Office documents — the corpus is "
            "entirely source_class: synthetic.",
            "FP8 equivalence, or that the FP8 gate passed. The treatment arm has never "
            "been generated.",
            "Teacher stability under batching other than concurrency 4. See "
            "calibration/VOLATILITY_v3.md.",
        ],
    }
    if backend is not None:
        manifest["backend"] = backend
        manifest["before_baseline"] = baseline
        manifest["training_authorization"] = authorization
        manifest["execution_authorized"] = execution_authorized
        manifest["preview"] = bool(args.preview)
        manifest["_preview_warning"] = (
            None if execution_authorized else
            "PREVIEW ONLY. Training is not justified and no valid same-base "
            "before baseline is pinned. This manifest cannot authorize API calls."
        )

    if not args.write:
        print("\ndry run — pass --write to apply")
        return
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.out}")
    print(f"  corpus  {manifest['corpus']['sha256'][:16]}…  {len(traces)} traces")
    print(f"  config  {manifest['training_config']['sha256'][:16]}…")
    print(f"  promote {manifest['promotion_manifest']['sha256'][:16]}…")
    print(f"  gate    exit {proc.returncode}, waiver {args.waiver}")


if __name__ == "__main__":
    main()
