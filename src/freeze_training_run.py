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
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCHEMA_VERSION = 2
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

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "_what": "The exact inputs and authorisation a training run started from. "
                 "Written before training, never edited after.",
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
        "claims_this_run_may_NOT_support": [
            "Any statement about performance on real Office documents — the corpus is "
            "entirely source_class: synthetic.",
            "FP8 equivalence, or that the FP8 gate passed. The treatment arm has never "
            "been generated.",
            "Teacher stability under batching other than concurrency 4. See "
            "calibration/VOLATILITY_v3.md.",
        ],
    }

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
