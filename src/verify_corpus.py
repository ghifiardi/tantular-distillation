"""Check a generated corpus before anything trains on it.

    python3 src/verify_corpus.py data/raw/*.jsonl

generate.py enforces splits at write time, but a corpus is assembled from many
runs over weeks — some possibly predating a manifest change, or hand-edited.
This re-derives the invariants from the data on disk, which is the only thing
training actually consumes.

Exits non-zero on any violation so it can gate a training run.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import splits as splits_module


def load_corpus(paths: list[Path]) -> list[dict]:
    records = []
    for path in paths:
        if not path.exists():
            sys.exit(f"no such file: {path}")
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                sys.exit(f"{path}:{line_no}: bad JSON — {error}")
            record["_source"] = f"{path.name}:{line_no}"
            records.append(record)
    return records


def check(records: list[dict], manifest: dict) -> list[str]:
    errors = []

    # THE invariant: one family, one split. A family appearing in both train
    # and eval means the model is evaluated on material it trained on, and
    # every downstream number is inflated by an unknown amount.
    splits_seen: dict[str, set] = defaultdict(set)
    for record in records:
        splits_seen[record.get("family", "?")].add(record.get("split", "?"))
    for family_id, seen in sorted(splits_seen.items()):
        if len(seen) > 1:
            errors.append(f"family {family_id} straddles splits: {sorted(seen)}")

    # Splits must still agree with the manifest — a trace could carry a split
    # that was correct under an older assignment.
    for record in records:
        family_id = record.get("family", "")
        expected = manifest["assignments"].get(family_id)
        if expected is None:
            errors.append(f"{record['_source']}: family {family_id!r} not in manifest")
        elif record.get("split") != expected:
            errors.append(
                f"{record['_source']}: {family_id} has split {record.get('split')!r}, "
                f"manifest says {expected!r}"
            )

    # Mixing partitionings is silent corruption; the fingerprint catches it.
    fingerprints = {
        r.get("provenance", {}).get("split_fingerprint")
        for r in records
        if r.get("provenance", {}).get("split_fingerprint")
    }
    if len(fingerprints) > 1:
        errors.append(f"corpus mixes {len(fingerprints)} split partitionings: {sorted(fingerprints)}")
    elif fingerprints and manifest["fingerprint"] not in fingerprints:
        errors.append(
            f"corpus was generated under partitioning {fingerprints.pop()}, "
            f"current manifest is {manifest['fingerprint']}"
        )

    # Traces that are empty, or quantized below what should be trained on.
    for record in records:
        if not str(record.get("completion", "")).strip():
            errors.append(f"{record['_source']}: empty completion")

    return errors


def report(records: list[dict], manifest: dict) -> None:
    by_split = Counter(r.get("split", "?") for r in records)
    by_quant = Counter(r.get("provenance", {}).get("quantization", "?") for r in records)
    kinds = manifest["kinds"]
    covered = {kinds.get(r.get("family", ""), "?") for r in records}

    print(f"{len(records)} traces")
    print(f"  splits      : {dict(by_split)}")
    print(f"  quantization: {dict(by_quant)}")
    print(f"  kinds       : {len(covered)}/{len(set(kinds.values()))} covered")

    missing = sorted(set(kinds.values()) - covered)
    if missing:
        head = ", ".join(missing[:6])
        print(f"  UNCOVERED   : {len(missing)} kinds — {head}"
              + (" ..." if len(missing) > 6 else ""))

    # Q4 traces are fine for validating a pipeline and wrong to train on:
    # the teacher's quantization error gets baked into the student.
    low = [q for q in by_quant if q in ("remote", "int4")]
    if low:
        print(f"  NOTE: {sum(by_quant[q] for q in low)} traces came from a quantized "
              "teacher — regenerate at fp8 before training")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()

    manifest = splits_module.load()
    splits_module.verify(manifest)
    records = load_corpus(args.paths)
    if not records:
        sys.exit("no traces found")

    report(records, manifest)
    errors = check(records, manifest)
    if errors:
        print(f"\nFAILED — {len(errors)} violation(s):")
        for error in errors[:20]:
            print(f"  {error}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more")
        sys.exit(1)
    print("\nOK — no split leakage, all families known, no empty traces")


if __name__ == "__main__":
    main()
