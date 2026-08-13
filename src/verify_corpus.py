"""Check a generated corpus before anything trains on it.

    python3 src/verify_corpus.py data/raw/*.jsonl              # smoke test
    python3 src/verify_corpus.py --gate data/raw/*.jsonl       # training gate

generate.py enforces splits at write time, but a corpus is assembled from many
runs over weeks — some possibly predating a manifest change, or hand-edited.
This re-derives the invariants from the data on disk, which is the only thing
training actually consumes.

Two strictness levels, because a 12-trace smoke test and a full corpus are
judged differently. Without --gate, coverage and teacher quantization are
reported. With --gate, they FAIL — a corpus is not training-ready if it misses
strata or carries traces from a quantized teacher, since quantization error in
the teacher is baked permanently into the student.

Split balance is reported as expected-vs-actual, weighted by how many families
each split actually holds. Deviation from a nominal 70/20/10 is not itself a
fault: splits partition FAMILIES, and families differ in how many prompts they
contribute, so trace counts legitimately drift from family counts. The check
is against the manifest's own weighting, not a global ratio.

Exits non-zero on any violation so it can gate a training run.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import splits as splits_module

ROOT = Path(__file__).resolve().parent.parent


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


# Teacher quantizations that must never reach training. "remote" means a
# gateway whose precision we do not control; Ollama-backed ones are ~Q4.
UNTRAINABLE_QUANTIZATION = {"remote", "int4", "int4_mlx", "int4_ollama"}


def volatile_families() -> dict[str, list[str]]:
    """Families a measured control arm showed flipping a metric between runs.

    Read from calibration/noise_floor.<arm>.json rather than hardcoded, so the
    rule follows the measurement instead of a comment going stale.
    """
    flipping = {}
    for path in sorted((ROOT / "calibration").glob("noise_floor.*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for family, detail in (payload.get("volatile_families") or {}).items():
            metrics = sorted(detail.get("metric_flips") or {})
            if metrics:
                flipping.setdefault(family, []).extend(metrics)
    return {f: sorted(set(m)) for f, m in flipping.items()}


def split_balance(records: list[dict], manifest: dict) -> list[tuple]:
    """Expected vs actual trace counts per split.

    Expected is derived from the manifest's own family weighting — the share
    of families assigned to each split — not from a hardcoded 70/20/10. A
    family contributing many prompts legitimately pulls trace counts away
    from family counts, so this is a drift signal, not a pass/fail.
    """
    family_counts = Counter(manifest["assignments"].values())
    total_families = sum(family_counts.values())
    actual = Counter(r.get("split", "?") for r in records)
    total_traces = len(records)

    rows = []
    for split in sorted(family_counts):
        share = family_counts[split] / total_families
        rows.append((
            split,
            family_counts[split],
            share * 100,
            actual.get(split, 0),
            (actual.get(split, 0) / total_traces * 100) if total_traces else 0.0,
        ))
    return rows


def check(records: list[dict], manifest: dict, gate: bool = False) -> list[str]:
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

    for record in records:
        if not str(record.get("completion", "")).strip():
            errors.append(f"{record['_source']}: empty completion")

    # Provenance must survive into every trace, or a later regression cannot
    # be attributed to the run that caused it.
    for record in records:
        provenance = record.get("provenance", {})
        for field in ("split_seed", "split_fingerprint", "teacher", "quantization"):
            if not provenance.get(field):
                errors.append(f"{record['_source']}: provenance missing {field}")
                break

    if not gate:
        return errors

    # --- training-gate-only checks -------------------------------------
    # A corpus missing whole strata trains a model that has never seen them,
    # and an eval over those strata says nothing.
    kinds = manifest["kinds"]
    all_kinds = set(kinds.values())
    covered = {kinds.get(r.get("family", ""), "?") for r in records}
    uncovered = sorted(all_kinds - covered)
    if uncovered:
        errors.append(
            f"{len(uncovered)}/{len(all_kinds)} kinds have no traces: "
            + ", ".join(uncovered[:10]) + (" ..." if len(uncovered) > 10 else "")
        )

    # Every kind should also appear in every split it was assigned to, or the
    # eval set silently omits strata the model was trained on.
    assigned = defaultdict(set)
    for family_id, split in manifest["assignments"].items():
        assigned[kinds[family_id]].add(split)
    present = defaultdict(set)
    for record in records:
        present[kinds.get(record.get("family", ""), "?")].add(record.get("split"))
    thin = sorted(k for k in covered if k in assigned and assigned[k] - present[k])
    if thin:
        errors.append(
            f"{len(thin)} kind(s) missing traces in an assigned split: "
            + ", ".join(f"{k}(-{sorted(assigned[k] - present[k])})" for k in thin[:6])
            + (" ..." if len(thin) > 6 else "")
        )

    # Signed waiver condition (calibration/INT4_WAIVER.md): a family whose
    # metric flips between runs must be represented by replicates or not at
    # all. One arbitrary trace from a bimodal family records whichever mode
    # that run happened to land in, and training on it teaches the coin toss
    # as if it were the teacher's judgment.
    flipping = volatile_families()
    if flipping:
        counts = Counter(r.get("family") for r in records)
        singletons = {f: m for f, m in flipping.items() if counts.get(f, 0) == 1}
        if singletons:
            errors.append(
                f"{len(singletons)} volatile family/families present with a SINGLE "
                "trace — replicate or exclude them (INT4_WAIVER.md): "
                + ", ".join(f"{f}({'/'.join(m)})" for f, m in sorted(singletons.items()))
            )

    # Quantization error in the teacher is baked permanently into the student.
    bad_quant = Counter(
        r.get("provenance", {}).get("quantization")
        for r in records
        if r.get("provenance", {}).get("quantization") in UNTRAINABLE_QUANTIZATION
    )
    if bad_quant:
        errors.append(
            f"{sum(bad_quant.values())} trace(s) from a quantized teacher "
            f"({dict(bad_quant)}) — regenerate at fp8 before training, or proceed "
            "under a SIGNED waiver (calibration/INT4_WAIVER.md). This gate stays "
            "FAILED either way: a waiver authorises proceeding despite the failure, "
            "it does not convert it into a pass, and no FP8 claim may be made."
        )

    return errors


def report(records: list[dict], manifest: dict, gate: bool) -> None:
    by_quant = Counter(r.get("provenance", {}).get("quantization", "?") for r in records)
    kinds = manifest["kinds"]
    all_kinds = set(kinds.values())
    covered = {kinds.get(r.get("family", ""), "?") for r in records}

    print(f"{len(records)} traces  ({'TRAINING GATE' if gate else 'smoke test'})")
    print(f"  quantization: {dict(by_quant)}")
    print(f"  kinds       : {len(covered)}/{len(all_kinds)} covered")

    print("  split balance (expected from family weighting vs actual traces):")
    for split, fam_n, fam_pct, trace_n, trace_pct in split_balance(records, manifest):
        print(f"    {split:<10} families {fam_n:>4} ({fam_pct:5.1f}%)   "
              f"traces {trace_n:>5} ({trace_pct:5.1f}%)")

    missing = sorted(all_kinds - covered)
    if missing:
        head = ", ".join(missing[:6])
        print(f"  uncovered   : {len(missing)} kinds — {head}"
              + (" ..." if len(missing) > 6 else ""))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--gate", action="store_true",
                        help="training-readiness gate: also fail on uncovered kinds "
                             "and traces from a quantized teacher")
    args = parser.parse_args()

    manifest = splits_module.load()
    splits_module.verify(manifest)
    records = load_corpus(args.paths)
    if not records:
        sys.exit("no traces found")

    report(records, manifest, args.gate)
    errors = check(records, manifest, gate=args.gate)
    if errors:
        print(f"\nFAILED — {len(errors)} violation(s):")
        for error in errors[:20]:
            print(f"  {error}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more")
        sys.exit(1)

    if args.gate:
        print("\nGATE PASSED — every kind covered, fp8 teacher, no leakage, "
              "provenance intact. Training may proceed.")
    else:
        print("\nOK — no split leakage, all families known, provenance intact. "
              "Run with --gate before training.")


if __name__ == "__main__":
    main()
