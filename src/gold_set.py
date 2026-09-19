"""Gold-set contract and loader for the RSI evaluation harness (Phase 1.1).

The Recursive Self-Improvement Report v2 puts the whole weight of a
self-improvement loop on one component: "What separates a method that compounds
from one that drifts is almost entirely the verifier", and "Your bottleneck is
not the algorithm. It is an Indonesian verifier you trust."

This module is the contract for the data that verifier runs on. It validates;
it never authors. The report's 300-item native gold set is human-authored,
human-approved input, exactly like `data/raw/`, and an agent that invented
Indonesian questions here would be manufacturing the evidence every later
measurement depends on.

Therefore, by construction:

* There is no code path that creates a production gold item.
* There is no code path that approves one. `source.approved_by` is a claim this
  module checks for structure and distinctness; it cannot supply it.
* An empty `data/gold/` is the CORRECT state today, and asking this module to
  gate on it refuses with "gold set is BLOCKED: 0 approved items".
* A structurally perfect synthetic fixture is still ineligible for the
  production gate. Fixtures exist to test the loader, never to stand in for
  gold.

NO NETWORK. NO MODEL. NO TRAINING. `train/TRAINING_BLOCKED.md` is controlling.

See `data/gold/SCHEMA.md` for the record contract this enforces.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parent.parent
GOLD_DIR = ROOT / "data" / "gold"

SCHEMA_VERSION = 1

SPLITS = ("reasoning", "instruction_following", "knowledge", "tool_use")
VERIFIER_TYPES = ("exact_match", "schema", "executor", "judge")
SCORE_ROLES = ("correctness", "style")

# The message the handoff names verbatim. Kept as a constant so the CLI, the
# library and the tests cannot drift apart.
BLOCKED_MESSAGE = "gold set is BLOCKED: 0 approved items"

# Top-level fields a record may carry. Unknown fields are rejected rather than
# ignored: silent schema drift changes what an evaluation MEANS while every
# number keeps reporting successfully.
ALLOWED_FIELDS = frozenset({
    "schema_version", "id", "split", "language", "prompt", "verifier",
    "expected", "score_role", "source_class", "source",
})

REQUIRED_FIELDS = (
    "schema_version", "id", "split", "language", "prompt", "verifier",
    "expected", "score_role", "source",
)

# Enumerated on purpose. "Never add fuzzy matching merely to make an answer
# pass" (SCHEMA.md); an open normalization list is how that happens by degrees.
ALLOWED_NORMALIZATIONS = ("trim", "unicode_nfc", "casefold", "collapse_space")

ALLOWED_RUNNERS = ("python",)

MAX_EXECUTOR_TIMEOUT_SECONDS = 10


class GoldSetError(Exception):
    """A gold set could not be loaded or does not satisfy the contract."""


def die(msg: str, code: int = 2) -> None:
    print(f"\nGOLD SET REFUSED: {msg}", file=sys.stderr)
    sys.exit(code)


# --- field-level checks -----------------------------------------------------


def _nonempty_str(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _parseable_date(value: Any) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    try:
        date.fromisoformat(value.strip())
    except ValueError:
        return False
    return True


def _check_expected(record: dict[str, Any], problems: list[str]) -> None:
    """Verifier-specific `expected`, per SCHEMA.md."""
    verifier = record.get("verifier")
    vtype = verifier.get("type") if isinstance(verifier, dict) else None
    expected = record.get("expected")

    if not isinstance(expected, dict):
        problems.append("expected: must be an object")
        return

    role = record.get("score_role")
    if role == "correctness" and not expected:
        problems.append("expected: must not be empty for a correctness item")

    if vtype == "exact_match":
        if not _nonempty_str(expected.get("answer")):
            problems.append("expected.answer: required for exact_match")
        norms = expected.get("normalization", [])
        if not isinstance(norms, list):
            problems.append("expected.normalization: must be a list")
        else:
            unknown = [n for n in norms if n not in ALLOWED_NORMALIZATIONS]
            if unknown:
                problems.append(
                    f"expected.normalization: unknown {unknown!r}; allowed "
                    f"{list(ALLOWED_NORMALIZATIONS)}"
                )
        check = expected.get("reasoning_check")
        if check is not None:
            if not isinstance(check, dict):
                problems.append("expected.reasoning_check: must be an object")
            elif check.get("type") != "required_claims":
                problems.append(
                    "expected.reasoning_check.type: only 'required_claims' "
                    "is implemented"
                )
            elif not [c for c in check.get("claims", []) if _nonempty_str(c)]:
                problems.append(
                    "expected.reasoning_check.claims: must list at least one claim"
                )

    elif vtype == "schema":
        schema = expected.get("json_schema")
        if not isinstance(schema, dict) or not schema:
            problems.append("expected.json_schema: required for schema verifier")

    elif vtype == "executor":
        runner = expected.get("runner")
        if runner not in ALLOWED_RUNNERS:
            problems.append(
                f"expected.runner: {runner!r} not in {list(ALLOWED_RUNNERS)}"
            )
        timeout = expected.get("timeout_seconds")
        if not isinstance(timeout, (int, float)) or isinstance(timeout, bool):
            problems.append("expected.timeout_seconds: required, numeric")
        elif not 0 < timeout <= MAX_EXECUTOR_TIMEOUT_SECONDS:
            problems.append(
                f"expected.timeout_seconds: must be within "
                f"(0, {MAX_EXECUTOR_TIMEOUT_SECONDS}]"
            )
        tests = expected.get("tests")
        if not isinstance(tests, list) or not tests:
            problems.append("expected.tests: required, non-empty")

    elif vtype == "judge":
        rubric = expected.get("rubric")
        if not isinstance(rubric, list) or not [r for r in rubric if _nonempty_str(r)]:
            problems.append("expected.rubric: required for judge")


def _check_source(record: dict[str, Any], problems: list[str]) -> None:
    source = record.get("source")
    if not isinstance(source, dict) or not source:
        problems.append("source: required, non-empty provenance object")
        return
    kind = source.get("kind")
    if not _nonempty_str(kind):
        problems.append("source.kind: required")
    elif kind not in ("human_authored", "synthetic_fixture"):
        problems.append(
            f"source.kind: {kind!r} is neither 'human_authored' nor "
            "'synthetic_fixture'"
        )
    if not _nonempty_str(source.get("evidence_ref")):
        problems.append("source.evidence_ref: required")


def validate_record(record: Any, *, index: int | None = None) -> list[str]:
    """Structural problems with one record. Empty list means structurally valid.

    Structural validity is NOT production eligibility: a synthetic fixture can
    be perfectly valid here and still never gate anything. See
    `production_problems`.
    """
    where = "" if index is None else f"line {index}: "
    if not isinstance(record, dict):
        return [f"{where}record must be a JSON object"]

    problems: list[str] = []

    unknown = sorted(set(record) - ALLOWED_FIELDS)
    if unknown:
        problems.append(f"{where}unknown field(s) {unknown!r}")

    for field in REQUIRED_FIELDS:
        if field not in record:
            problems.append(f"{where}{field}: required")

    if record.get("schema_version") != SCHEMA_VERSION:
        problems.append(
            f"{where}schema_version: expected {SCHEMA_VERSION}, "
            f"got {record.get('schema_version')!r}"
        )

    if not _nonempty_str(record.get("id")):
        problems.append(f"{where}id: required, non-empty")

    if record.get("split") not in SPLITS:
        problems.append(
            f"{where}split: {record.get('split')!r} not in {list(SPLITS)}"
        )

    # Translated English material does not become native gold by editing this
    # field, but a wrong value here is still a refusal.
    if record.get("language") != "id":
        problems.append(
            f"{where}language: must be 'id', got {record.get('language')!r}"
        )

    if not _nonempty_str(record.get("prompt")):
        problems.append(f"{where}prompt: required, non-empty")

    verifier = record.get("verifier")
    if not isinstance(verifier, dict):
        problems.append(f"{where}verifier: required object")
    else:
        vtype = verifier.get("type")
        if vtype not in VERIFIER_TYPES:
            problems.append(
                f"{where}verifier.type: {vtype!r} not in {list(VERIFIER_TYPES)}"
            )
        config = verifier.get("config", {})
        if not isinstance(config, dict):
            problems.append(f"{where}verifier.config: must be an object")

    role = record.get("score_role")
    if role not in SCORE_ROLES:
        problems.append(
            f"{where}score_role: {role!r} not in {list(SCORE_ROLES)}"
        )

    # The quarantine rule, stated in the report (§A: the model as judge is the
    # least trustworthy verifier) and in SCHEMA.md. A judge item may never
    # contribute to a correctness score.
    if isinstance(verifier, dict) and verifier.get("type") == "judge" \
            and role == "correctness":
        problems.append(
            f"{where}judge verifier with score_role 'correctness': a judge item "
            "is style-only until judge-human agreement is calibrated"
        )

    source_class = record.get("source_class", "internal")
    if not _nonempty_str(source_class):
        problems.append(f"{where}source_class: must be a non-empty string")

    sub: list[str] = []
    _check_expected(record, sub)
    _check_source(record, sub)
    problems.extend(f"{where}{p}" for p in sub)

    return problems


def production_problems(record: dict[str, Any]) -> list[str]:
    """Why this record may not gate a production measurement.

    Separate from `validate_record` on purpose: structural validity is a
    property of the file, eligibility is a property of the human review behind
    it, and conflating them is how a fixture becomes gold.
    """
    problems: list[str] = []
    source = record.get("source")
    if not isinstance(source, dict):
        return ["source: missing"]

    if source.get("kind") != "human_authored":
        problems.append(
            f"source.kind: {source.get('kind')!r} is not 'human_authored'"
        )
    if record.get("source_class", "internal") == "synthetic":
        problems.append("source_class: 'synthetic' is never production gold")

    for field in ("author_ref", "approved_by", "evidence_ref"):
        if not _nonempty_str(source.get(field)):
            problems.append(f"source.{field}: required for production")
    for field in ("authored_at", "approved_at"):
        if not _parseable_date(source.get(field)):
            problems.append(f"source.{field}: required, ISO date")

    author = str(source.get("author_ref") or "").strip()
    approver = str(source.get("approved_by") or "").strip()
    if author and approver and author == approver:
        # An author approving their own item is not an independent review.
        problems.append(
            "source.approved_by: must be distinct from source.author_ref"
        )

    return problems


# --- loading ----------------------------------------------------------------


def parse_jsonl(text: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            records.append(json.loads(stripped))
        except json.JSONDecodeError as error:
            raise GoldSetError(f"line {number}: invalid JSON: {error}") from error
    return records


def load_records(paths: Iterable[Path]) -> list[dict[str, Any]]:
    """Every record from every file, in file then line order."""
    out: list[dict[str, Any]] = []
    for path in paths:
        out.extend(parse_jsonl(Path(path).read_text(encoding="utf-8")))
    return out


def gold_files(directory: Path) -> list[Path]:
    return sorted(p for p in Path(directory).glob("*.jsonl") if p.is_file())


class GoldSet:
    """A loaded gold set and everything that is wrong with it."""

    def __init__(self, records: list[dict[str, Any]], source: str) -> None:
        self.source = source
        self.records = records
        self.problems: list[str] = []
        self.valid: list[dict[str, Any]] = []
        self.eligible: list[dict[str, Any]] = []
        self.ineligible: list[tuple[str, list[str]]] = []

        seen: dict[str, int] = {}
        for index, record in enumerate(records, start=1):
            issues = validate_record(record, index=index)
            if issues:
                self.problems.extend(issues)
                continue
            rid = str(record["id"])
            if rid in seen:
                self.problems.append(
                    f"line {index}: duplicate id {rid!r} (first seen at line "
                    f"{seen[rid]})"
                )
                continue
            seen[rid] = index
            self.valid.append(record)

            blockers = production_problems(record)
            if blockers:
                self.ineligible.append((rid, blockers))
            else:
                self.eligible.append(record)

    @property
    def approved_count(self) -> int:
        return len(self.eligible)

    def counts_by_split(self) -> dict[str, int]:
        counts = {s: 0 for s in SPLITS}
        for record in self.eligible:
            counts[record["split"]] += 1
        return counts

    def correctness_items(self) -> list[dict[str, Any]]:
        """Eligible items that may contribute to a correctness score."""
        return [r for r in self.eligible if r.get("score_role") == "correctness"]

    def summary(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "records": len(self.records),
            "structurally_valid": len(self.valid),
            "approved_items": self.approved_count,
            "ineligible_items": len(self.ineligible),
            "structural_problems": len(self.problems),
            "by_split": self.counts_by_split(),
            "correctness_items": len(self.correctness_items()),
            "gate_ready": False,  # set by gate(); never true by construction here
        }


def load(directory: Path = GOLD_DIR) -> GoldSet:
    """Load every .jsonl under `directory`. Does not gate; see `gate`."""
    directory = Path(directory)
    if not directory.exists():
        raise GoldSetError(f"gold directory does not exist: {directory}")
    files = gold_files(directory)
    records = load_records(files)
    return GoldSet(records, source=str(directory))


def gate(gold: GoldSet, *, require_splits: Iterable[str] = SPLITS,
         minimum_per_split: int = 1) -> list[str]:
    """Reasons this gold set may not gate a measurement. Empty means ready.

    Returns reasons rather than raising so a caller can report all of them at
    once; the CLI turns a non-empty list into a non-zero exit.
    """
    reasons: list[str] = []
    if gold.problems:
        reasons.append(
            f"{len(gold.problems)} structural problem(s); first: {gold.problems[0]}"
        )
    if gold.approved_count == 0:
        reasons.append(BLOCKED_MESSAGE)
        return reasons
    counts = gold.counts_by_split()
    for split in require_splits:
        if counts.get(split, 0) < minimum_per_split:
            reasons.append(
                f"split {split!r}: {counts.get(split, 0)} approved items, "
                f"minimum {minimum_per_split}"
            )
    if not gold.correctness_items():
        reasons.append(
            "no correctness items: a judge-only set cannot gate correctness"
        )
    return reasons


# --- CLI --------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Validate the native Indonesian gold set. Never authors it."
    )
    parser.add_argument("--dir", default=str(GOLD_DIR),
                        help="directory of .jsonl gold files")
    parser.add_argument("--gate", action="store_true",
                        help="exit non-zero unless the set can gate a measurement")
    parser.add_argument("--json", action="store_true", help="machine-readable summary")
    args = parser.parse_args(argv)

    try:
        gold = load(Path(args.dir))
    except GoldSetError as error:
        die(str(error))
        return

    summary = gold.summary()
    reasons = gate(gold) if args.gate else []
    summary["gate_ready"] = args.gate and not reasons

    if args.json:
        print(json.dumps({**summary, "gate_reasons": reasons},
                         ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(f"gold set: {summary['source']}")
        print(f"  records                {summary['records']}")
        print(f"  structurally valid     {summary['structurally_valid']}")
        print(f"  approved items         {summary['approved_items']}")
        print(f"  ineligible items       {summary['ineligible_items']}")
        print(f"  structural problems    {summary['structural_problems']}")
        for split, n in summary["by_split"].items():
            print(f"    {split:<24} {n}")
        for rid, blockers in gold.ineligible[:10]:
            print(f"  ineligible {rid}: {blockers[0]}")
        for problem in gold.problems[:10]:
            print(f"  problem: {problem}")

    if args.gate and reasons:
        die("; ".join(reasons))


if __name__ == "__main__":
    main()
