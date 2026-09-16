"""Bind a HUMAN approval record to an exact case set, or refuse.

`harness_eval.aggregate` refuses to treat an unapproved case set as a real
measurement, and `approved: true` is a boolean in a YAML file — which means
without this module the gate is one edit away from being defeated by whoever is
editing. This makes the flag evidence: it may be true only when a completed
human record exists whose digest matches the exact bytes that were reviewed.

MODELLED ON docs/licences/. Same shape, same refusals, same reason: a review of
something is only evidence about that something, so the record binds to a
digest and any change to the reviewed bytes invalidates it automatically.

WHAT AN AGENT MAY DO: prepare the record, fill the case-set name, digest, count
and split, and check that everything lines up. WHAT IT MAY NOT DO: supply the
reviewer identity, the review date, or the confirmations. An approval an agent
can grant itself is not an approval.

NO NETWORK, NO MODEL, NO OFFICE ACTION.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover
    raise SystemExit("pyyaml is required: pip install -r requirements.txt")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import harness_eval as he                                      # noqa: E402

RECORD_DIR = ROOT / "docs" / "case_sets"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

# Every confirmation must be present AND true. A record missing one is not a
# weaker approval; it is a review that did not finish.
REQUIRED_CONFIRMATIONS = (
    "privacy_redaction_reviewed",
    "product_representativeness_reviewed",
    "leakage_reviewed",
    "scorer_inputs_complete",
    "expected_outcomes_correct",
    "both_models_and_harnesses_executable",
    "at_least_320_independent_cases",
    "no_training_or_calibration_overlap",
)

# Anything matching these is a placeholder, not a person. Kept explicit so a
# half-filled template cannot pass by looking filled.
_PLACEHOLDER_RE = re.compile(
    r"^(REPLACE_WITH_[A-Z_]+|TBD|TODO|N/?A|NONE|UNKNOWN|-+|\?+)$", re.IGNORECASE)
# An agent must not appear as the reviewer. This is a floor, not a security
# control: someone determined to lie can type a human name. It stops the
# ACCIDENT of an automated reviewer field, which is the realistic failure.
_NON_HUMAN_RE = re.compile(
    r"(claude|gpt|chatgpt|copilot|assistant|\bai\b|agent|bot|automation|"
    r"anthropic|openai|script|generated)", re.IGNORECASE)

MATCHED = "MATCHED"
UNAPPROVED = "UNAPPROVED"


class ApprovalError(he.HarnessEvalError):
    """A fail-closed approval error."""


def _die(message: str, code: int = 2) -> None:
    label = "CASE-SET APPROVAL REFUSED" if code == 2 else "CASE SET NOT APPROVED"
    print(f"{label}: {message}", file=sys.stderr)
    raise SystemExit(code)


def parse_record(path: Path) -> dict[str, Any]:
    """Read the YAML block out of a human-written markdown record."""
    if not path.is_file():
        raise ApprovalError(f"no approval record at {path}")
    text = path.read_text(encoding="utf-8")
    blocks = re.findall(r"```ya?ml\n(.*?)```", text, re.DOTALL)
    if not blocks:
        raise ApprovalError(
            f"{path} contains no ```yaml block; the record is the block, and "
            "prose around it is not a decision")
    try:
        record = yaml.safe_load(blocks[0]) or {}
    except yaml.YAMLError as exc:
        raise ApprovalError(f"{path}: unreadable YAML block: {exc}") from exc
    if not isinstance(record, dict):
        raise ApprovalError(f"{path}: the YAML block must be a mapping")
    return record


def validate_record(record: dict[str, Any], *, source: str = "<record>") -> dict:
    """Check a record is COMPLETE and human, or refuse. Returns it unchanged."""
    for key in ("case_set", "case_set_sha256", "cases", "split",
                "reviewed_by", "reviewer_role", "reviewed_at", "confirmations"):
        if key not in record:
            raise ApprovalError(f"{source}: missing required field {key!r}")

    digest = str(record["case_set_sha256"])
    if not SHA256_RE.match(digest):
        raise ApprovalError(
            f"{source}: case_set_sha256 must be 64 lowercase hex, got "
            f"{digest!r}")
    if digest == "0" * 64:
        raise ApprovalError(
            f"{source}: case_set_sha256 is the template's all-zero placeholder; "
            "it binds to nothing")

    for key in ("reviewed_by", "reviewer_role", "reviewed_at"):
        value = str(record[key] or "").strip()
        if not value or _PLACEHOLDER_RE.match(value):
            raise ApprovalError(
                f"{source}: {key} is a placeholder ({record[key]!r}). A review "
                "nobody signed is not a review.")
    reviewer = str(record["reviewed_by"])
    if _NON_HUMAN_RE.search(reviewer):
        raise ApprovalError(
            f"{source}: reviewed_by {reviewer!r} names an automated actor. The "
            "approval gate exists so that a human decided; an agent approving "
            "its own case set is the failure this refuses.")

    confirmations = record["confirmations"]
    if not isinstance(confirmations, dict):
        raise ApprovalError(f"{source}: confirmations must be a mapping")
    missing = [c for c in REQUIRED_CONFIRMATIONS if c not in confirmations]
    if missing:
        raise ApprovalError(
            f"{source}: confirmations missing {sorted(missing)}; an incomplete "
            "review is not a weaker approval, it is an unfinished one")
    unconfirmed = [c for c in REQUIRED_CONFIRMATIONS
                   if confirmations[c] is not True]
    if unconfirmed:
        raise ApprovalError(
            f"{source}: not confirmed: {sorted(unconfirmed)}")

    cases = record["cases"]
    if not isinstance(cases, int) or cases < 1:
        raise ApprovalError(f"{source}: cases must be a positive integer")
    return record


def bind_to_case_set(record: dict[str, Any], case_set: dict[str, Any],
                     *, source: str = "<record>") -> dict[str, Any]:
    """Check the record describes THIS case set, or refuse."""
    validate_record(record, source=source)
    he.validate_case_set(case_set)
    digest = he.case_set_digest(case_set)

    if record["case_set"] != case_set.get("name"):
        raise ApprovalError(
            f"{source}: record approves {record['case_set']!r} but this set is "
            f"{case_set.get('name')!r}")
    if record["case_set_sha256"] != digest:
        raise ApprovalError(
            f"{source}: the approved digest {record['case_set_sha256'][:12]}... "
            f"is not this set's {digest[:12]}.... The reviewed bytes and these "
            "bytes are different, so the review says nothing about them.")
    if record["cases"] != len(case_set["cases"]):
        raise ApprovalError(
            f"{source}: record approves {record['cases']} case(s); this set has "
            f"{len(case_set['cases'])}")
    if record.get("split") != case_set.get("split"):
        raise ApprovalError(
            f"{source}: record approves split {record.get('split')!r}; this set "
            f"declares {case_set.get('split')!r}")
    return {
        "status": MATCHED,
        "case_set": record["case_set"],
        "case_set_sha256": digest,
        "cases": record["cases"],
        "reviewed_by": record["reviewed_by"],
        "reviewer_role": record["reviewer_role"],
        "reviewed_at": str(record["reviewed_at"]),
        "confirmations": {c: True for c in REQUIRED_CONFIRMATIONS},
    }


def record_path_for(name: str) -> Path:
    return RECORD_DIR / f"{name}.md"


def approval_for(case_set: dict[str, Any]) -> dict[str, Any] | None:
    """The binding approval for this set, or None. Never raises on absence:
    an unapproved set is an ordinary state, not an error."""
    name = str(case_set.get("name") or "")
    path = record_path_for(name)
    if not path.is_file():
        return None
    return bind_to_case_set(parse_record(path), case_set, source=str(path))


# Production approval is a CONJUNCTION, and each term is here because trusting
# the previous one alone failed somewhere: `leakage_reviewed: true` is a person
# saying they looked, and this requires the thing they looked at.
MINIMUM_ELIGIBLE_CASES = 320


def production_approval(case_set: dict[str, Any], *,
                        metric_contract: dict[str, Any] | None = None,
                        registry_path: Path | None = None) -> dict[str, Any]:
    """Everything a set must satisfy before it can decide anything.

    Returns a verdict rather than raising, because "not approved" is the normal
    state and a caller needs to report WHY rather than crash.
    """
    import case_set_registry as registry
    import score_capability as sc
    import verify_case_set_leakage_evidence as leakage

    problems: list[str] = []

    try:
        human = approval_for(case_set)
    except he.HarnessEvalError as exc:
        human = None
        problems.append(f"approval record does not bind: {exc}")
    if human is None and not problems:
        problems.append("no human approval record")

    evidence = None
    try:
        evidence = leakage.evidence_for(case_set)
    except he.HarnessEvalError as exc:
        problems.append(f"leakage evidence does not bind: {exc}")
    if evidence is None and "leakage evidence does not bind" not in \
            " ".join(problems):
        problems.append(
            "no leakage-evidence record. confirmations.leakage_reviewed is a "
            "person saying they looked; this is what they looked at.")
    elif evidence:
        if evidence["required_unmet"]:
            problems.append(
                f"leakage components not clean: {evidence['required_unmet']}")
        if evidence["office_components_blocked"]:
            problems.append(
                "document/full-case leakage coverage is unavailable "
                f"({evidence['office_components_blocked']}): no versioned "
                "extractor exists, so a reused document inside a rephrased "
                "prompt would not have been detected")

    eligible = [c for c in case_set["cases"]
                if c.get("eligible", True) is not False]
    if len(eligible) < MINIMUM_ELIGIBLE_CASES:
        problems.append(
            f"{len(eligible)} eligible case(s), below the "
            f"{MINIMUM_ELIGIBLE_CASES} the metric contract requires")

    digest = he.case_set_digest(case_set)
    allowed, why = registry.may_be_held_out(digest, registry.load(registry_path))
    if not allowed:
        problems.append(f"registered as development/calibration: {why}")

    if metric_contract is not None:
        spec, contract_digest = sc.load_contract()
        if metric_contract.get("digest") != contract_digest:
            problems.append(
                "metric-contract digest does not match the shipped contract; "
                "the set was qualified against a different definition")

    return {
        "approved_for_production": not problems,
        "case_set": case_set.get("name"),
        "case_set_sha256": digest,
        "human_approval": human,
        "leakage_evidence": evidence,
        "eligible_cases": len(eligible),
        "problems": problems,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cases", type=Path, help="the case-set file to check")
    parser.add_argument("--record", type=Path, default=None,
                        help="approval record (default: docs/case_sets/<name>.md)")
    args = parser.parse_args()
    try:
        case_set = he.load_case_set(args.cases)
        path = args.record or record_path_for(str(case_set["name"]))
        if not path.is_file():
            _die(f"no approval record at {path}. An agent may prepare one from "
                 "docs/case_sets/TEMPLATE.md; only a human may complete it.",
                 code=1)
        result = bind_to_case_set(parse_record(path), case_set, source=str(path))
    except he.HarnessEvalError as exc:
        _die(str(exc))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
