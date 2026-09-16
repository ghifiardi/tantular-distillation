"""Draft a case set from existing rows, and say what a human still owes.

Converting `prompts/*.jsonl` into the M10 case-set shape is mechanical for the
fields that exist and IMPOSSIBLE for the ones that do not. `must_preserve`,
`must_not_change`, `allowed_new_facts` and a target assertion are statements
about what the document means; inventing them would manufacture the very
declarations the scorer trusts, and every later measurement would inherit the
invention.

So this tool converts what it can, marks the rest as OWED, and always emits
`approved: false`. It cannot approve, cannot name a reviewer, and cannot drop a
case it could not convert -- a silently dropped case is a case set that looks
complete and is not.

NO NETWORK, NO MODEL, NO OFFICE ACTION. It reads only the paths it is given,
and deliberately does not touch the gitignored local corpora: leakage evidence
needs a tracked manifest that does not exist yet (see docs).
"""
from __future__ import annotations

import argparse
import hashlib
import json
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
import score_capability as sc                                  # noqa: E402

# What a human must supply per case, by expected action. The tool reports these
# as owed; it never fills them.
HUMAN_OWED = {
    sc.ACTION_EDIT: ("must_preserve", "must_not_change", "allowed_new_facts",
                     "target_assertion"),
    sc.ACTION_NO_EDIT: ("must_state_absence", "allowed_new_facts"),
}

# Set-level facts nobody can derive from a row.
SET_OWED = ("privacy_redaction_status", "reviewer_provenance",
            "training_calibration_exclusion_proof")


def source_digest(row: dict[str, Any]) -> str:
    return hashlib.sha256(he.canonical_json(row)).hexdigest()


def infer_expected_action(row: dict[str, Any]) -> str:
    """Read the action the SOURCE declared. Never guessed: a row that does not
    say is reported as owed rather than assumed to be an edit."""
    declared = row.get("expect")
    if declared == "absent":
        return sc.ACTION_NO_EDIT
    if declared == "edit":
        return sc.ACTION_EDIT
    return ""


def live_slice_problems(row: dict[str, Any], action: str) -> list[str]:
    """Why this case could not run the ONE-EDIT live Word protocol."""
    problems: list[str] = []
    if action == sc.ACTION_NO_EDIT:
        problems.append(
            "expected_action no_edit: OfficeLiveExecutor requires exactly one "
            "edit per case, so a correct decline cannot be executed live "
            "(contract live_support.no_edit is false)")
    if str(row.get("stratum") or "") == "multi-edit":
        problems.append(
            "declares stratum multi-edit: the live slice applies one edit, and "
            "Office.js has no transaction across context.sync() for a batch")
    if not str(row.get("document") or "").strip():
        problems.append("no document: there is nothing to edit")
    return problems


def convert_row(row: dict[str, Any], index: int) -> dict[str, Any]:
    case_id = str(row.get("id") or "").strip()
    owed: list[str] = []
    if not case_id:
        # Not invented. A generated id would be stable only within this run,
        # and the id is the join key across four arms.
        case_id = f"UNNAMED-{index:04d}"
        owed.append("case_id: the source row declares no stable id")

    action = infer_expected_action(row)
    if not action:
        owed.append("expected_action: the source row does not declare "
                    "expect: edit | absent")
        action = sc.ACTION_EDIT          # for reporting only; flagged as owed

    for field in HUMAN_OWED[action]:
        if field not in row:
            owed.append(f"{field}: required for expected_action {action!r}")

    problems = live_slice_problems(row, action)
    draft = {
        "case_id": case_id,
        "request": {k: row[k] for k in ("system", "user", "document")
                    if k in row},
        "expected": {"scorers": ["capability_pass_rate", "indonesian_voice",
                                 "edit_contract_output"]},
        "expects_state_change": action == sc.ACTION_EDIT,
        "requires_approval": action == sc.ACTION_EDIT,
        "expected_action": action,
        "source": {
            "row_sha256": source_digest(row),
            "declared_id": row.get("id"),
            "stratum": row.get("stratum"),
            "source_class": row.get("source_class"),
            "held_out_claimed": row.get("held_out"),
        },
        # Carried through verbatim when present; never fabricated when absent.
        **{k: row[k] for k in ("must_preserve", "must_not_change",
                               "allowed_new_facts", "structure",
                               "must_state_absence") if k in row},
        "human_input_required": owed,
        "live_slice_compatible": not problems,
        "live_slice_problems": problems,
    }
    return draft


def build(rows: list[dict[str, Any]], name: str, source_path: str
          ) -> dict[str, Any]:
    cases = [convert_row(row, i) for i, row in enumerate(rows, 1)]
    incomplete = [c["case_id"] for c in cases if c["human_input_required"]]
    incompatible = [c["case_id"] for c in cases if not c["live_slice_compatible"]]
    return {
        "schema_version": he.CASE_SET_SCHEMA,
        "name": name,
        # ALWAYS false. There is no argument, flag or code path that makes this
        # true: approval is a human record under docs/case_sets/, bound by
        # src/verify_case_set_approval.py.
        "approved": False,
        "split": "fixture",
        "source_class": "synthetic",
        "provenance": {
            "origin": f"drafted by src/build_case_set.py from {source_path}",
            "created_at": "unset",
            "note": "DRAFT. Not reviewed, not approved, and not usable for a "
                    "measurement. Every entry under human_input_required must "
                    "be supplied by a person before this set can be reviewed.",
        },
        "draft_status": {
            "cases": len(cases),
            "complete": len(cases) - len(incomplete),
            "needing_human_input": incomplete,
            "live_slice_incompatible": incompatible,
            "set_level_human_input_required": list(SET_OWED),
            "minimum_independent_cases": 320,
            "short_by": max(0, 320 - len(cases)),
        },
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path, help="a .jsonl of source rows")
    parser.add_argument("--name", required=True, help="draft case-set name")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if not args.source.is_file():
        raise SystemExit(f"no such source: {args.source}")
    rows = [json.loads(line) for line in
            args.source.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise SystemExit(f"{args.source} has no rows")

    draft = build(rows, args.name, str(args.source))
    text = yaml.safe_dump(draft, sort_keys=False, allow_unicode=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(json.dumps(draft["draft_status"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
