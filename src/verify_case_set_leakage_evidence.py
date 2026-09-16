"""Bind a leakage-evidence record to exactly what was checked, or refuse.

`confirmations.leakage_reviewed: true` in an approval record is a person
saying they looked. This is what they looked AT: a record carrying the
checker's own output, bound to five digests, so "reviewed" cannot drift from
"reviewed this".

TWO THINGS AN AGENT MAY NOT DO. It may not sign the human section, and it may
not turn `unverifiable` into `clean`. A component nobody could compare was not
found clean, and a record that said otherwise would make the leakage gate
strongest precisely where it knows least.

NO NETWORK, NO MODEL, NO CORPUS TEXT.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import check_case_leakage as ccl                               # noqa: E402
import corpus_identity as ci                                   # noqa: E402
import harness_eval as he                                      # noqa: E402
import verify_case_set_approval as approval                    # noqa: E402

RECORD_DIR = ROOT / "docs" / "case_sets"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

BOUND_DIGESTS = ("case_set_sha256", "public_manifest_sha256",
                 "private_manifest_sha256", "source_raw_sha256",
                 "checker_implementation_digest")
HUMAN_FIELDS = ("reviewed_by", "reviewer_role", "reviewed_at")
HUMAN_CONFIRMATIONS = ("limitations_understood",
                       "unverifiable_components_accepted")

# For a production held-out Office set these must be CLEAN, not merely present.
# Today none of the document-bearing components can reach that, which is the
# blocker this constant exists to make explicit rather than implicit.
OFFICE_REQUIRED_CLEAN = ("user_payload",)
OFFICE_BLOCKED_COMPONENTS = ("document", "full_office_case")


class LeakageEvidenceError(he.HarnessEvalError):
    """A fail-closed leakage-evidence error."""


def record_path_for(name: str) -> Path:
    return RECORD_DIR / f"{name}.leakage.md"


def parse_record(path: Path) -> dict[str, Any]:
    return approval.parse_record(path)          # same ```yaml block convention


def validate_record(record: dict[str, Any], *, source: str = "<record>"
                    ) -> dict[str, Any]:
    if str(record.get("case_set") or "") == "example-office-v1":
        raise LeakageEvidenceError(
            f"{source}: this is the shipped TEMPLATE. It is fixture data and "
            "binds to no case set.")
    for field in BOUND_DIGESTS:
        value = str(record.get(field) or "")
        if not SHA256_RE.match(value):
            raise LeakageEvidenceError(
                f"{source}: {field} must be 64 lowercase hex, got {value!r}")
        if value == "0" * 64:
            raise LeakageEvidenceError(
                f"{source}: {field} is the template's all-zero placeholder")
    if int(record.get("normalization_version", -1)) != ci.NORMALIZATION_VERSION:
        raise LeakageEvidenceError(
            f"{source}: recorded under normalization version "
            f"{record.get('normalization_version')!r}; this checker is "
            f"{ci.NORMALIZATION_VERSION}. Identities are only meaningful "
            "within one version.")

    for field in HUMAN_FIELDS:
        value = str(record.get(field) or "").strip()
        if not value or approval._PLACEHOLDER_RE.match(value):
            raise LeakageEvidenceError(
                f"{source}: {field} is a placeholder ({record.get(field)!r})")
    if approval._NON_HUMAN_RE.search(str(record["reviewed_by"])):
        raise LeakageEvidenceError(
            f"{source}: reviewed_by names an automated actor. An agent may "
            "generate the machine section; it may not sign the human one.")
    for field in HUMAN_CONFIRMATIONS:
        if record.get(field) is not True:
            raise LeakageEvidenceError(f"{source}: {field} is not confirmed")

    components = record.get("components")
    if not isinstance(components, dict) or not components:
        raise LeakageEvidenceError(f"{source}: components must be a mapping")
    for name, block in components.items():
        status = (block or {}).get("status")
        if status not in (ccl.CLEAN, ccl.OVERLAP, ccl.UNVERIFIABLE):
            raise LeakageEvidenceError(
                f"{source}: component {name!r} has status {status!r}; the "
                f"vocabulary is {ccl.CLEAN}/{ccl.OVERLAP}/{ccl.UNVERIFIABLE}")
        if status == ccl.CLEAN and not (block or {}).get("checked_cases"):
            raise LeakageEvidenceError(
                f"{source}: component {name!r} is recorded clean with zero "
                "checked cases. Nothing compared is not nothing found.")
    return record


def bind_to_case_set(record: dict[str, Any], case_set: dict[str, Any],
                     *, source: str = "<record>") -> dict[str, Any]:
    validate_record(record, source=source)
    he.validate_case_set(case_set)
    digest = he.case_set_digest(case_set)
    if record["case_set"] != case_set.get("name"):
        raise LeakageEvidenceError(
            f"{source}: evidence is for {record['case_set']!r}, not "
            f"{case_set.get('name')!r}")
    if record["case_set_sha256"] != digest:
        raise LeakageEvidenceError(
            f"{source}: the checked digest {record['case_set_sha256'][:12]}... "
            f"is not this set's {digest[:12]}.... The bytes changed after the "
            "check, so the evidence describes a different case set.")

    components = record["components"]
    overlap = [n for n, b in components.items() if b.get("status") == ccl.OVERLAP]
    if overlap:
        raise LeakageEvidenceError(
            f"{source}: overlap recorded in {sorted(overlap)}; this set shares "
            "material with a registered corpus")
    unmet = [n for n in OFFICE_REQUIRED_CLEAN
             if components.get(n, {}).get("status") != ccl.CLEAN]
    blocked = [n for n in OFFICE_BLOCKED_COMPONENTS
               if components.get(n, {}).get("status") != ccl.CLEAN]
    return {
        "status": "MATCHED",
        "case_set": record["case_set"],
        "case_set_sha256": digest,
        "source_raw_sha256": record["source_raw_sha256"],
        "hmac_key_id": record.get("hmac_key_id"),
        "reviewed_by": record["reviewed_by"],
        "reviewed_at": str(record["reviewed_at"]),
        "required_clean": list(OFFICE_REQUIRED_CLEAN),
        "required_unmet": unmet,
        # Reported, never silently satisfied: an Office held-out set needs
        # these and cannot have them until an extractor exists.
        "office_components_blocked": blocked,
        "components": {n: b.get("status") for n, b in components.items()},
    }


def evidence_for(case_set: dict[str, Any]) -> dict[str, Any] | None:
    path = record_path_for(str(case_set.get("name") or ""))
    if not path.is_file():
        return None
    return bind_to_case_set(parse_record(path), case_set, source=str(path))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cases", type=Path)
    parser.add_argument("--record", type=Path, default=None)
    args = parser.parse_args()
    try:
        case_set = he.load_case_set(args.cases)
        path = args.record or record_path_for(str(case_set["name"]))
        if not path.is_file():
            print(f"LEAKAGE EVIDENCE MISSING: no record at {path}",
                  file=sys.stderr)
            raise SystemExit(1)
        result = bind_to_case_set(parse_record(path), case_set, source=str(path))
    except he.HarnessEvalError as exc:
        print(f"LEAKAGE EVIDENCE REFUSED: {exc}", file=sys.stderr)
        raise SystemExit(2)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
