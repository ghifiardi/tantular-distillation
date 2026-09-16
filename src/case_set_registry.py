"""Append-only registry of what every case set has been USED for.

A set used to tune or validate the scorer is calibration data forever. It may
not later become the held-out set that decides anything -- not because the
bytes are tainted, but because the decision would be made against material the
decision-maker has already seen.

ONE-WAY BY CONSTRUCTION. A digest registered as `development` or `calibration`
can never be registered as `held_out`, renaming does not reset history (the
digest is the key, not the name), and modified bytes produce a NEW digest that
must still be checked for component overlap with everything already registered
-- otherwise editing one character would launder development material into a
held-out set.

NO NETWORK, NO MODEL, NO CORPUS TEXT.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import harness_eval as he                                      # noqa: E402

REGISTRY_PATH = ROOT / "docs" / "case_sets" / "REGISTRY.jsonl"

ROLE_DEVELOPMENT = "development"
ROLE_CALIBRATION = "calibration"
ROLE_HELD_OUT = "held_out"
ROLES = (ROLE_DEVELOPMENT, ROLE_CALIBRATION, ROLE_HELD_OUT)

# Registering as either of these permanently excludes a digest from held_out.
CONSUMING_ROLES = (ROLE_DEVELOPMENT, ROLE_CALIBRATION)


class RegistryError(he.HarnessEvalError):
    """A fail-closed registry error."""


def load(path: Path | None = None) -> list[dict[str, Any]]:
    path = path or REGISTRY_PATH
    if not path.is_file():
        return []
    entries = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise RegistryError(f"{path}:{number}: unreadable entry: {exc}")
    return entries


def history_for(digest: str, entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [e for e in entries if e.get("case_set_sha256") == digest]


def may_be_held_out(digest: str, entries: list[dict[str, Any]]) -> tuple[bool, str]:
    consumed = [e for e in history_for(digest, entries)
                if e.get("role") in CONSUMING_ROLES]
    if consumed:
        first = consumed[0]
        return False, (
            f"this digest was registered as {first['role']!r} on "
            f"{first.get('first_used_at')}: it has already been used to tune or "
            "validate, so a decision made against it would be made against "
            "material already seen. Renaming does not reset this -- the digest "
            "is the key.")
    return True, ""


def register(case_set: dict[str, Any], *, role: str, purpose: str,
             first_used_at: str, evidence: str | None = None,
             reviewer: str | None = None,
             path: Path | None = None) -> dict[str, Any]:
    """Append one entry. Never rewrites, never approves anything."""
    he.validate_case_set(case_set)
    if role not in ROLES:
        raise RegistryError(f"role must be one of {list(ROLES)}, got {role!r}")
    if not purpose.strip():
        raise RegistryError("purpose is required: a registration nobody can "
                            "explain is not a record")
    digest = he.case_set_digest(case_set)
    entries = load(path)

    if role == ROLE_HELD_OUT:
        allowed, why = may_be_held_out(digest, entries)
        if not allowed:
            raise RegistryError(f"cannot register as held_out: {why}")
        if reviewer is None:
            raise RegistryError(
                "a held_out registration names its reviewer; development and "
                "calibration do not")

    entry = {
        "case_set": case_set.get("name"),
        "case_set_sha256": digest,
        "role": role,
        "first_used_at": first_used_at,
        "purpose": purpose,
        "evidence": evidence,
        "reviewer": reviewer,
        "cases": len(case_set["cases"]),
    }
    target = path or REGISTRY_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:   # append-only
        handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
    return entry
