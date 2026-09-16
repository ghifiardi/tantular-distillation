"""The declared before/after checks, behind one protocol — or a refusal.

configs/harnesses/*.yaml declare four checks:

    before_action   request_schema, target_location
    after_action    edit_contract, faithful_edit

A name in YAML is not an implementation, and the gap between the two is
exactly the attribution ambiguity this repository exists to close. So every
declared check resolves through this module, and one that has no implementation
returns a REFUSAL rather than being skipped — a skipped check reads as a pass
to anyone counting, and an unimplemented check that reads as a pass is worse
than no check at all.

WHAT IS REAL TODAY:

  request_schema    implemented here. Shape-checks the edit contract the model
                    returned, using the add-in's documented limits.
  target_location   NOT implemented here, and cannot be: resolving a target
                    means reading the live document, which only the add-in can
                    do. It CONSUMES a resolution the add-in performed, and
                    refuses when none was supplied.
  edit_contract     real, via scripts/check_edit_contract.mjs against the
                    add-in's own parser. Operates on document TEXT.
  faithful_edit     real, via src/score_faithful_edit.py. Post-hoc over a
                    completion plus the case's declared spans.

NO MODEL CALL, NO NETWORK, NO OFFICE ACTION happens in this module.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from harness_eval import HarnessEvalError                      # noqa: E402

# Outcome vocabulary. `refused` is deliberately distinct from `fail`: a check
# that could not run has not judged anything, and folding it into a failure
# would let an absent verifier look like a caught defect.
PASSED = "passed"
FAILED = "failed"
REFUSED = "refused"

# From the add-in's editContract.js, which is the contract the product actually
# enforces. Mirrored rather than imported because this runs in Python; the
# numbers are asserted against the add-in's source in the tests.
MAX_EDITS = 20
MAX_FIND = 2000


def _result(check: str, outcome: str, detail: str = "",
            **extra: Any) -> dict[str, Any]:
    return {"check": check, "outcome": outcome, "passed": outcome == PASSED,
            "detail": detail, **extra}


# --- before_action ----------------------------------------------------------

def request_schema(payload: Any) -> dict[str, Any]:
    """Does the model's output parse as the edit contract the product accepts?

    Shape only. Whether the edits can be LOCATED is target_location's question,
    and conflating them produced a check that passed on valid JSON whose `find`
    strings appear nowhere.
    """
    check = "request_schema"
    if isinstance(payload, str):
        start, end = payload.find("{"), payload.rfind("}")
        if start == -1 or end <= start:
            return _result(check, FAILED, "no JSON object in the model output")
        try:
            payload = json.loads(payload[start:end + 1])
        except json.JSONDecodeError as exc:
            return _result(check, FAILED, f"unreadable JSON: {exc}")
    if not isinstance(payload, dict):
        return _result(check, FAILED, "contract root must be an object")

    edits = payload.get("edits")
    if not isinstance(edits, list) or not edits:
        return _result(check, FAILED, "contract declares no edits")
    if len(edits) > MAX_EDITS:
        return _result(check, FAILED,
                       f"{len(edits)} edits exceeds the contract maximum {MAX_EDITS}")
    problems: list[str] = []
    for index, edit in enumerate(edits, 1):
        if not isinstance(edit, dict):
            problems.append(f"#{index} is not an object")
            continue
        find = edit.get("find")
        if not isinstance(find, str) or not find:
            problems.append(f"#{index} has no usable 'find'")
            continue
        if len(find) > MAX_FIND:
            problems.append(f"#{index} 'find' exceeds {MAX_FIND} characters")
        if not isinstance(edit.get("replace"), str):
            problems.append(f"#{index} has no 'replace' string")
        occurrence = edit.get("occurrence", 1)
        if occurrence is not None and not (isinstance(occurrence, int)
                                           and not isinstance(occurrence, bool)
                                           and occurrence >= 1):
            problems.append(f"#{index} 'occurrence' must be a positive integer")
    if problems:
        return _result(check, FAILED, "; ".join(problems), edits=len(edits))
    return _result(check, PASSED, f"{len(edits)} edit(s) match the contract shape",
                   edits=len(edits))


def target_location(resolution: Any) -> dict[str, Any]:
    """Consume the add-in's LIVE resolution of where an edit lands.

    This module cannot resolve a target itself. Doing so would mean re-running
    locateEdit against a copy of the document, and a copy is not where the edit
    will land — the whole failure this check exists to catch is the document
    having moved. Only the add-in can answer, so this validates its answer and
    refuses when there is none.
    """
    check = "target_location"
    if resolution is None:
        return _result(check, REFUSED,
                       "no live resolution was supplied. This check reads the "
                       "add-in's locateEdit/searchOrdinalAt result against the "
                       "open document; it cannot be evaluated from a text copy, "
                       "and an unevaluated check is not a pass.")
    if not isinstance(resolution, dict):
        return _result(check, REFUSED, "resolution must be a mapping")
    if resolution.get("error"):
        error = str(resolution["error"])
        return _result(check, FAILED, f"the add-in could not anchor the edit: {error}",
                       error=error)
    ordinal = resolution.get("ordinal")
    target = resolution.get("target_digest")
    if not isinstance(ordinal, int) or ordinal < 0:
        return _result(check, REFUSED, "resolution carries no non-negative ordinal")
    if not isinstance(target, str) or len(target) != 64:
        # The digest, not the matched text: the text is document content and
        # must not reach a receipt.
        return _result(check, REFUSED,
                       "resolution carries no target_digest; without it the "
                       "receipt cannot say WHERE the edit landed")
    return _result(check, PASSED, "the add-in anchored the edit to one location",
                   ordinal=ordinal, target_digest=target)


# --- after_action -----------------------------------------------------------

def edit_contract(cases: list[dict], addin_src: Path,
                  checker: Path | None = None) -> dict[str, Any]:
    """The add-in's REAL parser, via scripts/check_edit_contract.mjs.

    Deliberately the same path run_gates.py and score_faithful_edit.py use, so
    the verifier cannot drift from the product's own contract.
    """
    check = "edit_contract"
    script = checker or (ROOT / "scripts" / "check_edit_contract.mjs")
    if not script.is_file():
        return _result(check, REFUSED, f"checker missing: {script}")
    if not (addin_src / "chat" / "editContract.js").is_file():
        return _result(check, REFUSED,
                       f"add-in parser missing under {addin_src}; the real "
                       "contract cannot be checked against a copy that is not there")
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False,
                                     encoding="utf-8") as handle:
        json.dump(cases, handle, ensure_ascii=False)
        path = Path(handle.name)
    try:
        proc = subprocess.run(["node", str(script), str(path), str(addin_src)],
                              capture_output=True, text=True)
    except FileNotFoundError:
        path.unlink(missing_ok=True)
        return _result(check, REFUSED, "node is not available to run the checker")
    path.unlink(missing_ok=True)
    if proc.returncode != 0 or not proc.stdout.strip():
        return _result(check, REFUSED,
                       f"checker failed: {proc.stderr.strip()[-300:]}")
    scored = json.loads(proc.stdout)
    rows = scored if isinstance(scored, list) else scored.get("cases", [])
    ok = all(row.get("contract_ok") is True for row in rows) if rows else False
    return _result(check, PASSED if ok else FAILED,
                   f"{sum(1 for r in rows if r.get('contract_ok'))}/{len(rows)} "
                   "case(s) parsed, located and applied", cases=len(rows))


def faithful_edit(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
    """Post-hoc faithfulness, via src/score_faithful_edit.py.

    Not wired into the live path in this slice. It needs a case's declared
    must_preserve / must_not_change spans, and the approved held-out set those
    would come from does not exist yet — so it refuses here rather than
    reporting a verdict it has no inputs for.
    """
    return _result("faithful_edit", REFUSED,
                   "faithful_edit scores a completion against a case's declared "
                   "must_preserve/must_not_change spans (src/score_faithful_edit.py). "
                   "No approved held-out Office case set exists, so there are no "
                   "declared spans to score against. Not run; not passed.")


# --- dispatch ---------------------------------------------------------------

_BEFORE = {"request_schema": request_schema, "target_location": target_location}
_AFTER = {"edit_contract": edit_contract, "faithful_edit": faithful_edit}


def run_declared(phase: str, names: list[str], inputs: dict[str, Any]
                 ) -> list[dict[str, Any]]:
    """Run every check a harness declares for one phase.

    An unknown name REFUSES. A harness declaring a check nobody implemented
    must not quietly get an empty list back: the list is what the receipt
    records, and an empty list of checks reads as "nothing to verify".
    """
    table = _BEFORE if phase == "before_action" else _AFTER
    if phase not in ("before_action", "after_action"):
        raise HarnessEvalError(f"unknown verification phase {phase!r}")
    results = []
    for name in names:
        function = table.get(name)
        if function is None:
            results.append(_result(name, REFUSED,
                                   f"the harness declares {name!r} for {phase}, "
                                   "and no implementation is registered. A "
                                   "declared check that cannot run is a refusal, "
                                   "never a silent pass."))
            continue
        try:
            results.append(function(inputs.get(name)))
        except TypeError:
            # A check whose inputs were not supplied in the shape it needs.
            results.append(_result(name, REFUSED,
                                   f"{name} was not given the inputs it requires"))
    return results
