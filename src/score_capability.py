"""Capability, composed from properties that already exist.

`capability_pass_rate` is the metric `configs/experiments/harness-before-weights.yaml`
decides on, and nothing computed it. This module supplies it WITHOUT writing a
new judge: every property it reads is produced by `src/score_faithful_edit.py`,
which is mechanical, deterministic, and already reviewed. What is new here is
the PARTITION — which properties belong to capability, which stay guardrails,
and what a failure is called.

Two properties are deliberately outside the Boolean. `contract` and `voice` are
the experiment's configured guardrails with their own minimums and their own
zero-regression rules; folding them in would make a voice regression move the
capability number too, and `evaluate()` could no longer tell
`harness_optimization_sufficient` from `guardrail_failures`.

One apparent exception is not one: an edit contract that does not parse, locate
and apply leaves NO applied document to inspect, so capability fails through
`model_output_invalid`. That is not the guardrail counted twice — it is the
absence of anything to judge, and `score_faithful_edit` already records the
five dependent properties as `_not_measured` rather than as failures for
exactly this reason.

NO MODEL CALL, NO NETWORK, NO OFFICE ACTION happens here.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import harness_distill as hd                                   # noqa: E402
from harness_eval import HarnessEvalError                      # noqa: E402

CONTRACT_PATH = ROOT / "configs" / "metrics" / "capability_pass_rate.v1.yaml"

ACTION_EDIT = "edit"
ACTION_NO_EDIT = "no_edit"

# Which declarations a case must carry before it can be scored AT ALL. Absence
# is not a failing case: it is a case nobody finished writing, and it must be
# caught before arm assignment so it cannot be discovered as a "failure" in one
# arm after the run. An EMPTY list is a real declaration ("nothing to
# preserve"); a missing key is not.
REQUIRED_DECLARATIONS = {
    ACTION_EDIT: ("document", "must_preserve", "must_not_change",
                  "allowed_new_facts", "target_assertion"),
    ACTION_NO_EDIT: ("document", "must_state_absence", "allowed_new_facts"),
}

# Deterministic precedence for the PRIMARY reason. Ordered by what explains the
# most: an executor that died explains every property that did not get
# measured, so reporting `preservation_failed` there would send someone to
# inspect text that was never produced. All applicable reasons are still
# returned in `failure_reasons`; this only decides which one leads.
PRECEDENCE = (
    "executor_error",
    "timeout",
    "verifier_error",
    "approval_rejected",
    "approval_invalid",
    "target_stale",
    "target_wrong",
    "request_schema_failed",
    "model_output_invalid",
    "edit_not_applied",
    "edit_partial",
    "unexpected_edit",
    "unexpected_no_edit",
    "preservation_failed",
    "structure_failed",
    "unsupported_fact",
)

# Informational, and a PASS: the case asked the model to decline, and it did.
CORRECT_NO_EDIT = "correct_no_edit"
# Run-level, never case-level. Declared in the contract so a report can name it
# with the same vocabulary the cases use.
STATISTICALLY_UNQUALIFIED = "statistically_unqualified"


def load_contract(path: Path | None = None) -> tuple[dict[str, Any], str]:
    """The metric definition and its canonical digest.

    Same digest construction as a harness: over the whole object minus any
    self-declared `digest`, because an artifact must not be trusted to describe
    itself.
    """
    spec = hd.load_yaml(path or CONTRACT_PATH)
    if spec.get("name") != "capability_pass_rate":
        raise HarnessEvalError(
            f"{path or CONTRACT_PATH} is not the capability metric contract")
    if spec.get("value_type") != "boolean":
        raise HarnessEvalError(
            "capability is a per-case boolean; a rate is an aggregate")
    return spec, hd.canonical_digest(spec)


def contract_binding(spec: dict[str, Any], digest: str) -> dict[str, Any]:
    return {"name": spec["name"], "version": spec["version"], "digest": digest}


def check_declarations(case: dict[str, Any]) -> None:
    """Refuse a case that cannot be scored, before any arm runs it."""
    action = case.get("expected_action", ACTION_EDIT)
    if action not in REQUIRED_DECLARATIONS:
        raise HarnessEvalError(
            f"case {case.get('case_id')!r}: unknown expected_action {action!r} "
            f"(have {sorted(REQUIRED_DECLARATIONS)})")
    missing = [key for key in REQUIRED_DECLARATIONS[action] if key not in case]
    if missing:
        raise HarnessEvalError(
            f"case {case.get('case_id')!r}: cannot be scored, missing "
            f"{sorted(missing)} for expected_action {action!r}. A case nobody "
            "finished writing must be refused before arm assignment, not "
            "discovered as a failure in one arm afterwards.")


def _execution_reasons(execution: dict[str, Any]) -> list[str]:
    """Reasons that come from HOW the run went, not from what it produced."""
    reasons: list[str] = []
    status = str(execution.get("status") or "ok")
    termination = str(execution.get("termination_reason") or "completed")

    if termination in ("budget_exceeded_wall_seconds",):
        reasons.append("timeout")
    if termination in ("approval_not_granted",):
        reasons.append("approval_rejected")
    if termination in ("tool_not_allowed", "repair_attempts_exceeded",
                       "budget_exceeded_steps"):
        reasons.append("executor_error")
    if execution.get("approval_invalid"):
        reasons.append("approval_invalid")
    if status == "error" and "timeout" not in reasons:
        reasons.append("executor_error")
    if status == "refused" and not reasons:
        # A refusal with no more specific cause is still a refusal, and it is
        # a capability failure: the case did not get done.
        reasons.append("approval_rejected")

    target = execution.get("target_location") or {}
    outcome = str(target.get("outcome") or "")
    if outcome == "refused":
        reasons.append("verifier_error")
    elif outcome == "failed":
        error = str(target.get("error") or "")
        reasons.append("target_stale" if error == "not_found" else "target_wrong")

    if execution.get("request_schema_failed"):
        reasons.append("request_schema_failed")
    if execution.get("edit_partial"):
        reasons.append("edit_partial")
    return reasons


def _property_reasons(findings: dict[str, Any]) -> list[str]:
    """Reasons that come from the applied document."""
    reasons: list[str] = []
    if "contract" in findings:
        # No applied document exists, so nothing downstream was judged.
        reasons.append("model_output_invalid")
    lands = findings.get("lands") or []
    if lands:
        # `lands` covers two different defects and they have different causes:
        # the edit never landed, or it landed on text that was protected.
        text = " ".join(str(x) for x in lands)
        matched = False
        if "unchanged" in text or "no applied document" in text:
            reasons.append("edit_not_applied")
            matched = True
        if "protected span altered" in text:
            reasons.append("preservation_failed")
            matched = True
        if not matched:
            # An unrecognised `lands` finding is still a real failure; report
            # the conservative reading rather than dropping it.
            reasons.append("edit_not_applied")
    if "preserves" in findings and "preservation_failed" not in reasons:
        reasons.append("preservation_failed")
    if "structure" in findings:
        reasons.append("structure_failed")
    if "no_new_facts" in findings:
        reasons.append("unsupported_fact")
    return reasons


def score_case(case: dict[str, Any], findings: dict[str, Any],
               execution: dict[str, Any] | None = None,
               spec: dict[str, Any] | None = None,
               digest: str | None = None) -> dict[str, Any]:
    """Capability for ONE case.

    `findings` is score_faithful_edit's output for this case, verbatim --
    including its `_not_measured` list. `execution` is what the controller
    observed. Nothing here re-judges text.
    """
    check_declarations(case)
    if spec is None or digest is None:
        spec, digest = load_contract()
    execution = dict(execution or {})
    action = case.get("expected_action", ACTION_EDIT)

    not_measured = list(findings.get("_not_measured") or [])
    detail = {k: v for k, v in findings.items() if k != "_not_measured"}

    reasons = _execution_reasons(execution)
    reasons += [r for r in _property_reasons(detail) if r not in reasons]

    # Expected action. Kept separate from the property reasons so "declined
    # when it should have edited" and "edited when it should have declined" are
    # never reported as the same thing.
    emitted = execution.get("edits_emitted")
    if emitted is not None:
        if action == ACTION_EDIT and emitted == 0:
            if "unexpected_no_edit" not in reasons:
                reasons.append("unexpected_no_edit")
        if action == ACTION_NO_EDIT and emitted > 0:
            if "unexpected_edit" not in reasons:
                reasons.append("unexpected_edit")

    required = set(spec["pass_condition"]["required"])
    conditional = set(spec["pass_condition"]["conditional"])
    # `structure` is scored only when the case declares checks. An undeclared
    # check must be neither a pass nor a failure -- counting it as a pass would
    # reward a case for declaring nothing.
    if "structure" in conditional and not (case.get("structure") or {}):
        reasons = [r for r in reasons if r != "structure_failed"]
        if "structure" not in not_measured:
            not_measured.append("structure")

    passed = not reasons
    if passed and action == ACTION_NO_EDIT:
        reasons = [CORRECT_NO_EDIT]          # informational, still a pass

    primary = None
    if not passed:
        for reason in PRECEDENCE:
            if reason in reasons:
                primary = reason
                break
        if primary is None:                  # unreachable: PRECEDENCE is closed
            raise HarnessEvalError(
                f"case {case.get('case_id')!r} failed with reasons {reasons} "
                "that have no precedence entry; the vocabulary is closed and a "
                "reason outside it is an error, not a new category")

    return {
        "case_id": case.get("case_id"),
        "expected_action": action,
        "passed": passed,
        "primary_failure_reason": primary,
        "failure_reasons": reasons,
        "properties": {k: (k not in detail) for k in
                       ("lands", "preserves", "structure", "no_new_facts")
                       if k not in not_measured},
        "property_findings": detail,
        "not_measured": sorted(set(not_measured)),
        "required_properties": sorted(required),
        "metric_contract": contract_binding(spec, digest),
    }
