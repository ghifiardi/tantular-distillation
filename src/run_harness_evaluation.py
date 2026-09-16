"""Bounded evaluation controller for the harness-before-weights experiment.

    # what would run, and against which identities
    python src/run_harness_evaluation.py plan \
        configs/experiments/harness-before-weights.yaml \
        --cases tests/fixtures/harness_cases/fixture-office-v1.yaml

    # execute every arm x case through an executor, one receipt each
    python src/run_harness_evaluation.py run \
        configs/experiments/harness-before-weights.yaml \
        --cases tests/fixtures/harness_cases/fixture-office-v1.yaml \
        --executor fake --output out/receipts

    # derive the measurement artifact from those receipts, and nothing else
    python src/run_harness_evaluation.py aggregate \
        configs/experiments/harness-before-weights.yaml \
        --cases tests/fixtures/harness_cases/fixture-office-v1.yaml \
        --receipts out/receipts --output out/measurements.json \
        --allow-fixture

    # then the existing gate reads that artifact
    python src/harness_distill.py evaluate \
        configs/experiments/harness-before-weights.yaml out/measurements.json

WHAT THIS DOES NOT DO. No model call, no network, no Office automation, no
training, no weight download. `plan` is the default posture and writes nothing.
A real executor additionally requires --real, and the only real executor in the
tree refuses because no Office adapter exists yet. Every artifact it writes
carries training_authorized: false.
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

import harness_distill as hd                                   # noqa: E402
import harness_eval as he                                      # noqa: E402
import harness_executors as hx                                 # noqa: E402

MODEL_DIR = ROOT / "configs" / "models"

# Exactly 40 lowercase hex: a full Git commit, never an abbreviation. Same rule
# the model registry and src/distill_plan.py already enforce.
COMMIT_RE = re.compile(r"[0-9a-f]{40}")

# arm -> (which model of the experiment, which harness of the experiment)
ARM_MATRIX = {
    "student_current": ("student_model", "current_harness"),
    "student_candidate": ("student_model", "candidate_harness"),
    "teacher_current": ("teacher_model", "current_harness"),
    "teacher_candidate": ("teacher_model", "candidate_harness"),
}


def _model_spec(registry_name: str) -> dict[str, Any]:
    spec = hd.load_yaml(MODEL_DIR / f"{registry_name}.yaml")
    model_id = spec.get("model_id")
    revision = spec.get("revision")
    if not isinstance(model_id, str) or not model_id.strip():
        raise he.HarnessEvalError(
            f"registry model {registry_name!r} declares no model_id")
    # A receipt that cannot name the exact checkpoint is an attribution to a
    # moving target. The identity milestones exist precisely so this is
    # available; refusing here keeps an unpinned model out of the evidence.
    if not isinstance(revision, str) or not COMMIT_RE.fullmatch(revision):
        raise he.HarnessEvalError(
            f"registry model {registry_name!r} has no pinned revision "
            f"({revision!r}); an evaluation receipt must name the exact "
            "checkpoint it ran against")
    return spec


def resolve_arms(experiment: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Bind every declared arm to a model identity and a harness definition."""
    arms = he.required_arms(experiment)
    unknown = [arm for arm in arms if arm not in ARM_MATRIX]
    if unknown:
        raise he.HarnessEvalError(
            f"experiment declares arm(s) this controller cannot bind: {unknown} "
            f"(known: {sorted(ARM_MATRIX)})")
    resolved: dict[str, dict[str, Any]] = {}
    for arm in arms:
        model_key, harness_key = ARM_MATRIX[arm]
        registry_name = str(experiment[model_key])
        harness_name = str(experiment[harness_key])
        spec = _model_spec(registry_name)
        harness = hd.load_harness(harness_name)
        # harness_provenance validates the harness and refuses an undeclared
        # model/harness pairing. Calling it here means an arm that could never
        # be attributed fails during planning, not after the executions.
        provenance = hd.harness_provenance(
            harness, execution_model_registry=registry_name)
        resolved[arm] = {
            "arm": arm,
            "model_registry": registry_name,
            "model_id": spec["model_id"],
            "model_revision": spec["revision"],
            "harness_name": harness["name"],
            "harness_digest": provenance["digest"],
            "harness": harness,
            "provenance": provenance,
        }
    return resolved


def plan_evaluation(experiment: dict[str, Any],
                    case_set: dict[str, Any]) -> dict[str, Any]:
    """What would run. Writes nothing, executes nothing."""
    plan = hd.build_plan(experiment)
    he.validate_case_set(case_set)
    resolved = resolve_arms(experiment)
    metrics = he.metric_names(experiment)
    ids = he.case_ids(case_set)

    declared = he.scorers_for(case_set)
    return {
        **plan,
        "case_set": {
            "name": case_set["name"],
            "digest": he.case_set_digest(case_set),
            "cases": len(ids),
            "case_ids": ids,
            "approved": case_set["approved"],
            "split": case_set["split"],
            "source_class": case_set["source_class"],
        },
        "metrics_required": metrics,
        "metrics_declared_by_cases": sorted(declared),
        "arms": {
            arm: {k: v for k, v in row.items()
                  if k not in ("harness", "provenance")}
            for arm, row in resolved.items()
        },
        "executions_planned": len(resolved) * len(ids),
        "real_execution_available": False,
        "real_execution_blocker": (
            "no Office harness adapter exists; see RealOfficeExecutor"),
        "training_authorized": False,
    }


# --- execution --------------------------------------------------------------

def _judge(request: hx.ExecutionRequest,
           result: hx.ExecutionResult) -> tuple[str, str, list[str]]:
    """Apply harness policy to what the executor reported.

    The executor observes; the controller judges. An executor that graded its
    own policy compliance would make the approval and budget gates advisory,
    and the first thing a broken adapter would report is that it behaved.
    """
    problems: list[str] = []
    if result.error:
        return he.STATUS_ERROR, hx.TERMINATION_EXECUTOR_ERROR, [str(result.error)]

    allowed = set(request.tool_policy.get("allow") or [])
    requires_approval = request.tool_policy.get(
        "state_change_requires_approval") is True

    for call in result.tool_calls:
        tool = str(call.get("tool", ""))
        if tool not in allowed:
            problems.append(
                f"tool {tool!r} is not in the harness allowlist {sorted(allowed)}")
            return he.STATUS_REFUSED, hx.TERMINATION_TOOL_NOT_ALLOWED, problems
        if requires_approval and hx.changes_state(tool) \
                and call.get("approved") is not True:
            problems.append(
                f"state-changing tool {tool!r} was called without an approval "
                "decision; the harness declares "
                "tools.state_change_requires_approval: true")
            return he.STATUS_REFUSED, hx.TERMINATION_APPROVAL_MISSING, problems

    max_steps = request.budgets["max_steps"]
    if result.steps > max_steps:
        problems.append(f"used {result.steps} steps against a budget of {max_steps}")
        return he.STATUS_REFUSED, hx.TERMINATION_BUDGET_STEPS, problems
    max_wall = request.budgets["max_wall_seconds"]
    if result.wall_seconds > max_wall:
        problems.append(
            f"used {result.wall_seconds}s against a budget of {max_wall}s")
        return he.STATUS_REFUSED, hx.TERMINATION_BUDGET_WALL, problems

    allowed_repairs = int(request.verification_policy.get("repair_attempts") or 0)
    if result.repair_attempts > allowed_repairs:
        problems.append(
            f"{result.repair_attempts} repair attempt(s) against a budget of "
            f"{allowed_repairs}")
        return he.STATUS_REFUSED, hx.TERMINATION_REPAIR_LIMIT, problems

    declared_scorers = set(request.case["expected"]["scorers"])
    absent = sorted(declared_scorers - set(result.scores))
    if absent:
        problems.append(f"no scorer result for {absent}")
        return he.STATUS_ERROR, hx.TERMINATION_EXECUTOR_ERROR, problems

    return he.STATUS_OK, hx.TERMINATION_COMPLETED, problems


def build_receipt(experiment: dict[str, Any], case_set: dict[str, Any],
                  bound: dict[str, Any], request: hx.ExecutionRequest,
                  result: hx.ExecutionResult, executor_identity: dict[str, Any],
                  ) -> dict[str, Any]:
    status, termination, problems = _judge(request, result)
    scores = dict(result.scores) if status == he.STATUS_OK else {}
    # Only the scorers this case declared. An executor returning an extra
    # metric must not widen the evidence beyond what was asked.
    if status == he.STATUS_OK:
        scores = {k: v for k, v in scores.items()
                  if k in set(request.case["expected"]["scorers"])}
    return {
        "schema_version": he.RECEIPT_SCHEMA,
        "experiment": experiment.get("name"),
        "experiment_digest": he.experiment_digest(experiment),
        "case_set_name": case_set["name"],
        "case_set_digest": he.case_set_digest(case_set),
        "case_id": request.case_id,
        "arm": request.arm,
        "model_registry": bound["model_registry"],
        "model_id": bound["model_id"],
        "model_revision": bound["model_revision"],
        "harness_name": bound["harness_name"],
        "harness_digest": bound["harness_digest"],
        "harness_provenance": bound["provenance"],
        "prompt_sha256": bound["provenance"]["prompt_sha256"],
        "prompt_verified": bound["provenance"]["prompt_verified"],
        "tools_offered": list(result.tools_offered),
        "tool_calls": list(result.tool_calls),
        "approvals": list(result.approvals),
        "before_action": list(result.before_action),
        "after_action": list(result.after_action),
        "repair_attempts": result.repair_attempts,
        "termination_reason": termination,
        "budgets": dict(request.budgets),
        "budget_consumed": {"steps": result.steps,
                            "wall_seconds": result.wall_seconds},
        # The OUTPUT's digest, not the output. A receipt is provenance, not a
        # copy of the document: recording the edit text would put customer
        # content into an evidence file that is meant to be shareable.
        "result_digest": he.digest_payload(result.output),
        "executor": dict(executor_identity),
        "run_id": request.run_id,
        "repetition": request.repetition,
        "started_at": result.started_at,
        "ended_at": result.ended_at,
        "status": status,
        "problems": problems,
        "scores": scores,
        # Supplied by the executor, not assumed by the controller: only the
        # executor knows whether it opened a document or a string.
        "execution_surface": executor_identity.get("execution_surface",
                                                   he.SURFACE_TEXT),
        "approval": result.approval,
        "training_authorized": False,
    }


def run_evaluation(experiment: dict[str, Any], case_set: dict[str, Any],
                   executor: Any, output: Path, *, run_id: str = "run-1",
                   ) -> dict[str, Any]:
    """Execute every arm x case and leave one receipt per pair.

    A failed execution leaves a FAILURE receipt. Disappearing on error would
    make a broken arm indistinguishable from an arm nobody ran, and the
    aggregation could not tell the difference either.
    """
    he.validate_case_set(case_set)
    resolved = resolve_arms(experiment)
    identity = executor.identity()
    written: list[str] = []
    counts = {status: 0 for status in he.RECEIPT_STATUSES}

    for arm in he.required_arms(experiment):
        bound = resolved[arm]
        harness = bound["harness"]
        for case in case_set["cases"]:
            request = hx.ExecutionRequest(
                case=case,
                arm=arm,
                model_registry=bound["model_registry"],
                model_id=bound["model_id"],
                model_revision=bound["model_revision"],
                harness_name=bound["harness_name"],
                harness_digest=bound["harness_digest"],
                prompt_sha256=bound["provenance"]["prompt_sha256"],
                prompt_verified=bound["provenance"]["prompt_verified"],
                tool_policy=dict(harness.get("tools") or {}),
                verification_policy=dict(harness.get("verification") or {}),
                budgets={
                    "max_steps": (harness.get("execution") or {})["max_steps"],
                    "max_wall_seconds":
                        (harness.get("execution") or {})["max_wall_seconds"],
                },
                run_id=run_id,
            )
            try:
                result = executor.execute(request)
            except he.HarnessEvalError:
                raise
            except Exception as exc:                  # noqa: BLE001 - see below
                # An executor that raises anything else still owes a record.
                # Losing the failure would leave a hole the aggregation reports
                # as "no receipt", which reads as "never attempted".
                result = hx.ExecutionResult(
                    error=f"{type(exc).__name__}: {exc}",
                    started_at=f"error:{run_id}:{arm}:{case['case_id']}:start",
                    ended_at=f"error:{run_id}:{arm}:{case['case_id']}:end")
            receipt = build_receipt(experiment, case_set, bound, request,
                                    result, identity)
            written.append(str(he.write_receipt(output, receipt)))
            counts[receipt["status"]] += 1

    return {
        "experiment": experiment.get("name"),
        "case_set_digest": he.case_set_digest(case_set),
        "executor": identity,
        "receipts_written": len(written),
        "status_counts": counts,
        "output": str(output),
        "training_authorized": False,
    }


# --- CLI --------------------------------------------------------------------

def _load_experiment(path: Path) -> dict[str, Any]:
    return hd.load_yaml(path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    p = commands.add_parser("plan", help="what would run (writes nothing)")
    p.add_argument("experiment", type=Path)
    p.add_argument("--cases", type=Path, required=True)

    r = commands.add_parser("run", help="execute arms and write receipts")
    r.add_argument("experiment", type=Path)
    r.add_argument("--cases", type=Path, required=True)
    r.add_argument("--executor", default="fake",
                   help=f"one of {sorted(hx.EXECUTORS)} (default: fake)")
    r.add_argument("--output", type=Path, required=True,
                   help="directory to write receipts into")
    r.add_argument("--run-id", default="run-1",
                   help="deterministic run/repetition identifier")
    r.add_argument("--real", action="store_true",
                   help="acknowledge a non-fixture executor; required for any "
                        "executor that claims to produce real measurements")

    a = commands.add_parser("aggregate", help="derive measurements from receipts")
    a.add_argument("experiment", type=Path)
    a.add_argument("--cases", type=Path, required=True)
    a.add_argument("--receipts", type=Path, required=True)
    a.add_argument("--output", type=Path)
    a.add_argument("--allow-fixture", action="store_true",
                   help="aggregate fixture-executor receipts into an explicitly "
                        "labelled fixture artifact")

    args = parser.parse_args(argv)
    try:
        experiment = _load_experiment(args.experiment)
        case_set = he.load_case_set(args.cases)

        if args.command == "plan":
            result = plan_evaluation(experiment, case_set)
        elif args.command == "run":
            executor = hx.get_executor(args.executor)
            identity = executor.identity()
            if identity.get("produces_real_measurements") and not args.real:
                raise he.HarnessEvalError(
                    f"executor {args.executor!r} claims to produce real "
                    "measurements. Pass --real to acknowledge that this is not "
                    "a dry run. The default posture of this tool is plan/replay.")
            if not identity.get("produces_real_measurements") and args.real:
                raise he.HarnessEvalError(
                    f"--real was passed but executor {args.executor!r} replays "
                    "fixtures. A fixture run is not a real measurement and will "
                    "not be labelled as one.")
            result = run_evaluation(experiment, case_set, executor, args.output,
                                    run_id=args.run_id)
        else:
            receipts = he.load_receipts(args.receipts)
            result = he.aggregate(experiment, case_set, receipts,
                                  allow_fixture=args.allow_fixture)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(
                    json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    except hd.HarnessPlanError as exc:
        print(f"HARNESS EVALUATION REFUSED: {exc}", file=sys.stderr)
        raise SystemExit(2)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
