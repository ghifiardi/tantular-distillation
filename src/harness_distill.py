"""Harness-aware distillation planner and evidence gate.

This module answers a narrower question than the weight trainer:

    Does an observed capability gap remain after a controlled harness change?

It performs no model call, network request, trace generation, or training.
Every result carries ``training_authorized: false``.  A positive result means
only that weight distillation is worth proposing under the repository's
existing authorization and provenance gates.

Examples:

    python src/harness_distill.py plan \
        configs/experiments/harness-before-weights.yaml

    python src/harness_distill.py evaluate \
        configs/experiments/harness-before-weights.yaml measurements.json

    python src/harness_distill.py audit-traces data/promoted/train.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    sys.exit("pyyaml is required: pip install -r requirements.txt")

ROOT = Path(__file__).resolve().parent.parent
HARNESS_DIR = ROOT / "configs" / "harnesses"

REQUIRED_ARMS = ("student_current", "student_candidate", "teacher_current")


class HarnessPlanError(ValueError):
    """A fail-closed harness plan or evidence error."""


def canonical_digest(payload: dict[str, Any]) -> str:
    """Digest a harness definition without trusting its self-declared digest."""
    value = dict(payload)
    value.pop("digest", None)
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise HarnessPlanError(f"missing YAML file: {path}")
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise HarnessPlanError(f"invalid YAML at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise HarnessPlanError(f"YAML root must be an object: {path}")
    return value


def load_harness(name: str) -> dict[str, Any]:
    return load_yaml(HARNESS_DIR / f"{name}.yaml")


def validate_harness(spec: dict[str, Any]) -> list[str]:
    """Return non-blocking warnings; raise on unsafe or incomplete policy."""
    required = (
        "schema_version",
        "name",
        "status",
        "model_contract",
        "tools",
        "memory",
        "execution",
        "verification",
        "mutation",
    )
    missing = [key for key in required if key not in spec]
    if missing:
        raise HarnessPlanError(
            f"harness {spec.get('name', '<unnamed>')!r} missing: {missing}"
        )

    mutation = spec["mutation"] or {}
    if mutation.get("production_self_modify") is not False:
        raise HarnessPlanError("production_self_modify must be false")
    if mutation.get("candidate_workspace_only") is not True:
        raise HarnessPlanError("candidate_workspace_only must be true")
    if mutation.get("evaluator_mutation_allowed") is not False:
        raise HarnessPlanError("evaluator_mutation_allowed must be false")
    if mutation.get("human_approval_required") is not True:
        raise HarnessPlanError("human_approval_required must be true")

    tools = spec["tools"] or {}
    if tools.get("state_change_requires_approval") is not True:
        raise HarnessPlanError(
            "state-changing tools must require approval"
        )

    execution = spec["execution"] or {}
    network = execution.get("network")
    if network != "denied_by_default" and not (
        isinstance(network, dict) and isinstance(network.get("allow"), list)
    ):
        raise HarnessPlanError(
            "execution.network must be denied_by_default or an explicit allowlist"
        )
    for key in ("max_steps", "max_wall_seconds"):
        value = execution.get(key)
        if not isinstance(value, int) or value <= 0:
            raise HarnessPlanError(f"execution.{key} must be a positive integer")

    warnings: list[str] = []
    prompt = ((spec.get("prompts") or {}).get("system") or {})
    if prompt.get("verified") is not True or not prompt.get("sha256"):
        warnings.append("system prompt identity is unverified")
    return warnings


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

# The block stamped onto a harness-aware trace. Version it separately from the
# harness schema: a trace written today must stay readable when the harness
# schema moves on.
HARNESS_PROVENANCE_SCHEMA = 1


def _policy_digest(payload: Any) -> str:
    """Digest one policy sub-object, so a trace can name the policy that
    produced it without carrying the whole thing."""
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def harness_provenance(spec: dict[str, Any], *,
                       execution_model_registry: str) -> dict[str, Any]:
    """The provenance block stamped onto every harness-aware trace.

    ONE SOURCE OF TRUTH. The digest here is the same canonical_digest that
    build_plan reports, computed the same way, because a trace and the
    experiment that reads it must agree about which harness ran — otherwise the
    attribution is decorative.

    The audit that motivated this found data/promoted/train.jsonl to be 136
    traces, 136 model-attributed, 0 harness-attributed: nothing in that corpus
    can say whether a result came from the model or from the scaffolding around
    it. This block is what makes the difference recordable.

    It never invents a prompt hash. An unverified harness records
    prompt_sha256: null and prompt_verified: false, and only
    src/verify_harness_identity.py may fill it in from a real file. A harness
    claiming verified: true without a digest is not believed.

    Raises HarnessPlanError for a harness that would not be allowed to run: a
    trace should not carry attribution for a configuration that is refused.
    """
    validate_harness(spec)                     # unsafe harness -> refuse

    # A harness is model-COMPATIBLE, not model-bound. The four-arm design runs
    # ONE harness against the student and the teacher, so pinning a single
    # registry model here would make the teacher arms inexpressible. The spec
    # declares what it may run against; the trace records what actually did.
    contract = spec.get("model_contract") or {}
    compatible = contract.get("compatible_registry_models")
    if not isinstance(compatible, list) or not compatible:
        raise HarnessPlanError(
            f"harness {spec.get('name')!r} declares no "
            "model_contract.compatible_registry_models")
    if not execution_model_registry:
        raise HarnessPlanError(
            "execution_model_registry is required: a harness-attributed trace "
            "must say which model produced it")
    if execution_model_registry not in compatible:
        raise HarnessPlanError(
            f"model {execution_model_registry!r} is not compatible with harness "
            f"{spec.get('name')!r} (declared: {compatible}). Attribution for an "
            "undeclared pairing would be a guess.")

    prompt = ((spec.get("prompts") or {}).get("system") or {})
    sha = prompt.get("sha256")
    verified = prompt.get("verified") is True and isinstance(sha, str) \
        and bool(SHA256_RE.match(sha))

    return {
        "schema_version": HARNESS_PROVENANCE_SCHEMA,
        "name": spec.get("name"),
        "status": spec.get("status"),
        "digest": canonical_digest(spec),
        "compatible_registry_models": list(compatible),
        "execution_model_registry": execution_model_registry,
        "prompt_sha256": sha if verified else None,
        "prompt_verified": verified,
        "tool_policy_digest": _policy_digest(spec.get("tools")),
        "verification_policy_digest": _policy_digest(spec.get("verification")),
    }


def build_plan(experiment: dict[str, Any]) -> dict[str, Any]:
    for key in (
        "student_model",
        "teacher_model",
        "current_harness",
        "candidate_harness",
        "decision",
    ):
        if not experiment.get(key):
            raise HarnessPlanError(f"experiment missing {key!r}")
    if experiment.get("training_authorized") is not False:
        raise HarnessPlanError("experiment.training_authorized must be false")

    current = load_harness(str(experiment["current_harness"]))
    candidate = load_harness(str(experiment["candidate_harness"]))
    current_warnings = validate_harness(current)
    candidate_warnings = validate_harness(candidate)

    arms = experiment.get("arms") or list(REQUIRED_ARMS)
    missing = [arm for arm in REQUIRED_ARMS if arm not in arms]
    if missing:
        raise HarnessPlanError(f"factorial plan missing required arms: {missing}")

    return {
        "experiment": experiment.get("name"),
        "models": {
            "student": experiment["student_model"],
            "teacher": experiment["teacher_model"],
        },
        "harnesses": {
            "current": {
                "name": current["name"],
                "digest": canonical_digest(current),
                "warnings": current_warnings,
            },
            "candidate": {
                "name": candidate["name"],
                "digest": canonical_digest(candidate),
                "warnings": candidate_warnings,
            },
        },
        "required_arms": list(arms),
        "measurements_required": {
            "metric": experiment["decision"].get("capability_metric"),
            "guardrails": sorted(
                (experiment["decision"].get("guardrails") or {}).keys()
            ),
        },
        "weight_distillation_assumed": False,
        "training_authorized": False,
    }


def _number(value: Any, label: str) -> float:
    if not isinstance(value, (int, float)):
        raise HarnessPlanError(f"{label} must be numeric")
    return float(value)


def _metric(arms: dict[str, Any], arm: str, metric: str) -> float:
    if arm not in arms or not isinstance(arms[arm], dict):
        raise HarnessPlanError(f"measurements missing arm {arm!r}")
    return _number(arms[arm].get(metric), f"{arm}.{metric}")


def evaluate(experiment: dict[str, Any], measurements: dict[str, Any]) -> dict[str, Any]:
    """Evaluate the harness-first gate without authorizing training."""
    plan = build_plan(experiment)
    arms = measurements.get("arms")
    if not isinstance(arms, dict):
        raise HarnessPlanError("measurements.arms must be an object")

    decision = experiment["decision"]
    metric = str(decision.get("capability_metric") or "")
    if not metric:
        raise HarnessPlanError("decision.capability_metric is required")
    if decision.get("higher_is_better") is not True:
        raise HarnessPlanError("only higher_is_better metrics are supported in v1")

    target = _number(decision.get("target"), "decision.target")
    minimum_gap = _number(
        decision.get("minimum_residual_gap"), "decision.minimum_residual_gap"
    )
    noise = _number(decision.get("noise_floor"), "decision.noise_floor")

    current = _metric(arms, "student_current", metric)
    candidate = _metric(arms, "student_candidate", metric)
    teacher = _metric(arms, "teacher_current", metric)

    guardrail_failures: list[str] = []
    for name, rule in (decision.get("guardrails") or {}).items():
        if not isinstance(rule, dict):
            raise HarnessPlanError(f"guardrail {name!r} must be an object")
        minimum = _number(rule.get("minimum"), f"guardrail {name}.minimum")
        max_regression = _number(
            rule.get("max_regression"), f"guardrail {name}.max_regression"
        )
        base_value = _metric(arms, "student_current", name)
        candidate_value = _metric(arms, "student_candidate", name)
        if candidate_value < minimum:
            guardrail_failures.append(
                f"{name}: {candidate_value:.4f} below minimum {minimum:.4f}"
            )
        regression = base_value - candidate_value
        if regression > max_regression:
            guardrail_failures.append(
                f"{name}: regression {regression:.4f} exceeds "
                f"{max_regression:.4f}"
            )

    observed_gap = current < target
    harness_sufficient = candidate >= target and not guardrail_failures
    teacher_passes = teacher >= target
    residual_gap = teacher - candidate
    significant_residual = residual_gap >= max(minimum_gap, noise)

    reasons: list[str] = []
    if not observed_gap:
        reasons.append("student_current already meets the capability target")
    if harness_sufficient:
        reasons.append("candidate harness closes the capability gap")
    if not teacher_passes:
        reasons.append("teacher_current does not meet the capability target")
    if not significant_residual:
        reasons.append(
            "teacher advantage over optimized student does not exceed "
            "minimum effect and noise floor"
        )
    if guardrail_failures:
        reasons.append("candidate harness violates product guardrails")

    candidate_for_weights = (
        observed_gap
        and not harness_sufficient
        and teacher_passes
        and significant_residual
        and not guardrail_failures
    )

    return {
        **plan,
        "scores": {
            "student_current": current,
            "student_candidate": candidate,
            "teacher_current": teacher,
            "harness_gain": candidate - current,
            "residual_model_gap": residual_gap,
        },
        "thresholds": {
            "target": target,
            "minimum_residual_gap": minimum_gap,
            "noise_floor": noise,
        },
        "guardrail_failures": guardrail_failures,
        "harness_optimization_sufficient": harness_sufficient,
        "weight_distillation_candidate": candidate_for_weights,
        "reasons": reasons,
        "training_authorized": False,
    }


def audit_traces(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise HarnessPlanError(f"missing trace file: {path}")
    total = model_attributed = harness_attributed = 0
    harnesses: dict[str, int] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise HarnessPlanError(
                f"{path}:{line_number}: invalid JSON: {exc}"
            ) from exc
        if not isinstance(row, dict):
            raise HarnessPlanError(f"{path}:{line_number}: row must be an object")
        total += 1
        provenance = row.get("provenance") or {}
        if provenance.get("teacher") or provenance.get("model_id"):
            model_attributed += 1
        harness = row.get("harness_provenance") or {}
        digest = harness.get("digest")
        if isinstance(digest, str) and digest:
            harness_attributed += 1
            harnesses[digest] = harnesses.get(digest, 0) + 1
    return {
        "path": str(path),
        "traces": total,
        "model_attributed": model_attributed,
        "harness_attributed": harness_attributed,
        "harness_coverage": harness_attributed / total if total else 0.0,
        "harness_digests": harnesses,
        "distillation_attribution_ready": total > 0 and harness_attributed == total,
        "limits": (
            []
            if total > 0 and harness_attributed == total
            else [
                "trace quality cannot be attributed between model and harness "
                "without harness_provenance.digest"
            ]
        ),
        "training_authorized": False,
    }


def _json_file(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise HarnessPlanError(f"missing JSON file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HarnessPlanError(f"invalid JSON at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise HarnessPlanError(f"JSON root must be an object: {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    p = commands.add_parser("plan")
    p.add_argument("experiment", type=Path)

    e = commands.add_parser("evaluate")
    e.add_argument("experiment", type=Path)
    e.add_argument("measurements", type=Path)

    a = commands.add_parser("audit-traces")
    a.add_argument("traces", type=Path)

    args = parser.parse_args()
    try:
        if args.command == "plan":
            result = build_plan(load_yaml(args.experiment))
        elif args.command == "evaluate":
            result = evaluate(
                load_yaml(args.experiment), _json_file(args.measurements)
            )
        else:
            result = audit_traces(args.traces)
    except HarnessPlanError as exc:
        print(f"HARNESS DISTILLATION REFUSED: {exc}", file=sys.stderr)
        raise SystemExit(2)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

