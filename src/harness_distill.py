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


def generation_support(spec: dict[str, Any]) -> list[str]:
    """Why src/generate.py may not stamp this harness onto a trace.

    generate.py sends a prompt through an ordinary chat client. It supplies no
    tools, runs no before/after verifier, enforces no memory or approval policy,
    and provides no companion-process isolation. A harness declaring any of
    those describes an execution that generate.py cannot perform, so stamping
    its full identity onto a trace would assert that policy ran when it did not
    — which is precisely the attribution ambiguity harness_provenance exists to
    remove.

    A harness must therefore OPT IN via trace_generation.supported_by_generate_py,
    and may only do so if it claims nothing generate.py cannot honour. Returns
    the reasons it may not; empty means it may.
    """
    reasons: list[str] = []
    declared = ((spec.get("trace_generation") or {})
                .get("supported_by_generate_py"))
    if declared is not True:
        reasons.append(
            "trace_generation.supported_by_generate_py is not true. "
            "src/generate.py has no harness executor: it cannot supply the "
            "declared tools, run the verifiers, or enforce the memory and "
            "approval policy, so it must not claim that it did.")
        return reasons

    tools = (spec.get("tools") or {}).get("allow") or []
    if tools:
        reasons.append(f"the harness declares tools {sorted(tools)}, which "
                       "generate.py does not supply to the model")
    verification = spec.get("verification") or {}
    for phase in ("before_action", "after_action"):
        declared_checks = verification.get(phase) or []
        if declared_checks:
            reasons.append(f"the harness declares {phase} {sorted(declared_checks)}, "
                           "which generate.py does not run")
    if (verification.get("repair_attempts") or 0):
        reasons.append("the harness declares a repair loop, which generate.py "
                       "does not run")
    return reasons


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


# The shape every consumer records. Fixed keys, present in both the attributed
# and the legacy case, so a reader never has to distinguish "absent" from
# "false" — that ambiguity is how a declaration quietly disappears between the
# pass manifest, promotion, the freeze and the trainer.
_LEGACY_SUMMARY = {
    "required": False,
    "attributed": False,
    "name": None,
    "digest": None,
    "prompt_verified": False,
    "prompt_sha256": None,
    "execution_model_registry": None,
}


def summarize_harness_attribution(traces: list[dict[str, Any]], *,
                                  required: bool) -> dict[str, Any]:
    """The single answer to "is this corpus harness-attributed, and by what?".

    Reused by the pass manifest, promotion, the freeze, the trainer, the audit
    and the corpus gate. Six implementations of this question would be six ways
    for those to disagree, and the disagreement would surface only as a corpus
    nobody can explain.

    `required` is DECLARED, never inferred from the traces. Inferring it would
    mean a corpus that lost its attribution reads as a valid legacy corpus,
    which is exactly the silent failure this prevents.

    Refuses rather than summarizing when the corpus is not internally coherent:
    a partially attributed file is neither a valid legacy corpus nor a valid
    harness-aware one, and two execution models in one file is the confound the
    factorial arms exist to remove — those arms are separate generation passes.
    """
    if not traces:
        raise HarnessPlanError(
            "cannot summarize harness attribution of an empty corpus")

    # PRESENCE, not truthiness. `{"harness_provenance": {}}` is not the same
    # claim as a trace that never carried the key: the first says "this was
    # attributed" and then says nothing, which is a malformed record, while the
    # second is an ordinary legacy trace. Treating them alike let an empty block
    # pass as legacy and defeated the partial-attribution guarantee.
    present = [("harness_provenance" in t) for t in traces]
    blocks = [t.get("harness_provenance") for t in traces]

    malformed = [i for i, (has, b) in enumerate(zip(present, blocks), 1)
                 if has and not (isinstance(b, dict) and b)]
    if malformed:
        raise HarnessPlanError(
            f"{len(malformed)} trace(s) carry a harness_provenance key that is "
            "not a non-empty mapping (first at record "
            f"{malformed[0]}). An attribution block that says nothing is a "
            "malformed record, not an absent one.")

    attributed = [b for b, has in zip(blocks, present) if has]

    if not required:
        if attributed:
            raise HarnessPlanError(
                f"{len(attributed)}/{len(blocks)} trace(s) carry harness "
                "attribution but harness-aware mode was not declared. Declare it "
                "explicitly (--harness-aware); a corpus is not treated as "
                "harness-aware by accident.")
        return dict(_LEGACY_SUMMARY)

    missing = len(blocks) - len(attributed)
    if missing:
        raise HarnessPlanError(
            f"harness-aware mode is declared but {missing}/{len(blocks)} "
            "trace(s) are not attributed. A partially attributed corpus is "
            "neither a valid legacy corpus nor a valid harness-aware one.")

    def one(field: str, label: str) -> Any:
        seen = sorted({json.dumps(b.get(field), sort_keys=True) for b in attributed})
        if len(seen) != 1:
            raise HarnessPlanError(
                f"the corpus mixes {len(seen)} {label}: "
                + ", ".join(s[:24] for s in seen))
        return json.loads(seen[0])

    digest = one("digest", "harness digests")
    name = one("name", "harness names")
    prompt_sha = one("prompt_sha256", "harness prompt hashes")
    model = one("execution_model_registry", "execution models")
    verified = one("prompt_verified", "prompt verification states")

    if verified is not True or not isinstance(prompt_sha, str) or \
            not re.fullmatch(r"[0-9a-f]{64}", prompt_sha):
        raise HarnessPlanError(
            "harness-aware corpus carries an unverified prompt identity "
            f"(prompt_verified={verified!r}). Attribution naming a prompt "
            "nobody checked is worse than none: it looks like evidence.")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise HarnessPlanError(f"harness digest is not a sha256: {digest!r}")
    if not model:
        raise HarnessPlanError("harness-aware corpus names no execution model")

    return {
        "required": True,
        "attributed": True,
        "name": name,
        "digest": digest,
        "prompt_verified": True,
        "prompt_sha256": prompt_sha,
        "execution_model_registry": model,
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
    total = model_attributed = harness_attributed = malformed = 0
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
        # PRESENCE, not truthiness — the same distinction the summary makes.
        # Counting a malformed block as "not attributed" reported a corpus with
        # broken attribution as a clean legacy corpus, which is the reading this
        # whole module exists to prevent.
        if "harness_provenance" in row:
            harness = row.get("harness_provenance")
            digest = harness.get("digest") if isinstance(harness, dict) else None
            if isinstance(digest, str) and digest:
                harness_attributed += 1
                harnesses[digest] = harnesses.get(digest, 0) + 1
            else:
                malformed += 1
    return {
        "path": str(path),
        "traces": total,
        "model_attributed": model_attributed,
        "harness_attributed": harness_attributed,
        "harness_malformed": malformed,
        "harness_coverage": harness_attributed / total if total else 0.0,
        "harness_digests": harnesses,
        "distillation_attribution_ready": (total > 0 and harness_attributed == total
                                           and not malformed),
        "limits": (
            [] if total > 0 and harness_attributed == total and not malformed
            else ([
                f"{malformed} trace(s) carry a harness_provenance key with no "
                "usable digest. That is a MALFORMED attribution, not an absent "
                "one, and it must not be read as a legacy corpus."
            ] if malformed else []) + ([
                "trace quality cannot be attributed between model and harness "
                "without harness_provenance.digest"
            ] if harness_attributed != total else [])
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

