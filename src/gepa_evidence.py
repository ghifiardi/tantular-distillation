"""Feed a GEPA result into the harness-before-weights gate (Task 2.2).

GEPA optimises the harness. `configs/experiments/harness-before-weights.yaml`
defines the decision that optimisation feeds: is the capability gap closed by
the harness, or is there a residual the weights would have to close? GEPA
supplies exactly one arm of that experiment — `student_candidate` — and
nothing else.

This module is deliberately thin, and deliberately refuses in two places:

1. **It does not decide anything.** `harness_distill.evaluate` is called
   unchanged. Harness gain and residual model gap are computed there, by the
   gate that already exists, with the rules it already has.

2. **It will not invent a guardrail measurement.** The gate checks
   `student_candidate` against `indonesian_voice` and `edit_contract_output`
   with `max_regression: 0.00`. GEPA optimises the capability metric and does
   not measure those. Copying the current arm's values across would assert "no
   regression" on evidence nobody gathered — the exact claim the guardrail
   exists to test — so a GEPA result without its own guardrail measurements is
   refused.

Every emitted verdict carries `training_authorized: false`, unchanged from
`harness_distill`. `train/TRAINING_BLOCKED.md` is controlling.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import harness_distill as hd  # noqa: E402

CANDIDATE_ARM = "student_candidate"


class GepaEvidenceError(Exception):
    """The GEPA result cannot be turned into an arm measurement."""


def die(msg: str, code: int = 2) -> None:
    print(f"\nGEPA EVIDENCE REFUSED: {msg}", file=sys.stderr)
    sys.exit(code)


def _held_out_score(result: dict[str, Any]) -> float:
    best = result.get("best")
    if not isinstance(best, dict):
        raise GepaEvidenceError("gepa result has no 'best' candidate")
    score = best.get("score")
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise GepaEvidenceError("gepa result best.score must be numeric")
    if not 0.0 <= float(score) <= 1.0:
        raise GepaEvidenceError(
            f"gepa result best.score {score} is not a rate in [0, 1]")
    return float(score)


def arm_from_gepa(result: dict[str, Any], *, metric: str,
                  guardrails: tuple[str, ...] | list[str]) -> dict[str, Any]:
    """The `student_candidate` arm measurement implied by a GEPA run.

    Requires the run to carry its own guardrail measurements. See the module
    docstring for why they are not defaulted.
    """
    if not isinstance(result, dict):
        raise GepaEvidenceError("gepa result must be an object")
    if result.get("training_authorized") is not False:
        raise GepaEvidenceError(
            "gepa result must carry training_authorized: false")
    for field in ("harness_identity", "model_identity"):
        value = result.get(field)
        if not isinstance(value, str) or not value.strip():
            raise GepaEvidenceError(
                f"gepa result carries no {field}; an unattributable result "
                "cannot populate an experiment arm")

    arm: dict[str, Any] = {metric: _held_out_score(result)}

    measured = result.get("guardrails")
    if not isinstance(measured, dict):
        measured = {}
    missing = [g for g in guardrails if g not in measured]
    if missing:
        raise GepaEvidenceError(
            f"gepa result does not measure guardrail(s) {missing!r}. The gate "
            "checks them with max_regression 0.00; carrying the current arm's "
            "values across would assert no regression on evidence nobody "
            "gathered. Measure them for the candidate harness and re-run.")
    for name in guardrails:
        value = measured[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise GepaEvidenceError(f"guardrail {name!r} must be numeric")
        arm[name] = float(value)

    arm["provenance"] = {
        "source": "gepa",
        "harness_identity": result["harness_identity"],
        "model_identity": result["model_identity"],
        "rollouts": result.get("rollouts"),
        "slice": result.get("slice"),
        "best_candidate": (result.get("best") or {}).get("id"),
    }
    return arm


def merge_candidate_arm(measurements: dict[str, Any], result: dict[str, Any], *,
                        metric: str,
                        guardrails: tuple[str, ...] | list[str]) -> dict[str, Any]:
    """Return measurements with `student_candidate` populated from GEPA.

    Does not mutate the caller's object. Refuses to overwrite an existing
    `student_candidate`: two different measurements of the same arm is a
    conflict a person should resolve, not something a script should pick a
    winner for.
    """
    if not isinstance(measurements, dict):
        raise GepaEvidenceError("measurements must be an object")
    arms = measurements.get("arms")
    if not isinstance(arms, dict):
        raise GepaEvidenceError("measurements.arms must be an object")
    if CANDIDATE_ARM in arms:
        raise GepaEvidenceError(
            f"measurements already contain {CANDIDATE_ARM!r}; refusing to "
            "overwrite an existing measurement with a GEPA result")

    merged = dict(measurements)
    merged["arms"] = {**arms,
                      CANDIDATE_ARM: arm_from_gepa(result, metric=metric,
                                                   guardrails=guardrails)}
    return merged


def evaluate_with_gepa(experiment: dict[str, Any], measurements: dict[str, Any],
                       result: dict[str, Any]) -> dict[str, Any]:
    """Populate the candidate arm, then hand the whole thing to the real gate."""
    decision = experiment.get("decision")
    if not isinstance(decision, dict):
        raise GepaEvidenceError("experiment.decision is required")
    metric = decision.get("capability_metric")
    if not isinstance(metric, str) or not metric:
        raise GepaEvidenceError("decision.capability_metric is required")
    guardrails = tuple((decision.get("guardrails") or {}).keys())

    merged = merge_candidate_arm(measurements, result,
                                 metric=metric, guardrails=guardrails)
    verdict = hd.evaluate(experiment, merged)

    # The gate's own decision is passed through untouched; only provenance is
    # added, so a reader can see which arm came from GEPA.
    verdict["candidate_arm_provenance"] = \
        merged["arms"][CANDIDATE_ARM]["provenance"]
    return verdict


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Feed a GEPA result into harness_distill.evaluate as the "
                    "student_candidate arm. Decides nothing itself.")
    parser.add_argument("--experiment", required=True, type=Path)
    parser.add_argument("--measurements", required=True, type=Path)
    parser.add_argument("--gepa", required=True, type=Path)
    args = parser.parse_args(argv)

    try:
        experiment = hd.load_yaml(args.experiment)
        measurements = json.loads(args.measurements.read_text(encoding="utf-8"))
        result = json.loads(args.gepa.read_text(encoding="utf-8"))
        verdict = evaluate_with_gepa(experiment, measurements, result)
    except (GepaEvidenceError, hd.HarnessPlanError) as error:
        die(str(error))
        return
    except (OSError, json.JSONDecodeError) as error:
        die(f"cannot read input: {error}")
        return

    print(json.dumps(verdict, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
