"""Separation and stability gate for the RSI evaluation harness (Task 1.3).

The Recursive Self-Improvement Report v2 makes this the gate that everything
downstream is worth: "the harness reliably separates Tantular-9B from
Tantular-4B, and re-runs are stable within a declared noise floor. Until that
is true, no later phase is worth its compute."

Two decisions, both fail-closed:

* **Separation** — on every split, the 9B's pass rate must exceed the 4B's by
  more than the declared effect. Paired, because the arms answer the SAME
  items: an unpaired comparison of two differently-sized item pools measures
  the pools.
* **Stability** — repeated runs of the same arm must vary by no more than
  `noise_floor` (0.025 in `configs/experiments/harness-before-weights.yaml`).
  A separation smaller than the harness's own re-run noise is not a finding.

Refusals, in the repo's `die()`/`sys.exit` style:

* a measurement missing a split, an endpoint, or a model identity;
* the same model identity in both arms (a model does not separate from itself);
* a teacher identity in a student slot — the failure `run_gates.verify_served_model`
  exists to prevent, checked here against the RECORDED identity because this
  command is offline and never contacts an endpoint;
* per-item results that do not pair.

The paired statistics are IMPORTED from `src/measurement_report.py`, never
re-implemented. That module lives on `origin/main`; on a branch without it this
command refuses with an actionable message rather than quietly substituting a
weaker test.

NO NETWORK. NO MODEL. NO TRAINING. `train/TRAINING_BLOCKED.md` is controlling;
every verdict carries `training_authorized: false`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import gold_set as gs  # noqa: E402

EXPERIMENT_PATH = ROOT / "configs" / "experiments" / "harness-before-weights.yaml"

REQUIRED_MEASUREMENT_FIELDS = ("arm", "model_identity", "splits")
REQUIRED_IDENTITY_FIELDS = ("expected", "served", "endpoint")


class EvalHarnessError(Exception):
    """The gate could not run. Never the same thing as the gate failing."""


def die(msg: str, code: int = 2) -> None:
    print(f"\nSEPARATION GATE REFUSED: {msg}", file=sys.stderr)
    sys.exit(code)


def _statistics_module():
    """The repo's paired statistics, imported.

    Deliberately not vendored. `paired_table` and `mcnemar` already exist and
    are already tested; a second copy here would drift, and then this gate
    would measure our copy of the rules instead of the rules.
    """
    try:
        import measurement_report  # noqa: PLC0415
    except ImportError as error:
        raise EvalHarnessError(
            "src/measurement_report.py is not importable on this branch, so the "
            "paired statistics this gate requires are unavailable. It exists on "
            "origin/main; merge or cherry-pick it rather than re-implementing "
            f"the tests here. ({error})"
        ) from error
    return measurement_report


# --- loading and validation -------------------------------------------------


def load_measurement(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise EvalHarnessError(f"measurement file does not exist: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise EvalHarnessError(f"{path}: invalid JSON: {error}") from error
    if not isinstance(data, dict):
        raise EvalHarnessError(f"{path}: measurement must be a JSON object")
    return data


def validate_measurement(data: dict[str, Any], *, label: str,
                         require_splits: tuple[str, ...] = gs.SPLITS) -> list[str]:
    """Everything wrong with one measurement file. Empty means usable."""
    problems: list[str] = []

    for field in REQUIRED_MEASUREMENT_FIELDS:
        if field not in data:
            problems.append(f"{label}: {field} is required")

    identity = data.get("model_identity")
    if not isinstance(identity, dict) or not identity:
        problems.append(f"{label}: model_identity is required")
    else:
        for field in REQUIRED_IDENTITY_FIELDS:
            value = identity.get(field)
            if field == "served":
                if not isinstance(value, list) or not value:
                    problems.append(
                        f"{label}: model_identity.served must be a non-empty list")
            elif not isinstance(value, str) or not value.strip():
                problems.append(f"{label}: model_identity.{field} is required")

    splits = data.get("splits")
    if not isinstance(splits, dict):
        problems.append(f"{label}: splits must be an object")
        return problems

    for split in require_splits:
        if split not in splits:
            # A missing split is not a zero. Scoring it as one would invent a
            # measurement nobody took.
            problems.append(f"{label}: split {split!r} is missing")
            continue
        entry = splits[split]
        if not isinstance(entry, dict):
            problems.append(f"{label}: split {split!r} must be an object")
            continue
        per_item = entry.get("per_item")
        if not isinstance(per_item, dict) or not per_item:
            problems.append(
                f"{label}: split {split!r} needs per_item results to pair on")
        elif any(not isinstance(v, bool) for v in per_item.values()):
            problems.append(
                f"{label}: split {split!r} per_item values must be booleans")

    unknown = sorted(set(splits) - set(require_splits))
    if unknown:
        problems.append(f"{label}: unknown split(s) {unknown!r}")

    return problems


def identity_is_teacher(identity: dict[str, Any], teachers: list[str]) -> str | None:
    """The teacher-in-a-student-slot check, against the RECORDED identity.

    `run_gates.verify_served_model` does this live against an endpoint. This
    command is offline by contract, so it checks what the measurement recorded
    — which is the only claim available after the fact anyway.
    """
    if not teachers:
        return None
    try:
        from model_ids import any_match  # noqa: PLC0415
    except ImportError:
        def any_match(expected: str, served: list[Any]) -> bool:  # type: ignore
            return any(str(s).strip() == str(expected).strip() for s in served)

    served = [s for s in identity.get("served", [])]
    for teacher in teachers:
        if any_match(teacher, served) or any_match(teacher, [identity.get("expected")]):
            return teacher
    return None


def pass_rate(entry: dict[str, Any]) -> float:
    per_item = entry["per_item"]
    if not per_item:
        return 0.0
    return sum(1 for v in per_item.values() if v) / len(per_item)


def run_variation(entry: dict[str, Any]) -> float | None:
    """Spread across repeated runs of the same arm, or None if not repeated.

    Range rather than standard deviation: with two or three re-runs the range
    is what an operator can check by eye against the noise floor, and sd of
    n=2 understates it.
    """
    runs = entry.get("runs")
    if not isinstance(runs, list) or len(runs) < 2:
        return None
    values: list[float] = []
    for run in runs:
        if isinstance(run, dict) and isinstance(run.get("pass_rate"), (int, float)):
            values.append(float(run["pass_rate"]))
        elif isinstance(run, (int, float)):
            values.append(float(run))
    if len(values) < 2:
        return None
    return max(values) - min(values)


# --- the gate ---------------------------------------------------------------


def separation_gate(experiment: dict[str, Any],
                    stronger: dict[str, Any],
                    weaker: dict[str, Any],
                    *, stronger_label: str = "student_9b",
                    weaker_label: str = "student_4b") -> dict[str, Any]:
    """Decide separation and stability. Raises on anything that cannot be decided."""
    decision = experiment.get("decision")
    if not isinstance(decision, dict):
        raise EvalHarnessError("experiment.decision is required")
    for field in ("minimum_residual_gap", "noise_floor"):
        if not isinstance(decision.get(field), (int, float)):
            raise EvalHarnessError(f"decision.{field} must be numeric")
    minimum_effect = float(decision["minimum_residual_gap"])
    noise_floor = float(decision["noise_floor"])

    problems = (validate_measurement(stronger, label=stronger_label)
                + validate_measurement(weaker, label=weaker_label))
    if problems:
        raise EvalHarnessError("; ".join(problems))

    a_identity = stronger["model_identity"]
    b_identity = weaker["model_identity"]
    if str(a_identity.get("expected")).strip() == str(b_identity.get("expected")).strip():
        raise EvalHarnessError(
            f"both arms report the same model identity "
            f"({a_identity.get('expected')!r}); a model does not separate from itself"
        )

    teachers = [t for t in (experiment.get("teacher_model"),) if t]
    for label, identity in ((stronger_label, a_identity), (weaker_label, b_identity)):
        found = identity_is_teacher(identity, teachers)
        if found:
            raise EvalHarnessError(
                f"{label} is serving the teacher {found!r}. Gating a teacher in a "
                "student slot measures nothing."
            )

    stats = _statistics_module()

    splits_report: dict[str, Any] = {}
    separated_everywhere = True
    stable_everywhere = True

    for split in gs.SPLITS:
        a_entry = stronger["splits"][split]
        b_entry = weaker["splits"][split]
        a_rate = pass_rate(a_entry)
        b_rate = pass_rate(b_entry)
        delta = a_rate - b_rate

        table = stats.paired_table(a_entry["per_item"], b_entry["per_item"])
        test = stats.mcnemar(table)

        a_var = run_variation(a_entry)
        b_var = run_variation(b_entry)
        # Stability is a property of BOTH arms. If either lacks repeated runs
        # its variance is unknown, and an unknown variance is not a small one:
        # taking the max of whatever happens to be present would report a
        # stable comparison on the strength of one arm's evidence.
        if a_var is None or b_var is None:
            worst_variation = None
            stable = None
        else:
            worst_variation = max(a_var, b_var)
            stable = worst_variation <= noise_floor

        # A separation the harness's own re-run noise could produce is not a
        # separation. Both conditions must hold.
        exceeds_effect = delta > minimum_effect
        above_noise = delta > noise_floor
        separated = bool(exceeds_effect and above_noise)

        if not separated:
            separated_everywhere = False
        if stable is False:
            stable_everywhere = False

        reasons: list[str] = []
        if not exceeds_effect:
            reasons.append(
                f"delta {delta:.4f} does not exceed declared effect {minimum_effect:.4f}")
        if not above_noise:
            reasons.append(
                f"delta {delta:.4f} is within the noise floor {noise_floor:.4f}")
        if stable is False:
            reasons.append(
                f"re-run variation {worst_variation:.4f} exceeds noise floor "
                f"{noise_floor:.4f}")
        if stable is None:
            which = [label for label, var in
                     ((stronger_label, a_var), (weaker_label, b_var)) if var is None]
            reasons.append(
                "stability unproven: fewer than two runs recorded for "
                f"{', '.join(which)} on this split")

        splits_report[split] = {
            f"{stronger_label}_pass_rate": a_rate,
            f"{weaker_label}_pass_rate": b_rate,
            "delta": delta,
            "minimum_effect": minimum_effect,
            "noise_floor": noise_floor,
            "separated": separated,
            "stable": stable,
            "run_variation": worst_variation,
            "paired": table,
            "mcnemar": test,
            "reasons": reasons,
        }

    unproven = [s for s, r in splits_report.items() if r["stable"] is None]
    passed = bool(separated_everywhere and stable_everywhere and not unproven)

    return {
        "gate": "separation",
        "experiment": experiment.get("name"),
        "arms": {
            stronger_label: a_identity.get("expected"),
            weaker_label: b_identity.get("expected"),
        },
        "splits": splits_report,
        "separated_on_every_split": separated_everywhere,
        "stable_on_every_split": stable_everywhere and not unproven,
        # "measured" is not "passed" — the run_gates distinction.
        "measured": True,
        "passed": passed,
        "training_authorized": False,
    }


# --- CLI --------------------------------------------------------------------


def _load_experiment(path: Path) -> dict[str, Any]:
    try:
        import yaml  # noqa: PLC0415
    except ImportError as error:  # pragma: no cover
        raise EvalHarnessError(f"pyyaml is required: {error}") from error
    path = Path(path)
    if not path.exists():
        raise EvalHarnessError(f"experiment file does not exist: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise EvalHarnessError(f"{path}: experiment must be a mapping")
    return data


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="RSI evaluation harness. Offline; contacts no endpoint.")
    sub = parser.add_subparsers(dest="command", required=True)

    gate = sub.add_parser(
        "separation-gate",
        help="decide whether the harness separates 9B from 4B, stably")
    gate.add_argument("--experiment", default=str(EXPERIMENT_PATH))
    gate.add_argument("--stronger", required=True,
                      help="measurement file for the larger model (9B)")
    gate.add_argument("--weaker", required=True,
                      help="measurement file for the smaller model (4B)")
    gate.add_argument("--stronger-label", default="student_9b")
    gate.add_argument("--weaker-label", default="student_4b")
    gate.add_argument("--json", action="store_true")

    args = parser.parse_args(argv)

    try:
        experiment = _load_experiment(Path(args.experiment))
        verdict = separation_gate(
            experiment,
            load_measurement(Path(args.stronger)),
            load_measurement(Path(args.weaker)),
            stronger_label=args.stronger_label,
            weaker_label=args.weaker_label,
        )
    except EvalHarnessError as error:
        die(str(error))
        return

    if args.json:
        print(json.dumps(verdict, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(f"separation gate: {verdict['experiment']}")
        for split, row in verdict["splits"].items():
            mark = "ok " if row["separated"] else "NO "
            print(f"  {mark} {split:<24} delta={row['delta']:+.4f} "
                  f"stable={row['stable']} p={row['mcnemar']['p_value']:.4f}")
            for reason in row["reasons"]:
                print(f"       - {reason}")
        print(f"  passed: {verdict['passed']}")
        print(f"  training_authorized: {verdict['training_authorized']}")

    if not verdict["passed"]:
        die("the harness does not yet separate the models stably on every split", 1)


if __name__ == "__main__":
    main()
