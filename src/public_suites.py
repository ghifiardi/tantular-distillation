"""SEA-HELM and IndoMMLU adapters — REPORTING ONLY (Phase 1, Task 1.4).

The Recursive Self-Improvement Report v2 §G is explicit about what these are
for: "Public suites exist and you should run them, but they are for reporting,
not for steering the loop."

So this module formats results that someone else produced. It does NOT:

* download, vendor, run, or shell out to either suite;
* contact the network, under any code path;
* contribute to `capability_pass_rate`, the separation gate, GEPA's acceptance
  decision, or any other number that steers the loop.

Every normalized result carries `steers_loop: false`, and
`attach_public_suites` writes them into a clearly separated `public_suites`
section of a report rather than mixing them into measured splits.

The report's own calibration warning is carried through as well: a July 2026
round-up placed Qwen3 VL 32B top of SEA-HELM Indonesian at 68.41, and the
report notes that is "not a target a 9B should be measured against". A number
here is context, not a goal.

Absence and breakage are treated differently, on purpose:

* **absent** — no file supplied is the normal state, and the suite is simply
  missing from the report. Nothing is fabricated, and no zero is invented.
* **present but malformed** — refuses. A result file that cannot be read is a
  broken measurement, and reporting it as missing would hide that.

NO NETWORK. NO MODEL. NO TRAINING. `train/TRAINING_BLOCKED.md` is controlling.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent

SUITES = ("sea_helm", "indommlu")

# SEA-HELM's five pillars (AI Singapore, Findings of ACL 2025). Listed so an
# unexpected pillar name is visible rather than silently averaged in.
SEA_HELM_PILLARS = (
    "nlp_classics", "llm_specifics", "sea_linguistics", "sea_culture", "safety",
)


class PublicSuiteError(Exception):
    """A supplied result file could not be read. Not the same as absent."""


def die(msg: str, code: int = 2) -> None:
    print(f"\nPUBLIC SUITE REFUSED: {msg}", file=sys.stderr)
    sys.exit(code)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise PublicSuiteError(f"{path}: not found") from error
    except json.JSONDecodeError as error:
        raise PublicSuiteError(f"{path}: invalid JSON: {error}") from error


SCALE_BOUNDS = {"0-1": (0.0, 1.0), "0-100": (0.0, 100.0)}


def _score(value: Any, where: str, scale: str) -> float:
    """A score, checked against the bounds of its OWN declared scale.

    Checking against the widest possible range would let 68.41 through a file
    declaring `scale: 0-1`, which is the precise confusion the `scale` field
    exists to prevent: the number is then silently a hundredfold wrong and
    every comparison drawn from it is meaningless.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PublicSuiteError(f"{where}: score must be numeric, got {value!r}")
    number = float(value)
    bounds = SCALE_BOUNDS.get(scale)
    if bounds is None:
        raise PublicSuiteError(f"{where}: unknown scale {scale!r}")
    low, high = bounds
    if not low <= number <= high:
        raise PublicSuiteError(
            f"{where}: score {number} is outside the declared scale "
            f"{scale!r} ({low}-{high})")
    return number


def normalize_sea_helm(data: Any, *, source: str = "") -> dict[str, Any]:
    """SEA-HELM results into the harness's report shape."""
    if not isinstance(data, dict):
        raise PublicSuiteError(f"{source or 'sea_helm'}: result must be an object")

    model = data.get("model")
    if not isinstance(model, str) or not model.strip():
        raise PublicSuiteError(f"{source or 'sea_helm'}: model is required")

    scale = data.get("scale")
    if scale not in ("0-1", "0-100"):
        raise PublicSuiteError(
            f"{source or 'sea_helm'}: scale must be '0-1' or '0-100', got "
            f"{scale!r}. Rescaling by guess turns 0.68 into 68.")

    pillars_in = data.get("pillars")
    if not isinstance(pillars_in, dict) or not pillars_in:
        raise PublicSuiteError(f"{source or 'sea_helm'}: pillars are required")

    unknown = sorted(set(pillars_in) - set(SEA_HELM_PILLARS))
    if unknown:
        raise PublicSuiteError(
            f"{source or 'sea_helm'}: unknown pillar(s) {unknown!r}; known "
            f"{list(SEA_HELM_PILLARS)}")

    pillars = {name: _score(value, f"sea_helm.pillars.{name}", scale)
               for name, value in sorted(pillars_in.items())}

    return {
        "suite": "sea_helm",
        "model": model,
        "scale": scale,
        "pillars": pillars,
        "language": data.get("language", "id"),
        "reported_at": data.get("reported_at"),
        "source_file": source,
        # The whole point of this module.
        "steers_loop": False,
        "role": "reporting_and_calibration_only",
    }


def normalize_indommlu(data: Any, *, source: str = "") -> dict[str, Any]:
    """IndoMMLU results into the harness's report shape."""
    if not isinstance(data, dict):
        raise PublicSuiteError(f"{source or 'indommlu'}: result must be an object")

    model = data.get("model")
    if not isinstance(model, str) or not model.strip():
        raise PublicSuiteError(f"{source or 'indommlu'}: model is required")

    scale = data.get("scale")
    if scale not in ("0-1", "0-100"):
        raise PublicSuiteError(
            f"{source or 'indommlu'}: scale must be '0-1' or '0-100'")

    subjects_in = data.get("subjects")
    if not isinstance(subjects_in, dict) or not subjects_in:
        raise PublicSuiteError(f"{source or 'indommlu'}: subjects are required")

    subjects = {name: _score(value, f"indommlu.subjects.{name}", scale)
                for name, value in sorted(subjects_in.items())}

    overall = data.get("overall")
    if overall is None:
        # Derived, and SAID to be derived. A mean of subject scores is not the
        # suite's own weighting, and presenting it as such would misreport.
        overall_value = _score(sum(subjects.values()) / len(subjects),
                               "indommlu.overall(derived)", scale)
        overall_origin = "derived_unweighted_mean"
    else:
        overall_value = _score(overall, "indommlu.overall", scale)
        overall_origin = "reported"

    return {
        "suite": "indommlu",
        "model": model,
        "scale": scale,
        "subjects": subjects,
        "overall": overall_value,
        "overall_origin": overall_origin,
        "reported_at": data.get("reported_at"),
        "source_file": source,
        "steers_loop": False,
        "role": "reporting_and_calibration_only",
    }


NORMALIZERS = {
    "sea_helm": normalize_sea_helm,
    "indommlu": normalize_indommlu,
}


def load_public_suites(paths: dict[str, Path | str | None]) -> dict[str, Any]:
    """Normalize whichever suites were supplied.

    A suite whose path is None or omitted is absent from the result. A suite
    whose path is given but unreadable raises.
    """
    out: dict[str, Any] = {}
    for suite, path in (paths or {}).items():
        if suite not in NORMALIZERS:
            raise PublicSuiteError(
                f"unknown suite {suite!r}; known {list(NORMALIZERS)}")
        if path is None:
            continue
        out[suite] = NORMALIZERS[suite](_read_json(Path(path)), source=str(path))
    return out


def attach_public_suites(report: dict[str, Any],
                         suites: dict[str, Any]) -> dict[str, Any]:
    """Add a separated `public_suites` section to a harness report.

    Returns a new dict; the caller's report is not mutated. The section is
    omitted entirely when nothing was supplied, so a report never carries an
    empty shell implying a measurement was attempted.
    """
    updated = dict(report)
    if not suites:
        updated.pop("public_suites", None)
        return updated
    updated["public_suites"] = {
        "note": (
            "Reporting and calibration only. Per the RSI report section G these "
            "suites do not steer the loop and contribute to no gate."
        ),
        "steers_loop": False,
        "results": dict(sorted(suites.items())),
    }
    return updated


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Normalize SEA-HELM / IndoMMLU results. Reporting only; "
                    "runs nothing and downloads nothing.")
    parser.add_argument("--sea-helm", default=None)
    parser.add_argument("--indommlu", default=None)
    parser.add_argument("--report", default=None,
                        help="optional harness report JSON to attach to")
    args = parser.parse_args(argv)

    try:
        suites = load_public_suites(
            {"sea_helm": args.sea_helm, "indommlu": args.indommlu})
        base = _read_json(Path(args.report)) if args.report else {}
        if not isinstance(base, dict):
            raise PublicSuiteError(f"{args.report}: report must be an object")
        output = attach_public_suites(base, suites)
    except PublicSuiteError as error:
        die(str(error))
        return

    print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
