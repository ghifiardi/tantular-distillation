"""Statistics for the four-arm comparison, and whether it qualifies at all.

Every case runs in ALL FOUR arms, so the comparison is PAIRED. Treating it as
two independent samples throws away most of the power: detecting a 0.05
difference near 0.90 needs ~686 cases per arm unpaired, against ~312 paired at
10% discordance. So the default test here is McNemar on the discordant pairs.

THE POINT OF THIS MODULE IS THE REFUSAL. A rate computed over ten cases is
arithmetic, not evidence: at n=10 the score can only move in steps of 0.1 -- 4x
coarser than the declared 0.025 noise floor -- and its 95% interval is about
+/-0.19, which spans the entire region the decision cares about. The report
still computes descriptive numbers, because seeing them is useful, but it says
`statistically_qualified: false` and names why.

n IS THE NUMBER OF UNIQUE CASES. Repetitions of the same case are correlated
observations of one question; adding them to n would shrink every interval by a
factor the data does not contain. Repetitions are aggregated per case first.

NO NETWORK, NO MODEL, NO OFFICE ACTION. Standard library only, so the numbers
do not depend on a version of anything.
"""
from __future__ import annotations

import math
import random
import sys
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from harness_eval import HarnessEvalError                      # noqa: E402

MINIMUM_INDEPENDENT_CASES = 320

# Splits that can never be production-qualified, whatever their size. A fixture
# replays declared outcomes and a pilot was not reviewed; either one reaching
# 320 cases would still not be a measurement of the product.
UNQUALIFIABLE_SPLITS = ("fixture", "pilot", "development", "preflight")


def wilson_interval(successes: int, n: int, z: float = 1.959964
                    ) -> tuple[float, float]:
    """Wilson score interval. Chosen over the normal approximation because the
    rates here sit near 0.9-1.0, where the normal interval runs past 1.0 and
    quietly reports an impossible bound."""
    if n <= 0:
        raise HarnessEvalError("a confidence interval over zero cases is not a "
                               "wide interval; it is no interval")
    if not 0 <= successes <= n:
        raise HarnessEvalError(f"successes {successes} outside 0..{n}")
    p = successes / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def paired_table(a: dict[str, bool], b: dict[str, bool]) -> dict[str, int]:
    """The 2x2 over cases present in BOTH arms.

    A case missing from either arm is not a zero; it is an unpaired
    observation, and silently treating it as a failure would invent data.
    """
    shared = sorted(set(a) & set(b))
    missing = sorted((set(a) ^ set(b)))
    if missing:
        raise HarnessEvalError(
            f"{len(missing)} case(s) appear in only one arm (first: "
            f"{missing[0]!r}). A paired test needs pairs; an unpaired case is "
            "not a failure.")
    both = sum(1 for c in shared if a[c] and b[c])
    only_a = sum(1 for c in shared if a[c] and not b[c])
    only_b = sum(1 for c in shared if not a[c] and b[c])
    neither = sum(1 for c in shared if not a[c] and not b[c])
    return {"both_pass": both, "only_first": only_a, "only_second": only_b,
            "neither": neither, "n": len(shared),
            "discordant": only_a + only_b}


def _chi2_sf_1df(x: float) -> float:
    """Upper tail of chi-square with 1 df = erfc(sqrt(x/2)).

    Exact via math.erfc rather than a table, so the p-value does not depend on
    a lookup nobody can check.
    """
    if x <= 0:
        return 1.0
    return math.erfc(math.sqrt(x / 2.0))


def mcnemar(table: dict[str, int], *, continuity: bool = True) -> dict[str, Any]:
    """McNemar's test on the discordant pairs.

    Only the discordant cells carry information: cases both arms got right, or
    both got wrong, say nothing about a DIFFERENCE between them.
    """
    b, c = table["only_first"], table["only_second"]
    discordant = b + c
    if discordant == 0:
        return {"statistic": 0.0, "p_value": 1.0, "discordant": 0,
                "continuity_correction": continuity,
                "note": "no discordant pairs: the arms agreed on every case, so "
                        "there is no difference to test"}
    delta = abs(b - c)
    if continuity:
        delta = max(0.0, delta - 1.0)      # Edwards' correction
    statistic = (delta * delta) / discordant
    return {"statistic": statistic, "p_value": _chi2_sf_1df(statistic),
            "discordant": discordant, "only_first": b, "only_second": c,
            "continuity_correction": continuity}


def collapse_repetitions(rows: Iterable[dict[str, Any]]) -> dict[str, bool]:
    """One outcome per case id.

    A case is a pass only if EVERY repetition passed. Any-pass would reward
    resampling until something works, which is the opposite of what
    repetitions are for.
    """
    by_case: dict[str, bool] = {}
    for row in rows:
        case_id = str(row["case_id"])
        passed = bool(row["passed"])
        by_case[case_id] = passed if case_id not in by_case else (
            by_case[case_id] and passed)
    return by_case


def paired_bootstrap_ci(a: dict[str, bool], b: dict[str, bool], *,
                        seed: int, resamples: int = 2000,
                        alpha: float = 0.05) -> dict[str, Any]:
    """CI for the paired difference, resampling CASES (not observations).

    Clustered by construction: a case is drawn whole, so its repetitions cannot
    be split across resamples and the correlation between them is preserved.
    The seed is required and reported -- an unseeded bootstrap is not a
    reproducible number.
    """
    cases = sorted(set(a) & set(b))
    if not cases:
        raise HarnessEvalError("no shared cases to bootstrap")
    rng = random.Random(seed)
    n = len(cases)
    diffs = []
    for _ in range(resamples):
        drawn = [cases[rng.randrange(n)] for _ in range(n)]
        diffs.append(sum(b[c] for c in drawn) / n - sum(a[c] for c in drawn) / n)
    diffs.sort()
    lo = diffs[int((alpha / 2) * resamples)]
    hi = diffs[min(resamples - 1, int((1 - alpha / 2) * resamples))]
    return {"low": lo, "high": hi, "seed": seed, "resamples": resamples,
            "alpha": alpha}


def qualification(independent_cases: int, split: str, *, approved: bool,
                  minimum: int = MINIMUM_INDEPENDENT_CASES) -> dict[str, Any]:
    """Whether this run may be read as a production measurement."""
    problems: list[str] = []
    if split in UNQUALIFIABLE_SPLITS:
        problems.append(
            f"split {split!r} can never be production-qualified: it is not a "
            "reviewed measurement of the product, whatever its size")
    if not approved:
        problems.append("the case set is not approved by a human record")
    if independent_cases < minimum:
        problems.append(
            f"{independent_cases} independent case(s) is below the {minimum} "
            "needed to detect a 0.05 paired difference at 80% power")
    return {"statistically_qualified": not problems,
            "independent_cases": independent_cases,
            "minimum_independent_cases": minimum,
            "granularity": (1.0 / independent_cases) if independent_cases else None,
            "qualification_problems": problems}


def report(arms: dict[str, list[dict[str, Any]]], *, split: str,
           approved: bool, seed: int = 20260916,
           capability_metric: str = "capability_pass_rate") -> dict[str, Any]:
    """The full four-arm statistical report.

    `arms` maps arm name -> per-observation rows of {case_id, passed}. Rows may
    repeat a case id; repetitions are collapsed first and never inflate n.
    """
    required = ("student_current", "student_candidate",
                "teacher_current", "teacher_candidate")
    missing = [arm for arm in required if arm not in arms]
    if missing:
        raise HarnessEvalError(f"report needs all four arms; missing {missing}")

    collapsed = {arm: collapse_repetitions(rows) for arm, rows in arms.items()}
    observations = {arm: len(rows) for arm, rows in arms.items()}
    sizes = {len(v) for v in collapsed.values()}
    if len(sizes) != 1:
        raise HarnessEvalError(
            f"arms cover different numbers of cases {sorted(sizes)}; an "
            "incomplete arm cannot be compared against a complete one")
    n = sizes.pop()

    rates: dict[str, Any] = {}
    for arm, outcomes in collapsed.items():
        passed = sum(1 for v in outcomes.values() if v)
        lo, hi = wilson_interval(passed, n)
        rates[arm] = {"passed": passed, "n": n, "rate": passed / n,
                      "wilson_95": [lo, hi],
                      "observations": observations[arm],
                      "repetitions_per_case": observations[arm] / n}

    student_gain = rates["student_candidate"]["rate"] - rates["student_current"]["rate"]
    teacher_gain = rates["teacher_candidate"]["rate"] - rates["teacher_current"]["rate"]

    pairs = {
        "student_harness": ("student_current", "student_candidate"),
        "teacher_harness": ("teacher_current", "teacher_candidate"),
        "residual_model_gap": ("student_candidate", "teacher_current"),
    }
    tests = {}
    for label, (first, second) in pairs.items():
        table = paired_table(collapsed[first], collapsed[second])
        tests[label] = {
            "arms": [first, second],
            "table": table,
            "mcnemar": mcnemar(table),
            "bootstrap_95": paired_bootstrap_ci(collapsed[first],
                                                collapsed[second], seed=seed),
        }

    return {
        "capability_metric": capability_metric,
        "independent_cases": n,
        "total_observations": sum(observations.values()),
        "rates": rates,
        "interaction": {
            "student_harness_gain": student_gain,
            "teacher_harness_gain": teacher_gain,
            "interaction": teacher_gain - student_gain,
            "residual_model_gap": rates["teacher_current"]["rate"]
            - rates["student_candidate"]["rate"],
        },
        "paired_tests": tests,
        "seed": seed,
        **qualification(n, split, approved=approved),
        "training_authorized": False,
    }
