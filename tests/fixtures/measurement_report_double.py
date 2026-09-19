"""Stand-in for `src/measurement_report.py` while it is absent from this branch.

`measurement_report` provides the repo's paired statistics and lives on
`origin/main`; the RSI branch does not carry it yet. `src/eval_harness.py`
IMPORTS it and refuses when it is missing — that refusal is itself tested.

This double exists so the gate's own logic (validation, identity rules,
separation and stability decisions, verdict shape) can be exercised offline on
this branch. It is NOT a second implementation of the statistics:

* it lives under `tests/fixtures/`, never `src/`;
* it implements only `paired_table` and `mcnemar`, matching the signatures and
  return keys on `origin/main` at the time of writing;
* a test asserts its results agree with hand-computed values, so a drift in the
  real module shows up as a disagreement rather than as silence.

When `measurement_report` lands on this branch, delete this file and the
`use_statistics_double` fixture; the gate needs no change.
"""
from __future__ import annotations

from typing import Any


class HarnessEvalError(Exception):
    """Mirrors the real module's error type for unpaired cases."""


def paired_table(a: dict[str, bool], b: dict[str, bool]) -> dict[str, int]:
    shared = sorted(set(a) & set(b))
    missing = sorted(set(a) ^ set(b))
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
    """Survival function of chi-square with 1 df, via erfc."""
    import math
    if x <= 0:
        return 1.0
    return math.erfc(math.sqrt(x / 2.0))


def mcnemar(table: dict[str, int], *, continuity: bool = True) -> dict[str, Any]:
    b, c = table["only_first"], table["only_second"]
    discordant = b + c
    if discordant == 0:
        return {"statistic": 0.0, "p_value": 1.0, "discordant": 0,
                "continuity_correction": continuity,
                "note": "no discordant pairs: the arms agreed on every case, so "
                        "there is no difference to test"}
    delta = abs(b - c)
    if continuity:
        delta = max(0.0, delta - 1.0)
    statistic = (delta * delta) / discordant
    return {"statistic": statistic, "p_value": _chi2_sf_1df(statistic),
            "discordant": discordant, "only_first": b, "only_second": c,
            "continuity_correction": continuity}
