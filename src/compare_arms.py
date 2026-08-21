"""Compare evaluation arms per ITEM and per DIMENSION, not by total score.

    ./.venv/bin/python src/compare_arms.py \
        --arm "Q4 greedy=data/gates/greedy/voice.score.json" \
        --arm "Q8 greedy=data/gates/q8/voice.score.json"

Two arms can score the same total while failing different items, and an arm can
improve on one dimension while regressing on another. A single rate hides both.

That is not hypothetical here: the Q4 shipped-sampling and Q4 greedy arms both
scored 0.9250, and the interesting fact was that they failed the SAME items with
the SAME findings — which is what eliminated sampling as the cause. A total
alone could not have said that.

Reports, for each pair of arms:
  fixed      failed in the first arm, passes in the second
  broken     passed in the first arm, fails in the second
  unchanged  same verdict in both
  dimensions per-dimension failure counts side by side
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load(spec: str) -> tuple[str, dict]:
    if "=" not in spec:
        sys.exit(f"--arm needs LABEL=path, got {spec!r}")
    label, path = spec.split("=", 1)
    p = Path(path)
    if not p.is_file():
        sys.exit(f"arm {label!r}: no such score file: {p}")
    return label, json.loads(p.read_text(encoding="utf-8"))


def verdicts(report: dict) -> dict[str, bool]:
    return {r["id"]: bool(r.get("passed")) for r in report.get("results", [])}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", action="append", required=True,
                   help="LABEL=path/to/score.json; repeatable, order matters")
    args = p.parse_args()
    if len(args.arm) < 2:
        sys.exit("at least two arms are needed to compare")

    arms = [load(a) for a in args.arm]

    print("=== TOTALS ===")
    for label, report in arms:
        rate = report.get("rate")
        print(f"  {label:<20}{report.get('passed')}/{report.get('items')}"
              f"   rate {rate:.4f}" if isinstance(rate, float) else
              f"  {label:<20}{report.get('passed')}/{report.get('items')}")

    base_label, base_report = arms[0]
    base = verdicts(base_report)
    for label, report in arms[1:]:
        other = verdicts(report)
        shared = sorted(set(base) & set(other))
        if not shared:
            sys.exit(f"{base_label} and {label} share no item ids")
        fixed = [i for i in shared if not base[i] and other[i]]
        broken = [i for i in shared if base[i] and not other[i]]
        same_fail = [i for i in shared if not base[i] and not other[i]]

        print(f"\n=== {base_label}  ->  {label} ===")
        print(f"  items compared   {len(shared)}")
        print(f"  fixed            {len(fixed)}  {fixed[:8]}")
        print(f"  broken           {len(broken)}  {broken[:8]}")
        print(f"  still failing    {len(same_fail)}  {same_fail[:8]}")
        if not fixed and not broken:
            print("  IDENTICAL VERDICTS on every shared item — the arms differ "
                  "in no item, not merely in no total.")

        bd = base_report.get("per_dimension_failures") or {}
        od = report.get("per_dimension_failures") or {}
        keys = sorted(set(bd) | set(od))
        if keys:
            print(f"\n  {'dimension':<20}{base_label:>14}{label:>14}")
            for k in keys:
                b, o = bd.get(k, 0), od.get(k, 0)
                mark = "" if b == o else ("  better" if o < b else "  WORSE")
                print(f"  {k:<20}{b:>14}{o:>14}{mark}")

        # Findings for items that fail in BOTH: same reason, or a different one?
        base_find = {r["id"]: r.get("findings") for r in base_report["results"]}
        other_find = {r["id"]: r.get("findings") for r in report["results"]}
        differing = [i for i in same_fail if base_find.get(i) != other_find.get(i)]
        if same_fail:
            print(f"\n  of the {len(same_fail)} still failing, "
                  f"{len(differing)} fail for a DIFFERENT reason")
            for i in differing[:5]:
                print(f"    {i}\n      {base_label}: "
                      f"{json.dumps(base_find.get(i), ensure_ascii=False)[:110]}"
                      f"\n      {label}: "
                      f"{json.dumps(other_find.get(i), ensure_ascii=False)[:110]}")


if __name__ == "__main__":
    main()
