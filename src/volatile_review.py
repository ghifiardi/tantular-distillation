"""Volatile-family review: report observed MODES, not averages.

    ./.venv/bin/python src/volatile_review.py \
        --replicates data/candidate/volatile.r*.jsonl data/candidate/traces.r0.jsonl \
        --prompts prompts/candidate.jsonl

The control arm's variance was shown to be bimodal, not diffuse: a third
replicate reproduced the first exactly while the second differed, and every
volatile family had exactly two distinct answers across three runs.

An average over a bimodal variable describes a state the system never
occupies. "refusal = 0.33" is not a family that refuses a third of the time in
any meaningful sense — it is a family that lands in one of two states, and the
mean only says how often each was sampled. So this reports the states.

Decision each volatile family needs before training:
  - INCLUDE ALL   modes are equivalent in quality; keep every replicate
  - INCLUDE ONE   one mode is correct and the other is not; keep the correct
                  one, and record why
  - EXCLUDE       the family cannot be represented faithfully by any single
                  trace
Never: take one arbitrary trace and treat it as the teacher's judgment.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import calibrate


def read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicates", nargs="+", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    args = parser.parse_args()

    prompts = {p["family"]: p for p in read(args.prompts)}
    by_family: dict[str, list[dict]] = defaultdict(list)
    for path in args.replicates:
        for record in read(path):
            by_family[record["family"]].append({**record, "_run": path.name})

    replicated = {f: rs for f, rs in by_family.items() if len(rs) > 1}
    print(f"families with >1 observation: {len(replicated)}\n")
    if not replicated:
        print("No replicated families. A volatile family represented by one trace "
              "records whichever mode that run landed in.")
        sys.exit(1)

    decisions = {}
    for family, records in sorted(replicated.items()):
        answers = [r.get("completion", "").strip() for r in records]
        modes = Counter(answers)
        print(f"=== {family} — {len(records)} observations, "
              f"{len(modes)} distinct answer(s) ===")
        for index, (answer, count) in enumerate(modes.most_common(), 1):
            runs = [r["_run"] for r in records if r.get("completion", "").strip() == answer]
            print(f"  mode {index}  seen {count}x  ({', '.join(runs)})")
            print(f"    {answer[:150]!r}")

        # Metric consequences per mode, since that is what training inherits.
        prompt = prompts.get(family, {})
        scored = [calibrate.score_trace({**r, "checks": prompt.get("checks") or {},
                                         **({"expected": prompt["expected"]}
                                            if prompt.get("expected") else {})})
                  for r in records]
        for metric in ("refusal", "constraints_ok", "source_preserved", "router_correct"):
            values = [s.get(metric) for s in scored if s.get(metric) is not None]
            if values and len(set(map(str, values))) > 1:
                print(f"  METRIC FLIP  {metric}: {values}")

        if len(modes) == 1:
            decisions[family] = "stable across observations — safe to include"
        else:
            decisions[family] = "MULTI-MODE — decide INCLUDE ALL / INCLUDE ONE / EXCLUDE"
        print()

    print("=== decisions required before training ===")
    for family, decision in sorted(decisions.items()):
        print(f"  {family:<32} {decision}")
    unresolved = [f for f, d in decisions.items() if d.startswith("MULTI-MODE")]
    if unresolved:
        print(f"\n{len(unresolved)} family/families need an explicit, recorded decision.")
        print("Until then this corpus is NOT promoted to training.")
        sys.exit(2)
    print("\nAll replicated families stable across observations.")


if __name__ == "__main__":
    main()
