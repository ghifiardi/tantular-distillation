"""Deployment gates: did an answer ARRIVE, and how expensively?

    ./.venv/bin/python src/deployment_gate.py \
        --traces data/gates/companion/voice.traces.jsonl \
        --traces data/gates/companion/edit.traces.jsonl \
        --latency-p95 30

WHY THIS IS SEPARATE FROM THE CAPABILITY GATES.

The capability gates measure whether an answer is GOOD. They never asked whether
one arrived. That is not a hypothetical distinction: on 2026-08-21 the shipped
profile scored voice 0.9500 and edit contract 0.9500 on one serving path while,
on the path the add-in actually used, it returned NOTHING on 7 of 10 Office
edit tasks — 21,808 characters of reasoning and an empty answer after 512
seconds. Every capability gate passed throughout. The defect lived for four
days.

Four conditions, all FAIL-CLOSED. Three are absolute because there is no
acceptable non-zero rate for them:

  empty answers        must be 0. An endpoint that answers with nothing is the
                       failure that survived two rentals on the FP8 arm and
                       four days here.
  finish_reason length must be 0. Truncation means the budget was spent before
                       the answer; the user sees nothing or half a thought.
  reasoning output     must be 0 characters. Reasoning is not an answer, and on
                       the local path it is controllable — think:false. A
                       non-zero value means the routing regressed.
  latency p95          under an agreed threshold, passed explicitly. No default
                       is baked in: a laptop and a server do not share one.

Reads the same trace files the capability gates read, so no separate run is
needed — the same generation is scored twice, for two different questions.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((pct / 100) * (len(ordered) - 1))))
    return ordered[index]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--traces", type=Path, action="append", required=True,
                   help="trace file; repeatable")
    p.add_argument("--latency-p95", type=float, required=True,
                   help="agreed p95 latency budget in seconds. Required and "
                        "explicit: no default can be right for every host.")
    p.add_argument("--json-out", type=Path)
    args = p.parse_args()

    rows = []
    for path in args.traces:
        if not path.is_file():
            sys.exit(f"trace file missing: {path}\n"
                     "A deployment gate with no traces is not a pass.")
        rows += [json.loads(l) for l in
                 path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not rows:
        sys.exit("no traces to judge — refusing to report a vacuous pass")

    empty = [r.get("family") or r.get("id") for r in rows
             if not (r.get("completion") or "").strip()]
    truncated = [r.get("family") or r.get("id") for r in rows
                 if (r.get("provenance") or {}).get("terminated_by") == "length"
                 or (r.get("provenance") or {}).get("truncated") is True]
    reasoning = {r.get("family") or r.get("id"): r.get("reasoning_chars", 0)
                 for r in rows if r.get("reasoning_chars", 0)}
    latencies = [(r.get("provenance") or {}).get("latency_s") for r in rows]
    latencies = [l for l in latencies if isinstance(l, (int, float))]

    print("=== DEPLOYMENT GATE ===")
    print(f"  traces            {len(rows)}")
    print(f"  empty answers     {len(empty)}  (must be 0)")
    print(f"  finish=length     {len(truncated)}  (must be 0)")
    print(f"  reasoning chars   {sum(reasoning.values())} across "
          f"{len(reasoning)} trace(s)  (must be 0)")
    if latencies:
        print(f"  latency           p50 {statistics.median(latencies):.1f}s  "
              f"p95 {percentile(latencies, 95):.1f}s  max {max(latencies):.1f}s"
              f"   (p95 budget {args.latency_p95}s)")
    else:
        print("  latency           NOT RECORDED")

    failures = []
    if empty:
        failures.append(f"{len(empty)} empty answer(s): {empty[:5]}")
    if truncated:
        failures.append(f"{len(truncated)} truncated: {truncated[:5]}")
    if reasoning:
        failures.append(f"reasoning emitted on {len(reasoning)} trace(s): "
                        f"{list(reasoning)[:5]}")
    if not latencies:
        # Not measurable is not a pass: the budget cannot be checked.
        failures.append("no latency recorded on any trace, so the p95 budget "
                        "cannot be checked")
    elif percentile(latencies, 95) > args.latency_p95:
        failures.append(f"p95 latency {percentile(latencies, 95):.1f}s exceeds "
                        f"the {args.latency_p95}s budget")

    verdict = "PASS" if not failures else "FAIL"
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps({
            "traces": len(rows), "verdict": verdict,
            "empty": empty, "truncated": truncated,
            "reasoning_chars_total": sum(reasoning.values()),
            "latency_p50": statistics.median(latencies) if latencies else None,
            "latency_p95": percentile(latencies, 95) if latencies else None,
            "latency_budget_p95": args.latency_p95,
            "failures": failures,
        }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"\nwrote {args.json_out}")

    if failures:
        print("\nDEPLOYMENT GATE FAILED:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("\nDEPLOYMENT GATE PASSED — every answer arrived, none truncated, "
          "no reasoning emitted,\nand p95 latency is within budget.")


if __name__ == "__main__":
    main()
