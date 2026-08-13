"""Repeated-crossover comparison: within-prompt deltas, noise reported separately.

    ./.venv/bin/python src/crossover.py schedule --arms ai19-ollama,rented-48gb \
        --prompts prompts/calibration.jsonl --replicates 3 --out data/crossover/run1
    ./.venv/bin/python src/crossover.py execute  --plan data/crossover/run1/plan.json
    ./.venv/bin/python src/crossover.py analyze  --plan data/crossover/run1/plan.json

WHY NOT A SINGLE PAIRED RUN. Two runs of the same arm were shown to differ on
34.6% of families, moving refusal_rate by 0.0385 and constraint_satisfaction by
0.0294 — both above the pre-registered 0.00 critical tolerance. One observation
per arm cannot separate a precision effect from that.

DESIGN.
  - Every prompt goes through BOTH arms, so prompt difficulty cancels in the
    within-prompt delta instead of inflating variance.
  - Arm order is randomized per prompt (seeded, so the schedule is
    reproducible) to prevent order effects — a warm cache or a drifting box
    favouring whichever arm ran second.
  - R replicates per arm per prompt, so within-arm variance is measured rather
    than assumed.

WHAT IS REPORTED.
  - runtime noise: spread across replicates WITHIN an arm. This is the floor.
  - effect: mean within-prompt (treatment - baseline).
  These are never combined into one number. An effect smaller than the floor is
  not evidence of a precision difference, and stating it as one would be the
  central error this whole design exists to avoid.

Thresholds in acceptance.yaml are NOT touched. They were pre-registered; this
changes how evidence is gathered, not what counts as passing.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import calibrate

ROOT = Path(__file__).resolve().parent.parent

# Metrics scored per-trace, so a within-prompt delta is meaningful. Rate-style
# aggregates (empty_rate) are derived from these at analysis time.
PAIRED_METRICS = ("constraints_ok", "source_preserved", "router_correct",
                  "indonesian", "refusal", "empty", "truncated")


def build_plan(arms: list[str], prompts_path: Path, replicates: int, seed: int) -> dict:
    prompts = [json.loads(l) for l in
               prompts_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    rng = random.Random(seed)
    observations = []
    for prompt in prompts:
        for replicate in range(replicates):
            # Randomize which arm runs first for this (prompt, replicate).
            order = arms[:]
            rng.shuffle(order)
            for position, arm in enumerate(order):
                observations.append({
                    "family": prompt["family"],
                    "arm": arm,
                    "replicate": replicate,
                    "order_position": position,
                })
    return {
        "arms": arms,
        "prompts": str(prompts_path),
        "replicates": replicates,
        "seed": seed,
        "observations": observations,
    }


def load_observations(plan: dict, out_dir: Path) -> dict[tuple, dict]:
    """Read whatever traces exist for this plan, keyed by (family, arm, replicate)."""
    found = {}
    for arm in plan["arms"]:
        for replicate in range(plan["replicates"]):
            path = out_dir / f"{arm}.r{replicate}.jsonl"
            if not path.exists():
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    record = json.loads(line)
                    found[(record["family"], arm, replicate)] = record
    return found


def per_trace_scores(record: dict, prompt: dict) -> dict:
    merged = dict(record)
    merged.setdefault("checks", prompt.get("checks") or {})
    if prompt.get("expected"):
        merged.setdefault("expected", prompt["expected"])
    scored = calibrate.score_trace(merged)
    return {k: scored.get(k) for k in PAIRED_METRICS}


def analyze(plan: dict, out_dir: Path) -> int:
    prompts = {json.loads(l)["family"]: json.loads(l) for l in
               Path(plan["prompts"]).read_text(encoding="utf-8").splitlines() if l.strip()}
    found = load_observations(plan, out_dir)
    arms = plan["arms"]
    if len(arms) != 2:
        sys.exit("analysis expects exactly two arms")
    baseline, treatment = arms

    missing = [o for o in plan["observations"]
               if (o["family"], o["arm"], o["replicate"]) not in found]
    print(f"observations planned {len(plan['observations'])}, "
          f"present {len(found)}, missing {len(missing)}")
    if missing:
        by_arm = defaultdict(int)
        for o in missing:
            by_arm[o["arm"]] += 1
        print(f"  missing by arm: {dict(by_arm)}")
        if all(o["arm"] == treatment for o in missing):
            print(f"  '{treatment}' has no observations — the treatment arm has not run "
                  "(expected until Ada/Hopper hardware exists)")
            return 2

    # --- runtime noise: variance WITHIN an arm across replicates ------------
    print("\n=== RUNTIME NOISE (within-arm, across replicates) ===")
    noise = {}
    for arm in arms:
        deltas = defaultdict(list)
        for family in prompts:
            reps = [found.get((family, arm, r)) for r in range(plan["replicates"])]
            reps = [r for r in reps if r]
            if len(reps) < 2:
                continue
            scores = [per_trace_scores(r, prompts[family]) for r in reps]
            for metric in PAIRED_METRICS:
                values = [s[metric] for s in scores if s[metric] is not None]
                if len(values) >= 2:
                    numeric = [float(v) for v in values]
                    deltas[metric].append(max(numeric) - min(numeric))
        if not deltas:
            print(f"  {arm}: insufficient replicates to measure noise")
            continue
        noise[arm] = {m: round(statistics.mean(v), 4) for m, v in deltas.items()}
        print(f"  {arm}")
        for metric, value in sorted(noise[arm].items()):
            print(f"    {metric:<20} mean within-prompt spread {value}")

    # --- effect: within-prompt paired difference ----------------------------
    print(f"\n=== EFFECT ({treatment} - {baseline}, within-prompt paired) ===")
    effects = defaultdict(list)
    for family in prompts:
        for replicate in range(plan["replicates"]):
            b = found.get((family, baseline, replicate))
            t = found.get((family, treatment, replicate))
            if not b or not t:
                continue
            bs = per_trace_scores(b, prompts[family])
            ts = per_trace_scores(t, prompts[family])
            for metric in PAIRED_METRICS:
                if bs[metric] is None or ts[metric] is None:
                    continue
                effects[metric].append(float(ts[metric]) - float(bs[metric]))
    if not effects:
        print("  no paired observations — cannot compute an effect")
        return 2
    for metric, values in sorted(effects.items()):
        mean = statistics.mean(values)
        floor = max((noise.get(a, {}).get(metric, 0.0) for a in arms), default=0.0)
        verdict = ("BELOW NOISE FLOOR" if abs(mean) <= floor
                   else "exceeds noise floor")
        print(f"  {metric:<20} mean delta {round(mean, 4):>8}  "
              f"n={len(values):<4} floor={floor:<8} {verdict}")

    print("\nNoise and effect are reported separately and never combined. An effect "
          "at or below the floor is not evidence of a precision difference.")
    return 0


def noise_report(arm: str, replicate_paths: list[Path], prompts_path: Path) -> int:
    """Within-arm noise floor across R replicates of one arm.

    This is control-side work: it needs no treatment arm and characterises the
    runtime rather than any precision difference. With R=2 only a range is
    observable; R>=3 gives a standard deviation, which is what a later effect
    must be compared against.
    """
    prompts = {json.loads(l)["family"]: json.loads(l) for l in
               prompts_path.read_text(encoding="utf-8").splitlines() if l.strip()}
    runs = []
    for path in replicate_paths:
        if not path.exists():
            sys.exit(f"missing replicate: {path}")
        runs.append({json.loads(l)["family"]: json.loads(l)
                     for l in path.read_text(encoding="utf-8").splitlines() if l.strip()})

    print(f"arm '{arm}' — {len(runs)} replicates over {len(prompts)} prompts")
    if len(runs) < 3:
        print("  NOTE: with fewer than 3 replicates only a range is observable, "
              "not a standard deviation")

    # Answer-level divergence, which is what a corpus would actually inherit.
    answer_varies = sum(
        1 for family in prompts
        if len({r[family]["completion"] for r in runs if family in r}) > 1)
    print(f"  answer text varies across replicates: {answer_varies}/{len(prompts)} prompts")

    print(f"\n  {'metric':<20}{'mean':>10}{'range':>10}{'stdev':>10}")
    floors = {}
    for metric in PAIRED_METRICS:
        spreads, stdevs = [], []
        for family in prompts:
            values = []
            for run in runs:
                record = run.get(family)
                if record:
                    score = per_trace_scores(record, prompts[family]).get(metric)
                    if score is not None:
                        values.append(float(score))
            if len(values) >= 2:
                spreads.append(max(values) - min(values))
                if len(values) >= 3:
                    stdevs.append(statistics.stdev(values))
        if not spreads:
            continue
        mean_spread = statistics.mean(spreads)
        floors[metric] = mean_spread
        stdev_txt = f"{round(statistics.mean(stdevs), 4):>10}" if stdevs else f"{'n/a':>10}"
        print(f"  {metric:<20}{round(mean_spread, 4):>10}"
              f"{round(max(spreads), 4):>10}{stdev_txt}")

    # --- volatile families, reported individually -------------------------
    # An aggregate mean of 0.0294 can mean "everything drifts slightly" or "one
    # prompt flips completely and the rest are stable". Those demand different
    # responses, and only the per-prompt view distinguishes them. A range of 1.0
    # on a boolean metric IS a full flip.
    volatile = {}
    for family in prompts:
        flips = {}
        for metric in PAIRED_METRICS:
            values = []
            for run in runs:
                record = run.get(family)
                if record:
                    score = per_trace_scores(record, prompts[family]).get(metric)
                    if score is not None:
                        values.append(float(score))
            if len(values) >= 2 and max(values) > min(values):
                flips[metric] = [round(v, 3) for v in values]
        answers = {run[family]["completion"] for run in runs if family in run}
        if flips or len(answers) > 1:
            volatile[family] = {"metric_flips": flips, "distinct_answers": len(answers)}

    print(f"\n  VOLATILE FAMILIES ({len(volatile)}/{len(prompts)}) — preserved "
          "individually, not averaged away")
    for family, detail in sorted(volatile.items()):
        flips = ", ".join(f"{m}={v}" for m, v in detail["metric_flips"].items())
        print(f"    {family:<34} answers={detail['distinct_answers']}"
              + (f"  {flips}" if flips else "  (answer text only)"))

    payload = {
        "arm": arm,
        "replicates": len(runs),
        "prompts": len(prompts),
        "answer_varies": answer_varies,
        "floors": {m: round(v, 4) for m, v in floors.items()},
        "volatile_families": volatile,
    }
    out_path = ROOT / "calibration" / f"noise_floor.{arm}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    print(f"\n  recorded -> {out_path.relative_to(ROOT)}")

    print("\n  This is the floor, and it is a DISTRIBUTION, not a scalar. A later "
          "treatment-vs-control\n  effect must exceed the corresponding control-side "
          "variability per metric — and a\n  volatile family cannot support a "
          f"per-prompt claim at all. Measured on {arm},\n  which is int4 on Ampere "
          "— NOT an FP8 arm.")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    s = sub.add_parser("schedule")
    s.add_argument("--arms", required=True, help="comma-separated host names")
    s.add_argument("--prompts", type=Path, required=True)
    s.add_argument("--replicates", type=int, default=3)
    s.add_argument("--seed", type=int, default=20260813)
    s.add_argument("--out", type=Path, required=True)

    for name in ("execute", "analyze"):
        p = sub.add_parser(name)
        p.add_argument("--plan", type=Path, required=True)

    n = sub.add_parser("noise", help="within-arm noise floor from R replicates")
    n.add_argument("--arm", required=True)
    n.add_argument("--replicates", nargs="+", type=Path, required=True)
    n.add_argument("--prompts", type=Path, required=True)

    args = parser.parse_args()

    if args.command == "schedule":
        plan = build_plan(args.arms.split(","), args.prompts, args.replicates, args.seed)
        args.out.mkdir(parents=True, exist_ok=True)
        path = args.out / "plan.json"
        path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {path}")
        print(f"  arms {plan['arms']}  replicates {plan['replicates']}  "
              f"seed {plan['seed']}")
        print(f"  {len(plan['observations'])} observations "
              f"({len(plan['observations']) // len(plan['arms'])} per arm)")
        print("\nExecute each (arm, replicate) with generate_normalized.py, writing to")
        print(f"  {args.out}/<arm>.r<replicate>.jsonl")
        print("then run `analyze`. Arm order per prompt is recorded in the plan.")
        return

    if args.command == "noise":
        sys.exit(noise_report(args.arm, args.replicates, args.prompts))

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    out_dir = args.plan.parent
    if args.command == "analyze":
        sys.exit(analyze(plan, out_dir))
    print("execute: run generate_normalized.py per (arm, replicate) into")
    print(f"  {out_dir}/<arm>.r<replicate>.jsonl")
    print("Kept manual so each run's provenance and preflight stay explicit.")


if __name__ == "__main__":
    main()
