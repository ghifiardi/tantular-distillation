"""Select v3 families for volatility replication, from v3's own evidence.

    ./.venv/bin/python src/volatile_probe.py \
        --passes data/v3-candidate/traces.r0.jsonl data/v3-pass-a/traces.r0.jsonl \
        --prompts prompts/expanded.v3.jsonl \
        --out data/v3-volatile/probe.jsonl

Prompt set v3 has no repeated prompts, so the within-run repeat groups that
measured v2's volatility do not exist. Volatility has to come from separate
observations of the same prompt.

This compares two or more completed v3 passes and selects every family whose
completions were not identical across them. Those are CANDIDATES, not volatile
families: two observations can only show that a family varied, never how it
varies. A family that disagreed once might have two equally good modes or might
be unstable across many, and one disagreement cannot tell those apart —
`volatile_review.py` needs several samples per family to report modes rather
than an average over a bimodal variable.

So the output is a probe prompt file for replicate generation. Feed it back
through the normal runner two or more times, then hand every replicate to
volatile_review.

The v1/v2 volatile list is deliberately not consulted. It was measured over
different documents behind different prompts; inheriting it would replicate
families chosen by evidence that no longer applies while leaving genuinely
volatile v3 families on a single sample.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

# The full plan a pass is generated under. Two passes agreeing on prompts but
# differing on, say, temperature would show "variation" that is a settings
# change, not model behaviour.
PINNED = ("split_fingerprint", "split_seed", "template_sha256", "repo",
          "quantization", "host", "reasoning_strength", "temperature", "seed")


def read(path: Path) -> list[dict]:
    if not path.exists():
        sys.exit(f"no such file: {path}")
    return [json.loads(l) for l in
            path.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--passes", type=Path, nargs="+", required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    if len(args.passes) < 2:
        sys.exit("at least two passes are needed; one pass shows no variation by "
                 "construction")

    prompts = {r["family"]: r for r in read(args.prompts)}
    passes = [read(p) for p in args.passes]

    # Passes must be observations of the same instrument, or a "difference" is
    # just the two files describing different prompts under different settings.
    # Both halves are checked: the decoding/plan configuration, and the prompt
    # each family actually saw.
    print("=== PLAN CONSISTENCY ===")
    mismatched = []
    for field in PINNED:
        values = [sorted({json.dumps(t.get("provenance", {}).get(field))
                          for t in traces}) for traces in passes]
        same = all(v == values[0] for v in values)
        shown = values[0][0] if len(values[0]) == 1 else values[0]
        print(f"  [{'OK ' if same else 'DIFFER'}] {field:<20} {shown}"
              + ("" if same else f"  vs  {values[1:]}"))
        if not same:
            mismatched.append(field)
    if mismatched:
        sys.exit(f"passes differ on {mismatched} — not observations of one instrument")

    digests = [{t["family"]: t["provenance"]["prompt_sha256"] for t in traces}
               for traces in passes]
    for other in digests[1:]:
        differing = [f for f in digests[0] if f in other and digests[0][f] != other[f]]
        if differing:
            sys.exit(f"passes disagree on the prompt for {len(differing)} families "
                     f"— they are not observations of one instrument: {differing[:3]}")
    shared_families = set(digests[0]).intersection(*[set(d) for d in digests[1:]])
    print(f"  [OK ] identical prompt digest for all {len(shared_families)} "
          f"shared families\n")

    answers = defaultdict(list)
    for traces in passes:
        for trace in traces:
            answers[trace["family"]].append(trace["completion"].strip())

    complete = {f: a for f, a in answers.items() if len(a) == len(passes)}
    partial = sorted(set(answers) - set(complete))
    candidates = sorted(f for f, a in complete.items() if len(set(a)) > 1)

    print(f"passes compared        : {len(passes)}")
    print(f"families in all passes : {len(complete)}")
    if partial:
        print(f"families missing from some pass: {len(partial)}  {partial[:3]}")
    print(f"families that VARIED   : {len(candidates)}/{len(complete)}")
    by_kind = defaultdict(int)
    for family in candidates:
        by_kind[family.split("::")[0]] += 1
    for kind, n in sorted(by_kind.items(), key=lambda kv: -kv[1]):
        checked = "checked" if (prompts.get(f"{kind}::0000", {}) or {}).get("checks") else "UNCHECKED"
        print(f"    {kind:<28}{n:>3}  ({checked} stratum)")

    if not candidates:
        print("\nNo family varied across these passes. That is NOT 'no volatile "
              "families':\ntwo observations cannot distinguish a stable family from "
              "one whose modes happen\nto agree twice. Volatility remains UNMEASURED "
              "until replicates are reviewed.")
        return

    probe = [prompts[f] for f in candidates if f in prompts]
    missing = [f for f in candidates if f not in prompts]
    if missing:
        sys.exit(f"{len(missing)} candidates have no prompt in {args.prompts}")

    print(f"\nprobe set: {len(probe)} prompts")
    print("Generate this file at least twice at identical settings, then pass every "
          "replicate\nto volatile_review.py. Two samples show THAT a family varies; "
          "modes need more.")

    if not args.write:
        print("\ndry run — pass --write to apply")
        return
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in probe) + "\n",
                        encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
