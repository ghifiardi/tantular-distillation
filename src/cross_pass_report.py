"""Across-run reproducibility: compare two independent passes. Never merges them.

    ./.venv/bin/python src/cross_pass_report.py \
        --pass-a data/expanded/traces.r0.jsonl \
        --pass-b data/expanded-r2/traces.r0.jsonl \
        --prompts prompts/expanded.jsonl

The within-run v2 floor came out 0.0 on every metric, but that is not a
reproducibility result: those repeats shared a session and its batching
conditions, and the v1 evidence tied mode selection plausibly to batching. Two
independent passes are what actually measures across-run variance.

Matching is by prompt_sha256, not by family. Families are an assignment
artifact; the prompt is what the model saw. Grouping by prompt also pools every
family sharing a prompt, giving a larger sample per comparison.

Two views, reported separately and never combined:
  UNIQUE-PROMPT     one vote per distinct prompt — the primary view
  FAMILY-WEIGHTED   what a training set would inherit, 3.3x repetition included

Plan consistency is verified before any comparison. Two passes generated under
different split fingerprints, templates or decoding settings are not two
observations of one thing.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import calibrate

METRICS = ("constraints_ok", "source_preserved", "router_correct", "indonesian",
           "refusal", "empty", "truncated")
PINNED = ("split_fingerprint", "split_seed", "template_sha256", "repo",
          "quantization", "host", "reasoning_strength", "temperature", "seed")


def read(path: Path) -> list[dict]:
    if not path.exists():
        sys.exit(f"no such file: {path}")
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def plan_of(traces: list[dict]) -> dict:
    return {f: sorted({json.dumps(t.get("provenance", {}).get(f)) for t in traces})
            for f in PINNED}


def score(trace: dict, prompt: dict) -> dict:
    merged = {**trace, "checks": trace.get("checks") or prompt.get("checks") or {}}
    if prompt.get("expected"):
        merged.setdefault("expected", prompt["expected"])
    return calibrate.score_trace(merged)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pass-a", type=Path, required=True)
    parser.add_argument("--pass-b", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    args = parser.parse_args()

    a, b = read(args.pass_a), read(args.pass_b)
    prompts = {p["family"]: p for p in read(args.prompts)}

    print(f"pass A {len(a)} traces   pass B {len(b)} traces\n")

    # --- plan consistency, before anything is compared --------------------
    print("=== PLAN CONSISTENCY ===")
    pa, pb = plan_of(a), plan_of(b)
    mismatched = [f for f in PINNED if pa[f] != pb[f]]
    for field in PINNED:
        same = pa[field] == pb[field]
        print(f"  [{'OK ' if same else 'DIFFER'}] {field:<20} "
              f"{pa[field][0] if len(pa[field]) == 1 else pa[field]}"
              + ("" if same else f"  vs  {pb[field]}"))
    if mismatched:
        sys.exit(f"\nPasses differ on {mismatched} — not two observations of one thing.")

    # --- group by prompt digest ------------------------------------------
    ga, gb = defaultdict(list), defaultdict(list)
    for trace in a:
        ga[trace["provenance"]["prompt_sha256"]].append(trace)
    for trace in b:
        gb[trace["provenance"]["prompt_sha256"]].append(trace)
    shared = sorted(set(ga) & set(gb))
    print(f"\n=== MATCHED GROUPS ===\n  {len(shared)} prompt digests present in both "
          f"passes (A had {len(ga)}, B had {len(gb)})")

    # --- across-pass answer agreement -------------------------------------
    # Three distinct outcomes, which one counter would blur:
    #   single_same  one answer, same in both passes — fully reproducible
    #   multi_same   several answers, but the SAME set in both — variance is
    #                internal to each pass, not across them
    #   set_differs  a mode appears in one pass and not the other — genuine
    #                across-run variance, invisible to a within-run measurement
    single_same = multi_same = 0
    mode_shift = []
    for digest in shared:
        set_a = {t["completion"].strip() for t in ga[digest]}
        set_b = {t["completion"].strip() for t in gb[digest]}
        if set_a == set_b:
            if len(set_a) == 1:
                single_same += 1
            else:
                multi_same += 1
        else:
            mode_shift.append((digest, len(set_a - set_b), len(set_b - set_a)))
    # Set membership alone is not enough. A prompt whose two modes appear 4:1 in
    # one pass and 1:4 in the other has an IDENTICAL mode set, so the counters
    # above call it "same set" and report nothing — while the mixture the
    # training corpus would inherit has inverted. Drift is the total-variation
    # distance between the two passes' mode frequencies, per prompt.
    drift = []
    for digest in shared:
        count_a = Counter(t["completion"].strip() for t in ga[digest])
        count_b = Counter(t["completion"].strip() for t in gb[digest])
        n_a, n_b = sum(count_a.values()), sum(count_b.values())
        if n_a and n_b:
            modes = set(count_a) | set(count_b)
            tv = sum(abs(count_a[m] / n_a - count_b[m] / n_b) for m in modes) / 2
            if tv:
                drift.append((tv, digest, sorted({t["family"] for t in ga[digest]})))

    print(f"  one answer, identical in both passes      : {single_same}/{len(shared)}")
    print(f"  several answers, SAME set in both passes  : {multi_same}/{len(shared)}")
    print(f"  answer set DIFFERS between passes         : {len(mode_shift)}/{len(shared)}")
    if mode_shift:
        print("    (a mode absent from one pass is across-run variance the "
              "within-run\n     measurement could not see)")

    print(f"\n=== MODE-FREQUENCY DRIFT ===")
    print(f"  prompts whose mode MIXTURE moved : {len(drift)}/{len(shared)}")
    if drift:
        drift.sort(reverse=True)
        print(f"  max total-variation distance     : {round(drift[0][0], 4)}")
        print(f"  mean over drifting prompts       : "
              f"{round(statistics.mean(d[0] for d in drift), 4)}")
        for tv, digest, fams in drift[:5]:
            print(f"    {round(tv, 3):<7} {digest[:16]}  {fams[0]}"
                  + (f" (+{len(fams) - 1} more)" if len(fams) > 1 else ""))
        print("  Drift is invisible to the set comparison above: a prompt can keep\n"
              "  the same mode set while the mixture a corpus inherits inverts.")
    else:
        print("  No prompt changed its mode mixture between passes.")

    # --- metric deltas, two views ----------------------------------------
    # Every metric prints its own denominator. A metric is None whenever the
    # prompt carries no check that could decide it, so the denominators are NOT
    # the row count and NOT equal to each other: constraints_ok is decidable
    # only where a structural check exists. Printing one number per metric
    # without its coverage would read as a claim over the whole set. A metric
    # nothing measures prints as NO COVERAGE rather than being skipped —
    # silently dropping it is how an unmeasured contract passes for a met one.
    def view(rows_a, rows_b, label, n_label):
        total = len(rows_a)
        print(f"\n  {label}  (n = {total} {n_label})")
        print(f"    {'metric':<20}{'pass A':>9}{'pass B':>9}{'|delta|':>9}{'covers':>12}")
        for metric in METRICS:
            va = [r[metric] for r in rows_a if r.get(metric) is not None]
            vb = [r[metric] for r in rows_b if r.get(metric) is not None]
            if not va or not vb:
                print(f"    {metric:<20}{'—':>9}{'—':>9}{'—':>9}"
                      f"{'NO COVERAGE':>12}")
                continue
            ma, mb = statistics.mean(map(float, va)), statistics.mean(map(float, vb))
            cover = f"{len(va)}/{total}" if len(va) == len(vb) else f"{len(va)}|{len(vb)}/{total}"
            print(f"    {metric:<20}{round(ma,4):>9}{round(mb,4):>9}"
                  f"{round(abs(ma-mb),4):>9}{cover:>12}")

    print("\n=== ACROSS-PASS METRIC COMPARISON ===")
    ua, ub = [], []
    for digest in shared:
        for source, dest in ((ga, ua), (gb, ub)):
            scored = [score(t, prompts.get(t["family"], {})) for t in source[digest]]
            agg = {}
            for metric in METRICS:
                vals = [s[metric] for s in scored if s.get(metric) is not None]
                agg[metric] = statistics.mean(map(float, vals)) if vals else None
            dest.append(agg)
    view(ua, ub, "UNIQUE-PROMPT (primary)", "distinct prompts")

    fa = [score(t, prompts.get(t["family"], {})) for t in a]
    fb = [score(t, prompts.get(t["family"], {})) for t in b]
    view(fa, fb, "FAMILY-WEIGHTED (3.3x repetition inflation)", "families")

    # --- what the metrics could not see ----------------------------------
    # An agreement rate is only as broad as the checks behind it. Strata whose
    # prompts carry no checks contribute to the answer-agreement counts above
    # (those compare raw text) but to none of the metric means.
    print("\n=== CHECK COVERAGE BY STRATUM ===")
    unchecked = sorted({fam.split("::")[0] for fam, p in prompts.items()
                        if not p.get("checks") and not p.get("expected")})
    checked = sorted({fam.split("::")[0] for fam, p in prompts.items()
                      if p.get("checks") or p.get("expected")})
    print(f"  {len(checked)} strata carry a check: {', '.join(checked) or '(none)'}")
    print(f"  {len(unchecked)} strata carry none : {', '.join(unchecked) or '(none)'}")
    if unchecked:
        print("  Metric agreement above is evidence about the checked strata only.\n"
              "  For the rest, two passes agreeing means the text matched — not that\n"
              "  either pass was correct.")

    print("\nThe two passes are NOT merged. Each is an independent observation; "
          "combining\nthem would double-count every prompt and describe neither pass.")


if __name__ == "__main__":
    main()
