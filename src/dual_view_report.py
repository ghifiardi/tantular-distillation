"""Report metrics under BOTH weightings, and verify repeat groups are real calls.

    ./.venv/bin/python src/dual_view_report.py \
        --traces data/expanded/traces.r0.jsonl --prompts prompts/expanded.jsonl

260 families resolve to 78 distinct prompts: a kind's families within one split
share that split's single artifact. So 182 traces answer a prompt some other
trace already answered.

Two weightings, two different questions:

  FAMILY-WEIGHTED   every family counts once. This is what a training set
                    inherits, duplicates included, so it reflects the corpus as
                    it would actually be consumed.

  UNIQUE-PROMPT     group by prompt_sha256, one vote per distinct prompt. This
                    is the honest measure of teacher behaviour: 182 duplicates
                    are not 182 independent quality observations, and reporting
                    them as such would inflate n roughly 3.3x and shrink every
                    confidence interval by a factor the data does not earn.

Neither is "the" number. Family-weighted overstates evidence; unique-prompt
understates corpus composition. Reported side by side, never merged.

REPEAT-GROUP INTEGRITY. A group of 8 traces sharing a prompt is only useful for
mode frequency if those were 8 independent model calls. This checks that they
carry distinct completion token counts or termination metadata rather than
being copies of one result — a pipeline that deduplicated silently would
produce identical rows and a fake mode distribution.
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


def read(path: Path) -> list[dict]:
    if not path.exists():
        sys.exit(f"no such file: {path}")
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def score_all(traces: list[dict], prompts: dict) -> list[tuple[dict, dict]]:
    out = []
    for trace in traces:
        prompt = prompts.get(trace["family"], {})
        merged = {**trace, "checks": trace.get("checks") or prompt.get("checks") or {}}
        if prompt.get("expected"):
            merged.setdefault("expected", prompt["expected"])
        out.append((trace, calibrate.score_trace(merged)))
    return out


def summarise(rows: list[dict], label: str, n_label: str) -> None:
    print(f"\n  {label}   (n = {len(rows)} {n_label})")
    for metric in METRICS:
        values = [r[metric] for r in rows if r.get(metric) is not None]
        if not values:
            continue
        mean = statistics.mean(float(v) for v in values)
        print(f"    {metric:<20} {round(mean, 4):>8}   from {len(values)} scored")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    args = parser.parse_args()

    traces = read(args.traces)
    prompts = {p["family"]: p for p in read(args.prompts)}
    scored = score_all(traces, prompts)

    by_prompt: dict[str, list] = defaultdict(list)
    missing_hash = 0
    for trace, score in scored:
        digest = trace.get("provenance", {}).get("prompt_sha256")
        if not digest:
            missing_hash += 1
            continue
        by_prompt[digest].append((trace, score))

    print(f"traces {len(traces)}   distinct prompt_sha256 {len(by_prompt)}"
          + (f"   MISSING HASH {missing_hash}" if missing_hash else ""))
    if missing_hash:
        print("  WARNING: traces without prompt_sha256 cannot be grouped; the "
              "unique-prompt view below excludes them")

    # --- the two views ----------------------------------------------------
    summarise([s for _, s in scored], "FAMILY-WEIGHTED", "families")
    print("    ^ what a training set inherits, duplicates included")

    unique_rows = []
    for digest, group in by_prompt.items():
        # One vote per distinct prompt: average within the group first.
        agg = {}
        for metric in METRICS:
            values = [s[metric] for _, s in group if s.get(metric) is not None]
            agg[metric] = statistics.mean(float(v) for v in values) if values else None
        unique_rows.append(agg)
    summarise(unique_rows, "UNIQUE-PROMPT", "distinct prompts")
    print("    ^ honest measure of teacher behaviour; duplicates are not "
          "independent observations")

    inflation = len(traces) / len(by_prompt) if by_prompt else 0
    print(f"\n  evidence inflation if duplicates were counted independently: "
          f"{inflation:.1f}x")

    # --- repeat-group integrity ------------------------------------------
    repeats = {d: g for d, g in by_prompt.items() if len(g) > 1}
    print(f"\nREPEAT GROUPS  {len(repeats)} prompt(s) answered more than once")
    suspicious = []
    for digest, group in sorted(repeats.items(), key=lambda kv: -len(kv[1]))[:6]:
        answers = Counter(t.get("completion", "").strip() for t, _ in group)
        tokens = {t.get("provenance", {}).get("completion_tokens") for t, _ in group}
        families = {t["family"].split("::")[0] for t, _ in group}
        independent = len(tokens) > 1 or len(answers) > 1
        print(f"  {digest[:12]}...  {len(group)} calls, {len(answers)} distinct "
              f"answer(s), {len(tokens)} distinct token count(s)  "
              f"[{'independent' if independent else 'SUSPECT: identical'}]")
        modes = {f"mode{i}": count for i, count in enumerate(answers.values(), 1)}
        print(f"    kind {', '.join(sorted(families))}   mode frequency {modes}")
        if not independent:
            suspicious.append(digest)

    if suspicious:
        print(f"\n  {len(suspicious)} group(s) show identical answers AND identical "
              "token counts.\n  That is consistent with copied traces rather than "
              "independent calls;\n  mode frequency from them would be fabricated.")
    elif repeats:
        print("\n  All inspected groups show variation in answer or token count — "
              "consistent with\n  independent model calls, so mode frequency is "
              "measurable from them.")
    else:
        # Saying "all groups look independent" when there are none is a vacuous
        # truth that reads as a positive finding.
        print("\n  No repeat groups in this corpus — mode frequency cannot be "
              "measured from it.")

    print("\nCorpus remains SYNTHETIC and UNPROMOTED. Neither view licenses a "
          "training decision.")


if __name__ == "__main__":
    main()
