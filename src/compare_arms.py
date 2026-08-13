"""Verify two arms are comparable, then compare them.

    ./.venv/bin/python src/compare_arms.py \
        --baseline data/calibration/int4-normalized-controlled/traces.jsonl \
        --treatment data/calibration/fp8/traces.jsonl \
        --prompts prompts/calibration.jsonl

Comparability is checked BEFORE any metric is printed, because a verdict from
arms that saw different prompts or different decoding is worse than no verdict:
it looks authoritative and means nothing.

Hard requirements — any failure blocks the comparison:
  - identical per-prompt rendered-prompt hashes (the same bytes reached both)
  - identical chat-template hash
  - identical reasoning_strength, temperature, seed, max_tokens, stop sequences
  - the same set of families

Termination is REPORTED, never assumed equivalent. Ollama's bare "stop" and
vLLM's "stop" are different vocabularies over different evidence: vLLM names
the stop string or token that fired and reports null for its own EOS, while
Ollama attributes nothing. Equating them without that evidence would assert
something neither runtime said.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import calibrate

# Settings that must match for a difference to be attributable to precision.
PINNED_SETTINGS = ("template_sha256", "reasoning_strength", "temperature",
                   "seed", "max_tokens", "stop_sequences")


def settings_of(traces: list[dict]) -> dict:
    out = {}
    for field in PINNED_SETTINGS:
        values = {json.dumps(t.get("provenance", {}).get(field), sort_keys=True)
                  for t in traces}
        out[field] = sorted(values)
    return out


def check_comparability(base: list[dict], treat: list[dict]) -> list[str]:
    errors = []

    base_by_family = {t["family"]: t for t in base}
    treat_by_family = {t["family"]: t for t in treat}
    only_base = sorted(set(base_by_family) - set(treat_by_family))
    only_treat = sorted(set(treat_by_family) - set(base_by_family))
    if only_base or only_treat:
        errors.append(f"family sets differ: {len(only_base)} only in baseline, "
                      f"{len(only_treat)} only in treatment")

    # THE check: identical rendered bytes. Anything else and the comparison
    # measures prompt construction as well as precision.
    mismatched = []
    for family in sorted(set(base_by_family) & set(treat_by_family)):
        b = base_by_family[family].get("provenance", {}).get("prompt_sha256")
        t = treat_by_family[family].get("provenance", {}).get("prompt_sha256")
        if not b or not t:
            errors.append(f"{family}: prompt hash missing "
                          f"(baseline={bool(b)}, treatment={bool(t)})")
        elif b != t:
            mismatched.append(family)
    if mismatched:
        errors.append(f"{len(mismatched)} prompt(s) rendered differently between arms — "
                      f"e.g. {', '.join(mismatched[:3])}")

    base_settings, treat_settings = settings_of(base), settings_of(treat)
    for field in PINNED_SETTINGS:
        if base_settings[field] != treat_settings[field]:
            errors.append(f"{field} differs: baseline={base_settings[field]} "
                          f"treatment={treat_settings[field]}")
    return errors


def termination_table(name: str, traces: list[dict]) -> None:
    raw = Counter(t.get("provenance", {}).get("raw_done_reason") for t in traces)
    stop = Counter(str(t.get("provenance", {}).get("raw_stop_reason")) for t in traces)
    derived = Counter(t.get("provenance", {}).get("terminated_by") for t in traces)
    marker = Counter(str(t.get("provenance", {}).get("ended_at_marker")) for t in traces)
    eos = Counter(str(t.get("provenance", {}).get("eos_applied_by_runtime")) for t in traces)
    print(f"  {name}")
    print(f"    raw finish/done reason  {dict(raw)}")
    print(f"    raw stop_reason         {dict(stop)}")
    print(f"    runtime applied own EOS {dict(eos)}")
    print(f"    text ended at marker    {dict(marker)}")
    print(f"    derived                 {dict(derived)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--treatment", type=Path, required=True)
    parser.add_argument("--prompts", type=Path)
    parser.add_argument("--force", action="store_true",
                        help="print metrics despite comparability failures (they will "
                             "still be reported as unattributable)")
    args = parser.parse_args()

    base = calibrate.load(args.baseline, args.prompts)
    treat = calibrate.load(args.treatment, args.prompts)

    print("=== COMPARABILITY ===")
    errors = check_comparability(base, treat)
    if errors:
        print(f"BLOCKED — {len(errors)} issue(s):")
        for error in errors:
            print(f"  {error}")
        if not args.force:
            print("\nNo verdict issued. A comparison across differing prompts or "
                  "decoding settings would attribute to precision what is actually "
                  "a protocol difference.")
            sys.exit(1)
        print("\n--force: metrics below are NOT attributable to precision.")
    else:
        print(f"OK — {len(base)} vs {len(treat)} traces, identical prompt bytes, "
              "template and decoding settings")

    print("\n=== TERMINATION (reported, not assumed equivalent) ===")
    termination_table("baseline ", base)
    termination_table("treatment", treat)
    print("  NOTE: a bare 'stop' with no stop_reason attribution stays `ambiguous`.")
    print("  Ollama and vLLM use different vocabularies over different evidence;")
    print("  equating 'stop' with 'eos' would assert what neither runtime said.")

    criteria = calibrate.yaml.safe_load(
        calibrate.ACCEPTANCE_PATH.read_text(encoding="utf-8"))
    base_agg, treat_agg = calibrate.aggregate(base), calibrate.aggregate(treat)
    calibrate.print_summary("baseline (int4)", base_agg)
    calibrate.print_summary("treatment (fp8)", treat_agg)
    sys.exit(calibrate.compare(base_agg, treat_agg, criteria))


if __name__ == "__main__":
    main()
