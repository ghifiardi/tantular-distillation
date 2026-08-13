"""Compare two runs of the SAME arm to confirm deterministic reproduction.

    ./.venv/bin/python src/determinism_check.py \
        --run-a data/calibration/int4-normalized/traces.jsonl \
        --run-b data/calibration/int4-normalized-controlled/traces.jsonl

Same weights, same rendered bytes, temperature 0, fixed seed: outputs should
reproduce. Whether they do is a property of the runtime worth knowing before
any cross-arm verdict, because a runtime that does not reproduce itself cannot
support conclusions about a difference between arms.

If the runs differ, BOTH are preserved and the difference is classified rather
than one replacing the other. Silently keeping the newer result would discard
the evidence that non-determinism exists.

Divergence classes, from least to most consequential:
  identical            - byte-identical answers and token counts
  token_count_only     - same answer text, different token accounting
  whitespace_only      - answers differ only in surrounding whitespace
  answer_divergent     - answer text differs materially
  structural           - one run malformed/missing where the other is not
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def read(path: Path) -> dict[str, dict]:
    if not path.exists():
        sys.exit(f"no such file: {path}")
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            out[record["family"]] = record
    return out


def classify(a: dict | None, b: dict | None) -> str:
    if a is None or b is None:
        return "structural"
    text_a, text_b = a.get("completion", ""), b.get("completion", "")
    tok_a = a.get("provenance", {}).get("completion_tokens")
    tok_b = b.get("provenance", {}).get("completion_tokens")
    if text_a == text_b:
        return "identical" if tok_a == tok_b else "token_count_only"
    if text_a.strip() == text_b.strip():
        return "whitespace_only"
    return "answer_divergent"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-a", type=Path, required=True)
    parser.add_argument("--run-b", type=Path, required=True)
    args = parser.parse_args()

    a, b = read(args.run_a), read(args.run_b)
    families = sorted(set(a) | set(b))
    verdicts = {f: classify(a.get(f), b.get(f)) for f in families}
    counts = Counter(verdicts.values())

    print(f"run A: {args.run_a.parent.name}  ({len(a)} traces)")
    print(f"run B: {args.run_b.parent.name}  ({len(b)} traces)")
    print(f"\ncompared {len(families)} families")
    for verdict, n in sorted(counts.items()):
        print(f"  {verdict:<20} {n}")

    # Prompt bytes must match too, or a divergence is explained by the input
    # rather than by the runtime.
    # Determinism is only meaningful if both runs received identical input.
    # Missing hashes count as unproven, not as a match: a run that does not
    # record what it sent cannot testify that it sent the same thing.
    unproven = []
    for family in families:
        if family not in a or family not in b:
            continue
        ha = a[family].get("provenance", {}).get("prompt_sha256")
        hb = b[family].get("provenance", {}).get("prompt_sha256")
        if not ha or not hb or ha != hb:
            unproven.append(family)
    if unproven:
        print(f"\nINDETERMINATE — {len(unproven)} family/families cannot be shown to "
              "have received identical input (hash missing or different).")
        print("Determinism is undefined across differing inputs. Identical outputs "
              "here would prove nothing, and different outputs would not be "
              "attributable to the runtime.")
        sys.exit(2)

    divergent = [f for f, v in verdicts.items() if v not in ("identical",)]
    if not divergent:
        print("\nDETERMINISTIC — byte-identical reproduction across runs.")
        print("The runtime reproduces itself, so a cross-arm difference can be "
              "attributed to the arms rather than to run-to-run noise.")
        return

    print(f"\nNON-DETERMINISTIC — {len(divergent)} of {len(families)} families differ.")
    print("BOTH runs are preserved; neither replaces the other.")
    for family in divergent[:8]:
        verdict = verdicts[family]
        print(f"\n  {family}  [{verdict}]")
        if family in a and family in b and verdict == "answer_divergent":
            print(f"    A: {a[family]['completion'][:90]!r}")
            print(f"    B: {b[family]['completion'][:90]!r}")
    if len(divergent) > 8:
        print(f"\n  ... and {len(divergent) - 8} more")
    print("\nA runtime that does not reproduce itself sets a noise floor: any "
          "int4-vs-FP8 difference smaller than this floor is not evidence of a "
          "precision effect. Record the floor before issuing a verdict.")


if __name__ == "__main__":
    main()
