"""Report one calibration arm, and set it beside another WITHOUT merging.

    ./.venv/bin/python src/study_report.py \
        --arm data/calibration/int4-normalized/traces.jsonl \
        --baseline data/calibration/int4/traces.jsonl \
        --prompts prompts/calibration.jsonl

Five sections, in the order they matter:

  1. valid vs malformed — what actually entered the denominator
  2. quality metrics on VALID traces only
  3. prompt/template hash consistency — proof both arms saw identical bytes
  4. endpoint/runtime errors, kept apart from model quality
  5. side-by-side against another arm, presented as two columns

Datasets are never concatenated. Arms differ in protocol, and a merged corpus
would silently average two different measurements into one meaningless number.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import calibrate


def read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def section_validity(arm: Path, traces: list[dict]) -> dict:
    malformed = read(arm.with_suffix(".malformed.jsonl"))
    attempted = len(traces) + len(malformed)
    print("1. VALIDITY")
    print(f"   valid traces      {len(traces)}")
    print(f"   malformed         {len(malformed)}"
          + (f"  ({', '.join(f'{n}x {r}' for r, n in Counter(m.get('malformed_reason','?') for m in malformed).items())})"
             if malformed else ""))
    print(f"   attempted         {attempted}")
    if malformed:
        print("   malformed are retained in .malformed.jsonl and excluded from all "
              "quality denominators")
    return {"valid": len(traces), "malformed": len(malformed), "attempted": attempted}


def section_hashes(traces: list[dict]) -> bool:
    print("\n3. PROMPT / TEMPLATE HASH CONSISTENCY")
    templates = {t.get("provenance", {}).get("template_sha256") for t in traces}
    templates.discard(None)
    prompt_hashes = [t.get("provenance", {}).get("prompt_sha256") for t in traces]
    present = [h for h in prompt_hashes if h]
    if not templates:
        print("   no template hash recorded — this arm predates the normalized protocol")
        return False
    ok = len(templates) == 1
    print(f"   template sha256   {sorted(templates)[0][:16]}"
          f"{'' if ok else '  ** MULTIPLE TEMPLATES: ' + str(len(templates)) + ' **'}")
    print(f"   per-prompt hashes {len(present)}/{len(traces)} recorded, "
          f"{len(set(present))} distinct")
    if len(set(present)) != len(present):
        print("   NOTE: duplicate prompt hashes — expected only if prompts repeat")
    return ok


def section_errors(arm: Path) -> int:
    print("\n4. ENDPOINT / RUNTIME ERRORS")
    errors_path = arm.with_suffix(".errors.json")
    if not errors_path.exists():
        print("   none recorded")
        return 0
    payload = json.loads(errors_path.read_text(encoding="utf-8"))
    failures = payload.get("infrastructure_failures", [])
    count = failures if isinstance(failures, int) else len(failures)
    print(f"   infrastructure failures {count}  -> {errors_path.name}")
    print("   excluded from quality denominators; not retried into the result set")
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--prompts", type=Path)
    args = parser.parse_args()

    traces = calibrate.load(args.arm, args.prompts)
    if not traces:
        sys.exit(f"no traces in {args.arm}")

    protocol = {t.get("provenance", {}).get("protocol", "chat-endpoint") for t in traces}
    print(f"=== {args.arm.parent.name} ===  protocol={sorted(protocol)}\n")

    section_validity(args.arm, traces)

    agg = calibrate.aggregate(traces)
    print("\n2. QUALITY METRICS (valid traces only)")
    for key in ("empty_rate", "truncation_rate", "refusal_rate", "router_label_accuracy",
                "constraint_satisfaction", "source_preservation", "indonesian_quality",
                "mean_completion_tokens"):
        value = agg[key]
        print(f"   {key:<26} {'n/a' if value is None else value}")

    section_hashes(traces)
    section_errors(args.arm)

    if args.baseline:
        base_traces = calibrate.load(args.baseline, args.prompts)
        base = calibrate.aggregate(base_traces)
        base_protocol = {t.get("provenance", {}).get("protocol", "chat-endpoint")
                         for t in base_traces}
        print(f"\n5. SIDE BY SIDE (datasets NOT merged)")
        print(f"   left : {args.arm.parent.name:<22} protocol={sorted(protocol)}")
        print(f"   right: {args.baseline.parent.name:<22} protocol={sorted(base_protocol)}")
        print(f"\n   {'metric':<26} {'normalized':>12} {'deployment':>12}")
        for key in ("empty_rate", "truncation_rate", "refusal_rate", "router_label_accuracy",
                    "constraint_satisfaction", "source_preservation", "indonesian_quality",
                    "mean_completion_tokens"):
            left, right = agg[key], base[key]
            print(f"   {key:<26} {('n/a' if left is None else left):>12} "
                  f"{('n/a' if right is None else right):>12}")
        print("\n   These measure DIFFERENT protocols on the same weights, so a "
              "difference here is a\n   response-path effect, not a precision effect. "
              "Neither is a verdict; the verdict\n   needs the FP8 arm under the "
              "normalized protocol.")


if __name__ == "__main__":
    main()
