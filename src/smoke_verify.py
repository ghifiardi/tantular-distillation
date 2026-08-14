"""Verify PIPELINE properties of the smoke run. Not model quality.

    ./.venv/bin/python src/smoke_verify.py \
        --traces data/smoke/traces.jsonl --prompts prompts/smoke.jsonl

Checks only that the path works end to end. Deliberately silent on how good the
answers are: this corpus cannot support that judgment, and reporting a quality
number from it would invite exactly the misreading the run was designed to
avoid.

Two results are PREDICTED artifacts of the source pack, recorded before the run
in calibration/SMOKE_RUN_EXPECTATIONS.md, and are reported as such rather than
as findings:
  - router accuracy near 1/8 — all 8 router prompts are byte-identical while
    carrying 8 different expected labels
  - near-duplicate document answers — 4 document kinds share one generic source
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

EXPECTED_TEACHER = "muse-glimmer"
EXPECTED_MODEL = "muse-glimmer:30b"
EXPECTED_QUANT = "int4_ollama"
EXPECTED_HOST = "ai19-ollama"


def read(path: Path) -> list[dict]:
    if not path.exists():
        sys.exit(f"no such file: {path}")
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    args = parser.parse_args()

    traces = read(args.traces)
    prompts = {p["family"]: p for p in read(args.prompts)}
    malformed_path = args.traces.with_suffix(".malformed.jsonl")
    errors_path = args.traces.with_suffix(".errors.json")
    malformed = read(malformed_path) if malformed_path.exists() else []

    failures = []
    print("PIPELINE PROPERTIES (not model quality)\n")

    # 1. completeness
    ok = len(traces) == len(prompts)
    print(f"  [{'PASS' if ok else 'FAIL'}] traces generated        {len(traces)}/{len(prompts)}")
    if not ok:
        failures.append("incomplete generation")

    # 2. no malformed or empty
    empty = [t for t in traces if not str(t.get("completion", "")).strip()]
    ok = not malformed and not empty
    print(f"  [{'PASS' if ok else 'FAIL'}] malformed / empty       "
          f"{len(malformed)} malformed, {len(empty)} empty")
    if not ok:
        failures.append("malformed or empty outputs")

    # 3. source text actually embedded — the failure that wasted the first seed
    #    run was prompts REFERRING to documents they did not contain.
    unembedded = []
    for trace in traces:
        prompt = prompts.get(trace["family"])
        if not prompt:
            continue
        if prompt.get("expected"):
            continue  # router prompts are requests, not document tasks
        body = prompt["user"].split("\n\n", 1)
        if len(body) < 2 or len(body[1].strip()) < 100:
            unembedded.append(trace["family"])
    ok = not unembedded
    print(f"  [{'PASS' if ok else 'FAIL'}] source embedded         "
          f"{len(traces) - len(unembedded)}/{len(traces)} prompts carry their source")
    if not ok:
        failures.append(f"source not embedded: {', '.join(unembedded[:4])}")

    # 4. provenance identifies the waiver-covered teacher
    wrong = Counter()
    for trace in traces:
        p = trace.get("provenance", {})
        if (p.get("teacher") != EXPECTED_TEACHER or p.get("repo") != EXPECTED_MODEL
                or p.get("quantization") != EXPECTED_QUANT or p.get("host") != EXPECTED_HOST):
            wrong[(p.get("repo"), p.get("quantization"), p.get("host"))] += 1
    ok = not wrong
    print(f"  [{'PASS' if ok else 'FAIL'}] provenance              "
          f"{len(traces) - sum(wrong.values())}/{len(traces)} carry "
          f"{EXPECTED_MODEL} / {EXPECTED_QUANT} / {EXPECTED_HOST}")
    if not ok:
        failures.append(f"provenance mismatch: {dict(wrong)}")

    # 5. infrastructure errors kept separate
    if errors_path.exists():
        payload = json.loads(errors_path.read_text(encoding="utf-8"))
        count = payload.get("infrastructure_failures")
        count = count if isinstance(count, int) else len(count or [])
        print(f"  [INFO] infrastructure errors  {count} (excluded from quality denominators)")

    # --- predicted artifacts, reported as such -----------------------------
    print("\nPREDICTED SOURCE-PACK ARTIFACTS (not model-quality results)")
    router = [t for t in traces if prompts.get(t["family"], {}).get("expected")]
    if router:
        correct = sum(1 for t in router
                      if prompts[t["family"]]["expected"].upper()
                      in str(t.get("completion", "")).upper())
        print(f"  router label accuracy   {correct}/{len(router)} — capped at 1/8 by "
              "8 byte-identical prompts")
        print("    This measures the source pack, NOT the teacher. Not a regression.")
    docs = [t for t in traces if t["family"].startswith("document:")]
    if docs:
        distinct = len({t.get("completion", "").strip() for t in docs})
        print(f"  document answers        {distinct} distinct of {len(docs)} — "
              "4 kinds share one generic source")

    print("\nCORPUS ROLE: pipeline_smoke — synthetic pipeline smoke corpus, "
          "NOT suitable as a training corpus.")
    print("Excluded from training. Source diversification is required first.")

    if failures:
        print(f"\nPIPELINE FAILED — {len(failures)}: {'; '.join(failures)}")
        sys.exit(1)
    print("\nPIPELINE OK — inventory -> seeds -> ai19 -> traces works end to end.")
    print("This says nothing about corpus quality or the student it would produce.")


if __name__ == "__main__":
    main()
