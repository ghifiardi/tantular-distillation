"""Score generated answers against the Indonesian-voice rubric.

    ./.venv/bin/python src/score_voice.py \
        --traces data/voice-eval/before/traces.jsonl \
        --items prompts/voice_eval.v1.jsonl

The `indonesian_voice` gate `train/qlora_9b.yaml` requires. Distillation
reliably costs Indonesian voice, so this runs before and after a training run
and an adapter that regresses is not promoted.

RUBRIC v2, approved 2026-08-19 (v1 rubric revised the same day after review).

THIS GATE DOES NOT TEST FAITHFULNESS. It checks voice only. A fabricated figure
written in fluent professional Indonesian PASSES here. Faithfulness is covered by
the mechanical checks in calibrate.py and by office_json_contract; a voice pass
must never be read as a faithfulness pass.

RUBRIC, approved 2026-08-19. Derived from a survey of accepted material
(`calibration/VOICE_EVAL_PROPOSAL.md`), with register decided as a product
question because the surveyed corpus is consumer-support ("Baik Kak") while
Office writes to management.

  register        no informal address (Kak/kamu/lo), no greeting filler opening
                  a document answer, no customer-service opener.
  baku            no colloquial contractions (gak/udah/dapet/nunggu/...).
  terminology     generic anglicisms flagged where a standard Indonesian term
                  exists; ESTABLISHED domain terms are allow-listed and never
                  penalised; terms the SOURCE requires are exempt entirely.
  language        answer is Indonesian-dominant, reusing calibrate's metric —
                  except where the task itself asks for English.

An item PASSES only if every pass/fail dimension passes. The gate uses the
overall item rate; per-dimension rates are reported because they say WHERE a
regression happened, which a single number cannot.

Nothing here judges style with another model. Every check is a deterministic
string or ratio test, so a failure can be pointed at.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import calibrate

RUBRIC_VERSION = "v2"

# --- register ---------------------------------------------------------------
INFORMAL_ADDRESS = ["kak", "kakak", "kamu", "lo", "lu", "gan", "bro", "sis"]
# Openers that mark a customer-service turn rather than a document answer.
GREETING_OPENERS = ["baik", "oke", "ok", "siap", "halo", "hai", "selamat pagi",
                    "selamat siang", "selamat sore", "terima kasih sudah"]

# --- baku -------------------------------------------------------------------
COLLOQUIAL = ["gak", "nggak", "udah", "udh", "dapet", "nunggu", "kelar", "bakal",
              "ngecek", "dateng", "soalnya", "bikin", "kayak", "gimana", "kenapa sih",
              "banget", "aja", "yg", "dgn", "tdk"]

# --- terminology ------------------------------------------------------------
# Flagged: generic anglicisms with a standard Indonesian equivalent in use.
GENERIC_ANGLICISM = {
    "setting": "pengaturan", "backup": "pencadangan", "user": "pengguna",
    "di-disable": "dinonaktifkan", "disable": "nonaktifkan",
    # `update` is allow-listed as an established noun ("melakukan update"), but
    # `di-update` is a mixed-affix verb — Indonesian prefix on an English root —
    # whose standard form is `diperbarui`. Approved 2026-08-19: the noun is
    # established usage, the affixed verb is not.
    "di-update": "diperbarui", "maintenance": "pemeliharaan", "meeting": "rapat",
    "deadline": "batas waktu", "report": "laporan",
    "schedule": "jadwal", "budget": "anggaran", "approve": "menyetujui",
}
# Never flagged: established domain terms the surveyed corpus itself retains,
# plus Office formats and proper nouns.
# `file` and `update` were added 2026-08-19 after review: both are established in
# Indonesian office usage in a way `setting`/`backup`/`user` are not, and the
# deny-list was producing false failures on natural professional Indonesian. A
# gate that cries wolf gets loosened under pressure, so the fix is a correct list
# rather than a lower threshold.
ALLOWED_TERMS = ["otp", "pin", "cvv", "apk", "aplikasi", "call center",
                 "mobile banking", "internet banking", "helpdesk", "pdf", "xlsx",
                 "docx", "kwh", "email", "sla", "npwp", "dukcapil",
                 "file", "update"]


def word_present(text: str, term: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(term)}(?!\w)", text, re.IGNORECASE) is not None


def score_answer(answer: str, item: dict) -> dict:
    text = (answer or "").strip()
    lowered = text.lower()
    findings = {}

    findings["register_address"] = [t for t in INFORMAL_ADDRESS if word_present(text, t)]

    opener = []
    for g in GREETING_OPENERS:
        if lowered.startswith(g):
            opener.append(g)
    findings["register_opener"] = opener

    findings["baku"] = [t for t in COLLOQUIAL if word_present(text, t)]

    # Terminology: a term the SOURCE requires is exempt — preserving source
    # wording is correct even when the term is an anglicism, and penalising it
    # would push the model to paraphrase figures and names it must keep.
    exempt = {t.lower() for t in item.get("source_terms_must_preserve", [])}
    flagged = []
    for term in GENERIC_ANGLICISM:
        if term.lower() in exempt or term.lower() in ALLOWED_TERMS:
            continue
        if word_present(text, term):
            flagged.append(f"{term} -> {GENERIC_ANGLICISM[term]}")
    findings["terminology"] = flagged

    # Language: skip where the task asks for English.
    wants_english = "inggris" in item.get("user", "").lower()
    scored = calibrate.score_trace({"completion": text, "family": item["id"], "checks": {}})
    ratio = scored.get("indonesian")
    findings["language"] = ([] if wants_english or ratio is None or ratio >= 0.6
                            else [f"indonesian ratio {ratio:.2f} < 0.60"])

    findings["empty"] = ["no answer"] if not text else []

    dimensions = {k: not v for k, v in findings.items()}
    return {"id": item["id"], "focus": item.get("dimension_focus"),
            "passed": all(dimensions.values()),
            "dimensions": dimensions,
            "findings": {k: v for k, v in findings.items() if v}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    items = {r["id"]: r for r in (json.loads(l) for l in
             args.items.read_text(encoding="utf-8").splitlines() if l.strip())}
    traces = [json.loads(l) for l in
              args.traces.read_text(encoding="utf-8").splitlines() if l.strip()]

    results, missing = [], []
    for item_id, item in items.items():
        trace = next((t for t in traces if t.get("family") == item_id
                      or t.get("id") == item_id), None)
        if trace is None:
            missing.append(item_id)
            continue
        results.append(score_answer(trace.get("completion", ""), item))

    if missing:
        sys.exit(f"{len(missing)} eval items have no trace: {missing[:3]}\n"
                 "The gate scores every item or none; a partial run is not a rate.")

    passed = sum(1 for r in results if r["passed"])
    rate = passed / len(results) if results else 0.0

    print(f"=== ITEM RESULT (the gate) ===")
    print(f"  {passed}/{len(results)} items passed   rate {rate:.4f}   "
          f"threshold {args.threshold}")

    print(f"\n=== PER-DIMENSION (reported, not gated) ===")
    dim_fail = defaultdict(int)
    for r in results:
        for dim, ok in r["dimensions"].items():
            if not ok:
                dim_fail[dim] += 1
    for dim in ("register_address", "register_opener", "baku", "terminology",
                "language", "empty"):
        n = dim_fail.get(dim, 0)
        print(f"  {dim:<20}{len(results)-n:>3}/{len(results)} pass")

    by_focus = defaultdict(lambda: [0, 0])
    for r in results:
        by_focus[r["focus"]][1] += 1
        by_focus[r["focus"]][0] += int(r["passed"])
    print(f"\n=== BY FOCUS ===")
    for focus, (ok, total) in sorted(by_focus.items()):
        print(f"  {focus:<16}{ok:>3}/{total}")

    failures = [r for r in results if not r["passed"]]
    if failures:
        print(f"\n=== FAILURES ({len(failures)}) ===")
        for r in failures[:10]:
            reasons = "; ".join(f"{k}: {', '.join(map(str, v))}"
                                for k, v in r["findings"].items())
            print(f"  {r['id']}  {reasons}")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps({
            "items": len(results), "passed": passed, "rate": rate,
            "threshold": args.threshold,
            "verdict": "PASS" if rate >= args.threshold else "FAIL",
            "per_dimension_failures": dict(dim_fail),
            "results": results,
        }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"\nwrote {args.json_out}")

    verdict = "PASS" if rate >= args.threshold else "FAIL"
    print(f"\n{verdict} — {passed}/{len(results)} at threshold {args.threshold}")
    sys.exit(0 if verdict == "PASS" else 1)


if __name__ == "__main__":
    main()
