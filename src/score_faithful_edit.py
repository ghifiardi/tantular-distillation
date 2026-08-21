"""Score the faithful-constrained-editing eval.

    ./.venv/bin/python src/score_faithful_edit.py \
        --items prompts/faithful_edit_pilot.v1.jsonl \
        --traces data/gates/fce/traces.jsonl \
        --addin-src ../tantular_office_addin/src

Objective, approved 2026-08-21: can the model change an Indonesian document as
an Office instruction asks, while preserving the figures, names, terms,
structure and facts it must not touch, and without adding information that was
not there?

SIX PROPERTIES, all mechanical. No model judges another model.

  1 lands        the requested span changed, and every `must_not_change` span
                 is byte-identical in the applied document
  2 preserves    every `must_preserve` string survives with the SAME count
  3 structure    the declared calibrate.py checks hold on the applied document
  4 no_new_facts no number, date or entity in the output that is absent from the
                 source and undeclared (src/faithful_facts.py)
  5 contract     parsed AND located AND applied, via the add-in's real parser
  6 voice        rubric v2 on the text the model wrote

An item passes only if ALL SIX hold. Per-property rates are reported because
they say WHERE a failure is, which one number cannot.

TWO ITEM KINDS. `expect: "edit"` asks for a change. `expect: "absent"` asks for
something the document does not contain — the correct answer STATES that and
emits no edits. Inventing a plausible value is the worst failure this eval can
detect, and the reason it exists: a fabricated figure in an Office document is
worse than a clumsy sentence.

NO THRESHOLD IS DEFINED HERE. Approved 2026-08-21: thresholds come after the
pilot and a base measurement, never from reading a score and adding a margin.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import calibrate
import faithful_facts
import score_voice


def run_contract_checker(cases: list[dict], addin_src: Path) -> dict[str, dict]:
    """The add-in's REAL parser, so the eval cannot drift from the product."""
    checker = ROOT / "scripts" / "check_edit_contract.mjs"
    if not checker.is_file():
        sys.exit(f"checker missing: {checker}")
    if not (addin_src / "chat" / "editContract.js").is_file():
        sys.exit(f"add-in parser missing under {addin_src}")
    tmp = ROOT / "data" / "gates" / "fce" / "_cases.json"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps(cases, ensure_ascii=False), encoding="utf-8")
    proc = subprocess.run(["node", str(checker), str(tmp), str(addin_src)],
                          capture_output=True, text=True, cwd=ROOT,
                          start_new_session=True, timeout=300)
    if proc.returncode != 0:
        sys.exit(f"contract checker failed:\n{proc.stderr[-500:]}")
    return {r["id"]: r for r in json.loads(proc.stdout)["results"]}


def count_of(needle: str, haystack: str) -> int:
    return (haystack or "").count(needle)


def score_edit_item(item: dict, completion: str, contract: dict) -> dict:
    source = item["document"]
    applied = contract.get("applied")
    findings: dict[str, list[str]] = {}

    # 5. contract — everything else depends on the edit having applied at all.
    #
    # When it has not, there is no applied document to inspect, and scoring the
    # raw JSON against `must_preserve` reports four extra failures that are all
    # one defect. The item still FAILS (a finding is present), but the dependent
    # properties are recorded as not measured rather than counted as failures,
    # so the per-property view says where the real problem is. Measured on the
    # pilot: every failing item showed contract + preserves + no_new_facts, and
    # only contract was real.
    if not contract.get("contract_ok"):
        return {"contract": [contract.get("error") or "contract not applied"],
                "_not_measured": ["lands", "preserves", "structure",
                                  "no_new_facts", "voice"]}
    text = applied if applied else completion

    # 1. lands: the target changed, the protected spans did not.
    if applied is not None:
        if applied == source:
            findings["lands"] = ["the document is unchanged"]
        unchanged = [span for span in item.get("must_not_change", [])
                     if count_of(span, applied) != count_of(span, source)]
        if unchanged:
            findings.setdefault("lands", []).extend(
                [f"protected span altered: {s[:60]!r}" for s in unchanged])
    elif "contract" not in findings:
        findings["lands"] = ["no applied document to inspect"]

    # 2. preserves: same string, same count.
    lost = [s for s in item.get("must_preserve", [])
            if count_of(s, text) != count_of(s, source)]
    if lost:
        findings["preserves"] = [f"{s!r}: {count_of(s, source)} -> "
                                 f"{count_of(s, text)}" for s in lost]

    # 3. structure
    findings.update(structure_findings(item, text))

    # 4. no new facts
    introduced = faithful_facts.new_facts(text, source,
                                          item.get("allowed_new_facts"))
    if introduced:
        findings["no_new_facts"] = [f"{k}: {v}" for k, v in introduced.items()]

    # 6. voice — on what the MODEL wrote, not on the untouched source around it.
    written = " ".join(e.get("replace", "")
                       for e in contract.get("edit_details") or [])
    findings.update(voice_findings(item, written or completion))
    return findings


def score_absent_item(item: dict, completion: str, contract: dict) -> dict:
    """The document does not contain what was asked for. Saying so is correct."""
    findings: dict[str, list[str]] = {}
    source = item["document"]

    if contract.get("parse_ok") and (contract.get("edit_details") or contract.get("edits")):
        findings["lands"] = ["emitted edits for information the document does "
                             "not contain"]
    for phrase in item.get("must_state_absence", []):
        if phrase.lower() not in (completion or "").lower():
            findings.setdefault("structure", []).append(
                f"reply does not state absence ({phrase!r} missing)")
    introduced = faithful_facts.new_facts(completion, source,
                                          item.get("allowed_new_facts"))
    if introduced:
        findings["no_new_facts"] = [f"{k}: {v}" for k, v in introduced.items()]
    findings.update(voice_findings(item, completion))
    return findings


# Exactly the check names calibrate.score_trace acts on. A name outside this
# set is silently ignored by calibrate, which is how the structure check first
# shipped as a no-op: the item declared {"bullets": 3}, calibrate supports
# `max_bullets` and `exact_bullets` but not `bullets`, and every answer passed.
# Measured 2026-08-21 while testing that each check FIRES.
SUPPORTED_CHECKS = {"max_sentences", "max_bullets", "exact_bullets",
                    "must_contain", "must_contain_any", "closed_set",
                    "max_paragraphs", "must_not_contain", "table_rows"}


def structure_findings(item: dict, text: str) -> dict:
    checks = item.get("structure") or {}
    if not checks:
        return {}
    unsupported = sorted(set(checks) - SUPPORTED_CHECKS)
    if unsupported:
        # Loud, not silent. An unrecognised check that scored as a pass would
        # mean the item declares a constraint nobody is testing.
        return {"structure": [f"UNSUPPORTED CHECK {u!r} — calibrate.score_trace "
                              "ignores it, so it would never fail"
                              for u in unsupported]}
    scored = calibrate.score_trace({"completion": text, "family": item["id"],
                                    "checks": checks})
    ok = scored.get("constraints_ok")
    if ok is None:
        return {"structure": ["declared checks produced no verdict"]}
    return {} if ok else {"structure": [f"failed {sorted(checks)}"]}


def voice_findings(item: dict, text: str) -> dict:
    scored = score_voice.score_answer(text, {"id": item["id"],
                                             "user": item.get("user", ""),
                                             "source_terms_must_preserve":
                                                 item.get("must_preserve", [])})
    bad = {k: v for k, v in scored["findings"].items() if v and k != "empty"}
    return {"voice": [f"{k}: {v}" for k, v in bad.items()]} if bad else {}


PROPERTIES = ("lands", "preserves", "structure", "no_new_facts", "contract",
              "voice")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--items", type=Path, required=True)
    p.add_argument("--traces", type=Path, required=True)
    p.add_argument("--addin-src", type=Path,
                   default=ROOT.parent / "tantular_office_addin" / "src")
    p.add_argument("--json-out", type=Path)
    args = p.parse_args()

    items = [json.loads(l) for l in
             args.items.read_text(encoding="utf-8").splitlines() if l.strip()]
    traces = [json.loads(l) for l in
              args.traces.read_text(encoding="utf-8").splitlines() if l.strip()]
    by_id = {t.get("family") or t.get("id"): t.get("completion", "")
             for t in traces}

    missing = [i["id"] for i in items if i["id"] not in by_id]
    if missing:
        sys.exit(f"{len(missing)} item(s) have no completion: {missing[:5]}\n"
                 "Every item is scored or none; a partial run is not a rate.")

    cases = [{"id": i["id"], "document": i["document"],
              "completion": by_id[i["id"]]} for i in items]
    contracts = run_contract_checker(cases, args.addin_src.resolve())

    results = []
    for item in items:
        completion = by_id[item["id"]]
        contract = contracts.get(item["id"], {})
        findings = (score_absent_item(item, completion, contract)
                    if item.get("expect") == "absent"
                    else score_edit_item(item, completion, contract))
        results.append({"id": item["id"], "stratum": item.get("stratum"),
                        "expect": item.get("expect", "edit"),
                        "passed": not {k: v for k, v in findings.items()
                                       if k != "_not_measured"},
                        "findings": findings})

    passed = sum(1 for r in results if r["passed"])
    rate = passed / len(results) if results else 0.0
    print(f"=== FAITHFUL CONSTRAINED EDITING ===")
    print(f"  {passed}/{len(results)} items passed   rate {rate:.4f}")
    print("  (no threshold: set after the pilot and a base measurement)")

    print("\n=== PER-PROPERTY (denominator = items where it could be measured) ===")
    fails = defaultdict(int)
    measurable = defaultdict(int)
    for r in results:
        skipped = set(r["findings"].get("_not_measured", []))
        for prop in PROPERTIES:
            if prop in skipped:
                continue
            measurable[prop] += 1
            if prop in r["findings"]:
                fails[prop] += 1
    for prop in PROPERTIES:
        total = measurable[prop]
        if not total:
            print(f"  {prop:<14}NOT MEASURABLE on any item")
            continue
        note = "" if total == len(results) else f"   ({len(results) - total} not measurable)"
        print(f"  {prop:<14}{total - fails[prop]:>3}/{total} pass{note}")

    print("\n=== BY STRATUM ===")
    by_stratum = defaultdict(lambda: [0, 0])
    for r in results:
        by_stratum[r["stratum"]][1] += 1
        by_stratum[r["stratum"]][0] += int(r["passed"])
    for stratum, (ok, total) in sorted(by_stratum.items()):
        print(f"  {stratum:<28}{ok:>3}/{total}")

    failures = [r for r in results if not r["passed"]]
    if failures:
        print(f"\n=== FAILURES ({len(failures)}) ===")
        for r in failures:
            reasons = "; ".join(f"{k}: {', '.join(map(str, v))}"
                                for k, v in r["findings"].items()
                                if k != "_not_measured")
            print(f"  {r['id']} [{r['stratum']}] {reasons[:200]}")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(
            {"items": len(results), "passed": passed, "rate": rate,
             "threshold": None, "per_property_failures": dict(fails),
             "results": results}, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8")
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
