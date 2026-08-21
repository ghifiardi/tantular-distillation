"""Score a calibration arm, and compare two arms against pre-registered criteria.

    python3 src/calibrate.py score  data/calibration/int4/traces.jsonl
    python3 src/calibrate.py compare data/calibration/int4/traces.jsonl \
                                     data/calibration/fp8/traces.jsonl

Metrics are deliberately mechanical. "Is this trace usable" must not depend on
taste, or the comparison measures the scorer rather than the runtime. Anything
genuinely subjective (fluency, register) is left to blind pairwise review,
which this module prepares but does not adjudicate.

Thresholds come from calibration/acceptance.yaml, which was committed before
any results existed.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.exit("pyyaml is required: pip install -r requirements.txt")

ROOT = Path(__file__).resolve().parent.parent
ACCEPTANCE_PATH = ROOT / "calibration" / "acceptance.yaml"

# A model that asks for the source text, or declines, produced no trace. These
# are the openings such answers actually start with, in both languages.
REFUSAL_MARKERS = re.compile(
    r"(mohon (?:tempelkan|kirimkan|lampirkan|berikan)|silakan (?:tempelkan|kirim)"
    r"|saya (?:tidak dapat|tidak bisa|belum dapat)|tidak dapat (?:saya )?(?:bantu|memproses)"
    r"|please (?:provide|paste|share)|i (?:cannot|can't|am unable))",
    re.IGNORECASE,
)

# Crude but objective Indonesian signal: function words that appear in almost
# any Indonesian sentence and in no English one. Used to detect language drift
# (an Indonesian task answered in English), not to judge elegance.
ID_MARKERS = re.compile(
    r"\b(yang|dan|untuk|dengan|pada|dari|tidak|akan|dalam|adalah|ini|itu|"
    r"sudah|dapat|harus|telah|serta|atau)\b", re.IGNORECASE)
EN_MARKERS = re.compile(
    r"\b(the|and|for|with|from|this|that|will|have|been|are|is|of|to)\b",
    re.IGNORECASE)


def sentences(text: str) -> int:
    return len([s for s in re.split(r"[.!?]+(?:\s|$)", text.strip()) if s.strip()])


def bullets(text: str) -> int:
    """Lines that are list items.

    A bullet marker must be followed by whitespace, and a doubled asterisk is
    excluded: the old pattern accepted a bare leading `*`, so a Markdown bold
    heading like `**Ringkasan**` counted as a bullet. That charged outputs an
    extra item they never emitted — a compliant 5-bullet slide scored as 6 and
    failed a max_bullets of 5.
    """
    return len([l for l in text.splitlines()
                if re.match(r"^\s*(?:[-•]|\*(?!\*))\s|^\s*\d+[.)]\s", l)])


def paragraphs(text: str) -> int:
    """Blank-line-separated blocks. 'Satu paragraf' is a real instruction in the
    summarisation strata, and max_bullets cannot express it: it requires at
    least one bullet, so it can never assert that prose stayed unbroken."""
    return len([b for b in re.split(r"\n\s*\n", text.strip()) if b.strip()])


def absent(text: str, terms: list[str]) -> list[str]:
    """Terms that must NOT survive, matched on word boundaries.

    The terminology and spelling-correction tasks are defined by what has to
    disappear, not by what has to remain — 'replace setting, hacker, backup,
    user, di-disable with standard Indonesian' is only satisfied if those words
    are gone. must_contain cannot state that, so an unedited copy of the source
    scored as a perfect edit. Word boundaries keep 'user' from matching inside
    an unrelated word while still catching the plural.
    """
    return [t for t in terms
            if re.search(rf"(?<!\w){re.escape(t)}(?!\w)", text, re.IGNORECASE)]


def score_trace(record: dict) -> dict:
    """Per-trace mechanical scoring. Any check absent scores None, not 0 —
    a task with no bullet limit must not be counted as failing one."""
    text = str(record.get("completion", ""))
    checks = record.get("checks") or {}
    provenance = record.get("provenance", {})
    result = {
        "family": record.get("family", "?"),
        "empty": not text.strip(),
        "truncated": bool(provenance.get("truncated")),
        "refusal": bool(REFUSAL_MARKERS.search(text)),
        "latency_s": provenance.get("latency_s"),
        "completion_tokens": provenance.get("completion_tokens"),
        "router_correct": None,
        "constraints_ok": None,
        "source_preserved": None,
        "indonesian": None,
    }

    expected = record.get("expected")
    if expected:
        got = re.sub(r"[^A-Z_]", "", text.upper())
        result["router_correct"] = (got == re.sub(r"[^A-Z_]", "", expected.upper()))

    # Constraint satisfaction: every applicable structural rule must hold.
    applicable = []
    if "max_sentences" in checks:
        applicable.append(sentences(text) <= checks["max_sentences"])
    if "max_bullets" in checks:
        applicable.append(0 < bullets(text) <= checks["max_bullets"])
    if "exact_bullets" in checks:
        # "Exactly three bullets" is a real Office instruction; max_bullets
        # cannot express it, and a rewrite that returns prose or four bullets
        # has not followed the instruction. Added 2026-08-21 for the
        # faithful-editing eval.
        applicable.append(bullets(text) == checks["exact_bullets"])
    if "must_contain" in checks:
        applicable.append(all(s.lower() in text.lower() for s in checks["must_contain"]))
    if "must_contain_any" in checks:
        applicable.append(any(s.lower() in text.lower() for s in checks["must_contain_any"]))
    if "closed_set" in checks:
        applicable.append(re.sub(r"[^A-Z_]", "", text.upper()) in checks["closed_set"])
    if "max_paragraphs" in checks:
        applicable.append(paragraphs(text) <= checks["max_paragraphs"])
    if "must_not_contain" in checks:
        applicable.append(not absent(text, checks["must_not_contain"]))
    if applicable:
        result["constraints_ok"] = all(applicable)

    # Source preservation: figures and dates that must survive an edit intact.
    if "preserve" in checks:
        kept = [token for token in checks["preserve"] if token.lower() in text.lower()]
        result["source_preserved"] = len(kept) / len(checks["preserve"])

    # Table fidelity: a row counts only if all of its cells appear TOGETHER on
    # one line. A flat `preserve` list cannot express this — cell values like
    # "1" or "8" match somewhere in almost any prose, so a table could score as
    # fully preserved while its rows were shuffled, merged, or invented. The
    # co-occurrence requirement is what makes the digit mean the right row.
    if "table_rows" in checks:
        lines = [l.lower() for l in text.splitlines()]
        intact = sum(1 for row in checks["table_rows"]
                     if any(all(str(cell).lower() in line for cell in row)
                            for line in lines))
        result["source_preserved"] = intact / len(checks["table_rows"])

    # Language integrity — only meaningful on substantial prose, and skipped
    # for translation tasks, where English is the correct output.
    if len(text.split()) >= 20 and "terjemah" not in result["family"].lower() \
            and "TERJEMAH" not in str(expected or ""):
        id_hits = len(ID_MARKERS.findall(text))
        en_hits = len(EN_MARKERS.findall(text))
        result["indonesian"] = id_hits / (id_hits + en_hits) if (id_hits + en_hits) else None

    return result


def _mean(values):
    values = [v for v in values if v is not None]
    return round(statistics.mean(values), 4) if values else None


def aggregate(records: list[dict]) -> dict:
    scored = [score_trace(r) for r in records]
    n = len(scored)
    quant = {r.get("provenance", {}).get("quantization") for r in records}
    return {
        "n": n,
        "quantization": sorted(q for q in quant if q),
        "empty_rate": round(sum(s["empty"] for s in scored) / n, 4) if n else None,
        "truncation_rate": round(sum(s["truncated"] for s in scored) / n, 4) if n else None,
        "refusal_rate": round(sum(s["refusal"] for s in scored) / n, 4) if n else None,
        "router_label_accuracy": _mean([s["router_correct"] for s in scored]),
        "constraint_satisfaction": _mean([s["constraints_ok"] for s in scored]),
        "source_preservation": _mean([s["source_preserved"] for s in scored]),
        "indonesian_quality": _mean([s["indonesian"] for s in scored]),
        "median_latency_s": _mean([s["latency_s"] for s in scored]),
        "mean_completion_tokens": _mean([s["completion_tokens"] for s in scored]),
        "_scored": scored,
    }


def load(path: Path, prompts_path: Path | None = None) -> list[dict]:
    if not path.exists():
        sys.exit(f"no such file: {path}")
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
               if line.strip()]
    # Traces generated before checks/expected were carried through can still be
    # scored by joining on family id, which is unique per calibration prompt.
    if prompts_path:
        by_family = {}
        for line in prompts_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                prompt = json.loads(line)
                by_family[prompt["family"]] = prompt
        for record in records:
            prompt = by_family.get(record.get("family"))
            if not prompt:
                continue
            record.setdefault("checks", prompt.get("checks") or {})
            if prompt.get("expected"):
                record.setdefault("expected", prompt["expected"])
    return records


def print_summary(name: str, agg: dict) -> None:
    print(f"\n=== {name} ===  n={agg['n']}  quantization={agg['quantization']}")
    for key in ("empty_rate", "truncation_rate", "refusal_rate", "router_label_accuracy",
                "constraint_satisfaction", "source_preservation", "indonesian_quality",
                "median_latency_s", "mean_completion_tokens"):
        value = agg[key]
        print(f"  {key:<26} {'n/a' if value is None else value}")


def compare(baseline: dict, treatment: dict, criteria: dict) -> int:
    """Judge baseline (int4) against treatment (fp8) using pre-registered bounds."""
    failures = []
    print("\n=== VERDICT vs pre-registered criteria ===")

    for band in ("critical", "quality"):
        for metric, rule in (criteria.get(band) or {}).items():
            base, treat = baseline.get(metric), treatment.get(metric)
            if base is None or treat is None:
                print(f"  [{band}] {metric:<26} SKIPPED (not measured in both arms)")
                continue
            higher_better = rule.get("direction", "higher_better") == "higher_better"
            # Regression is always "how much worse baseline is than treatment".
            regression = (treat - base) if higher_better else (base - treat)
            bound = rule.get("max_regression_abs")
            ok = True
            detail = f"int4={base} fp8={treat} regression={round(regression, 4)}"
            if bound is not None and regression > bound:
                ok = False
                detail += f" > allowed {bound}"
            if "min_absolute" in rule and base < rule["min_absolute"]:
                ok = False
                detail += f" (below floor {rule['min_absolute']})"
            print(f"  [{band}] {metric:<26} {'PASS' if ok else 'FAIL'}  {detail}")
            if not ok:
                failures.append(metric)

    if failures:
        print(f"\nFAIL — {len(failures)} criterion/criteria breached: {failures}")
        print(criteria["verdict"]["on_fail"])
        return 1
    print("\nPASS — " + criteria["verdict"]["on_pass"])
    print("This licenses a recorded quality waiver for a specific run. "
          "It does not remove the FP8 gate.")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("score", help="score one arm")
    s.add_argument("path", type=Path)
    s.add_argument("--prompts", type=Path, help="join checks/expected from the prompt set")
    c = sub.add_parser("compare", help="baseline (int4) vs treatment (fp8)")
    c.add_argument("baseline", type=Path)
    c.add_argument("treatment", type=Path)
    c.add_argument("--prompts", type=Path, help="join checks/expected from the prompt set")
    args = parser.parse_args()

    criteria = yaml.safe_load(ACCEPTANCE_PATH.read_text(encoding="utf-8"))

    if args.command == "score":
        agg = aggregate(load(args.path, args.prompts))
        print_summary(args.path.parent.name or "arm", agg)
        print("\nBaseline recorded. The FP8 arm cannot be measured until an "
              "Ada/Hopper host exists; no verdict is possible from one arm.")
    else:
        base = aggregate(load(args.baseline, args.prompts))
        treat = aggregate(load(args.treatment, args.prompts))
        print_summary("baseline (int4)", base)
        print_summary("treatment (fp8)", treat)
        if base["n"] != treat["n"]:
            print(f"\nWARNING: arms differ in size ({base['n']} vs {treat['n']}) — "
                  "they must run the identical prompt set to be comparable")
        sys.exit(compare(base, treat, criteria))


if __name__ == "__main__":
    main()
