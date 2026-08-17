"""Blind pairwise review between two teacher arms.

    # 1. after both arms exist, build the blinded packet
    ./.venv/bin/python src/blind_review.py prepare \
        --baseline data/calibration/int4/traces.jsonl \
        --treatment data/calibration/fp8/traces.jsonl \
        --prompts prompts/calibration.jsonl \
        --out data/calibration/_review --salt <any string> --write

    # 2. review data/calibration/_review/REVIEW.md, fill in VERDICTS.csv
    # 3. score it
    ./.venv/bin/python src/blind_review.py score --dir data/calibration/_review

`acceptance.yaml` gates on `pairwise_win_rate`, and it is the only metric in the
study that can see what the mechanical ones cannot. Quantization damage shows up
as weaker inference, blander drafts and shallower summaries — none of which move
constraint satisfaction, figure preservation or a router label. An arm can score
1.0 on every automated check and still lose a preference test badly.

calibrate.py deliberately does not adjudicate this. Neither does this module: it
blinds, it collects, it scores. The judgement is a human's.

BLINDING. Each item shows two completions as A and B. Which arm is which is
decided by sha256(salt + prompt digest), so it is deterministic and auditable
afterwards but carries no pattern a reviewer can learn — arms do not alternate,
and one arm is not systematically first. The mapping goes in KEY.json, which the
reviewer must not open until VERDICTS.csv is filled in. The key also records a
digest of REVIEW.md, so verdicts cannot be silently matched against an edited
packet.

TIES are allowed and count as half a win. Forcing a preference on two acceptable
answers manufactures signal that is not there.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.exit("pyyaml is required: pip install -r requirements.txt")

ROOT = Path(__file__).resolve().parent.parent
ACCEPTANCE = ROOT / "calibration" / "acceptance.yaml"


def read(path: Path) -> list[dict]:
    if not path.exists():
        sys.exit(f"no such file: {path}")
    return [json.loads(l) for l in
            path.read_text(encoding="utf-8").splitlines() if l.strip()]


def digest_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(args) -> None:
    baseline = {t["family"]: t for t in read(args.baseline)}
    treatment = {t["family"]: t for t in read(args.treatment)}
    prompts = {p["family"]: p for p in read(args.prompts)}

    shared = sorted(set(baseline) & set(treatment))
    if not shared:
        sys.exit("the two arms share no families")
    missing = sorted((set(baseline) | set(treatment)) - set(shared))

    # acceptance.yaml: require_identical_prompts. Different serving stacks will
    # not produce byte-identical OUTPUT, but they must have been given
    # byte-identical INPUT or the comparison is between two different questions.
    #
    # Only the normalized protocol records the input digest, and only it renders
    # the prompt client-side so vLLM and Ollama receive the same bytes. An arm
    # generated the old way cannot be compared across runtimes at all — the
    # server applied its own template, so the two arms answered differently
    # worded questions.
    for label, arm, path in (("baseline", baseline, args.baseline),
                             ("treatment", treatment, args.treatment)):
        without = [f for f, t in arm.items()
                   if "prompt_sha256" not in t.get("provenance", {})]
        if without:
            sys.exit(
                f"{len(without)} {label} traces in {path.name} carry no "
                f"prompt_sha256 — they predate the normalized protocol "
                f"(protocol: {arm[without[0]].get('provenance', {}).get('protocol')}).\n"
                "Cross-runtime comparison needs client-side rendering, so use a "
                "normalized arm\n(e.g. data/calibration/int4-normalized/traces.jsonl) "
                "or regenerate with src/generate_normalized.py."
            )

    differing = [f for f in shared
                 if baseline[f]["provenance"].get("prompt_sha256")
                 != treatment[f]["provenance"].get("prompt_sha256")]
    if differing:
        sys.exit(f"{len(differing)} families were given different prompts across arms "
                 f"— require_identical_prompts is violated: {differing[:3]}")

    quant = ({t["provenance"].get("quantization") for t in baseline.values()},
             {t["provenance"].get("quantization") for t in treatment.values()})
    print(f"baseline  {args.baseline.name:<28} quantization {sorted(quant[0])}")
    print(f"treatment {args.treatment.name:<28} quantization {sorted(quant[1])}")
    if quant[0] == quant[1]:
        sys.exit("both arms report the same quantization — this is not a comparison. "
                 "An A6000 silently falling back to int4 looks exactly like this.")
    if missing:
        print(f"NOTE {len(missing)} families are in only one arm and are excluded")

    # Blinding by hash RANK, not hash parity. Parity is only balanced in
    # expectation: on the first attempt here it put the baseline in slot A for
    # nine consecutive items, and a reviewer who starts recognising a house
    # style in position A has partially unblinded themselves. Ranking the salted
    # digests and giving the lower half one order guarantees an exact split,
    # while rank order stays uncorrelated with item order so the assignment is
    # still scattered and unlearnable. Deterministic and auditable from the salt.
    ranked = sorted(shared, key=lambda f: hashlib.sha256(
        f"{args.salt}:{baseline[f]['provenance']['prompt_sha256']}".encode()).hexdigest())
    baseline_first = set(ranked[:len(ranked) // 2])

    items, key = [], {}
    for index, family in enumerate(shared, 1):
        prompt = prompts.get(family, {})
        first, second = (("baseline", "treatment") if family in baseline_first
                         else ("treatment", "baseline"))
        arms = {"baseline": baseline[family], "treatment": treatment[family]}
        items.append({
            "item": index,
            "family": family,
            "system": prompt.get("system", ""),
            "user": prompt.get("user", baseline[family].get("user", "")),
            "A": arms[first]["completion"],
            "B": arms[second]["completion"],
        })
        key[str(index)] = {"family": family, "A": first, "B": second}

    out = args.out
    if not args.write:
        print(f"\n{len(items)} items would be written to {out}")
        print("dry run — pass --write to apply")
        return
    out.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Blind pairwise review",
        "",
        f"{len(items)} items. For each, read the task then both answers, and record "
        "your verdict in `VERDICTS.csv` as **A**, **B**, or **tie**.",
        "",
        "You are not told which system produced which answer, and the order is not "
        "alternating. Do not open `KEY.json` until every verdict is filled in.",
        "",
        "Judge overall usefulness as an answer to the task: correctness of any "
        "reasoning, faithfulness to the source, and whether it would be usable as "
        "written. A tie is a legitimate verdict — most answers to an easy task are "
        "genuinely equivalent, and forcing a preference invents signal.",
        "",
        "---",
        "",
    ]
    for item in items:
        lines += [f"## Item {item['item']}", ""]
        if item["system"]:
            lines += ["**System:**", "", "```", item["system"].strip(), "```", ""]
        lines += ["**Task:**", "", "```", item["user"].strip(), "```", "",
                  "### Answer A", "", item["A"].strip(), "",
                  "### Answer B", "", item["B"].strip(), "", "---", ""]
    review_path = out / "REVIEW.md"
    review_path.write_text("\n".join(lines), encoding="utf-8")

    verdicts_path = out / "VERDICTS.csv"
    with verdicts_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["item", "verdict", "note"])
        for item in items:
            writer.writerow([item["item"], "", ""])

    (out / "KEY.json").write_text(json.dumps({
        "_warning": "DO NOT OPEN until VERDICTS.csv is complete. Reading this "
                    "before judging destroys the blinding, and a blinded review "
                    "cannot be un-spoiled — it would have to be redone with a new "
                    "salt and a different reviewer.",
        "salt": args.salt,
        "review_sha256": digest_of(review_path),
        "baseline_file": str(args.baseline),
        "treatment_file": str(args.treatment),
        "items": key,
    }, indent=2) + "\n", encoding="utf-8")

    print(f"\nwrote {review_path}")
    print(f"wrote {verdicts_path}  <- fill this in")
    print(f"wrote {out / 'KEY.json'}  <- do not open until then")


def score(args) -> None:
    out = args.dir
    key = json.loads((out / "KEY.json").read_text(encoding="utf-8"))
    review_path = out / "REVIEW.md"

    if digest_of(review_path) != key["review_sha256"]:
        sys.exit("REVIEW.md has changed since the packet was prepared — verdicts "
                 "cannot be matched to the items that were judged.")

    rows = list(csv.DictReader((out / "VERDICTS.csv").open(encoding="utf-8")))
    filled = [r for r in rows if (r.get("verdict") or "").strip()]
    blank = len(rows) - len(filled)
    if not filled:
        sys.exit("no verdicts recorded yet")

    tally = {"baseline": 0.0, "treatment": 0.0, "tie": 0}
    invalid = []
    for row in filled:
        verdict = row["verdict"].strip().lower()
        mapping = key["items"].get(str(row["item"]).strip())
        if mapping is None:
            invalid.append(row["item"])
            continue
        if verdict == "tie":
            tally["tie"] += 1
            tally["baseline"] += 0.5
            tally["treatment"] += 0.5
        elif verdict in ("a", "b"):
            tally[mapping[verdict.upper()]] += 1
        else:
            invalid.append(row["item"])
    if invalid:
        sys.exit(f"{len(invalid)} verdict(s) are not A, B or tie: {invalid[:5]}")

    n = len(filled) - 0  # ties already counted as half to each
    win_rate = tally["baseline"] / n

    criteria = yaml.safe_load(ACCEPTANCE.read_text(encoding="utf-8"))
    threshold = criteria["quality"]["pairwise_win_rate"]["min_absolute"]

    print(f"items judged     : {len(filled)}" + (f"  ({blank} still blank)" if blank else ""))
    print(f"ties             : {tally['tie']}")
    print(f"baseline (int4)  : {tally['baseline']:.1f} / {n}")
    print(f"treatment (fp8)  : {tally['treatment']:.1f} / {n}")
    print(f"\npairwise_win_rate (baseline) = {win_rate:.4f}")
    print(f"acceptance threshold         >= {threshold}")
    if blank:
        print("\nPARTIAL — some items are unjudged; this is not the study result.")
    print("\n" + ("PASS — int4 is within the agreed preference margin"
                  if win_rate >= threshold else
                  "FAIL — int4 loses by more than the agreed margin; "
                  "acceptance.yaml says: generate at fp8 before training"))
    print("\nThis is one metric of two gates. The critical metrics are scored by "
          "calibrate.py compare;\nboth must pass for the study verdict.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prepare", help="build a blinded review packet")
    p.add_argument("--baseline", type=Path, required=True, help="int4 arm traces")
    p.add_argument("--treatment", type=Path, required=True, help="fp8 arm traces")
    p.add_argument("--prompts", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--salt", required=True,
                   help="any string; recorded in KEY.json so blinding is auditable")
    p.add_argument("--write", action="store_true")

    s = sub.add_parser("score", help="score completed verdicts")
    s.add_argument("--dir", type=Path, required=True)

    args = parser.parse_args()
    (prepare if args.command == "prepare" else score)(args)


if __name__ == "__main__":
    main()
