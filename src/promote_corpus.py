"""Promote mechanically-verifiable traces into a training corpus.

    ./.venv/bin/python src/promote_corpus.py \
        --traces data/v3-candidate/traces.r0.jsonl \
        --prompts prompts/expanded.v3.jsonl \
        --out-dir data/promoted \
        --manifest train/RUN_MANIFEST.v1-mechanical.json \
        --frozen-at 2026-08-19T10:00:00+07:00 --write

The README points at ../tantular/finetune/{judge,dedup,review_promote}.py, but
only one of those is usable here. `judge.py` has no CLI and is a teacher-as-judge
injected into gen_edit for the edit axis's fallback subtypes; `review_promote.py`
consumes a `review_queue.jsonl` this pipeline never produces. `dedup.py` IS
reusable — a pure, deterministic shingle-Jaccard library — and is imported below.

THE BAR IS MIXED, BY DECISION (2026-08-19):

  strata WITH mechanical checks   promoted when provenance is complete, the
                                  source digest matches the pack, and no
                                  mechanical check failed.

  strata WITHOUT checks           NOT promoted. Five open-ended prose strata
                                  (explain / draft / re-register / assess /
                                  answer) have no objective rule, so nothing
                                  here can establish their quality. Promoting
                                  them on "no check failed" would manufacture
                                  quality from the absence of a test.

Using Muse Glimmer to judge its own output was considered and rejected: it
measures a model against itself. Those strata need an independent judge or human
review, and until then exclusion is the honest option — a smaller corpus that
means something beats a larger one that does not.

SPLITS ARE NOT RE-DERIVED. `split` and `split_fingerprint` come from the
manifest that governed generation. Re-deriving them would risk moving a family
across a boundary its traces were generated under. `challenge` is held out of
both outputs: it is neither training nor ordinary eval.

DEDUP IS AN AUDIT, NOT A FILTER. Near-duplicates are reported per split and
never removed automatically — two families can legitimately produce the same
answer to different instructions, and discarding one would delete a valid
example. Only EXACT duplicates of (prompt, completion) are dropped, which is
deterministic and cannot discard distinct content.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import calibrate  # noqa: E402
import splits as splits_module  # noqa: E402

# dedup.py lives in the sibling repo and has no CLI; import it as a library.
FINETUNE_DIR = ROOT.parent / "tantular" / "finetune"


def load_dedup():
    if not (FINETUNE_DIR / "dedup.py").exists():
        return None
    sys.path.insert(0, str(FINETUNE_DIR))
    import dedup  # noqa: E402
    return dedup


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--frozen-at", required=True)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    traces = [json.loads(l) for l in
              args.traces.read_text(encoding="utf-8").splitlines() if l.strip()]
    prompts = {r["family"]: r for r in
               (json.loads(l) for l in
                args.prompts.read_text(encoding="utf-8").splitlines() if l.strip())}
    manifest = splits_module.load()
    assignments = manifest["assignments"]

    # --- which strata can be judged mechanically at all --------------------
    checked, unchecked = set(), set()
    for family, prompt in prompts.items():
        stratum = family.split("::")[0]
        (checked if (prompt.get("checks") or prompt.get("expected")) else unchecked
         ).add(stratum)
    unchecked -= checked  # a stratum counts as checked if any prompt carries one

    print("=== BAR ===")
    print(f"  {len(checked)} strata have mechanical checks -> eligible")
    print(f"  {len(unchecked)} strata have none -> EXCLUDED: {', '.join(sorted(unchecked))}")

    # --- per-trace verdicts -------------------------------------------------
    promoted, rejected = [], []
    reasons = Counter()
    for trace in traces:
        family = trace.get("family", "")
        stratum = family.split("::")[0]
        prompt = prompts.get(family, {})
        why = None

        if stratum in unchecked:
            why = "stratum has no mechanical check (excluded by decision)"
        elif not trace.get("provenance"):
            why = "provenance missing"
        elif not trace.get("source_sha256"):
            why = "source_sha256 missing"
        elif prompt.get("source_sha256") != trace.get("source_sha256"):
            why = "source digest does not match the prompt's document"
        elif family not in assignments:
            why = "family absent from the split manifest"
        else:
            scored = calibrate.score_trace({
                **trace, "checks": prompt.get("checks") or {},
                **({"expected": prompt["expected"]} if prompt.get("expected") else {})})
            if scored["empty"]:
                why = "empty completion"
            elif scored["truncated"]:
                why = "truncated"
            elif scored["refusal"]:
                why = "refusal"
            elif scored["constraints_ok"] is False:
                why = "constraint violated"
            elif scored["router_correct"] is False:
                why = "router label wrong"
            elif scored["source_preserved"] is not None and scored["source_preserved"] < 1.0:
                why = f"source preservation {scored['source_preserved']:.2f} < 1.0"

        if why:
            rejected.append({"family": family, "reason": why})
            reasons[why] += 1
        else:
            promoted.append(trace)

    print(f"\n=== VERDICTS ===\n  promoted {len(promoted)}/{len(traces)}")
    for reason, n in reasons.most_common():
        print(f"  rejected {n:>3}  {reason}")

    # --- exact duplicates: deterministic, safe to drop ----------------------
    seen, exact_dupes, kept = {}, [], []
    for trace in promoted:
        key = hashlib.sha256(
            (prompts.get(trace["family"], {}).get("user", "") + "\x00" +
             trace.get("completion", "")).encode()).hexdigest()
        if key in seen:
            exact_dupes.append({"family": trace["family"], "duplicate_of": seen[key]})
        else:
            seen[key] = trace["family"]
            kept.append(trace)
    print(f"\n=== EXACT DUPLICATES (prompt+completion) ===\n  {len(exact_dupes)} removed")

    # --- near-duplicates: AUDIT ONLY ---------------------------------------
    dedup = load_dedup()
    near = {}
    print("\n=== NEAR-DUPLICATE AUDIT (reported, never removed) ===")
    if dedup is None:
        print("  dedup.py not found; audit SKIPPED")
        near = None
    else:
        by_split = defaultdict(list)
        for trace in kept:
            by_split[assignments[trace["family"]]].append(trace)
        for split, rows in sorted(by_split.items()):
            texts = [r.get("completion", "") for r in rows]
            # near_duplicates returns the INDICES IT WOULD REJECT — later members
            # of a cluster, first occurrence always kept. For an audit that is
            # only half the story: naming what each flagged trace resembles is
            # what makes a human review possible, so the partner is recovered
            # with the library's own similarity function rather than guessed.
            flagged = dedup.near_duplicates(texts)
            shingles = [dedup._shingles(dedup._normalize(t)) for t in texts]
            entries = []
            for i in sorted(flagged):
                best_j, best_s = None, 0.0
                for j in range(i):
                    s = dedup._jaccard(shingles[i], shingles[j])
                    if s > best_s:
                        best_j, best_s = j, s
                entries.append({
                    "flagged": rows[i]["family"],
                    "resembles": rows[best_j]["family"] if best_j is not None else None,
                    "similarity": round(float(best_s), 4),
                })
            # Router answers are single closed-set labels — every correct answer
            # for an intent IS the same string, so near-duplicate detection flags
            # ~100% of them by construction. dedup.py's own docstring scopes it to
            # "teacher-generated prose". Separating these keeps a reviewer from
            # working through dozens of non-findings to reach the real ones.
            for e in entries:
                e["expected_by_construction"] = e["flagged"].startswith("router:")
            near[split] = entries
            real = [e for e in entries if not e["expected_by_construction"]]
            label = [e for e in entries if e["expected_by_construction"]]
            print(f"  {split:<10} {len(rows):>3} traces, {len(entries)} flagged "
                  f"({len(real)} to review, {len(label)} closed-set labels)")
            for e in real[:3]:
                print(f"      {e['similarity']}  {e['flagged']}  ~  {e['resembles']}")
        print("  Not removed: two families may answer different instructions the same")
        print("  way, and discarding one would delete a valid example.")

    # --- split the promoted set --------------------------------------------
    out = defaultdict(list)
    for trace in kept:
        out[assignments[trace["family"]]].append(trace)
    print(f"\n=== SPLITS (from manifest {manifest['fingerprint']}, not re-derived) ===")
    for split in ("train", "eval", "challenge"):
        note = "  <- held out of both outputs" if split == "challenge" else ""
        print(f"  {split:<10} {len(out[split]):>3}{note}")

    if not args.write:
        print("\ndry run — pass --write to apply")
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    for split, name in (("train", "train.jsonl"), ("eval", "eval.jsonl")):
        path = args.out_dir / name
        path.write_text("\n".join(json.dumps(t, ensure_ascii=False) for t in out[split]) + "\n",
                        encoding="utf-8")
        written[split] = {
            "path": str(path), "traces": len(out[split]),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "families": sorted(t["family"] for t in out[split]),
        }

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps({
        "_what": "Families promoted into a training corpus under the MECHANICAL bar. "
                 "Does not supersede RUN_MANIFEST.v1.json, which freezes the full "
                 "260-trace corpus as generated.",
        "frozen_at": args.frozen_at,
        "source_corpus": {
            "path": str(args.traces),
            "sha256": hashlib.sha256(args.traces.read_bytes()).hexdigest(),
            "traces": len(traces),
        },
        "bar": {
            "mechanical_strata_promoted": sorted(checked),
            "unchecked_strata_EXCLUDED": sorted(unchecked),
            "_why_excluded": "No objective rule exists for these, so nothing here "
                             "can establish their quality. Muse Glimmer was NOT used "
                             "to judge its own output; that measures a model against "
                             "itself. They need an independent judge or human review.",
        },
        "splits": {
            "fingerprint": manifest["fingerprint"],
            "_not_re_derived": "split and split_fingerprint come from the manifest that "
                               "governed generation.",
            "challenge_held_out": len(out["challenge"]),
        },
        "promoted": written,
        "rejected": {"count": len(rejected), "by_reason": dict(reasons),
                     "families": rejected},
        "exact_duplicates_removed": exact_dupes,
        "near_duplicate_audit": near,
        "_near_duplicates_not_removed": "Reported for review only. Removal is a human "
                                        "decision; identical answers to different "
                                        "instructions can both be valid.",
        "_near_duplicate_caveat": "Entries with expected_by_construction=true are "
                                  "router strata, whose answers are single closed-set "
                                  "labels — every correct answer for an intent is the "
                                  "same string, so near-duplicate detection flags them "
                                  "at ~100% by construction. dedup.py is scoped to "
                                  "prose by its own docstring. Only the false entries "
                                  "are a genuine review surface.",
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"\nwrote {written['train']['path']}  {written['train']['traces']} traces")
    print(f"wrote {written['eval']['path']}   {written['eval']['traces']} traces")
    print(f"wrote {args.manifest}")


if __name__ == "__main__":
    main()
