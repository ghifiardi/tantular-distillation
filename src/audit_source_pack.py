"""Audit a per-family source pack for distinctness and split integrity.

    ./.venv/bin/python src/audit_source_pack.py --pack ~/tantular-source-pack-v3

Four invariants, all of which must hold before a pack is generated against.
Exits 1 on any failure, so this can gate a pipeline rather than merely inform:

  260 families        every family in the split manifest has an artifact
  260 digests         and they are all different
  0 shared            no digest is used by more than one family
  0 straddling        no digest appears in more than one split

The third and fourth are not the same check. Two families in the SAME split
sharing a document is the repetition that made source pack v2's "260/260 ready"
misleading — it inflates a training set without adding information. Two families
in DIFFERENT splits sharing a document is split leakage, which invalidates every
eval number downstream. A pack can pass one and fail the other.

Distinctness is also checked at the level that matters. Identical digests are
the coarse failure; near-identical documents that differ only in a serial number
would pass a digest check while still being paraphrases, so the audit also
reports how much the pack's vocabulary actually varies per split, and whether
the split WORLDS remain separable rather than converging on one another.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import splits as splits_module


def tokens(text: str) -> set[str]:
    """Content words, lowercased. Digits dropped: two documents differing only
    in their figures are still the same document for distinctness purposes."""
    return {w for w in re.findall(r"[A-Za-zÀ-ÿ]{4,}", text.lower())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, required=True)
    args = parser.parse_args()

    pack = args.pack.expanduser()
    digests_path = pack / "digests.json"
    if not digests_path.exists():
        sys.exit(f"no digests.json in {pack}")
    digests = json.loads(digests_path.read_text(encoding="utf-8"))

    manifest = splits_module.load()
    assignments = manifest["assignments"]

    failures = []

    # --- 1. coverage ------------------------------------------------------
    missing = sorted(set(assignments) - set(digests))
    extra = sorted(set(digests) - set(assignments))
    print("=== COVERAGE ===")
    print(f"  families in manifest : {len(assignments)}")
    print(f"  artifacts in pack    : {len(digests)}")
    if missing:
        failures.append(f"{len(missing)} families have no artifact: {missing[:3]}")
    if extra:
        failures.append(f"{len(extra)} artifacts match no family: {extra[:3]}")

    # Recompute rather than trust digests.json — it is written by the same tool
    # that wrote the files, so believing it would only confirm it agrees with
    # itself.
    recomputed, unreadable = {}, []
    for family in digests:
        path = pack / family.replace("::", "__") / "source.txt"
        if not path.exists():
            unreadable.append(family)
            continue
        recomputed[family] = hashlib.sha256(path.read_bytes()).hexdigest()
    if unreadable:
        failures.append(f"{len(unreadable)} artifacts listed but absent on disk")
    mismatched = [f for f, d in recomputed.items() if digests.get(f) != d]
    if mismatched:
        failures.append(f"{len(mismatched)} digests do not match file contents")
    print(f"  digests recomputed from disk: {len(recomputed)}, "
          f"mismatched: {len(mismatched)}")

    # --- 2 & 3. distinctness ---------------------------------------------
    by_digest = defaultdict(list)
    for family, digest in recomputed.items():
        by_digest[digest].append(family)
    shared = {d: f for d, f in by_digest.items() if len(f) > 1}
    print("\n=== DISTINCTNESS ===")
    print(f"  distinct digests     : {len(by_digest)}/{len(recomputed)}")
    print(f"  digests used by >1 family : {len(shared)}")
    if shared:
        for digest, fams in list(shared.items())[:5]:
            print(f"    {digest[:16]} <- {len(fams)} families: {fams[:4]}")
        failures.append(f"{len(shared)} documents are reused across families")

    # --- 4. split integrity ----------------------------------------------
    straddling = {d: sorted({assignments[f] for f in fams})
                  for d, fams in by_digest.items()
                  if len({assignments[f] for f in fams if f in assignments}) > 1}
    print("\n=== SPLIT INTEGRITY ===")
    print(f"  digests spanning >1 split : {len(straddling)}")
    if straddling:
        failures.append(f"{len(straddling)} documents straddle a split boundary")
    per_split = Counter(assignments[f] for f in recomputed if f in assignments)
    print(f"  artifacts per split  : {dict(sorted(per_split.items()))}")

    # --- 5. are the split worlds still substantively apart? ---------------
    # A pack can satisfy every count above and still be paraphrase-thin, if all
    # three splits drift toward one vocabulary. Jaccard overlap of content words
    # between splits is a blunt but objective read on whether the worlds stayed
    # separate.
    vocab = defaultdict(set)
    per_family_tokens = {}
    for family in recomputed:
        text = (pack / family.replace("::", "__") / "source.txt").read_text(encoding="utf-8")
        toks = tokens(text)
        per_family_tokens[family] = toks
        vocab[assignments[family]] |= toks
    print("\n=== SPLIT WORLD SEPARATION ===")
    names = sorted(vocab)
    # Raw pairwise overlap is reported but is NOT the instrument. Every split
    # contains all 26 kinds, so all three share the same scaffolding — "MEMO
    # INTERNAL", "Permintaan penyuntingan", the synthetic footer — which pins
    # overlap near 0.8 no matter how different the domains are. Judging
    # separation by that number would flag a healthy pack as merged.
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            inter, union = len(vocab[a] & vocab[b]), len(vocab[a] | vocab[b])
            print(f"  {a:<10} vs {b:<10} raw overlap {inter / union:.3f} "
                  f"(scaffolding-dominated; not the test)")
    # Exclusive vocabulary is the test: words this split has and the others do
    # not. A split whose world had collapsed into its neighbours would have
    # almost none, whatever the raw overlap said.
    print()
    thin = []
    for split in names:
        others = set().union(*[v for k, v in vocab.items() if k != split])
        exclusive = sorted(vocab[split] - others)
        print(f"  {split:<10} {len(exclusive):>3} EXCLUSIVE words: "
              f"{', '.join(exclusive[:8])}")
        if len(exclusive) < 10:
            thin.append(split)
    if thin:
        failures.append(f"splits with <10 exclusive words (worlds have merged): {thin}")

    # Within-split variety: if families in one split were paraphrases, their
    # pairwise vocabulary overlap would sit near 1.0.
    print("\n=== WITHIN-SPLIT VARIETY (paraphrase check) ===")
    for split in names:
        fams = sorted(f for f in recomputed if assignments[f] == split)
        pairs, total = 0, 0.0
        for i in range(0, min(len(fams), 40)):
            for j in range(i + 1, min(len(fams), 40)):
                ta, tb = per_family_tokens[fams[i]], per_family_tokens[fams[j]]
                if ta | tb:
                    total += len(ta & tb) / len(ta | tb)
                    pairs += 1
        mean = total / pairs if pairs else 0.0
        flag = "  <-- near-duplicate" if mean > 0.9 else ""
        print(f"  {split:<10} mean pairwise overlap {mean:.3f} "
              f"over {pairs} pairs{flag}")

    print()
    if failures:
        for f in failures:
            print(f"FAIL: {f}")
        sys.exit(f"\n{len(failures)} invariant(s) violated — do not generate against this pack")
    print("AUDIT PASSES — 260 families, 260 distinct digests, "
          "0 reused across families, 0 straddling a split")


if __name__ == "__main__":
    main()
