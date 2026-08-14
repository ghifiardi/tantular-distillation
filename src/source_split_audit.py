"""Audit source-to-split assignment BEFORE any generation.

    ./.venv/bin/python src/source_split_audit.py

Answers one question: would generating from this inventory put the same source
document into more than one split?

That check has to happen here, not after generation. The gate catches it in a
finished corpus, but by then the traces exist, the teacher time is spent, and
the temptation is to rationalise rather than rebuild.

Families are partitioned by the manifest; sources are not. A kind-level
inventory row backs every family of its kind, and those families span all three
splits — so one row silently places one document in train, eval and challenge
while every family-level invariant still passes.

Exits non-zero if any source digest spans splits, or if a kind lacks coverage
in a split it was assigned.
"""
from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import inventory as inventory_module
import splits as splits_module


def main() -> None:
    manifest = splits_module.load()
    rows = inventory_module.load_inventory()
    assignments = manifest["assignments"]
    kinds = manifest["kinds"]

    # Which source digest would back each family, and in which split.
    digest_splits: dict[str, set] = defaultdict(set)
    digest_families: dict[str, list] = defaultdict(list)
    unbacked = []
    for family_id, split in sorted(assignments.items()):
        key, row = inventory_module.resolve_for_family(rows, family_id)
        if not row or not row.get("source_sha256"):
            unbacked.append(family_id)
            continue
        digest = row["source_sha256"]
        digest_splits[digest].add(split)
        digest_families[digest].append((family_id, split, key))

    print(f"families      {len(assignments)}")
    print(f"distinct source digests {len(digest_splits)}")
    grain = Counter(
        inventory_module.resolve_for_family(rows, f)[0] == f for f in assignments)
    print(f"backed by family-level rows: {grain.get(True, 0)}, "
          f"by kind-level rows: {grain.get(False, 0)}")

    failures = []

    if unbacked:
        failures.append(f"{len(unbacked)} family/families have no source digest")
        print(f"\nUNBACKED  {len(unbacked)} — e.g. {', '.join(unbacked[:4])}")

    straddling = {d: sorted(s) for d, s in digest_splits.items() if len(s) > 1}
    print(f"\nsource digests spanning >1 split: {len(straddling)}")
    for digest, spans in list(straddling.items())[:6]:
        fams = digest_families[digest]
        backed_by = fams[0][2]
        print(f"  {digest[:12]}...  {spans}  ({len(fams)} families, row '{backed_by}')")
    if len(straddling) > 6:
        print(f"  ... and {len(straddling) - 6} more")
    if straddling:
        failures.append(
            f"{len(straddling)} source document(s) would appear in multiple splits")

    # Every kind must be able to supply traces in each split it was assigned,
    # from a DIFFERENT document per split.
    per_kind: dict[str, dict] = defaultdict(lambda: defaultdict(set))
    for digest, entries in digest_families.items():
        for family_id, split, _ in entries:
            per_kind[kinds[family_id]][split].add(digest)
    thin = {}
    for kind, splits_map in sorted(per_kind.items()):
        assigned = {s for f, s in assignments.items() if kinds[f] == kind}
        shortfall = {s: len(splits_map.get(s, set())) for s in sorted(assigned)}
        if any(n < 1 for n in shortfall.values()) or len(
                {d for ds in splits_map.values() for d in ds}) < len(assigned):
            thin[kind] = shortfall
    if thin:
        print(f"\nkinds without a distinct document per assigned split: {len(thin)}")
        for kind, shortfall in list(thin.items())[:6]:
            print(f"  {kind:<28} {shortfall}")
        if len(thin) > 6:
            print(f"  ... and {len(thin) - 6} more")
        failures.append(f"{len(thin)} kind(s) lack a distinct document per split")

    if failures:
        print(f"\nAUDIT FAILED — {len(failures)}:")
        for failure in failures:
            print(f"  {failure}")
        print("\nDo not generate. Author distinct source artifacts per split and add "
              "family-level inventory rows first.")
        sys.exit(1)
    print("\nAUDIT PASSED — no source digest spans a split boundary, and every kind "
          "has a distinct document per assigned split.")


if __name__ == "__main__":
    main()
