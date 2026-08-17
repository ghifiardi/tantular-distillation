"""Persist the router pool-versus-family report as a regression baseline.

    ./.venv/bin/python src/write_router_pool_baseline.py --write

Router artifacts are parameterised by topic alone, so a router kind's available
distinct documents in a split equal that split's topic pool. Source pack v3
currently sits at margin 0 on its tightest group: router:CEK_AMAN has 8 train
families against 8 train topics. It is distinct today because index spacing
happened to land on every topic — not because the pool is large enough to
guarantee it.

The baseline records that state, plus the two inputs that would invalidate it:
the split fingerprint and instances_per_kind. tests/ asserts against this file,
so raising instances_per_kind or reseeding the splits fails a test at the point
of change rather than silently emitting two identical documents and relying on
someone re-running the pack audit to notice.

Regenerate deliberately, never to make a red test go green: a changed baseline
is a claim that the new margins were reviewed and accepted.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import author_sources_v3
import splits as splits_module

ROOT = Path(__file__).resolve().parent.parent
BASELINE = ROOT / "calibration" / "ROUTER_POOL_BASELINE.json"


def build() -> dict:
    manifest = splits_module.load()
    report = author_sources_v3.router_pool_report(manifest)
    return {
        "_what": "Router topic pool vs families drawing on it, per (kind, split). "
                 "Router documents vary by TOPIC ALONE, so pool size is the ceiling "
                 "on how many distinct router documents a split can yield for one "
                 "kind. margin = topics - families; margin 0 means distinctness "
                 "rests on index spacing rather than on having room.",
        "_invalidated_by": "A change to either input below, or to WORLDS[split]['topics']. "
                           "tests/test_router_pool_regression.py fails when they move.",
        "split_fingerprint": manifest["fingerprint"],
        "split_seed": manifest["seed"],
        "instances_per_kind": manifest["instances_per_kind"],
        "topics_per_split": {split: len(world["topics"])
                             for split, world in sorted(author_sources_v3.WORLDS.items())},
        "tightest_margin": report[0]["margin"] if report else None,
        "groups": report,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    baseline = build()
    print(f"split_fingerprint  {baseline['split_fingerprint']}")
    print(f"instances_per_kind {baseline['instances_per_kind']}")
    print(f"topics per split   {baseline['topics_per_split']}")
    print(f"tightest margin    {baseline['tightest_margin']}")
    print(f"\n{'kind':<24}{'split':<12}{'families':>9}{'topics':>8}{'margin':>8}")
    for row in baseline["groups"][:6]:
        print(f"{row['kind']:<24}{row['split']:<12}{row['families']:>9}"
              f"{row['topics']:>8}{row['margin']:>8}")

    if not args.write:
        print("\ndry run — pass --write to apply")
        return
    BASELINE.write_text(json.dumps(baseline, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {BASELINE}")


if __name__ == "__main__":
    main()
