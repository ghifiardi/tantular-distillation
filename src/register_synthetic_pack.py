"""Write family-level inventory rows for a per-family synthetic source pack.

    ./.venv/bin/python src/register_synthetic_pack.py \
        --pack ~/tantular-source-pack-v3 \
        --approval SYNTHETIC-SOURCE-PACK-V3-2026-08-17 \
        --generator-version "author_sources_v3.py (per-family scenario composition)"

The synthetic counterpart to register_source.py. That tool registers a
sanitized copy of REAL material and refuses to invent any approval reference;
this one registers material that was generated from scratch, where the
"approval" is a pack identifier rather than a human sign-off on a real
document.

The distinction is kept explicit in every row: `source_class: synthetic`,
`redaction_record: not_applicable_synthetic_from_scratch`. Synthetic material
supports pipeline work and behavioural training and supports NO claim about
performance on real Office documents, however many strata it covers. Nothing
here upgrades that.

One row per family, each pointing at that family's own artifact. Refuses to
write if two families would end up pointing at the same document — the failure
that let a 78-document pack report 260/260 ready.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.exit("pyyaml is required: pip install -r requirements.txt")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import inventory as inventory_module
import splits as splits_module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--approval", required=True,
                        help="pack identifier recorded as the approval reference")
    parser.add_argument("--generator-version", required=True)
    parser.add_argument("--generated-at", required=True,
                        help="ISO timestamp, supplied rather than read from the clock")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    pack = args.pack.expanduser()
    digests = json.loads((pack / "digests.json").read_text(encoding="utf-8"))
    manifest = splits_module.load()
    assignments = manifest["assignments"]

    missing = sorted(set(assignments) - set(digests))
    if missing:
        sys.exit(f"{len(missing)} families have no artifact in the pack: {missing[:3]}")

    # The invariant this tool exists to enforce, checked before anything is
    # written rather than reported afterwards.
    reused = {d: n for d, n in Counter(digests.values()).items() if n > 1}
    if reused:
        sys.exit(f"{len(reused)} document(s) are shared by more than one family — "
                 "per-family rows would misreport this pack as distinct")

    generator_sha = hashlib.sha256(
        (Path(__file__).resolve().parent / "author_sources_v3.py").read_bytes()).hexdigest()

    rows = {}
    for family in sorted(assignments):
        source_path = pack / family.replace("::", "__") / "source.txt"
        rows[family] = {
            "approval": args.approval,
            "egress_ai19_approved": True,
            "egress_reference": args.approval,
            "generator": {
                "generated_at": args.generated_at,
                "model": "none",
                "prompt_template_sha256": generator_sha,
                "service": "local-script",
                "version": args.generator_version,
            },
            "notes": f"split={assignments[family]}; one artifact per family; "
                     f"split worlds differ by domain, families differ by entity and quantity",
            "redacted": False,
            "redaction_record": "not_applicable_synthetic_from_scratch",
            "source": f"local-synthetic:{source_path}",
            "source_class": "synthetic",
            "source_sha256": digests[family],
        }

    print(f"{len(rows)} family rows")
    print(f"distinct source documents: {len(set(digests.values()))}/{len(rows)}")
    print(f"distinct source pointers : "
          f"{len({r['source'] for r in rows.values()})}/{len(rows)}")

    if not args.write:
        print("\ndry run — pass --write to apply")
        return

    path = inventory_module.INVENTORY_PATH
    header = ("# Source inventory — FAMILY-level rows, one artifact per family.\n"
              "# Split worlds differ by domain; families within a split differ by\n"
              "# entity and quantity. No document is shared by two families, and no\n"
              "# document spans a split. Audited by src/audit_source_pack.py.\n"
              "#\n"
              "# All rows are source_class: synthetic. This supports pipeline work and\n"
              "# behavioural training, and supports NO claim about real Office documents.\n\n")
    path.write_text(header + yaml.safe_dump(rows, sort_keys=True, allow_unicode=True),
                    encoding="utf-8")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
