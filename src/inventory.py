"""Source inventory: which approved material backs each stratum.

    python3 src/inventory.py refresh                 # sync rows with the manifest
    python3 src/inventory.py status data/raw/*.jsonl # what is blocked, and on what

Coverage is not a number to be satisfied — it is a claim that each stratum is
backed by real, approved, redacted source material. Writing invented documents
to turn 12/26 into 26/26 produces a model that scores well on a corpus
describing a company that does not exist.

So this file records, per kind: is there an approved source, has it been
redacted, and under whose approval. Kinds without that are BLOCKED, and the
correct project state is "blocked on data coverage" rather than synthetic
coverage presented as production data.

Human-filled fields (`source`, `approval`, `redacted`, `notes`) are preserved
across refreshes; only the manifest-derived fields are rewritten.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.exit("pyyaml is required: pip install -r requirements.txt")

import splits as splits_module

ROOT = Path(__file__).resolve().parent.parent
INVENTORY_PATH = ROOT / "inventory" / "sources.yaml"

TEMPLATE_ROW = {
    # Where the real examples come from. A path, a ticket, a system of record —
    # never "generated" or "synthetic" for a production corpus.
    "source": None,
    # Who approved this material for teacher generation, and under what scope.
    "approval": None,
    # Has identifying/confidential content been removed or replaced?
    "redacted": False,
    # Does the approval permit sending this material off-premises?
    "egress_approved": False,
    "notes": "",
}


def load_inventory() -> dict:
    if not INVENTORY_PATH.exists():
        return {}
    return yaml.safe_load(INVENTORY_PATH.read_text(encoding="utf-8")) or {}


def refresh(manifest: dict) -> dict:
    """Rebuild rows from the manifest, preserving anything a human filled in."""
    existing = load_inventory()
    kinds = sorted(set(manifest["kinds"].values()))
    families_per_kind = Counter(manifest["kinds"].values())

    inventory = {}
    for kind in kinds:
        row = dict(TEMPLATE_ROW)
        row.update({k: v for k, v in (existing.get(kind) or {}).items() if k in TEMPLATE_ROW})
        row["families"] = families_per_kind[kind]
        inventory[kind] = row
    return inventory


def save(inventory: dict) -> Path:
    INVENTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# Source inventory — which approved material backs each stratum.\n"
        "#\n"
        "# Fill in `source`, `approval`, and `redacted` before writing seed\n"
        "# prompts for a kind. A kind without an approved, redacted source is\n"
        "# BLOCKED: do not invent documents to satisfy coverage.\n"
        "#\n"
        "# `egress_approved` gates sending this material to an off-premises\n"
        "# host (rented GPU, HF Jobs). Default false — approval must be\n"
        "# explicit, per kind, not assumed.\n"
        "#\n"
        "# Regenerate rows with: python3 src/inventory.py refresh\n"
        "# (human-filled fields are preserved)\n\n"
    )
    INVENTORY_PATH.write_text(
        header + yaml.safe_dump(inventory, sort_keys=True, allow_unicode=True),
        encoding="utf-8",
    )
    return INVENTORY_PATH


def is_ready(row: dict) -> bool:
    return bool(row.get("source") and row.get("approval") and row.get("redacted"))


def status(inventory: dict, corpus_paths: list[Path]) -> None:
    covered = set()
    for path in corpus_paths:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                family = json.loads(line).get("family", "")
                covered.add(family.split("::")[0])

    ready, blocked, generated = [], [], []
    for kind, row in sorted(inventory.items()):
        if kind in covered:
            generated.append(kind)
        elif is_ready(row):
            ready.append(kind)
        else:
            blocked.append(kind)

    print(f"{len(inventory)} kinds")
    print(f"  traces generated : {len(generated)}")
    print(f"  approved, no seeds yet : {len(ready)}")
    print(f"  BLOCKED on source/approval : {len(blocked)}")

    if ready:
        print("\nready to seed:")
        for kind in ready:
            print(f"  {kind}  <- {inventory[kind]['source']}")
    if blocked:
        print("\nblocked — needs approved, redacted source material:")
        for kind in blocked:
            row = inventory[kind]
            missing = [f for f in ("source", "approval") if not row.get(f)]
            if not row.get("redacted"):
                missing.append("redacted")
            print(f"  {kind:<28} missing: {', '.join(missing)}")
        print("\nThis is the correct state to report. Do not write invented "
              "documents to close the gap.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("refresh", help="sync rows with the split manifest")
    status_cmd = sub.add_parser("status", help="report ready vs blocked kinds")
    status_cmd.add_argument("paths", nargs="*", type=Path)
    args = parser.parse_args()

    manifest = splits_module.load()
    if args.command == "refresh":
        inventory = refresh(manifest)
        save(inventory)
        print(f"wrote {INVENTORY_PATH} — {len(inventory)} kinds")
        unfilled = sum(1 for row in inventory.values() if not is_ready(row))
        print(f"{unfilled} kind(s) still need an approved, redacted source")
    else:
        status(load_inventory(), args.paths)


if __name__ == "__main__":
    main()
