"""Split assignment: deterministic, persisted, and enforced before generation.

The design rule this exists to uphold: families are partitioned into
train/eval/challenge BEFORE any teacher call, so no template or document ever
straddles a split boundary. Assigning splits after generation — or letting a
prompt file declare its own — silently leaks training material into eval and
makes every later number meaningless.

`families.py` in the tantular repo is the single source of truth for
enumeration and assignment; this module does not reimplement it. It locates
that module, calls it, and persists the result as a manifest so a corpus
generated today and one generated next month agree on which family is eval.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = ROOT / "splits" / "manifest.json"

# The pipeline lives in a sibling repo. Overridable so this works from a
# standalone clone or CI, rather than assuming one directory layout.
DEFAULT_FINETUNE_DIR = ROOT.parent / "tantular" / "finetune"


def _load_families_module():
    finetune_dir = Path(os.environ.get("TANTULAR_FINETUNE_DIR", DEFAULT_FINETUNE_DIR))
    if not (finetune_dir / "families.py").exists():
        raise SystemExit(
            f"cannot find families.py under {finetune_dir}\n"
            "set TANTULAR_FINETUNE_DIR to the tantular/finetune directory"
        )
    sys.path.insert(0, str(finetune_dir))
    import families  # noqa: E402
    return families


def build(seed: str, instances_per_kind: int) -> dict:
    """Enumerate families, assign splits, and return a manifest dict."""
    families = _load_families_module()
    enumerated = families.enumerate_families(instances_per_kind)
    assignment = families.assign_splits(enumerated, seed=seed)

    kind_of = {f["id"]: f["kind"] for f in enumerated}
    manifest = {
        "seed": seed,
        "instances_per_kind": instances_per_kind,
        "splits": sorted(set(assignment.values())),
        "assignments": dict(sorted(assignment.items())),
        "kinds": dict(sorted(kind_of.items())),
    }
    manifest["fingerprint"] = fingerprint(manifest)
    verify(manifest)
    return manifest


def fingerprint(manifest: dict) -> str:
    """Stable hash of the assignment itself.

    Lets a corpus record which partitioning produced it, so traces generated
    under a different seed can never be mixed into one training set unnoticed.
    """
    payload = json.dumps(manifest["assignments"], sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def verify(manifest: dict) -> None:
    """Fail loudly on the invariants the whole split policy rests on."""
    assignments = manifest["assignments"]
    kinds = manifest["kinds"]

    # A family in two splits is the exact leak this module exists to prevent.
    # A dict cannot express it, so the real check is that every family we
    # enumerated is present exactly once and resolves to a known split.
    valid = set(manifest["splits"])
    for family_id, split in assignments.items():
        if split not in valid:
            raise SystemExit(f"family {family_id} has unknown split {split!r}")
        if family_id not in kinds:
            raise SystemExit(f"family {family_id} has no kind")

    # Every stratum must appear in every split, or an eval score says nothing
    # about the kinds missing from it.
    by_kind: dict[str, set] = {}
    for family_id, split in assignments.items():
        by_kind.setdefault(kinds[family_id], set()).add(split)
    thin = {k: sorted(v) for k, v in by_kind.items() if set(v) != valid}
    if thin:
        raise SystemExit(
            "these kinds do not cover every split (raise instances_per_kind):\n  "
            + "\n  ".join(f"{k}: {v}" for k, v in sorted(thin.items()))
        )


def save(manifest: dict, path: Path = MANIFEST_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def load(path: Path = MANIFEST_PATH) -> dict:
    if not path.exists():
        raise SystemExit(
            f"no split manifest at {path}\n"
            "create one first:  python3 src/splits.py build --seed <seed>"
        )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    # A hand-edited manifest would silently repartition an existing corpus.
    actual = fingerprint(manifest)
    if manifest.get("fingerprint") != actual:
        raise SystemExit(
            f"manifest fingerprint mismatch: recorded {manifest.get('fingerprint')}, "
            f"computed {actual}. The assignment was edited after it was written; "
            "rebuild it or restore the original."
        )
    return manifest


def split_of(family_id: str, manifest: dict) -> str:
    split = manifest["assignments"].get(family_id)
    if split is None:
        kind = family_id.split("::")[0]
        known = sum(1 for k in manifest["kinds"].values() if k == kind)
        hint = (
            f"kind {kind!r} has {known} families; ids look like {kind}::0000"
            if known else
            f"kind {kind!r} is not in the manifest at all"
        )
        raise SystemExit(f"family {family_id!r} is not in the split manifest — {hint}")
    return split


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    build_cmd = sub.add_parser("build", help="enumerate families and assign splits")
    build_cmd.add_argument("--seed", required=True,
                           help="assignment seed; changing it repartitions everything")
    build_cmd.add_argument("--instances-per-kind", type=int, default=10)
    build_cmd.add_argument("--force", action="store_true",
                           help="overwrite an existing manifest")

    sub.add_parser("show", help="summarize the current manifest")

    args = parser.parse_args()

    if args.command == "build":
        if MANIFEST_PATH.exists() and not args.force:
            raise SystemExit(
                f"{MANIFEST_PATH} already exists. Rebuilding repartitions every "
                "family and invalidates any corpus generated under it. Pass "
                "--force if that is genuinely what you want."
            )
        manifest = build(args.seed, args.instances_per_kind)
        save(manifest)
        print(f"wrote {MANIFEST_PATH}")
        print(f"  seed={manifest['seed']} fingerprint={manifest['fingerprint']}")
        print(f"  {len(manifest['assignments'])} families, "
              f"{len(set(manifest['kinds'].values()))} kinds")
        print(f"  balance: {dict(Counter(manifest['assignments'].values()))}")
    else:
        manifest = load()
        verify(manifest)
        print(f"seed={manifest['seed']} fingerprint={manifest['fingerprint']}")
        print(f"{len(manifest['assignments'])} families across "
              f"{len(set(manifest['kinds'].values()))} kinds")
        print(f"balance: {dict(Counter(manifest['assignments'].values()))}")


if __name__ == "__main__":
    main()
