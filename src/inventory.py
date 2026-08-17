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
import pathlib
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
    # "real" — drawn from actual Office documents, or
    # "synthetic" — fabricated from scratch, no real entity anywhere in it.
    # The distinction bounds what the trained student can be claimed to do.
    "source_class": None,
    # Where the examples come from. A path, a ticket, a system of record, or for
    # synthetic material the generating artifact: tinker/<project>/<artifact>.
    "source": None,
    # Hash of the exported source artifact, so a corpus can be traced to the
    # exact documents that produced it.
    "source_sha256": None,
    # Who approved this material for teacher generation, and under what scope.
    # For synthetic material this is the approval for substituting fabricated
    # documents in place of real ones.
    "approval": None,
    # Has identifying/confidential content been removed or replaced?
    "redacted": False,
    # HOW. "Synthetic from scratch; no real personal or company data" is a
    # different claim from "names replaced in a real document", and only the
    # first survives an external service boundary safely.
    "redaction_record": None,
    # Who owns the source material, and under what authority it is used.
    # Unknown ownership is a blocker, not a formality.
    "owner": None,
    # Where the ORIGINALS live. Must be outside Git and outside data/raw/.
    # Recorded so a sanitized artifact can be traced back under supervision,
    # never so the originals get copied into the repo.
    "originals_location": None,
    # Sending sanitized local material to ai19 over the SSH tunnel is still an
    # egress event and needs its own authorization.
    "egress_ai19_approved": False,
    "egress_reference": None,
    # For synthetic material: what generated it. An external service is an
    # external boundary regardless of its terms, so the artifact must be
    # attributable.
    "generator": None,   # {service, model, version, prompt_template_sha256, generated_at}
    "notes": "",
}

# Fields every stratum needs before seeds may be written for it.
REQUIRED_ALWAYS = ("source_class", "source", "source_sha256", "approval",
                   "redaction_record")
# Real material additionally needs an owner, actual redaction, and egress
# authorization — the SSH tunnel to ai19 is an external boundary crossing.
REQUIRED_REAL = ("owner", "originals_location")
# Fabricated material needs attributable provenance, but no redaction: there is
# nothing real in it to redact.
REQUIRED_SYNTHETIC = ("generator",)
GENERATOR_FIELDS = ("service", "model", "version", "prompt_template_sha256", "generated_at")
# "local_real"  — an actual document, held locally, sanitized before use
# "synthetic"   — fabricated; nothing real in it
VALID_SOURCE_CLASSES = ("local_real", "synthetic")


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


MIN_SOURCE_BYTES = 120
SHELL_ARTIFACT_MARKERS = ("pbpaste", "pbcopy", "cat >", "cat <<", "echo >", "EOF")


def inspect_artifact(source: str) -> str | None:
    """Return a problem description, or None if the artifact looks like content."""
    raw = str(source)
    for prefix in ("local-synthetic:", "local:", "drive:"):
        if raw.startswith(prefix):
            raw = raw[len(prefix):]
            break
    if raw.startswith("drive/"):
        return None  # a Drive reference, not a local artifact to inspect
    path = pathlib.Path(raw).expanduser()
    if not path.exists():
        return "artifact missing on disk"
    if path.is_dir():
        return None
    size = path.stat().st_size
    try:
        head = path.read_text(encoding="utf-8", errors="replace")[:400]
    except OSError as error:
        return f"unreadable ({error})"
    if any(m in head for m in SHELL_ARTIFACT_MARKERS) and size < 400:
        return (f"captured shell command, not a document "
                f"({size}B: {head.strip()[:52]!r})")
    if size < MIN_SOURCE_BYTES:
        return f"implausibly small for a source document ({size}B)"
    return None


def _artifact_path(source: str) -> pathlib.Path:
    raw = str(source)
    for prefix in ("local-synthetic:", "local:", "drive:"):
        if raw.startswith(prefix):
            raw = raw[len(prefix):]
            break
    return pathlib.Path(raw).expanduser()


def hash_artifact(source: str) -> str | None:
    path = _artifact_path(source)
    if not path.is_file():
        return None
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def missing_fields(row: dict) -> list[str]:
    """What this stratum still needs. Empty means ready to seed."""
    missing = [f for f in REQUIRED_ALWAYS if not row.get(f)]

    source_class = row.get("source_class")
    if source_class and source_class not in VALID_SOURCE_CLASSES:
        missing.append(f"source_class(invalid:{source_class})")

    if source_class == "local_real":
        missing.extend(f for f in REQUIRED_REAL if not row.get(f))
        # Real material must actually be sanitized, and the sanitized artifact
        # still crosses a boundary when it reaches ai19.
        if not row.get("redacted"):
            missing.append("redacted")
        if not row.get("egress_ai19_approved"):
            missing.append("egress_ai19_approved")
        if not row.get("egress_reference"):
            missing.append("egress_reference")
        # Originals in the repo defeat the point of keeping them out of Git.
        location = str(row.get("originals_location") or "")
        if location and not location.startswith("<") and _inside_repo(location):
            missing.append("originals_location(INSIDE REPO — move them out)")
    elif source_class == "synthetic":
        missing.extend(f for f in REQUIRED_SYNTHETIC if not row.get(f))
        generator = row.get("generator") or {}
        missing.extend(f"generator.{f}" for f in GENERATOR_FIELDS
                       if not generator.get(f))
        # Nothing real to redact, but if generation crossed a service boundary
        # that still needed authorizing.
        if (generator.get("service") or "").lower() not in ("", "local", "handwritten"):
            if not row.get("egress_reference"):
                missing.append("egress_reference")

    # Content sanity. Metadata completeness says nothing about content: a
    # 37-byte file containing the shell command that was meant to create it
    # passes existence, hashing and every required field, and would be
    # reported READY.
    source = row.get("source")
    if source:
        problem = inspect_artifact(source)
        if problem:
            missing.append(f"source(ARTIFACT: {problem})")
        # A recorded hash that no longer matches its file means the artifact
        # changed after approval. The row then attests to content that is no
        # longer there — silently, since every other field still looks fine.
        recorded = row.get("source_sha256")
        if recorded and not problem:
            actual = hash_artifact(source)
            if actual and actual != recorded:
                missing.append(
                    f"source_sha256(STALE: recorded {recorded[:12]}..., "
                    f"file is {actual[:12]}... — artifact changed after approval)")
    return missing


def _inside_repo(location: str) -> bool:
    try:
        Path(location).expanduser().resolve().relative_to(ROOT)
        return True
    except (ValueError, OSError):
        return False


def is_ready(row: dict) -> bool:
    return not missing_fields(row)


def resolve_for_family(inventory: dict, family_id: str) -> tuple[str, dict] | tuple[None, None]:
    """A family-specific row wins over its kind's row.

    Rows may be keyed by KIND (`document:memo`, covering all its families) or by
    FAMILY (`document:memo::0003`, covering exactly one). Different families of
    a kind usually come from different documents, so the family key is the more
    honest granularity — but a single document legitimately backing a whole kind
    should not require ten identical rows.
    """
    if family_id in inventory:
        return family_id, inventory[family_id]
    kind = family_id.split("::")[0]
    if kind in inventory:
        return kind, inventory[kind]
    return None, None


def family_coverage(inventory: dict, manifest: dict) -> dict:
    """Per-family readiness, which is what generation actually needs."""
    out = {}
    for family_id in manifest["assignments"]:
        key, row = resolve_for_family(inventory, family_id)
        out[family_id] = {
            "key": key,
            "granularity": ("family" if key == family_id else
                            "kind" if key else None),
            "missing": missing_fields(row) if row else ["no inventory row"],
            "source": (row or {}).get("source"),
        }
    return out


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

    classes = Counter(row.get("source_class") or "unset" for row in inventory.values())
    print(f"{len(inventory)} kinds   source_class: {dict(classes)}")
    if classes.get("synthetic"):
        print("  NOTE: strata marked `synthetic` support pipeline generation and "
              "behavioural\n        training, but NOT any claim about performance on "
              "real Office documents.")
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
            print(f"  {kind:<28} missing: {', '.join(missing_fields(row))}")
        print("\nThis is the correct state to report. Do not write invented "
              "documents to close the gap.")

    # Per-family view. Kind-level rows cover all families of that kind; a
    # family-level row overrides for one. Generation needs family granularity,
    # so report it explicitly rather than letting a kind row imply that ten
    # families are all backed by the same approved document.
    try:
        manifest = splits_module.load()
    except SystemExit:
        return
    coverage = family_coverage(inventory, manifest)
    ready = [f for f, c in coverage.items() if not c["missing"]]
    by_grain = Counter(c["granularity"] for c in coverage.values() if not c["missing"])
    print(f"\nper-family: {len(ready)}/{len(coverage)} families ready"
          + (f"  (backed by: {dict(by_grain)})" if by_grain else ""))
    if ready and by_grain.get("kind"):
        print("  NOTE: families backed by a KIND-level row all inherit one source. "
              "If different\n        families of that kind come from different "
              "documents, add family-level rows.")

    # Readiness is not distinctness, and reporting only the former is how a pack
    # of 78 documents behind 260 families read as "260/260 ready". Family-level
    # ROWS do not imply distinct DOCUMENTS: v2 had 260 family rows whose source
    # pointers resolved to 78 files, so the granularity note above stayed silent
    # while every document was inherited 3.33x. Count the pointers.
    sources = [coverage[f]["source"] for f in ready if coverage[f].get("source")]
    if sources:
        distinct = len(set(sources))
        shared = Counter(sources)
        reused = {s: n for s, n in shared.items() if n > 1}
        print(f"  distinct source documents : {distinct}/{len(sources)}")
        if reused:
            worst = max(reused.values())
            print(f"  REUSED: {len(reused)} document(s) back more than one family "
                  f"(max {worst} families share one)")
            print(f"          mean {len(sources) / distinct:.2f} families per document — "
                  f"a training set\n          inherits each document that many times "
                  f"behind an identical prompt.")
            print("          Per-family artifacts are NOT in place despite the "
                  "readiness count above.")
        else:
            print("  every ready family has its own document — no reuse")


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
