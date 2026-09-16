"""Build the public and private manifests for one corpus file.

Streams the source one record at a time, holds no plaintext beyond the record
in hand, and NEVER writes to the source. What it emits is two artifacts: a
public one safe to commit, and a private one of HMAC identities that is not.

WHAT IT REFUSES TO GUESS. If a component has no explicit extractor it is
recorded as `unverifiable` with a reason, never derived heuristically. Parsing
a prompt to guess where the instruction ends and the document begins would
produce identities that look authoritative and are not, and a later leakage
check would inherit the guess without knowing it had been made.

NO NETWORK, NO MODEL, NO TRAINING. It reads a file and writes two.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

try:
    import yaml
except ImportError:  # pragma: no cover
    raise SystemExit("pyyaml is required: pip install -r requirements.txt")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import corpus_identity as ci                                   # noqa: E402

MANIFEST_SCHEMA = 1
# Fields a source row must carry before it can be identified at all.
REQUIRED_SOURCE_FIELDS = ("family", "system", "user", "completion")
# Components this generator can actually compute.
COMPUTED_COMPONENTS = ("system_payload", "user_payload", "completion_payload",
                       "canonical_row")
# Where each computed component reads from.
COMPONENT_SOURCE = {"system_payload": "system", "user_payload": "user",
                    "completion_payload": "completion"}

EXIT_OK, EXIT_REFUSED = 0, 2


class ManifestError(ValueError):
    """A fail-closed manifest error."""


def _die(message: str) -> None:
    print(f"CORPUS MANIFEST REFUSED: {message}", file=sys.stderr)
    raise SystemExit(EXIT_REFUSED)


# --- key handling -----------------------------------------------------------

def read_key(path: Path) -> bytes:
    """Load the HMAC key, or refuse.

    The key never appears in output, logs or error text -- only its path and
    its length, because a refusal that quoted the key would leak it to whoever
    could already read the error.
    """
    if not path.is_file():
        raise ManifestError(f"no HMAC key file at {path}")
    if path.is_symlink():
        raise ManifestError(
            f"{path} is a symlink; a key must be a regular file so its "
            "permissions are the ones that were checked")
    mode = path.stat().st_mode
    if mode & (stat.S_IRWXG | stat.S_IRWXO):
        raise ManifestError(
            f"{path} is group- or world-accessible (mode {stat.S_IMODE(mode):04o}). "
            "Set 0600 or stricter; a readable key is not a key.")
    material = path.read_bytes().strip()
    if len(material) < ci.MINIMUM_KEY_BYTES:
        raise ManifestError(
            f"{path} holds {len(material)} bytes; at least "
            f"{ci.MINIMUM_KEY_BYTES} of real key material are required")
    return material


def validate_key_id(key_id: str) -> str:
    key_id = str(key_id or "").strip()
    if not key_id:
        raise ManifestError("--hmac-key-id is required and identifies the key")
    if len(key_id) > 64 or any(c.isspace() for c in key_id):
        raise ManifestError("hmac_key_id must be a short whitespace-free label")
    return key_id


# --- streaming --------------------------------------------------------------

def stream_rows(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    """One record at a time. The file is opened read-only and never written."""
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ManifestError(
                    f"{path}:{index}: invalid JSON ({exc.msg}). A corpus with an "
                    "unreadable row cannot be summarised honestly; nothing is "
                    "written.") from exc
            if not isinstance(row, dict):
                raise ManifestError(f"{path}:{index}: row must be a JSON object")
            yield index, row


def raw_digest(path: Path) -> tuple[str, int]:
    sha = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            sha.update(chunk)
            size += len(chunk)
    return sha.hexdigest(), size


def atomic_write(path: Path, text: str, *, mode: int | None = None) -> None:
    """Write via a temp file in the same directory, then rename.

    A half-written manifest that a later check read as authoritative would be
    the worst possible artifact: it would look like evidence.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        if mode is not None:
            os.chmod(temporary, mode)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


# --- build ------------------------------------------------------------------

def build(source: Path, *, key: bytes, key_id: str, role: str,
          repository: str = "ghifiardi/tantular-distillation",
          checkout_head: str | None = None,
          now: str | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    digest, size = raw_digest(source)

    rows: list[dict[str, Any]] = []
    leaves: list[str] = []
    seen_ids: set[str] = set()
    source_classes: dict[str, int] = {}
    models: set[str] = set()
    harnesses: set[str] = set()
    attributed = 0
    present: dict[str, int] = {c: 0 for c in COMPUTED_COMPONENTS}

    for index, row in stream_rows(source):
        missing = [f for f in REQUIRED_SOURCE_FIELDS if f not in row]
        if missing:
            raise ManifestError(
                f"{source}:{index}: missing required field(s) {sorted(missing)}; "
                "a row that cannot be identified must not be summarised as one "
                "that was")
        stable = str(row["family"])
        if stable in seen_ids:
            raise ManifestError(
                f"{source}:{index}: duplicate stable id. Ids are the join key "
                "for a leakage check; a duplicate makes overlap uncountable.")
        seen_ids.add(stable)

        components = {}
        for component in COMPUTED_COMPONENTS:
            if component == "canonical_row":
                components[component] = ci.row_identity(key, row)
            else:
                components[component] = ci.text_identity(
                    key, component, row.get(COMPONENT_SOURCE[component]))
            present[component] += 1

        rows.append({"source_index": index,
                     "stable_id_hmac": ci.stable_id_identity(key, stable),
                     "components": components})
        leaves.append(components["canonical_row"])

        cls = str(row.get("source_class") or "unknown")
        source_classes[cls] = source_classes.get(cls, 0) + 1
        provenance = row.get("provenance") or {}
        if isinstance(provenance, dict) and provenance.get("teacher"):
            models.add(str(provenance["teacher"]))
        harness = row.get("harness_provenance")
        if isinstance(harness, dict) and harness.get("name"):
            harnesses.add(str(harness["name"]))
            attributed += 1
        # The row leaves scope here. Nothing retains its text.

    if not rows:
        raise ManifestError(f"{source} has no rows; an empty manifest describes "
                            "nothing and must not be written")

    root = ci.merkle_root(leaves)
    private = {
        "schema_version": MANIFEST_SCHEMA,
        "normalization_version": ci.NORMALIZATION_VERSION,
        "privacy_scheme": ci.PRIVACY_SCHEME,
        "hmac_key_id": key_id,
        "source_raw_sha256": digest,
        "rows": rows,
        "merkle_root": root,
    }
    private_bytes = ci.canonical_json(private)
    private_digest = hashlib.sha256(private_bytes).hexdigest()

    if attributed == 0:
        attribution = "unattributed"
    elif attributed == len(rows):
        attribution = "attributed"
    else:
        # Partial attribution is neither state, and reporting it as either
        # would lose the only fact that matters about it.
        attribution = "partially_attributed"

    components_block: dict[str, Any] = {}
    for component in COMPUTED_COMPONENTS:
        components_block[component] = {
            "available": present[component] == len(rows),
            "status": "available" if present[component] == len(rows) else "unverifiable",
            "rows_with_identity": present[component],
        }
    for component, reason in ci.UNVERIFIABLE_COMPONENTS.items():
        components_block[component] = {"available": False,
                                       "status": "unverifiable",
                                       "reason": reason}

    public = {
        "schema_version": MANIFEST_SCHEMA,
        "normalization_version": ci.NORMALIZATION_VERSION,
        "privacy_scheme": ci.PRIVACY_SCHEME,
        "hmac_key_id": key_id,
        "source": {
            "kind": "machine_local_gitignored",
            "repository": repository,
            "checkout_head": checkout_head,
            # Said in the artifact, not only in the docs: no commit contains
            # or reproduces these bytes.
            "checkout_head_is_not_a_binding": True,
            "logical_path": str(source),
            "bytes": size,
            "mtime_utc": datetime.fromtimestamp(source.stat().st_mtime,
                                                timezone.utc)
            .strftime("%Y-%m-%dT%H:%M:%SZ"),
            "rows": len(rows),
            "raw_sha256": digest,
        },
        "corpus": {
            "role": role,
            "source_classes": source_classes,
            "model_identities": sorted(models),
            "harness_identities": sorted(harnesses),
            "harness_attribution_status": attribution,
            "private_manifest_sha256": private_digest,
            "private_merkle_root": root,
        },
        "components": components_block,
        "generator": {
            "implementation_digest": implementation_digest(),
            "generated_at": now or datetime.now(timezone.utc)
            .strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
    }
    return public, private


def implementation_digest() -> str:
    """Digest of this generator's own source, so a manifest names the code
    that produced it."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path)
    parser.add_argument("--hmac-key-file", type=Path, required=True)
    parser.add_argument("--hmac-key-id", required=True)
    parser.add_argument("--public-out", type=Path, required=True)
    parser.add_argument("--private-out", type=Path, required=True)
    parser.add_argument("--role", default="training",
                        choices=("training", "evaluation", "calibration",
                                 "development"))
    parser.add_argument("--checkout-head", default=None,
                        help="provenance only; never a binding to the bytes")
    parser.add_argument("--fixture", action="store_true",
                        help="this is a FIXTURE run with a non-production key; "
                             "required for any key id containing 'fixture'")
    args = parser.parse_args()

    try:
        if not args.source.is_file():
            raise ManifestError(f"no such source: {args.source}")
        if args.public_out.resolve() == args.private_out.resolve():
            raise ManifestError("public and private outputs must be different "
                                "files; the private one is not committable")
        for out in (args.public_out, args.private_out):
            if out.resolve() == args.source.resolve():
                raise ManifestError("refusing to write over the source corpus")
        key_id = validate_key_id(args.hmac_key_id)
        looks_fixture = "fixture" in key_id.lower() or "test" in key_id.lower()
        if looks_fixture and not args.fixture:
            raise ManifestError(
                f"key id {key_id!r} looks like a fixture key. Pass --fixture to "
                "say so explicitly; a fixture manifest must never be mistaken "
                "for a production one.")
        if args.fixture and not looks_fixture:
            raise ManifestError(
                "--fixture was passed but the key id does not say so. Name it "
                "with 'fixture' so the artifact is self-describing.")
        key = read_key(args.hmac_key_file)
        public, private = build(args.source, key=key, key_id=key_id,
                                role=args.role,
                                checkout_head=args.checkout_head)
    except (ManifestError, ci.IdentityError) as exc:
        _die(str(exc))

    atomic_write(args.public_out,
                 yaml.safe_dump(public, sort_keys=False, allow_unicode=True))
    # 0600: the private manifest is not committable and not shareable.
    atomic_write(args.private_out,
                 yaml.safe_dump(private, sort_keys=False, allow_unicode=True),
                 mode=0o600)
    print(json.dumps({
        "public": str(args.public_out), "private": str(args.private_out),
        "rows": public["source"]["rows"],
        "raw_sha256": public["source"]["raw_sha256"],
        "private_manifest_sha256": public["corpus"]["private_manifest_sha256"],
        "merkle_root": public["corpus"]["private_merkle_root"],
        "fixture": bool(args.fixture),
    }, indent=2))
    raise SystemExit(EXIT_OK)


if __name__ == "__main__":
    main()
