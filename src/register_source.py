"""Hash a sanitized local copy and draft its inventory row.

    ./.venv/bin/python src/register_source.py \
        --file ~/tantular-private-sources/memo-sanitized.docx \
        --kind document:memo \
        --drive-id 1AbC...xyz \
        --owner "Legal, K. Putri" \
        --approval TKT-1041 \
        --redaction-record TKT-1041-redaction \
        --egress-reference TKT-1042

Automates steps 5 and 6 only. It does NOT approve anything: every approval
reference must be supplied and is copied verbatim. Placeholders are rejected —
a row that looks complete but carries `<fill me>` is worse than a blocked one,
because `status` would report it ready.

The file must live outside the repository. Originals stay in Drive untouched;
this registers a SANITIZED copy.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.exit("pyyaml is required")

import inventory as inventory_module

ROOT = Path(__file__).resolve().parent.parent

# Anything that looks like a stand-in rather than a real reference.
PLACEHOLDER_MARKERS = ("<", ">", "tbd", "todo", "xxx", "placeholder", "fixme",
                       "example", "your-", "fill")


def looks_like_placeholder(value: str) -> bool:
    lowered = str(value).strip().lower()
    return (not lowered) or any(marker in lowered for marker in PLACEHOLDER_MARKERS)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", type=Path, required=True,
                        help="the SANITIZED local copy, outside the repo")
    parser.add_argument("--kind", required=True,
                        help="kind (document:memo) or family id (document:memo::0003)")
    parser.add_argument("--drive-id", required=True, help="Drive file id or URL")
    parser.add_argument("--owner", required=True)
    parser.add_argument("--approval", required=True)
    parser.add_argument("--redaction-record", required=True)
    parser.add_argument("--egress-reference", required=True,
                        help="authorization to send the sanitized artifact to ai19")
    parser.add_argument("--sensitivity", default="")
    args = parser.parse_args()

    path = args.file.expanduser().resolve()
    if not path.exists():
        sys.exit(f"no such file: {path}")

    # Originals and sanitized copies both belong outside version control.
    try:
        path.relative_to(ROOT)
        sys.exit(f"{path} is INSIDE the repository. Keep source material outside "
                 "Git — move it to a private directory and re-run.")
    except ValueError:
        pass

    references = {
        "owner": args.owner,
        "approval": args.approval,
        "redaction_record": args.redaction_record,
        "egress_reference": args.egress_reference,
        "drive-id": args.drive_id,
    }
    bad = [name for name, value in references.items() if looks_like_placeholder(value)]
    if bad:
        sys.exit(f"placeholder or empty value for: {', '.join(bad)}\n"
                 "Supply real references, or leave the stratum blocked. A row that "
                 "looks complete but is not would be reported ready.")

    row = {
        "source_class": "local_real",
        "source": f"drive/{args.drive_id}",
        "source_sha256": sha256_file(path),
        "originals_location": str(path),
        "owner": args.owner,
        "approval": args.approval,
        "redacted": True,
        "redaction_record": args.redaction_record,
        "egress_ai19_approved": True,
        "egress_reference": args.egress_reference,
        "notes": f"sensitivity: {args.sensitivity}" if args.sensitivity else "",
    }

    missing = inventory_module.missing_fields(row)
    if missing:
        sys.exit(f"drafted row is still incomplete: {', '.join(missing)}")

    print(f"# verified complete — paste into inventory/sources.yaml under '{args.kind}'")
    print(f"# sanitized copy: {path}")
    print(f"# sha256 covers the SANITIZED copy, not the Drive original\n")
    print(yaml.safe_dump({args.kind: row}, sort_keys=True, allow_unicode=True))
    print("# `redacted: true` and `egress_ai19_approved: true` assert that "
          "redaction was\n# performed and transfer authorized. Do not paste this "
          "unless both are actually true.")


if __name__ == "__main__":
    main()
