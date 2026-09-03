"""Build the exact conversation payload sent to Tinker.

This module is deliberately dependency-free.  The schema-v3 freezer and the
Tinker runner both import it, so the bytes frozen before a run are the bytes
later written for upload.

The upload JSONL contains only the format accepted by
``FromConversationFileBuilder``::

    {"messages": [{"role": "system", ...}, {"role": "user", ...},
                  {"role": "assistant", ...}]}

Source classification and provenance stay in a local sidecar.  Each sidecar
row pins the SHA-256 of its corresponding upload line; this lets the egress
check prove that the classified row and the bytes being sent are the same
artifact without transmitting internal provenance metadata to the service.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Iterable


class PayloadError(ValueError):
    """The promoted corpus cannot be rendered safely for Tinker."""


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _required_text(row: dict, field: str, family: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise PayloadError(
            f"{family}: {field} must be a non-empty string; refusing to render"
        )
    return value


def render_rows(rows: Iterable[dict], *, split: str) -> tuple[bytes, bytes, dict]:
    """Return upload bytes, audit-sidecar bytes, and a deterministic summary."""
    upload_lines: list[str] = []
    audit_lines: list[str] = []
    families: set[str] = set()
    source_classes: Counter[str] = Counter()
    max_content_bytes = 0

    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise PayloadError(f"{split} row {index}: expected an object")
        family = row.get("family")
        if not isinstance(family, str) or not family.strip():
            raise PayloadError(f"{split} row {index}: family must be a non-empty string")
        if family in families:
            raise PayloadError(f"{split}: duplicate family {family!r}")
        families.add(family)

        declared_split = row.get("split")
        if declared_split != split:
            raise PayloadError(
                f"{family}: row declares split {declared_split!r}, expected {split!r}"
            )

        # Exact comparison is intentional.  Missing, differently-cased, real,
        # and otherwise unclassified material all fail closed at the external
        # service boundary.
        source_class = row.get("source_class")
        if not isinstance(source_class, str) or source_class != "synthetic":
            raise PayloadError(
                f"{family}: source_class must be exactly 'synthetic' for Tinker; "
                f"got {source_class!r}"
            )
        source_classes[source_class] += 1

        system = _required_text(row, "system", family)
        user = _required_text(row, "user", family)
        assistant = _required_text(row, "completion", family)
        payload = {
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant},
            ]
        }
        upload_line = canonical_json(payload)
        upload_digest = sha256_bytes(upload_line.encode("utf-8"))
        upload_lines.append(upload_line)

        source_sha = row.get("source_sha256")
        if not isinstance(source_sha, str) or len(source_sha) != 64:
            raise PayloadError(f"{family}: source_sha256 is missing or malformed")
        audit_lines.append(canonical_json({
            "line": index + 1,
            "family": family,
            "split": split,
            "source_class": source_class,
            "source_sha256": source_sha,
            "upload_line_sha256": upload_digest,
        }))
        max_content_bytes = max(
            max_content_bytes,
            len(system.encode("utf-8"))
            + len(user.encode("utf-8"))
            + len(assistant.encode("utf-8")),
        )

    if not upload_lines:
        raise PayloadError(f"{split}: no rows to render")

    upload = ("\n".join(upload_lines) + "\n").encode("utf-8")
    audit = ("\n".join(audit_lines) + "\n").encode("utf-8")
    return upload, audit, {
        "rows": len(upload_lines),
        "families": sorted(families),
        "source_classes": dict(sorted(source_classes.items())),
        "sha256": sha256_bytes(upload),
        "audit_sha256": sha256_bytes(audit),
        "max_content_utf8_bytes": max_content_bytes,
    }


def verify_upload_against_audit(upload: bytes, audit: bytes) -> None:
    """Prove every classified sidecar row names the exact upload line."""
    upload_lines = upload.decode("utf-8").splitlines()
    audit_lines = audit.decode("utf-8").splitlines()
    if len(upload_lines) != len(audit_lines):
        raise PayloadError(
            f"upload/audit row count differs: {len(upload_lines)} vs {len(audit_lines)}"
        )
    for index, (upload_line, audit_line) in enumerate(
        zip(upload_lines, audit_lines, strict=True), start=1
    ):
        try:
            payload = json.loads(upload_line)
            record = json.loads(audit_line)
        except json.JSONDecodeError as exc:
            raise PayloadError(f"rendered row {index} is not valid JSON: {exc}") from exc
        if set(payload) != {"messages"}:
            raise PayloadError(
                f"upload row {index}: expected only a messages field, got {sorted(payload)}"
            )
        messages = payload["messages"]
        if (
            not isinstance(messages, list)
            or [m.get("role") for m in messages if isinstance(m, dict)]
            != ["system", "user", "assistant"]
        ):
            raise PayloadError(
                f"upload row {index}: roles must be system, user, assistant"
            )
        if record.get("line") != index:
            raise PayloadError(f"audit row {index}: line number mismatch")
        if record.get("source_class") != "synthetic":
            raise PayloadError(
                f"audit row {index}: source_class is not exactly 'synthetic'"
            )
        actual = sha256_bytes(upload_line.encode("utf-8"))
        if record.get("upload_line_sha256") != actual:
            raise PayloadError(
                f"audit row {index}: upload line digest mismatch"
            )


def render_files(train_path: Path, eval_path: Path) -> dict:
    def load(path: Path) -> list[dict]:
        try:
            return [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except (OSError, json.JSONDecodeError) as exc:
            raise PayloadError(f"invalid JSONL at {path}: {exc}") from exc

    train_upload, train_audit, train_summary = render_rows(
        load(train_path), split="train"
    )
    eval_upload, eval_audit, eval_summary = render_rows(
        load(eval_path), split="eval"
    )
    overlap = set(train_summary["families"]) & set(eval_summary["families"])
    if overlap:
        raise PayloadError(
            f"train/eval family overlap: {', '.join(sorted(overlap)[:5])}"
        )
    verify_upload_against_audit(train_upload, train_audit)
    verify_upload_against_audit(eval_upload, eval_audit)
    return {
        "train": {
            "upload": train_upload,
            "audit": train_audit,
            **train_summary,
        },
        "eval": {
            "upload": eval_upload,
            "audit": eval_audit,
            **eval_summary,
        },
    }
