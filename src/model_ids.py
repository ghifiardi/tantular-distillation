"""Exact model-id matching shared by gate and freeze checks.

Substring matching is unsafe for model families whose identifiers extend one
another, for example ``Qwen3.5-9B`` and ``Qwen3.5-9B-Base``.  Only the complete
repository id or its complete basename is accepted, case-insensitively.
"""
from __future__ import annotations

from pathlib import PurePosixPath


def aliases(model_id: str) -> set[str]:
    value = str(model_id or "").strip().lower()
    if not value:
        return set()
    return {value, PurePosixPath(value).name}


def matches(expected: str, served: object) -> bool:
    return str(served or "").strip().lower() in aliases(expected)


def any_match(expected: str, served_ids: list[object]) -> bool:
    return any(matches(expected, served) for served in served_ids)
