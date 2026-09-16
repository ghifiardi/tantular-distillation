"""Normalized, domain-separated identities for corpus rows and case fields.

A leakage check compares one corpus against another by hash, and every way of
getting that wrong is a way of being confidently useless:

  NORMALIZE TOO LITTLE and a CRLF line ending makes the same text look like new
  text, so real overlap goes unseen.

  NORMALIZE TOO MUCH and materially different Office documents collide. A
  checker that over-matches blocks legitimate cases while looking rigorous,
  which is worse than none: it produces refusals nobody can act on.

  FRAME BADLY and ("ab", "c") hashes the same as ("a", "bc"), so an identity
  means whatever the concatenation happened to produce.

So: case is preserved because in an Office edit it is content; paragraph breaks
are preserved because they are structure; only whitespace RUNS collapse; and
every HMAC input carries a version-tagged domain and an explicit length, so
components cannot run together.

THE VERSION IS PART OF THE IDENTITY. Comparing identities computed under
different normalization versions is refused rather than silently rehashed --
rehashing would answer a question nobody asked, about text nobody normalized
the same way.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import re
import unicodedata
from typing import Any

NORMALIZATION_VERSION = 1
PRIVACY_SCHEME = "hmac-sha256"

# Domains, one per component. Version-tagged so a normalization change cannot
# quietly reuse an old identity space.
DOMAIN_SYSTEM = "tantular/corpus/system_payload/v1"
DOMAIN_USER = "tantular/corpus/user_payload/v1"
DOMAIN_COMPLETION = "tantular/corpus/completion_payload/v1"
DOMAIN_ROW = "tantular/corpus/canonical_row/v1"
DOMAIN_STABLE_ID = "tantular/corpus/stable_id/v1"
DOMAIN_MERKLE_LEAF = "tantular/corpus/merkle_leaf/v1"
DOMAIN_MERKLE_NODE = "tantular/corpus/merkle_node/v1"

DOMAINS = {
    "system_payload": DOMAIN_SYSTEM,
    "user_payload": DOMAIN_USER,
    "completion_payload": DOMAIN_COMPLETION,
    "canonical_row": DOMAIN_ROW,
}

# Components this module can produce, and the ones it deliberately cannot.
# `request` and `document` would need a versioned extractor that parses a
# prompt into instruction and document; guessing where one ends and the other
# begins is exactly the heuristic this milestone refuses to ship.
UNVERIFIABLE_COMPONENTS = {
    "request": "no_versioned_extractor",
    "document": "no_versioned_extractor",
    "expected_target": "source_has_no_structured_field",
    "expected_outcome": "no_versioned_extractor",
    "full_office_case": "required_components_unavailable",
}

MINIMUM_KEY_BYTES = 32          # 256 bits


class IdentityError(ValueError):
    """A fail-closed identity or normalization error."""


# --- normalization ----------------------------------------------------------

_SPACES = re.compile(r"[ \t]+")
_BLANKS = re.compile(r"\n{3,}")


def normalize_text(value: Any) -> str:
    """Normalization v1. See the module docstring for what is NOT done."""
    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFC", text)
    # CRLF and bare CR are transport artifacts, not content.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse runs of horizontal whitespace only: a line break survives.
    text = _SPACES.sub(" ", text)
    # Trailing spaces before a newline are invisible and not meaningful.
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    # Three or more blank lines are a formatting accident; one blank line is a
    # paragraph break and is kept.
    text = _BLANKS.sub("\n\n", text)
    return text.strip()


def canonical_json(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def normalize_row(row: dict[str, Any]) -> bytes:
    """The whole row, with every string normalized, as canonical JSON."""
    def walk(value: Any) -> Any:
        if isinstance(value, str):
            return normalize_text(value)
        if isinstance(value, dict):
            return {k: walk(v) for k, v in value.items()}
        if isinstance(value, list):
            return [walk(v) for v in value]
        return value
    return canonical_json(walk(row))


# --- framing ----------------------------------------------------------------

def framed(domain: str, payload: bytes) -> bytes:
    """domain || 0x00 || decimal-length || 0x00 || payload.

    The length is what stops ("ab", "c") and ("a", "bc") producing one value.
    Without it an identity would mean whatever the concatenation happened to
    be, and two different component pairs could share one hash.
    """
    if "\x00" in domain:
        raise IdentityError("a domain may not contain a NUL byte")
    return domain.encode("utf-8") + b"\x00" + str(len(payload)).encode() \
        + b"\x00" + payload


def identity(key: bytes, domain: str, payload: bytes) -> str:
    """HMAC-SHA256 over the framed payload.

    Keyed, not a plain digest: the corpus holds short templated requests, and a
    public sha256 of one is a membership oracle -- anyone can hash a guess and
    test it. Same reasoning as the add-in's lookup audit key.
    """
    if not isinstance(key, (bytes, bytearray)) or len(key) < MINIMUM_KEY_BYTES:
        raise IdentityError(
            f"HMAC key must be at least {MINIMUM_KEY_BYTES} bytes of real key "
            "material")
    return hmac.new(bytes(key), framed(domain, payload), hashlib.sha256).hexdigest()


def text_identity(key: bytes, component: str, value: Any) -> str:
    domain = DOMAINS.get(component)
    if domain is None:
        raise IdentityError(f"no domain registered for component {component!r}")
    return identity(key, domain, normalize_text(value).encode("utf-8"))


def row_identity(key: bytes, row: dict[str, Any]) -> str:
    return identity(key, DOMAIN_ROW, normalize_row(row))


def stable_id_identity(key: bytes, stable_id: Any) -> str:
    return identity(key, DOMAIN_STABLE_ID,
                    normalize_text(stable_id).encode("utf-8"))


# --- merkle -----------------------------------------------------------------

def merkle_root(leaves: list[str]) -> str:
    """Deterministic binary Merkle root over SORTED leaf identities.

    Sorted so the root describes the SET of rows rather than the order they
    happened to be written in -- two manifests of the same corpus must agree.
    An odd node is promoted rather than duplicated: duplicating a leaf lets a
    2n-leaf tree collide with an n-leaf one.
    """
    if not leaves:
        return hashlib.sha256(framed(DOMAIN_MERKLE_NODE, b"")).hexdigest()
    level = [hashlib.sha256(framed(DOMAIN_MERKLE_LEAF, leaf.encode())).hexdigest()
             for leaf in sorted(leaves)]
    while len(level) > 1:
        nxt = []
        for i in range(0, len(level) - 1, 2):
            joined = framed(DOMAIN_MERKLE_NODE,
                            level[i].encode() + b"\x00" + level[i + 1].encode())
            nxt.append(hashlib.sha256(joined).hexdigest())
        if len(level) % 2:
            nxt.append(level[-1])
        level = nxt
    return level[0]


# --- version and key guards -------------------------------------------------

def require_same_normalization(left: int, right: int, *, what: str) -> None:
    if left != right:
        raise IdentityError(
            f"{what}: normalization version {left} cannot be compared with "
            f"{right}. Identities are only meaningful within one version; "
            "rehashing here would answer a question nobody asked.")


def require_same_key_id(left: str | None, right: str | None, *, what: str) -> None:
    if left != right:
        raise IdentityError(
            f"{what}: hmac_key_id {left!r} cannot be compared with {right!r}. "
            "Different keys produce different identity spaces, so equality "
            "across them means nothing and inequality proves nothing.")
