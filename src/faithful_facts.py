"""Fact extraction for the faithful-editing eval.

Property 4 is "no new facts". Capitalisation alone cannot carry that check —
Indonesian capitalises sentence openings and month names, so a capitalisation
scan flags correct answers constantly. Approved 2026-08-21: compare facts
EXTRACTED from the output against facts ALLOWED by the source.

Three fact kinds, each extracted the same way from source and output:

  numbers    digits in any Indonesian format, including currency and percent.
             Compared on a NORMALISED form so "Rp12.500.000" and
             "Rp 12.500.000" are the same fact and "12.500.001" is not.
  dates      day + Indonesian month + optional year.
  entities   multi-word capitalised sequences, AFTER removing sentence-initial
             position and month names — the two things that make a naive scan
             useless here.

A fact in the output that is in neither the source nor the item's declared
allow-list is a NEW fact, which is what property 4 forbids.
"""
from __future__ import annotations

import re

MONTHS = ["januari", "februari", "maret", "april", "mei", "juni", "juli",
          "agustus", "september", "oktober", "november", "desember"]

# Words that legitimately open a sentence in Indonesian office prose and would
# otherwise be read as entities. Not exhaustive by design: the entity check only
# ever flags a token ABSENT from the source, so a missed word here can only
# cause a false positive on genuinely new text, which is the case we want seen.
SENTENCE_OPENERS = {
    "berikut", "dokumen", "informasi", "data", "mohon", "silakan", "sesuai",
    "berdasarkan", "dengan", "untuk", "pada", "dalam", "tidak", "belum",
    "maaf", "catatan", "perubahan", "hasil", "ringkasan", "adapun", "namun",
    "selain", "setelah", "sebelum", "jika", "karena", "oleh", "agar", "saya",
    "kami", "ini", "itu", "terima", "kalimat", "teks", "bagian", "laporan",
    "memo", "notulen", "rapat", "anggaran", "jumlah", "total", "nilai",
}

NUMBER_RE = re.compile(r"(?:Rp\s*)?\d[\d.,]*\s*(?:%|persen|kwh|kg|km|jam|hari|"
                       r"bulan|tahun|orang|unit|lembar|buah)?", re.IGNORECASE)
DATE_RE = re.compile(r"\b(\d{1,2})\s+(" + "|".join(MONTHS) + r")\s*(\d{4})?\b",
                     re.IGNORECASE)
# Two or more capitalised words in a row, or a single ALL-CAPS token.
ENTITY_RE = re.compile(r"\b(?:[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)+|[A-Z]{2,})\b")


def normalise_number(token: str) -> str:
    """Digits and unit only: 'Rp 12.500.000' and 'Rp12.500.000' are one fact."""
    token = token.strip().lower()
    digits = re.sub(r"[^\d]", "", token)
    unit = re.sub(r"[\d\s.,]", "", token)
    return f"{digits}|{unit}"


def numbers(text: str) -> set[str]:
    out = set()
    for match in NUMBER_RE.finditer(text or ""):
        raw = match.group(0).strip()
        if re.search(r"\d", raw):
            out.add(normalise_number(raw))
    return out


def dates(text: str) -> set[str]:
    return {f"{d}|{m.lower()}|{y or ''}" for d, m, y in DATE_RE.findall(text or "")}


def entities(text: str) -> set[str]:
    """Capitalised sequences, minus the two sources of false positives."""
    out = set()
    for sentence in re.split(r"(?<=[.!?:\n])\s+", text or ""):
        stripped = sentence.strip()
        if not stripped:
            continue
        for match in ENTITY_RE.finditer(stripped):
            token = match.group(0).strip()
            lowered = token.lower()
            # Sentence-initial: a capital there carries no information.
            if stripped.startswith(token) and " " not in token:
                continue
            if lowered in SENTENCE_OPENERS:
                continue
            if any(month in lowered for month in MONTHS):
                continue
            # Bullet and list markers capitalise their first word too.
            if re.match(r"^[-*•]\s", stripped) and stripped[2:].startswith(token):
                continue
            out.add(lowered)
    return out


def extract(text: str) -> dict[str, set[str]]:
    return {"numbers": numbers(text), "dates": dates(text),
            "entities": entities(text)}


def new_facts(output: str, source: str, allowed: dict | None = None) -> dict[str, list[str]]:
    """Facts present in the output and in neither the source nor the allow-list.

    The item's `allowed_new_facts` exists because some instructions legitimately
    introduce a value — "change the deadline to 20 March" puts a date in the
    output that the source never had. Anything NOT declared that way is a
    fabrication.
    """
    allowed = allowed or {}
    src = extract(source)
    out = extract(output)

    # A declared fact licenses EVERY fact it contains, not only facts of the
    # kind it was filed under. Declaring "31 Maret 2026" under `dates` used to
    # permit the date and still flag the bare number 31 — which failed a
    # hand-authored CORRECT answer during the pilot dry run. A scorer that
    # rejects correct work is worse than no scorer: it teaches people to ignore
    # it. Measured 2026-08-21.
    permitted = {kind: set(src[kind]) for kind in ("numbers", "dates", "entities")}
    for declared_list in allowed.values():
        for declared in declared_list:
            for kind, facts in extract(str(declared)).items():
                permitted[kind] |= facts

    found: dict[str, list[str]] = {}
    for kind in ("numbers", "dates", "entities"):
        introduced = sorted(out[kind] - permitted[kind])
        if introduced:
            found[kind] = introduced
    return found
