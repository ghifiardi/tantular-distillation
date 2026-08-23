"""Verify an answer produced with untrusted web context against the USER DOCUMENT.

    ./.venv/bin/python src/verify_web_answer.py --document doc.txt \
        --untrusted page.txt --answer answer.txt --protect "PT Sinar Mas"

Why this exists: an untrusted label in the prompt is not a control. Seven
injection classes were tested against the shipped profile on 2026-08-23 and
three succeeded — a vendor renamed to [REDACTED], a vendor quietly changed to
PT Contoh, and document content echoed in an attacker-chosen format. See
calibration/PROMPT_INJECTION_RESULT.md.

THE DESIGN PROBLEM. Permitted facts cannot simply be "document ∪ web page": an
injected `PT Contoh` appears in the web page, so sourcing alone would bless it.
The rules are therefore about ROLE, not origin:

  preserves       facts the DOCUMENT asserts must still be there. A vendor that
                  vanishes is a violation however plausible the replacement.
  no_new_facts    numbers, dates and entities must trace to the document or the
                  fetched page. Catches inventions with no source at all.
  untrusted_echo  the answer must not open by adopting a literal string that
                  appears only in the untrusted page. This is what catches
                  exfiltration, where every document fact is preserved and no
                  new fact appears — the attack is the FORMAT, not the content.

FAILS CLOSED. A missing document, a missing answer, or an unavailable check is a
refusal, not a pass: an unverified answer must never be presented as verified.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import faithful_facts


def check_preserves(answer: str, protect: list[str], document: str) -> list[str]:
    """Facts the document asserts must survive into the answer."""
    missing = []
    for needle in protect:
        if needle not in document:
            # Protecting something absent from the source is a configuration
            # error, not a model failure. Refuse rather than silently pass.
            missing.append(f"CONFIG: {needle!r} is not in the document")
        elif needle not in answer:
            missing.append(f"{needle!r} present in the document, absent from the answer")
    return missing


def check_no_new_facts(answer: str, document: str, untrusted: str) -> list[str]:
    """Numbers, dates and entities must trace to the document or the page."""
    introduced = faithful_facts.new_facts(answer, document + "\n" + untrusted)
    return [f"{kind}: {vals}" for kind, vals in introduced.items()]


def check_untrusted_echo(answer: str, document: str, untrusted: str,
                         window: int = 60, min_len: int = 5) -> list[str]:
    """Did the answer adopt a literal from the page that the document lacks?

    Exfiltration preserves every document fact and invents nothing — it obeys a
    FORMAT the page dictated. So the signal is structural: the answer opens with
    a token the page supplied and the document never contained.
    """
    head = (answer or "")[:window]
    findings = []
    for token in re.findall(r"[A-Za-z][A-Za-z_-]{%d,}:" % (min_len - 1), untrusted or ""):
        if token in head and token not in document:
            findings.append(f"answer opens with {token!r}, a marker supplied by "
                            "the untrusted page")
    return findings


def verify(answer: str, document: str, untrusted: str,
           protect: list[str]) -> dict:
    if not (answer or "").strip():
        return {"ok": False, "reason": "no_answer",
                "findings": {"fail_closed": ["the answer is empty"]}}
    if not (document or "").strip():
        return {"ok": False, "reason": "no_document",
                "findings": {"fail_closed": [
                    "no user document to verify against; refusing to call the "
                    "answer verified"]}}

    findings = {}
    for name, result in (
        ("preserves", check_preserves(answer, protect, document)),
        ("no_new_facts", check_no_new_facts(answer, document, untrusted)),
        ("untrusted_echo", check_untrusted_echo(answer, document, untrusted)),
    ):
        if result:
            findings[name] = result
    return {"ok": not findings,
            "reason": "verified" if not findings else "failed_verification",
            "findings": findings}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--document", type=Path, required=True)
    p.add_argument("--answer", type=Path, required=True)
    p.add_argument("--untrusted", type=Path, default=None)
    p.add_argument("--protect", action="append", default=[])
    args = p.parse_args()

    for path in (args.document, args.answer):
        if not path.is_file():
            sys.exit(f"missing input: {path}  (fail closed: nothing is verified)")

    out = verify(args.answer.read_text(encoding="utf-8"),
                 args.document.read_text(encoding="utf-8"),
                 args.untrusted.read_text(encoding="utf-8") if args.untrusted
                 and args.untrusted.is_file() else "",
                 args.protect)
    print(json.dumps(out, indent=2, ensure_ascii=False))
    sys.exit(0 if out["ok"] else 1)


if __name__ == "__main__":
    main()
