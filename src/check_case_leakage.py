"""Compare a case set against a corpus manifest, component by component.

Exit codes are the interface:

    0   every REQUIRED component was checked and no overlap was found
    1   overlap found
    2   refused, or a required component could not be checked

2 is the important one. A component with no identities on one side cannot be
compared, and reporting that as `clean` would make the gate strongest-looking
exactly when it knows least. `unverifiable` is therefore never `clean`, and it
never produces exit 0 for a required component.

WHAT A CLEAN RESULT MEANS: no identical normalized component was found under
normalization v1. It does NOT mean no related material exists. Exact hashing
does not detect paraphrase, translation or a lightly-edited document, and this
tool claims none of that.

WHAT user_payload EQUALITY DOES NOT PROVE: the promoted corpora carry no
document field and no versioned extractor exists, so `request` and `document`
are unverifiable. A clean `user_payload` says the whole prompt differs; it says
nothing about a reused document inside a rephrased prompt.

NO NETWORK, NO MODEL, NO CORPUS TEXT. Identities only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import corpus_identity as ci                                   # noqa: E402
import harness_distill as hd                                   # noqa: E402
import harness_eval as he                                      # noqa: E402

EXIT_CLEAN, EXIT_OVERLAP, EXIT_REFUSED = 0, 1, 2

CLEAN, OVERLAP, UNVERIFIABLE = "clean", "overlap", "unverifiable"

# Which case field feeds which component. Only components with an explicit
# mapping can be compared; everything else is unverifiable by construction.
CASE_COMPONENT_SOURCE = {
    "system_payload": ("request", "system"),
    "user_payload": ("request", "user"),
}
# What a production held-out check must satisfy before it may return 0.
REQUIRED_COMPONENTS = ("user_payload",)

# Components whose overlap is EXPECTED and therefore not blocking.
#
# Every production case reuses one of the product's own system prompts -- the
# promoted corpora draw on four of them across 183 rows -- so a shared
# system_payload means "this case uses the product's prompt", not "this case
# leaked". Treating it as overlap would fail every legitimate case set, and a
# gate that always fires is a gate nobody keeps. It is still reported, because
# an UNEXPECTED system prompt is worth seeing.
INFORMATIONAL_COMPONENTS = ("system_payload",)


class LeakageError(ValueError):
    """A fail-closed leakage-check error."""


def _die(message: str) -> None:
    print(f"LEAKAGE CHECK REFUSED: {message}", file=sys.stderr)
    raise SystemExit(EXIT_REFUSED)


def implementation_digest() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def read_key(path: Path) -> bytes:
    import build_corpus_manifest as bcm
    return bcm.read_key(path)


def load_manifests(public_path: Path, private_path: Path
                   ) -> tuple[dict[str, Any], dict[str, Any]]:
    public = hd.load_yaml(public_path)
    if not private_path.is_file():
        raise LeakageError(
            f"no private manifest at {private_path}. The public manifest holds "
            "no identities, so there is nothing to compare; a check without it "
            "is not a clean result.")
    private = hd.load_yaml(private_path)

    ci.require_same_normalization(
        int(public.get("normalization_version", -1)),
        int(private.get("normalization_version", -2)),
        what="public vs private manifest")
    ci.require_same_key_id(public.get("hmac_key_id"), private.get("hmac_key_id"),
                           what="public vs private manifest")

    declared = str(public.get("corpus", {}).get("private_manifest_sha256") or "")
    actual = hashlib.sha256(ci.canonical_json(private)).hexdigest()
    if declared != actual:
        raise LeakageError(
            "the private manifest does not match the digest the public one "
            f"declares ({declared[:12]}... vs {actual[:12]}...). One of them "
            "was changed after generation; neither can be trusted.")
    declared_root = str(public.get("corpus", {}).get("private_merkle_root") or "")
    if declared_root != str(private.get("merkle_root") or ""):
        raise LeakageError("the Merkle roots disagree between manifests")
    if str(public.get("source", {}).get("raw_sha256") or "") != \
            str(private.get("source_raw_sha256") or ""):
        raise LeakageError("the manifests describe different source bytes")
    return public, private


def check(case_set: dict[str, Any], public: dict[str, Any],
          private: dict[str, Any], key: bytes, key_id: str,
          *, required: tuple[str, ...] = REQUIRED_COMPONENTS,
          expected_source_sha256: str | None = None) -> dict[str, Any]:
    he.validate_case_set(case_set)
    ci.require_same_normalization(int(public["normalization_version"]),
                                  ci.NORMALIZATION_VERSION,
                                  what="manifest vs this checker")
    ci.require_same_key_id(public.get("hmac_key_id"), key_id,
                           what="manifest vs supplied key")
    if expected_source_sha256 is not None and \
            expected_source_sha256 != public["source"]["raw_sha256"]:
        raise LeakageError(
            "the corpus digest changed since the previous conclusion "
            f"({expected_source_sha256[:12]}... -> "
            f"{public['source']['raw_sha256'][:12]}...). Every earlier leakage "
            "result described different bytes and is now void.")

    corpus_index: dict[str, set[str]] = {}
    for row in private.get("rows") or []:
        for component, value in (row.get("components") or {}).items():
            corpus_index.setdefault(component, set()).add(value)

    results: dict[str, Any] = {}
    for component in sorted(set(list(CASE_COMPONENT_SOURCE) +
                                list(ci.UNVERIFIABLE_COMPONENTS))):
        declared = (public.get("components") or {}).get(component) or {}
        if declared.get("status") != "available" or component not in corpus_index:
            results[component] = {
                "status": UNVERIFIABLE, "checked_cases": 0, "matches": [],
                "reason": declared.get("reason")
                or f"the manifest declares {component} "
                   f"{declared.get('status', 'absent')!r}; there is nothing to "
                   "compare, which is not the same as finding nothing"}
            continue

        where = CASE_COMPONENT_SOURCE.get(component)
        if where is None:
            results[component] = {
                "status": UNVERIFIABLE, "checked_cases": 0, "matches": [],
                "reason": "no mapping from a case field to this component"}
            continue

        matches = []
        checked = 0
        for case in case_set["cases"]:
            container = case.get(where[0]) or {}
            if where[1] not in container:
                continue                      # this case does not carry it
            checked += 1
            value = ci.text_identity(key, component, container[where[1]])
            if value in corpus_index[component]:
                # The identity, never the text. It is HMAC-keyed, so it
                # discloses nothing to a reader without the key.
                matches.append({"case_id": case["case_id"],
                                "identity": value})
        results[component] = {
            "status": OVERLAP if matches else CLEAN,
            "checked_cases": checked, "matches": matches,
            "reason": None if checked else
            "no case in this set carries the field this component reads",
        }
        if not checked:
            results[component]["status"] = UNVERIFIABLE

    overlap = [c for c, r in results.items()
               if r["status"] == OVERLAP and c not in INFORMATIONAL_COMPONENTS]
    informational = [c for c, r in results.items()
                     if r["status"] == OVERLAP and c in INFORMATIONAL_COMPONENTS]
    unmet = [c for c in required if results.get(c, {}).get("status") != CLEAN]
    return {
        "case_set": case_set["name"],
        "case_set_sha256": he.case_set_digest(case_set),
        "public_manifest_sha256": hashlib.sha256(ci.canonical_json(public)).hexdigest(),
        "private_manifest_sha256": public["corpus"]["private_manifest_sha256"],
        "source_raw_sha256": public["source"]["raw_sha256"],
        "normalization_version": ci.NORMALIZATION_VERSION,
        "hmac_key_id": key_id,
        "checker_implementation_digest": implementation_digest(),
        "components": results,
        "required_components": list(required),
        "required_unmet": unmet,
        "overlap_components": overlap,
        # Reported, not blocking: see INFORMATIONAL_COMPONENTS.
        "expected_overlap_components": informational,
        "limitations": [
            "exact identity under normalization v1 only; paraphrase, "
            "translation and lightly-edited documents are NOT detected",
            "request and document are unverifiable: no versioned extractor "
            "exists, so user_payload equality says nothing about a reused "
            "document inside a rephrased prompt",
            "system_payload overlap is expected and non-blocking: every "
            "production case reuses one of the product's own system prompts",
        ],
        "exit_code": EXIT_OVERLAP if overlap else
                     (EXIT_REFUSED if unmet else EXIT_CLEAN),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cases", type=Path)
    parser.add_argument("--public-manifest", type=Path, required=True)
    parser.add_argument("--private-manifest", type=Path, required=True)
    parser.add_argument("--hmac-key-file", type=Path, required=True)
    parser.add_argument("--hmac-key-id", required=True)
    parser.add_argument("--expect-source-sha256", default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    try:
        case_set = he.load_case_set(args.cases)
        public, private = load_manifests(args.public_manifest,
                                         args.private_manifest)
        key = read_key(args.hmac_key_file)
        report = check(case_set, public, private, key, args.hmac_key_id,
                       expected_source_sha256=args.expect_source_sha256)
    except (LeakageError, ci.IdentityError, he.HarnessEvalError) as exc:
        _die(str(exc))

    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    raise SystemExit(report["exit_code"])


if __name__ == "__main__":
    main()
