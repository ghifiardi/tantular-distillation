"""Build prompt set v3: v2's instructions over per-family source documents.

    ./.venv/bin/python src/make_prompt_set_v3.py --write

v2's 260 prompts carried only 78 distinct documents, so 260 families asked 78
distinct questions. v3 keeps every instruction byte-identical and swaps in each
family's own artifact from source pack v3, giving 260 distinct prompts.

Instructions are LIFTED FROM v2, not re-authored. The tasks must stay the same
or nothing measured against v2 informs anything about v3 — the point of this
change is more material, not different work. Each kind's instruction is taken
from a v2 prompt of that kind and asserted identical across all ten of them
before use, so a stratum whose prompts disagreed would fail loudly rather than
have one of them silently chosen.

The split-boundary rule still holds by construction: source pack v3 assigns each
family its own document within its own split, so no document reaches a prompt
outside its split.

This is a NEW instrument. Its results share no baseline with v1 or v2 — new
source pack AND new prompt set — and need their own measurement before any
reproducibility claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import splits as splits_module

SOURCE_PACK = Path.home() / "tantular-source-pack-v3"


def body_of(source: str) -> str:
    """A source artifact's content, minus its own header line.

    Every artifact opens with a label ("Teks sumber sintetis:", "Permintaan
    pengguna sintetis:") followed by a blank line.
    """
    _, sep, body = source.partition("\n\n")
    return (body if sep else source).strip()


def instruction_of(user: str, source: str) -> str:
    """The task text a prompt wraps around its source document.

    Two prompt shapes exist in this corpus and conflating them would corrupt a
    stratum:

      non-router  instruction, blank line, then the artifact VERBATIM
      router      the request itself, with the artifact's header stripped and
                  no instruction at all — the user's words are the whole prompt

    Detected by testing against the artifact rather than assumed from the kind
    name, so a stratum that changes shape later fails here instead of silently
    producing a prompt with a duplicated or missing document.
    """
    if source.strip() in user:
        head, sep, _ = user.partition(source.strip())
        return head.rstrip("\n") if sep else ""
    if body_of(source) in user:
        return ""
    raise ValueError("prompt does not contain its source artifact in either shape")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-set", type=Path, default=Path("prompts/expanded.v2.jsonl"))
    parser.add_argument("--pack", type=Path, default=SOURCE_PACK)
    parser.add_argument("--out", type=Path, default=Path("prompts/expanded.v3.jsonl"))
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    pack = args.pack.expanduser()
    digests = json.loads((pack / "digests.json").read_text(encoding="utf-8"))
    manifest = splits_module.load()
    assignments = manifest["assignments"]

    v2 = [json.loads(l) for l in
          args.from_set.read_text(encoding="utf-8").splitlines() if l.strip()]

    # Resolve each v2 prompt against ITS OWN v2 artifact to recover the
    # instruction, which requires the v2 pack the prompts were built from.
    v2_pack = Path.home() / "tantular-source-pack-v2"
    v2_digests = json.loads((v2_pack / "digests.json").read_text(encoding="utf-8"))
    v2_by_digest = {d: k for k, d in v2_digests.items()}

    by_kind = defaultdict(set)
    systems = defaultdict(set)
    variants = {}
    for row in v2:
        kind = row["family"].split("::")[0]
        key = v2_by_digest.get(row["source_sha256"])
        if not key:
            sys.exit(f"{row['family']}: source digest not found in the v2 pack")
        v2_kind, v2_split = key.split("|")
        source = (v2_pack / v2_kind / f"{v2_split}.txt").read_text(encoding="utf-8")
        by_kind[kind].add(instruction_of(row["user"], source))
        systems[kind].add(row.get("system", ""))
        if row.get("task_variant"):
            variants[kind] = row["task_variant"]

    inconsistent = {k: len(v) for k, v in by_kind.items() if len(v) != 1}
    if inconsistent:
        sys.exit(f"strata whose prompts disagree on their instruction: {inconsistent}")
    inconsistent_sys = {k: len(v) for k, v in systems.items() if len(v) != 1}
    if inconsistent_sys:
        sys.exit(f"strata whose prompts disagree on their system text: {inconsistent_sys}")
    print(f"lifted {len(by_kind)} instructions from {args.from_set}, "
          f"each consistent across its 10 prompts")

    rows = []
    for family in sorted(assignments):
        kind = family.split("::")[0]
        source = (pack / family.replace("::", "__") / "source.txt").read_text(encoding="utf-8")
        instruction = next(iter(by_kind[kind]))
        # Reassemble in the same shape the stratum used in v2: an empty
        # instruction means a router prompt, which carries the request body
        # alone and must not gain a wrapper it never had.
        user = (f"{instruction}\n\n{source.strip()}\n" if instruction
                else f"{body_of(source)}")
        row = {
            "family": family,
            "source_class": "synthetic",
            "corpus_role": "synthetic_candidate",
            "source_sha256": digests[family],
            "system": next(iter(systems[kind])),
            "user": user,
            "checks": {},
        }
        if kind in variants:
            row["task_variant"] = variants[kind]
        # Router prompts carry the intent they should elicit. Without it
        # router_correct is None for every router trace and the stratum silently
        # stops contributing to the metric it exists to measure — which is what
        # happened on the first v3 build.
        if kind.startswith("router:"):
            row["expected"] = kind.split(":", 1)[1]
        rows.append(row)

    # The digest recorded must be the digest of the file actually embedded.
    for row in rows:
        family = row["family"]
        raw = (pack / family.replace("::", "__") / "source.txt").read_bytes()
        if hashlib.sha256(raw).hexdigest() != row["source_sha256"]:
            sys.exit(f"{family}: source_sha256 does not match the embedded document")

    distinct_prompts = len({(r.get("system", ""), r["user"]) for r in rows})
    distinct_sources = len({r["source_sha256"] for r in rows})
    print(f"{len(rows)} prompts | {distinct_prompts} distinct | "
          f"{distinct_sources} distinct source digests")
    if distinct_prompts != len(rows) or distinct_sources != len(rows):
        sys.exit("prompt set is not one-per-family — refusing to write")

    if not args.write:
        print("\ndry run — pass --write to apply")
        return
    args.out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                        encoding="utf-8")
    print(f"\nwrote {args.out}")
    print("NEW INSTRUMENT: new source pack and new prompt set. No v1 or v2 result "
          "carries over;\nreproducibility must be measured again before it is claimed.")


if __name__ == "__main__":
    main()
