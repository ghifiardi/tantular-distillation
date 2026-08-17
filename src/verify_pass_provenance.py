"""Verify a trace file really came from the source pack and prompt set claimed.

    ./.venv/bin/python src/verify_pass_provenance.py \
        --traces data/v3-candidate/traces.r0.jsonl \
        --prompts prompts/expanded.v3.jsonl \
        --pack ~/tantular-source-pack-v3 \
        --other-pack ~/tantular-source-pack-v2

`verify_corpus.py` checks corpus structure — split balance, known families,
provenance presence. This checks IDENTITY: that every trace was produced by the
instrument named on the tin, and that no trace from a different pack or prompt
set has been mixed in.

The distinction matters once more than one instrument exists. A v2 trace sitting
in a v3 corpus passes every structural check — right families, right splits,
intact provenance — and corrupts every number computed from it, because it
answers a different prompt about a different document.

Exits 1 on any failure. Checks:

  families        every manifest family present exactly once
  prompt identity every trace's prompt_sha256 equals the digest of the named
                  prompt set rendered through the chat template
  source identity every trace's source_sha256 belongs to the named pack, and to
                  that trace's own family
  no foreign      no trace carries a digest belonging to a different pack
  fields          closed_set / expected / checks / source_sha256 populated where
                  the stratum requires them
  digest hygiene  no source digest shared by two families or spanning splits
  role            corpus_role is what it claims to be
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import jinja2

sys.path.insert(0, str(Path(__file__).resolve().parent))
import splits as splits_module

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = ROOT / "calibration" / "parity" / "chat_template.jinja"


def render(template, system: str, user: str, strength: str) -> str:
    messages = ([{"role": "system", "content": system}] if system else []) + \
               [{"role": "user", "content": user}]
    return template.render(messages=messages, add_generation_prompt=True,
                           reasoning_strength=strength)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--other-pack", type=Path, action="append", default=[],
                        help="a pack these traces must NOT come from; repeatable")
    parser.add_argument("--expect-role", default="synthetic_candidate")
    args = parser.parse_args()

    failures, notes = [], []

    traces = [json.loads(l) for l in
              args.traces.read_text(encoding="utf-8").splitlines() if l.strip()]
    prompts = {r["family"]: r for r in
               (json.loads(l) for l in
                args.prompts.read_text(encoding="utf-8").splitlines() if l.strip())}
    pack = args.pack.expanduser()
    digests = json.loads((pack / "digests.json").read_text(encoding="utf-8"))
    manifest = splits_module.load()
    assignments = manifest["assignments"]
    template = jinja2.Template(TEMPLATE.read_text(encoding="utf-8"))

    # --- families ---------------------------------------------------------
    counts = Counter(t.get("family") for t in traces)
    duplicates = {f: n for f, n in counts.items() if n > 1}
    missing = sorted(set(assignments) - set(counts))
    unknown = sorted(set(counts) - set(assignments))
    print("=== FAMILIES ===")
    print(f"  traces {len(traces)} | distinct families {len(counts)} | "
          f"manifest families {len(assignments)}")
    if missing:
        failures.append(f"{len(missing)} manifest families have no trace: {missing[:3]}")
    if unknown:
        failures.append(f"{len(unknown)} traces name families not in the manifest")
    if duplicates:
        notes.append(f"{len(duplicates)} families appear more than once "
                     f"(expected only if this file carries replicates)")

    # --- prompt identity --------------------------------------------------
    print("\n=== PROMPT IDENTITY ===")
    mismatched, unprompted = [], []
    for trace in traces:
        prompt = prompts.get(trace.get("family"))
        if prompt is None:
            unprompted.append(trace.get("family"))
            continue
        expected = hashlib.sha256(render(
            template, prompt.get("system", ""), prompt["user"],
            trace.get("provenance", {}).get("reasoning_strength", "high"),
        ).encode()).hexdigest()
        if expected != trace.get("provenance", {}).get("prompt_sha256"):
            mismatched.append(trace["family"])
    print(f"  traces whose prompt_sha256 matches {args.prompts.name}: "
          f"{len(traces) - len(mismatched) - len(unprompted)}/{len(traces)}")
    if unprompted:
        failures.append(f"{len(unprompted)} traces have no prompt in the named set")
    if mismatched:
        failures.append(f"{len(mismatched)} traces do not match the named prompt set "
                        f"— they answer different prompts: {mismatched[:3]}")

    # --- source identity --------------------------------------------------
    print("\n=== SOURCE IDENTITY ===")
    pack_digests = set(digests.values())
    foreign, wrong_family = [], []
    for trace in traces:
        digest = trace.get("source_sha256")
        if digest not in pack_digests:
            foreign.append(trace.get("family"))
        elif digests.get(trace.get("family")) != digest:
            wrong_family.append(trace.get("family"))
    print(f"  traces whose source_sha256 is in {pack.name}: "
          f"{len(traces) - len(foreign)}/{len(traces)}")
    print(f"  traces carrying their OWN family's document: "
          f"{len(traces) - len(foreign) - len(wrong_family)}/{len(traces)}")
    if foreign:
        failures.append(f"{len(foreign)} traces carry a digest not in this pack")
    if wrong_family:
        failures.append(f"{len(wrong_family)} traces carry another family's document")

    # --- no mixing with other packs ---------------------------------------
    for other in args.other_pack:
        other = Path(other).expanduser()
        other_file = other / "digests.json"
        if not other_file.exists():
            notes.append(f"other pack {other.name} has no digests.json; skipped")
            continue
        other_digests = set(json.loads(other_file.read_text(encoding="utf-8")).values())
        bleed = [t.get("family") for t in traces if t.get("source_sha256") in other_digests]
        print(f"  traces originating from {other.name}: {len(bleed)}")
        if bleed:
            failures.append(f"{len(bleed)} traces come from {other.name} — corpora mixed")

    # --- required fields --------------------------------------------------
    print("\n=== REQUIRED FIELDS ===")
    field_gaps = defaultdict(list)
    for trace in traces:
        family = trace.get("family", "?")
        kind = family.split("::")[0]
        prompt = prompts.get(family, {})
        if not trace.get("source_sha256"):
            field_gaps["source_sha256"].append(family)
        if not trace.get("provenance"):
            field_gaps["provenance"].append(family)
        if trace.get("corpus_role") != args.expect_role:
            field_gaps[f"corpus_role != {args.expect_role}"].append(family)
        if kind.startswith("router:"):
            if not prompt.get("expected"):
                field_gaps["expected (router)"].append(family)
            if not (prompt.get("checks") or {}).get("closed_set"):
                field_gaps["closed_set (router)"].append(family)
    checked_strata = {f.split("::")[0] for f, p in prompts.items() if p.get("checks")}
    print(f"  strata carrying checks: {len(checked_strata)}/26")
    for field, families in sorted(field_gaps.items()):
        print(f"  MISSING {field}: {len(families)} traces  e.g. {families[:2]}")
        failures.append(f"{len(families)} traces missing {field}")
    if not field_gaps:
        print("  source_sha256, provenance, corpus_role present on every trace; "
              "router expected + closed_set present")

    # --- digest hygiene ---------------------------------------------------
    print("\n=== DIGEST HYGIENE ===")
    by_digest = defaultdict(set)
    for family, digest in digests.items():
        by_digest[digest].add(family)
    shared = {d: f for d, f in by_digest.items() if len(f) > 1}
    straddling = {d: {assignments[x] for x in f if x in assignments}
                  for d, f in by_digest.items()}
    straddling = {d: s for d, s in straddling.items() if len(s) > 1}
    print(f"  digests shared by >1 family : {len(shared)}")
    print(f"  digests spanning >1 split   : {len(straddling)}")
    if shared:
        failures.append(f"{len(shared)} documents are shared across families")
    if straddling:
        failures.append(f"{len(straddling)} documents straddle a split")

    print()
    for note in notes:
        print(f"NOTE: {note}")
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        sys.exit(f"\n{len(failures)} provenance check(s) failed")
    print(f"PROVENANCE OK — every trace answers a {args.prompts.name} prompt over a "
          f"{pack.name} document\nbelonging to its own family; no foreign corpus mixed in.")


if __name__ == "__main__":
    main()
