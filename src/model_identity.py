"""Record what a served Ollama model IS — three independent pieces of evidence.

    ./.venv/bin/python src/model_identity.py ghifidanukusumo/tantular:latest \
        --out data/gates/registry-pull/identity.json

WHY NOT JUST THE DIGEST.

Ollama recomputes a model's manifest on push: pushing a tag whose digest was
ad43e5078243 and pulling it back gave 0ed8471c2c9e, because the PARAMETER lines
came back in a different order. Nothing about the model had changed — same
weights, same prompt, same parameters, identical behaviour on 60 items.

So a raw manifest digest cannot answer "is this the artifact we measured?"
across a push/pull boundary. It answers "is this byte-identical packaging?",
which is a different and less useful question. Recorded 2026-08-22.

THREE PIECES, kept separate because they fail independently:

  1 registry identity   the tag and the digest the registry currently serves.
                        Not an equivalence test — a pointer, so a later
                        investigation knows which artifact was in play.
  2 weights blob        sha256 of the GGUF the model resolves to. If this
                        changes, the weights changed, and nothing else matters.
  3 canonical profile   sha256 of SYSTEM/TEMPLATE/RENDERER/PARSER with the
                        PARAMETER lines SORTED, so ordering imposed by the
                        registry cannot masquerade as a difference.

Behavioural equivalence over the eval set remains the primary verification.
This exists so that when behaviour differs, there is enough evidence to say
what changed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def ollama(*args: str) -> str:
    proc = subprocess.run(["ollama", *args], capture_output=True, text=True)
    if proc.returncode != 0:
        sys.exit(f"ollama {' '.join(args)} failed: {proc.stderr.strip()[:200]}")
    return proc.stdout


def registry_identity(tag: str) -> dict:
    """Tag and digest as the local store currently holds them."""
    import urllib.request
    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=30) as r:
            models = json.load(r).get("models", [])
    except Exception as e:
        sys.exit(f"cannot reach the Ollama API to read tags: {e}")
    for m in models:
        if m.get("name") == tag:
            return {"tag": tag, "digest": m.get("digest"),
                    "size_bytes": m.get("size"),
                    "modified_at": m.get("modified_at"),
                    "_note": "a pointer, not an equivalence test: this digest "
                             "changes across a push/pull round-trip"}
    sys.exit(f"tag not present locally: {tag}")


def split_modelfile(tag: str) -> tuple[list[str], list[str], str | None]:
    """Returns (profile lines, PARAMETER lines, weights blob name)."""
    profile, params, blob = [], [], None
    for line in ollama("show", "--modelfile", tag).splitlines():
        if line.startswith("LICENSE"):
            break                      # multi-line block; everything after is licence
        if line.startswith("FROM "):
            candidate = line[len("FROM "):].strip()
            if "/" in candidate:
                blob = candidate.rsplit("/", 1)[-1]
            continue
        if line.startswith("PARAMETER "):
            params.append(line[len("PARAMETER "):].strip())
            continue
        if line.startswith("#"):
            continue
        profile.append(line)
    return profile, params, blob


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("tag")
    p.add_argument("--out", type=Path)
    p.add_argument("--compare-to", help="second tag; reports equivalence")
    args = p.parse_args()

    def evidence(tag: str) -> dict:
        profile, params, blob = split_modelfile(tag)
        # SORTED: ordering imposed by the registry is not a difference.
        canonical = "\n".join(profile + sorted(params))
        return {
            "registry": registry_identity(tag),
            "weights_blob": blob,
            "canonical_profile_sha256": sha(canonical),
            "parameters_sorted": sorted(params),
            "profile_only_sha256": sha("\n".join(profile)),
        }

    record = {"recorded_utc": None, "primary": evidence(args.tag)}
    e = record["primary"]
    print(f"=== {args.tag} ===")
    print(f"  registry digest    {e['registry']['digest']}")
    print(f"  weights blob       {e['weights_blob']}")
    print(f"  canonical profile  {e['canonical_profile_sha256'][:16]}")

    if args.compare_to:
        other = evidence(args.compare_to)
        record["compared_to"] = other
        same_weights = e["weights_blob"] == other["weights_blob"]
        same_profile = (e["canonical_profile_sha256"]
                        == other["canonical_profile_sha256"])
        record["equivalent"] = bool(same_weights and same_profile)
        print(f"\n=== {args.compare_to} ===")
        print(f"  registry digest    {other['registry']['digest']}")
        print(f"  weights blob       {other['weights_blob']}")
        print(f"  canonical profile  {other['canonical_profile_sha256'][:16]}")
        print("\n=== EQUIVALENCE ===")
        print(f"  same weights blob      {same_weights}")
        print(f"  same canonical profile {same_profile}")
        print(f"  same registry digest   "
              f"{e['registry']['digest'] == other['registry']['digest']}"
              "   <- expected to differ across a push/pull; not a failure")
        print(f"\n  VERDICT: {'equivalent' if record['equivalent'] else 'DIFFERENT'}")
        print("  Behavioural equivalence over the eval set remains the primary "
              "check;\n  this says WHAT changed when it does.")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
