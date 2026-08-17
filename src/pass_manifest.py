"""Record what a generation pass produced, without committing the corpus itself.

    ./.venv/bin/python src/pass_manifest.py data/expanded-r2

`.gitignore` excludes `*.jsonl`, so trace files live on disk and in HF, never in
git. That leaves a gap: a measurement citing two passes is unverifiable later if
nothing in the repo pins WHICH bytes were compared. This writes a MANIFEST.json
next to the traces holding the digest, line count and provenance pins of each
file — small enough to commit, specific enough to detect a swapped or
regenerated pass.

The digest is over raw file bytes. Two passes are expected to differ here: that
is the point of an independent pass, not a fault.

Provenance pins are collapsed per field. A field with more than one value across
a file means the pass was not generated under one configuration, so the manifest
records the full set rather than a single value that would imply consistency
that does not hold.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

PINNED = ("split_fingerprint", "split_seed", "template_sha256", "repo",
          "quantization", "host", "reasoning_strength", "temperature", "seed")


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            sha.update(chunk)
    return sha.hexdigest()


def describe(path: Path) -> dict:
    traces = [json.loads(line) for line in
              path.read_text(encoding="utf-8").splitlines() if line.strip()]
    pins = {}
    for field in PINNED:
        seen = sorted({json.dumps(t.get("provenance", {}).get(field)) for t in traces})
        pins[field] = json.loads(seen[0]) if len(seen) == 1 else [json.loads(s) for s in seen]
    # A resumed run appends. If a chunk were replayed, the file would hold two
    # traces for one family and every mean over it would be silently weighted
    # toward the replayed slice. Counted rather than assumed.
    family_counts = Counter(t.get("family") for t in traces)
    duplicates = {f: n for f, n in family_counts.items() if n > 1}
    return {
        "sha256": digest(path),
        "bytes": path.stat().st_size,
        "traces": len(traces),
        "distinct_prompt_digests": len({t.get("provenance", {}).get("prompt_sha256")
                                        for t in traces}),
        "families": len(family_counts),
        "duplicate_families": duplicates or None,
        "all_traces_unique": not duplicates,
        "provenance": pins,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pass_dir", type=Path)
    parser.add_argument("--note", default="", help="what this pass is, in one line")
    parser.add_argument("--sessions", type=int, default=1,
                        help="how many generation sessions produced this pass")
    parser.add_argument("--resumed-at", type=int, action="append", default=[],
                        metavar="N", help="trace index where a session boundary falls")
    parser.add_argument("--resume-reason", default="",
                        help="why the run was resumed rather than restarted")
    args = parser.parse_args()

    if not args.pass_dir.is_dir():
        sys.exit(f"no such directory: {args.pass_dir}")

    files = sorted(args.pass_dir.glob("*.jsonl"))
    if not files:
        sys.exit(f"no .jsonl traces in {args.pass_dir}")

    manifest = {
        "_what": "Digests of an on-disk generation pass. The traces themselves are "
                 "gitignored (see .gitignore: *.jsonl); this pins which bytes a "
                 "measurement referred to.",
        "_note": args.note,
        "location": str(args.pass_dir),
        "files": {path.name: describe(path) for path in files},
    }

    # Session structure is part of what a pass IS, not incidental logistics. A
    # pass generated across two sessions is not the same object as one
    # generated continuously: the sessions differ in batching conditions, and
    # batching is the leading explanation for the answer-selection variance
    # these passes exist to measure. Recording it here keeps a later reader
    # from calling a two-session pass a single-session result.
    manifest["sessions"] = {
        "count": args.sessions,
        "resumed_at_trace": args.resumed_at or None,
        "resume_reason": args.resume_reason or None,
        "_continuous": args.sessions == 1,
    }
    if args.sessions > 1 and not args.resumed_at:
        sys.exit("--sessions > 1 requires --resumed-at: an undisclosed boundary "
                 "is the thing this field exists to prevent")
    errors = sorted(args.pass_dir.glob("*.errors.json"))
    if errors:
        manifest["_errors_recorded_in"] = [p.name for p in errors]

    out = args.pass_dir / "MANIFEST.json"
    out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    for name, detail in manifest["files"].items():
        print(f"  {name:<24}{detail['traces']:>5} traces  {detail['sha256'][:16]}…")


if __name__ == "__main__":
    main()
