"""Freeze the exact corpus, config and gate verdict a training run starts from.

    ./.venv/bin/python src/freeze_training_run.py \
        --corpus data/v3-candidate/traces.r0.jsonl \
        --config train/qlora_9b.yaml \
        --waiver calibration/INT4_WAIVER.md \
        --out train/RUN_MANIFEST.v1.json --write

A training run is a claim about what produced a checkpoint. Six months later the
adapter exists and the question is what it was trained on — which corpus bytes,
which hyperparameters, and under what authorisation. Reconstructing that from
memory produces a story, not a record.

THE GATE IS RUN, NOT ASKED ABOUT. This executes `verify_corpus.py --gate` and
records its real exit code and stderr. It does not accept an exit code as an
argument, because the one thing this file must never do is let a failing gate be
written down as a pass. int4 traces FAIL the gate; a waiver authorises
proceeding despite that failure and does not convert it into a pass. The
manifest therefore records `gate_exit_code: 1` alongside the waiver reference,
which is the honest shape of the decision.

Refuses to write if a waiver is required (gate failed) and none was supplied.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            sha.update(chunk)
    return sha.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--waiver", type=Path,
                        help="required when the gate fails; recorded by reference")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--frozen-at", required=True,
                        help="ISO timestamp, supplied rather than read from the clock")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    for path in (args.corpus, args.config):
        if not path.exists():
            sys.exit(f"no such file: {path}")

    traces = [json.loads(l) for l in
              args.corpus.read_text(encoding="utf-8").splitlines() if l.strip()]
    pinned = ("split_fingerprint", "split_seed", "template_sha256", "repo",
              "quantization", "host", "reasoning_strength", "temperature", "seed")
    provenance = {}
    for field in pinned:
        seen = sorted({json.dumps(t.get("provenance", {}).get(field)) for t in traces})
        provenance[field] = json.loads(seen[0]) if len(seen) == 1 else \
            [json.loads(s) for s in seen]

    # Run the gate. Its exit code is evidence, not a parameter.
    proc = subprocess.run(
        [sys.executable, str(ROOT / "src" / "verify_corpus.py"), str(args.corpus), "--gate"],
        capture_output=True, text=True)
    gate_out = (proc.stdout + proc.stderr).strip()
    violations = [l.strip() for l in gate_out.splitlines()
                  if l.strip().startswith(("260 trace", "-", "  ")) and "violation" not in l]
    failed = proc.returncode != 0

    print(f"corpus     {args.corpus}")
    print(f"gate exit  {proc.returncode}  ({'FAILED' if failed else 'passed'})")
    if failed and not args.waiver:
        sys.exit("\nThe gate FAILED and no --waiver was given. A failing gate may only be\n"
                 "proceeded past under a recorded waiver. Supply one or fix the corpus;\n"
                 "this tool will not write a manifest that omits the authorisation.")

    manifest = {
        "_what": "The exact inputs and authorisation a training run started from. "
                 "Written before training, never edited after.",
        "frozen_at": args.frozen_at,
        "corpus": {
            "path": str(args.corpus),
            "sha256": digest(args.corpus),
            "traces": len(traces),
            "families": len({t.get("family") for t in traces}),
            "provenance": provenance,
        },
        "training_config": {
            "path": str(args.config),
            "sha256": digest(args.config),
        },
        "gate": {
            "command": f"src/verify_corpus.py {args.corpus} --gate",
            "exit_code": proc.returncode,
            "verdict": "FAILED" if failed else "passed",
            "_not_a_pass": "A non-zero exit is the correct and expected state for an "
                           "int4 corpus. The waiver authorises proceeding DESPITE this "
                           "failure; it does not convert it into a pass, and no FP8 "
                           "claim may be made on the strength of it.",
            "output": gate_out.splitlines()[-6:] if gate_out else [],
        },
        "waiver": {
            "path": str(args.waiver) if args.waiver else None,
            "sha256": digest(args.waiver) if args.waiver else None,
        },
        "claims_this_run_may_NOT_support": [
            "Any statement about performance on real Office documents — the corpus is "
            "entirely source_class: synthetic.",
            "FP8 equivalence, or that the FP8 gate passed. The treatment arm has never "
            "been generated.",
            "Teacher stability under batching other than concurrency 4. See "
            "calibration/VOLATILITY_v3.md.",
        ],
    }

    if not args.write:
        print("\ndry run — pass --write to apply")
        return
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.out}")
    print(f"  corpus  {manifest['corpus']['sha256'][:16]}…  {len(traces)} traces")
    print(f"  config  {manifest['training_config']['sha256'][:16]}…")
    print(f"  gate    exit {proc.returncode}, waiver {args.waiver}")


if __name__ == "__main__":
    main()
