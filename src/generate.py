"""Generate teacher traces for a task family.

    python3 src/generate.py --teacher muse-glimmer --host ai19 \
        --prompts prompts/office_edit.jsonl --out data/raw/muse.office_edit.jsonl

Every record carries full provenance (teacher, repo, host, quantization) so a
later regression can be traced to the run that caused it.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from bridge_client import TeacherClient, write_traces
from config import base_url, resolve

ROOT = Path(__file__).resolve().parent.parent


def load_prompts(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"no prompt file at {path}")
    records = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise SystemExit(f"{path}:{line_no}: bad JSON — {error}")
    return records


async def run(args: argparse.Namespace) -> None:
    resolved = resolve(args.teacher, args.host)
    client = TeacherClient(
        base_url=args.base_url or base_url(resolved, args.serve_host),
        model=resolved["TEACHER_SERVED_MODEL_NAME"],
        api_key=args.api_key,
        sampling=resolved["SAMPLING"],
    )

    if not await client.health():
        raise SystemExit(
            f"no teacher answering at {client.base_url}\n"
            f"start one:  ./scripts/serve_teacher.sh {args.teacher} {args.host}"
        )

    prompts = load_prompts(Path(args.prompts))

    # A validation host exists to check trace SHAPE cheaply, not to build a
    # corpus. Cap it loudly rather than letting an int4 run quietly become
    # the training set.
    limit = args.limit
    if resolved["HOST_VALIDATE_ONLY"]:
        cap = int(resolved["HOST_MAX_PROMPTS"] or 50)
        if len(prompts) > cap:
            print(f"host '{args.host}' is validation-only: using {cap} of {len(prompts)} prompts")
            limit = cap
    if limit:
        prompts = prompts[:limit]

    print(f"{args.teacher} @ {args.host} ({resolved['HOST_QUANTIZATION']}) — {len(prompts)} prompts")

    message_sets = [
        [
            {"role": "system", "content": record.get("system", "")},
            {"role": "user", "content": record["user"]},
        ]
        for record in prompts
    ]
    completions = await client.complete_many(message_sets, concurrency=args.concurrency)

    records, failed = [], 0
    for prompt, completion in zip(prompts, completions):
        if completion is None:
            failed += 1
            continue
        records.append({
            "family": prompt.get("family", "unknown"),
            "system": prompt.get("system", ""),
            "user": prompt["user"],
            "completion": completion,
            "provenance": {
                "teacher": resolved["TEACHER_NAME"],
                "repo": resolved["TEACHER_REPO"],
                "license": resolved["TEACHER_LICENSE"],
                "host": resolved["HOST_NAME"],
                "quantization": resolved["HOST_QUANTIZATION"],
            },
        })

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = write_traces(out_path, records)
    print(f"wrote {written} traces to {out_path}" + (f" ({failed} failed)" if failed else ""))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--serve-host", default="localhost",
                        help="hostname the teacher is served on (for a remote box)")
    parser.add_argument("--base-url", default="",
                        help="override the composed URL entirely")
    parser.add_argument("--api-key", default="",
                        help="only needed behind a gateway; local vLLM needs none")
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
