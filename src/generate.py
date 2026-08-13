"""Generate teacher traces for a task family.

    python3 src/generate.py --teacher muse-glimmer --host gateway \
        --prompts prompts/office_seed.jsonl --out data/raw/muse.seed.jsonl

Two things this does beyond calling the teacher:

BATCHING (--batch N). A thinking model spends its reasoning budget per
REQUEST, not per token of output. A router label costs ~300 tokens to emit
one word, so ~95% is overhead thrown away. Asking for N labels in one call
amortizes that thinking across N answers. Only safe for short, independent
outputs — never for prose, where items would bleed into each other.

VERIFICATION. A prompt may carry `expected`. If the teacher disagrees, the
trace is quarantined rather than kept or dropped: sometimes the teacher is
right and the label is wrong (verified case: "make this more formal" is
UBAH_NADA, not EDIT_TEKS). Both need a human, so neither is silently trusted.

Every record carries full provenance so a later regression can be traced to
the run that caused it.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import splits as splits_module
from bridge_client import TeacherClient, write_traces
from config import base_url, resolve

ROOT = Path(__file__).resolve().parent.parent

BATCH_INSTRUCTION = (
    "Berikut {n} permintaan independen. Jawab SETIAP permintaan pada baris "
    "terpisah dengan format `<nomor>. <jawaban>`. Jangan menambah penjelasan, "
    "komentar, atau baris lain."
)
NUMBERED_LINE = re.compile(r"^\s*(\d+)\s*[.)]\s*(.+?)\s*$")


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


def build_batches(prompts: list[dict], size: int) -> list[dict]:
    """Group prompts into batched calls.

    Only prompts sharing an identical system prompt may share a call — a
    batch has ONE system message, so mixing instructions would silently apply
    the wrong one. Indices are kept so answers map back to their prompt.
    """
    if size <= 1:
        return [{"system": p.get("system", ""), "items": [(i, p)]}
                for i, p in enumerate(prompts)]

    by_system: dict[str, list] = defaultdict(list)
    for index, prompt in enumerate(prompts):
        by_system[prompt.get("system", "")].append((index, prompt))

    batches = []
    for system, items in by_system.items():
        for start in range(0, len(items), size):
            batches.append({"system": system, "items": items[start:start + size]})
    return batches


def batch_messages(batch: dict) -> list[dict]:
    items = batch["items"]
    if len(items) == 1:
        return [
            {"role": "system", "content": batch["system"]},
            {"role": "user", "content": items[0][1]["user"]},
        ]
    numbered = "\n\n".join(f"{n}. {p['user']}" for n, (_, p) in enumerate(items, 1))
    return [
        {"role": "system", "content": batch["system"]},
        {"role": "user", "content": f"{BATCH_INSTRUCTION.format(n=len(items))}\n\n{numbered}"},
    ]


def unpack_batch(batch: dict, content: str) -> dict[int, str]:
    """Map a batched response back to prompt indices.

    A batch that does not yield exactly one answer per item is rejected
    wholesale — a partial parse would misalign answers against prompts and
    silently mislabel training data, which is worse than losing the batch.
    """
    items = batch["items"]
    if len(items) == 1:
        return {items[0][0]: content}

    answers: dict[int, str] = {}
    for line in content.splitlines():
        match = NUMBERED_LINE.match(line)
        if match:
            answers[int(match.group(1))] = match.group(2)

    if len(answers) != len(items) or set(answers) != set(range(1, len(items) + 1)):
        return {}
    return {index: answers[n] for n, (index, _) in enumerate(items, 1)}


def normalize(value: str) -> str:
    return re.sub(r"[^A-Z_]", "", str(value or "").upper())


async def run(args: argparse.Namespace) -> None:
    resolved = resolve(args.teacher, args.host)

    # A gateway host carries its own URL and names the env var holding its
    # key; a self-hosted one is addressed by port on localhost.
    host_url = resolved["HOST_BASE_URL"]
    api_key = args.api_key
    if resolved["HOST_API_KEY_ENV"] and not api_key:
        api_key = os.environ.get(resolved["HOST_API_KEY_ENV"], "")
        if not api_key:
            raise SystemExit(
                f"host '{args.host}' needs a key in ${resolved['HOST_API_KEY_ENV']}\n"
                f"  export {resolved['HOST_API_KEY_ENV']}=sk-..."
            )

    prompts = load_prompts(Path(args.prompts))

    # Splits come from the manifest, never from the prompt file. A prompt that
    # declares its own split can put the same family in train here and eval
    # there, which is precisely the leak the split-before-generate policy
    # exists to prevent. Resolving every family up front also means an unknown
    # family id fails now rather than after an hour of generation.
    manifest = splits_module.load()
    splits_module.verify(manifest)
    for prompt in prompts:
        family_id = prompt.get("family", "")
        resolved_split = splits_module.split_of(family_id, manifest)
        declared = prompt.get("split")
        if declared and declared != resolved_split:
            raise SystemExit(
                f"{family_id}: prompt file says split {declared!r} but the manifest "
                f"says {resolved_split!r}. The manifest wins — remove the field."
            )
        prompt["split"] = resolved_split

    # Data handling. Prompts carrying real Office material must not reach an
    # off-premises host without explicit approval. Unclassified defaults to
    # `internal`, so forgetting to label real corpus material fails closed
    # rather than shipping it to a rented GPU.
    egress = resolved["HOST_DATA_EGRESS"]
    if egress != "internal":
        carried = {p.get("source_class", "internal") for p in prompts}
        needs_approval = sorted(c for c in carried if c != "synthetic")
        if needs_approval and not args.egress_approval:
            reason = {
                "external": "runs on third-party hardware",
                "operator_visible": "is operated by a third party who sees and "
                                    "may retain every prompt",
            }.get(egress, f"is classified {egress}")
            raise SystemExit(
                f"host '{args.host}' {reason} (data_egress: {egress}), and "
                f"{len(prompts)} prompt(s) are classified {needs_approval}.\n"
                "Real or unclassified Office material needs explicit approval to go "
                "there. Either mark prompts \"source_class\": \"synthetic\", use an "
                "approved internal host (ai19), or pass --egress-approval <reference> "
                "once that approval exists."
            )
        if needs_approval:
            print(f"EGRESS APPROVED [{args.egress_approval}]: sending "
                  f"{needs_approval} material to off-premises host '{args.host}'")

    # Only now touch the network. Everything above is free and local, so a
    # malformed prompt file or an unknown family should surface immediately
    # rather than behind a connection error — or worse, an hour into a run.
    client = TeacherClient(
        base_url=args.base_url or host_url or base_url(resolved, args.serve_host),
        # On a gateway the served name IS the remote model id, prefix included.
        model=resolved["TEACHER_REPO"] if host_url else resolved["TEACHER_SERVED_MODEL_NAME"],
        api_key=api_key,
        sampling=resolved["SAMPLING"],
    )

    if not await client.health():
        hint = (
            "check the gateway is up and the key is valid/scoped to this model"
            if host_url
            else f"start one:  ./scripts/serve_teacher.sh {args.teacher} {args.host}"
        )
        raise SystemExit(f"no teacher answering at {client.base_url}\n{hint}")

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

    batches = build_batches(prompts, args.batch)
    saved = len(prompts) - len(batches)
    print(f"{args.teacher} @ {args.host} ({resolved['HOST_QUANTIZATION']}) — "
          f"{len(prompts)} prompts in {len(batches)} calls"
          + (f" ({saved} fewer round-trips)" if saved > 0 else ""))

    concurrency = args.concurrency or int(resolved["HOST_CONCURRENCY"])
    results = await client.complete_many(
        [batch_messages(b) for b in batches], concurrency=concurrency
    )

    # Three ways a trace is unusable, all of which look like success at the
    # HTTP layer: a thinking model that spends its whole budget reasoning
    # returns 200 with empty content; one that runs out mid-JSON returns 200
    # with a fragment; a batch can come back misaligned. Reject all here
    # rather than letting judge.py sort it out later.
    answers: dict[int, dict] = {}
    rejected = {"failed": 0, "empty": 0, "truncated": 0, "unparsed_batch": 0}
    for batch, result in zip(batches, results):
        count = len(batch["items"])
        if result is None:
            rejected["failed"] += count
            continue
        if not result["content"]:
            rejected["empty"] += count
            continue
        if result["truncated"]:
            rejected["truncated"] += count
            continue
        unpacked = unpack_batch(batch, result["content"])
        if not unpacked:
            rejected["unparsed_batch"] += count
            continue
        for index, text in unpacked.items():
            answers[index] = {"content": text, "tokens": result["completion_tokens"]}

    records, mismatches = [], []
    for index, prompt in enumerate(prompts):
        answer = answers.get(index)
        if not answer:
            continue
        record = {
            "family": prompt.get("family", "unknown"),
            "split": prompt.get("split", "train"),
            "system": prompt.get("system", ""),
            "user": prompt["user"],
            "completion": answer["content"],
            "provenance": {
                "teacher": resolved["TEACHER_NAME"],
                "repo": resolved["TEACHER_REPO"],
                "license": resolved["TEACHER_LICENSE"],
                "host": resolved["HOST_NAME"],
                "quantization": resolved["HOST_QUANTIZATION"],
                "completion_tokens": answer["tokens"],
                "batched": len(batches) < len(prompts),
                # Which partitioning produced this trace. Traces generated
                # under different seeds must never be mixed into one training
                # set, and this is what makes that detectable.
                "split_seed": manifest["seed"],
                "split_fingerprint": manifest["fingerprint"],
            },
        }
        expected = prompt.get("expected")
        if expected and normalize(expected) != normalize(answer["content"]):
            record["expected"] = expected
            mismatches.append(record)
            continue
        records.append(record)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = write_traces(out_path, records)

    dropped = ", ".join(f"{n} {k}" for k, n in rejected.items() if n)
    print(f"wrote {written}/{len(prompts)} traces to {out_path}"
          + (f" — dropped {dropped}" if dropped else ""))

    if mismatches:
        # Quarantined, not discarded: the teacher may be right and the label
        # wrong. Either way a human decides, and neither is trusted silently.
        quarantine = out_path.with_suffix(".mismatch.jsonl")
        write_traces(quarantine, mismatches)
        print(f"QUARANTINED {len(mismatches)} label mismatch(es) -> {quarantine}")
        for record in mismatches[:5]:
            print(f"  {record['family']}: expected {record['expected']!r}, "
                  f"got {record['completion'][:40]!r}")

    if written < len(prompts) / 2:
        print("WARNING: over half the prompts produced nothing usable — "
              "check max_tokens and whether prompts embed their source text")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--concurrency", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch", type=int, default=0,
                        help="items per call; only for short outputs (router labels), never prose")
    parser.add_argument("--serve-host", default="localhost",
                        help="hostname the teacher is served on (for a remote box)")
    parser.add_argument("--base-url", default="",
                        help="override the composed URL entirely")
    parser.add_argument("--egress-approval", default="",
                        help="approval reference permitting non-synthetic material "
                             "on an off-premises host")
    parser.add_argument("--api-key", default="",
                        help="only needed behind a gateway; local vLLM needs none")
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
