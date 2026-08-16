"""Derive prompt set v2 from v1 by giving document:spreadsheet-text a task that
can actually test table fidelity.

    ./.venv/bin/python src/make_prompt_set_v2.py            # report
    ./.venv/bin/python src/make_prompt_set_v2.py --write

v1 asks "Sebutkan status dan pemilik yang tercantum, apa adanya." Correct
answers to that name the statuses and the owner and say nothing about the 16
cell values — so no check attached to v1 can measure whether cell values
survive. Demanding them anyway scored correct completions at 0.264, which read
as table infidelity that was not there.

Table fidelity needs a prompt that asks for the table back. That is a different
task, not a stricter check, so it changes the prompt set: the 10 spreadsheet
prompts get a new prompt_sha256 and their v1 traces stop being observations of
them. The other 250 prompts are copied byte-for-byte.

CONSEQUENCE, deliberately not smoothed over: v2 results must not be merged into
the v1 floor. The two sets are not the same instrument. A reproducibility claim
over v2 needs two fresh passes over v2 — the traces on disk were generated
against v1 and answer a question v2 no longer asks.

Only this one stratum changes. Rewriting more of the set would invalidate more
traces than the gap requires, and the gap is specific.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REVISED_KIND = "document:spreadsheet-text"

# Names the columns explicitly so a partial reproduction is a visible failure
# rather than a defensible reading of a vague instruction. "Pertahankan urutan
# baris" makes row order checkable too.
TABLE_INSTRUCTION = (
    "Salin ulang tabel di bawah ini secara utuh, apa adanya. Sertakan setiap "
    "baris beserta seluruh nilai pada kolom Kol A, Kol B, Kol C, Nilai, dan "
    "Status. Jangan mengubah, membulatkan, menambah, atau menghilangkan satu "
    "angka pun, dan pertahankan urutan baris seperti pada sumber."
)


def revise(row: dict) -> dict:
    """Swap the instruction line, keep the source document byte-identical.

    The instruction is the first line; everything after the first blank line is
    the source table. Splitting on the first blank line rather than reflowing
    the whole prompt keeps the document — and therefore source_sha256 — exactly
    as authored.
    """
    head, sep, body = row["user"].partition("\n\n")
    if not sep:
        sys.exit(f"{row['family']}: no blank line separating instruction from source")
    return {**row,
            "user": f"{TABLE_INSTRUCTION}{sep}{body}",
            "task_variant": "table-reproduction",
            "checks": {}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("prompts/expanded.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("prompts/expanded.v2.jsonl"))
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    rows = [json.loads(l) for l in
            args.source.read_text(encoding="utf-8").splitlines() if l.strip()]

    out, revised = [], 0
    for row in rows:
        if row["family"].split("::")[0] == REVISED_KIND:
            out.append(revise(row))
            revised += 1
        else:
            out.append(row)

    untouched = [a for a, b in zip(rows, out) if a == b]
    print(f"{len(rows)} prompts: {revised} revised, {len(untouched)} byte-identical to v1")
    if revised != 10:
        sys.exit(f"expected 10 {REVISED_KIND} prompts, found {revised}")

    # The revised prompts must still contain their source table verbatim; a
    # instruction swap that clipped the document would silently shrink the task.
    for row in out:
        if row.get("task_variant") == "table-reproduction":
            if "Kol A" not in row["user"] or "Ambang penilaian" not in row["user"]:
                sys.exit(f"{row['family']}: source table did not survive the swap")
    print("all revised prompts still carry their full source table")

    print(f"\nsample revised instruction:\n  {TABLE_INSTRUCTION[:88]}…")

    if not args.write:
        print("\ndry run — pass --write to apply")
        return
    args.out.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in out) + "\n",
        encoding="utf-8")
    print(f"\nwrote {args.out}")
    print("v2 is a DIFFERENT instrument from v1. Do not merge v2 results into the "
          "v1 floor;\nthe 10 revised prompts have new digests and no valid traces yet.")


if __name__ == "__main__":
    main()
