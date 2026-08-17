"""Attach mechanical checks to the expanded prompt set.

    ./.venv/bin/python src/author_checks.py --prompts prompts/expanded.jsonl
    ./.venv/bin/python src/author_checks.py --prompts prompts/expanded.jsonl --write

Only 8 of 26 strata carried a check (`closed_set`, all router). The other 18
contributed to no metric, so `constraints_ok 1.0` described the router alone and
`source_preserved` was computed on zero traces — while the edit prompts were
literally instructing "Pertahankan angka persis seperti sumber" with nothing
verifying it.

Checks are DERIVED, never read off the prose. Each prompt's `source_sha256`
maps through the source pack's digests.json to a (kind, split), and that split
selects the same SCENARIOS entry `author_sources.py` used to compose the text.
So a `preserve` list contains the figures that provably went in, rather than
figures a human believed they saw. Transcribing them by eye would produce a
check that fails when the transcription is wrong and passes when it is wrong in
the same way as the source.

`system` and `user` are never touched. `prompt_sha256` is the digest of the
RENDERED prompt (system + user through the chat template) — checks are scoring
metadata and are not rendered. Existing traces therefore stay valid evidence:
adding checks re-scores the passes already on disk instead of requiring a
regeneration. This asserts that byte-identity rather than assuming it.

Two instruments, deliberately different in severity:

  constraints_ok   pass/fail, and only where the instruction STATES the rule —
                   "TEPAT dua kalimat", "maksimal 5 bullet", the three required
                   section labels, the terms that must disappear.
  source_preserved a ratio, for figures that ought to survive. Fractional on
                   purpose: a summary that legitimately drops one figure should
                   register as 0.8, not as a failure. Where a task does not
                   entail preservation at all, nothing is claimed.

Strata whose instruction entails no objective rule keep an empty check rather
than a padded one. An invented constraint that the task never imposed produces
failures that mean nothing.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import author_sources
import author_sources_v3
import splits as splits_module

SOURCE_PACK = Path.home() / "tantular-source-pack-v2"

# Informal terms the ubah_istilah source plants and the instruction names.
INFORMAL = ["setting", "hacker", "backup", "user", "di-disable"]
# Misspellings the koreksi source plants; the task is precisely their removal.
MISSPELLED = ["peninjuan", "kemis", "hasil nya", "di kirim", "di lengkapi",
              "nomer", "Kordinator", "di selesaikan"]


def figures(*texts: str) -> list[str]:
    """Rupiah amounts and bare numbers, in first-appearance order.

    Rupiah is matched before bare digits so "Rp 612.000.000" is preserved as one
    token; splitting it would let an answer keep "612" while inventing a
    different magnitude and still score as preserved.
    """
    found, seen = [], set()
    for text in texts:
        for match in re.finditer(r"Rp\s?[\d.,]+\d|\b\d+(?:[.,]\d+)*\b", str(text)):
            token = match.group(0).strip()
            if token not in seen:
                seen.add(token)
                found.append(token)
    return found


def checks_for(kind: str, scenario: dict, task_variant: str = "",
               router_intents: tuple[str, ...] = ()) -> dict:
    """The checks a stratum's own instruction entails. Nothing more."""
    s = scenario
    finding_figures = figures(*s["findings"])

    # Router: the answer must be one of the known intents. v2 carried this as a
    # hand-authored closed_set that this tool preserved rather than derived,
    # which silently dropped it for any prompt set built without one. Derived
    # now from the manifest's own router kinds, so a new prompt set gets it
    # automatically and cannot quietly lose router_correct coverage.
    if kind.startswith("router:"):
        if not router_intents:
            raise ValueError("router intents not supplied — refusing to write an "
                             "empty closed_set that would score every answer correct")
        return {"closed_set": list(router_intents)}

    # Prompt set v2 asks for the table back, so cell values become checkable.
    # Keyed on the task variant, not the kind: both variants are
    # document:spreadsheet-text over the same source, and only the instruction
    # tells them apart. `table_rows` requires a row's cells to appear together
    # on one line, which a flat preserve list cannot express — see calibrate.
    if task_variant == "table-reproduction":
        return {"table_rows": [[row[0], str(row[1]), str(row[2]), str(row[3]),
                                row[4], row[5]] for row in s["items"]]}

    # --- document ---------------------------------------------------------
    if kind == "document:email":
        # "Ringkas isi pesan ini dalam satu paragraf tanpa menambah fakta."
        return {"max_paragraphs": 1, "preserve": figures(*s["findings"][:3])}
    if kind == "document:memo":
        # "Apa saja ketentuan ... dan sejak kapan berlaku?" The memo states a
        # deadline, not a commencement date, so an answer that says the memo
        # does not give one is defensible. Hence a ratio, not a pass/fail rule.
        return {"preserve": [s["deadline"]]}
    if kind == "document:report":
        return {"preserve": finding_figures + [s["deadline"]]}
    if kind == "document:slide-text":
        # "Ubah isi berikut menjadi maksimal 5 bullet ringkas tanpa menambah fakta."
        return {"max_bullets": 5, "preserve": finding_figures}
    if kind == "document:spreadsheet-text":
        # "Sebutkan status dan pemilik yang tercantum, apa adanya."
        #
        # Row labels, statuses and the owner — and deliberately NOT the 16 cell
        # values or the 75,0 threshold. Requiring those first scored this
        # stratum at 0.264, which read as catastrophic table infidelity; the
        # completions were in fact correct and complete answers that omitted
        # numbers nobody asked for. The instruction requests status and owner,
        # so that is the whole of what preservation can mean here.
        #
        # Table-value fidelity is therefore still UNMEASURED. Measuring it needs
        # a prompt that asks for the table back, which is a new prompt and hence
        # a new generation — not a check that can be bolted onto this one.
        names = [row[0] for row in s["items"]]
        statuses = sorted({row[5] for row in s["items"]})
        return {"must_contain": [s["owner"]], "preserve": names + statuses}

    # --- edit -------------------------------------------------------------
    if kind == "edit:koreksi":
        # "Perbaiki ejaan dan tata bahasa. Pertahankan seluruh angka, tanggal,
        #  dan nama bagian persis seperti sumber."
        return {"must_not_contain": MISSPELLED,
                "preserve": [str(s["count"]), str(s["count2"]), s["deadline"], s["unit"]]}
    if kind == "edit:perjelas":
        return {"preserve": [s["date"]]}
    if kind == "edit:elaborasi":
        # "Jangan menambah tanggal atau angka baru" is a prohibition on unseen
        # values; no primitive can express "no number outside this set", so only
        # the survival of the two source values is asserted.
        return {"preserve": [s["date"], str(s["days"])]}
    if kind == "edit:ringkas_bagian":
        # "Padatkan menjadi TEPAT dua kalimat tanpa menghilangkan satu pun angka."
        return {"max_sentences": 2, "preserve": finding_figures}
    if kind == "edit:ubah_istilah":
        # Source plants "enkripsi {count}6 bit" and "tiap {days} jam".
        return {"must_not_contain": INFORMAL,
                "preserve": [f"{s['count']}6", str(s["days"])]}
    if kind == "edit:restrukturisasi":
        # "Susun ulang menjadi tiga bagian berlabel Capaian, Temuan, dan Rekomendasi."
        return {"must_contain": ["Capaian", "Temuan", "Rekomendasi"],
                "preserve": finding_figures}

    # --- prose ------------------------------------------------------------
    if kind == "prose:ringkas":
        return {"max_paragraphs": 1, "preserve": figures(*s["findings"][:4])}
    if kind == "prose:terjemah":
        # "Terjemahkan ... pertahankan angka." Figures must survive translation;
        # `indonesian` is already skipped for this stratum by score_trace.
        return {"preserve": figures(*s["findings"][:4])}
    if kind in ("prose:tanyaDokumen", "prose:umum", "prose:cekAman",
                "prose:draftTeks", "prose:ubahNada"):
        # Open-ended: explain, draft, re-register, assess. None of these entail
        # a countable structure or a figure that must survive. Left empty on
        # purpose — see the module docstring.
        return {}
    return {}


def unsatisfiable(checks: dict, source: str) -> list[str]:
    """Checks that cannot mean what they claim, given the source text.

    A derived check is only as good as the derivation. Two ways it can be
    quietly wrong, both of which produce a number that looks like a measurement:

      preserve token absent from the source — nothing could ever preserve it,
      so the stratum's source_preserved is capped below 1.0 by the scorer's own
      bug rather than by the model.

      must_not_contain term absent from the source — there was nothing to
      remove, so the check passes for every output including a verbatim copy,
      and reports a constraint as satisfied that was never tested.
    """
    problems = []
    for token in checks.get("preserve", []):
        if str(token).lower() not in source.lower():
            problems.append(f"preserve {token!r} does not occur in the source")
    for row in checks.get("table_rows", []):
        for cell in row:
            if str(cell).lower() not in source.lower():
                problems.append(f"table_rows cell {cell!r} does not occur in the source")
    for term in checks.get("must_not_contain", []):
        if term.lower() not in source.lower():
            problems.append(f"must_not_contain {term!r} is already absent — vacuous")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, default=Path("prompts/expanded.jsonl"))
    parser.add_argument("--source-pack", type=Path, default=SOURCE_PACK)
    parser.add_argument("--write", action="store_true",
                        help="rewrite the prompt file; otherwise report only")
    args = parser.parse_args()

    digests = json.loads((args.source_pack.expanduser() / "digests.json")
                         .read_text(encoding="utf-8"))
    by_digest = {digest: key for key, digest in digests.items()}

    rows = [json.loads(l) for l in
            args.prompts.read_text(encoding="utf-8").splitlines() if l.strip()]
    split_manifest = splits_module.load()
    # The router intents, taken from the family enumeration rather than
    # hardcoded, so the closed_set follows the taxonomy instead of a copy of it.
    router_intents = tuple(sorted({k.split(":", 1)[1]
                                   for k in split_manifest["kinds"].values()
                                   if k.startswith("router:")}))

    updated, unmapped, per_kind, problems = [], [], {}, []
    for row in rows:
        key = by_digest.get(row["source_sha256"])
        if not key:
            unmapped.append(row["family"])
            updated.append(row)
            continue
        # Two pack layouts. v2 keyed artifacts by "<kind>|<split>" and shared
        # one scenario across a split; v3 keys them by family and gives each its
        # own. Detected from the key shape rather than a flag, so the wrong
        # scenario can never be silently paired with a prompt.
        if "|" in key:
            kind, split = key.split("|")
            scenario = author_sources.SCENARIOS[split]
        else:
            if key != row["family"]:
                sys.exit(f"{row['family']} carries the source digest of {key} — "
                         "refusing to derive checks from another family's document")
            kind = row["family"].split("::")[0]
            scenario = author_sources_v3.scenario_for_family(key, split_manifest)
        if kind != row["family"].split("::")[0]:
            sys.exit(f"{row['family']} maps to source of kind {kind} — refusing to guess")

        # Router prompts carry a hand-authored closed_set that this tool does
        # not derive, so theirs is kept. Everything else is re-derived on every
        # run, making the tool authoritative and idempotent: a correction to
        # checks_for() must actually take effect on a file that already has
        # checks. Falling back to "keep whatever is there" froze the first
        # write and silently discarded a fix to the spreadsheet stratum.
        if kind.startswith("router:") and row.get("checks"):
            new = row["checks"]
        else:
            new = checks_for(kind, scenario, row.get("task_variant", ""),
                             router_intents)
        problems.extend(f"{row['family']}: {p}" for p in unsatisfiable(new, row["user"]))
        row = {**row, "checks": new}
        per_kind.setdefault(kind, set()).update(new)
        updated.append(row)

    if unmapped:
        print(f"WARNING {len(unmapped)} prompts did not map to the source pack: "
              f"{unmapped[:3]}")

    print(f"{'stratum':<30}{'checks'}")
    for kind in sorted(per_kind):
        got = sorted(per_kind[kind])
        print(f"  {kind:<28}{', '.join(got) if got else '(none — no objective rule)'}")
    covered = sum(1 for k, v in per_kind.items() if v)
    print(f"\n{covered}/{len(per_kind)} strata now carry at least one check")

    if problems:
        print(f"\n{len(problems)} UNSATISFIABLE CHECKS:")
        for problem in problems[:20]:
            print(f"  {problem}")
        sys.exit("refusing to write checks that cannot measure what they claim")
    print("all preserve tokens occur in their source; "
          "all must_not_contain terms are present to be removed")

    if not args.write:
        print("\ndry run — pass --write to apply")
        return

    # The prompt text must not move: prompt_sha256 is taken over the rendered
    # system+user, and a change there would invalidate every trace on disk.
    before = {(r["family"], r.get("system", ""), r["user"]) for r in rows}
    after = {(r["family"], r.get("system", ""), r["user"]) for r in updated}
    if before != after:
        sys.exit("prompt text changed — refusing to write; existing traces would be void")

    args.prompts.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in updated) + "\n",
        encoding="utf-8")
    # Scoped claim: this tool did not move system/user, so traces generated
    # against THIS prompt file stay valid. It says nothing about traces
    # generated against a different prompt file — v2 revised the spreadsheet
    # instruction, and no amount of care here makes a v1 trace an observation
    # of a v2 prompt.
    print(f"\nwrote {args.prompts} — this tool changed only `checks`; "
          f"system/user are byte-identical,\nso traces generated against THIS "
          f"prompt file stay valid and need only re-scoring")


if __name__ == "__main__":
    main()
