"""Build the 26-kind pipeline smoke seed set from the approved source pack.

    ./.venv/bin/python src/make_smoke_seeds.py --out prompts/smoke.jsonl

SCOPE — this produces a SYNTHETIC PIPELINE SMOKE CORPUS, not a training
corpus. It exists to exercise the full path end to end: inventory -> seeds ->
ai19 generation -> gate. Every record is marked so downstream tooling and any
later reader can tell.

Why it is not a training set: 260 families rest on ~23 distinct documents, and
four document kinds share one byte-identical generic memo. A spreadsheet
stratum backed by prose teaches nothing about spreadsheets. Source
diversification is a separate, prior step to any training run.

Each prompt embeds its source document inline. A prompt that merely refers to a
document makes the teacher ask for the missing text — observed directly, 4 of 5
traces unusable in the first seed run.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import inventory as inventory_module
import splits as splits_module

ROOT = Path(__file__).resolve().parent.parent

# Per-axis instruction and machine-checkable expectations. Kept mechanical so
# scoring measures the teacher, not the scorer's taste.
DOC_SYSTEM = ("Anda adalah Tantular Office. Jawab HANYA berdasarkan dokumen yang "
              "diberikan. Jika informasi tidak ada, katakan demikian. Jangan mengarang.")
EDIT_SYSTEM = ("Anda adalah Tantular Office. Perbaiki teks sesuai instruksi. "
               "Pertahankan SEMUA angka, nama, dan tanggal persis seperti sumber. "
               "Jangan menambah informasi baru. Jawab hanya dengan teks hasilnya.")
PROSE_SYSTEM = ("Anda adalah Tantular Office. Kerjakan permintaan berdasarkan teks "
                "yang diberikan. Pertahankan fakta dan angka.")
ROUTER_SYSTEM = ("Klasifikasikan permintaan pengguna ke salah satu intent: "
                 "TANYA_DOKUMEN, EDIT_TEKS, DRAFT_TEKS, TERJEMAH, RINGKAS, "
                 "UBAH_NADA, CEK_AMAN, UMUM. Jawab HANYA nama intent.")

DOC_INSTRUCTION = {
    "document:memo": "Apa saja ketentuan pelaksanaan yang disebutkan, dan sejak kapan berlaku?",
    "document:email": "Ringkas isi pesan ini dalam satu paragraf tanpa menambah fakta.",
    "document:report": "Sebutkan temuan utama dan tindakan yang direkomendasikan.",
    "document:slide-text": "Ubah isi berikut menjadi maksimal 5 bullet ringkas tanpa menambah fakta.",
    "document:spreadsheet-text": "Sebutkan status dan pemilik yang tercantum, apa adanya.",
}
PROSE_INSTRUCTION = {
    "prose:ringkas": "Ringkas teks berikut menjadi satu paragraf.",
    "prose:draftTeks": "Buat draft pengantar singkat berdasarkan teks berikut.",
    "prose:tanyaDokumen": "Berdasarkan teks berikut, apa tujuan utama yang disebutkan?",
    "prose:terjemah": "Terjemahkan teks berikut ke bahasa Inggris, pertahankan angka.",
    "prose:ubahNada": "Ubah teks berikut menjadi nada formal untuk pimpinan.",
    "prose:cekAman": "Periksa apakah teks berikut aman dibagikan ke pihak luar. Sebutkan temuan.",
    "prose:umum": "Jelaskan secara singkat apa yang dibahas teks berikut.",
}


def build_prompt(kind: str, source_text: str) -> dict:
    axis = kind.split(":", 1)[0]
    if axis == "router":
        intent = kind.split(":", 1)[1]
        # The source file opens with a header naming the route, then a blank
        # line, then the request. The header MUST be stripped: leaving it in
        # puts the answer inside the question, and router accuracy would
        # measure copying rather than classification.
        body = source_text.split("\n\n", 1)
        request = (body[1] if len(body) > 1 else body[0]).strip()
        if intent in request:
            raise SystemExit(
                f"{kind}: the intent label still appears in the request text — "
                "the router prompt would leak its own answer")
        return {"system": ROUTER_SYSTEM, "user": request,
                "expected": intent,
                "checks": {"closed_set": ["TANYA_DOKUMEN", "EDIT_TEKS", "DRAFT_TEKS",
                                          "TERJEMAH", "RINGKAS", "UBAH_NADA",
                                          "CEK_AMAN", "UMUM"]}}
    if axis == "document":
        return {"system": DOC_SYSTEM,
                "user": f"{DOC_INSTRUCTION.get(kind, 'Ringkas dokumen berikut.')}\n\n{source_text}",
                "checks": {}}
    if axis == "edit":
        subtype = kind.split(":", 1)[1]
        return {"system": EDIT_SYSTEM,
                "user": f"Kerjakan penyuntingan bertipe '{subtype}' pada teks berikut, "
                        f"sesuai permintaan yang tertulis di dalamnya.\n\n{source_text}",
                "checks": {}}
    return {"system": PROSE_SYSTEM,
            "user": f"{PROSE_INSTRUCTION.get(kind, 'Kerjakan permintaan pada teks berikut.')}"
                    f"\n\n{source_text}",
            "checks": {}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--role", default="pipeline_smoke",
                        choices=("pipeline_smoke", "synthetic_candidate"),
                        help="corpus_role stamped on every record. Neither is "
                             "production-ready training data.")
    parser.add_argument("--all-families", action="store_true",
                        help="emit every family assignment, not one instance per kind")
    parser.add_argument("--only", default="",
                        help="comma-separated family ids, for volatile-family replicates")
    parser.add_argument("--instance", type=int, default=0,
                        help="which family instance per kind (default ::0000)")
    args = parser.parse_args()

    rows = inventory_module.load_inventory()
    manifest = splits_module.load()

    only = {f.strip() for f in args.only.split(",") if f.strip()}
    # --all-families emits every family assignment; otherwise one instance per kind.
    keys = sorted(rows) if not args.all_families else sorted(
        f for f in manifest["assignments"] if f in rows or f.split("::")[0] in rows)
    records, skipped = [], []
    for kind in keys:
        row = rows.get(kind) or rows.get(kind.split("::")[0])
        missing = inventory_module.missing_fields(row)
        if missing:
            skipped.append((kind, missing))
            continue
        family_id = kind if "::" in kind else f"{kind}::{args.instance:04d}"
        if only and family_id not in only:
            continue
        if family_id not in manifest["assignments"]:
            skipped.append((kind, [f"{family_id} not in split manifest"]))
            continue
        source_path = Path(str(row["source"]).replace("local-synthetic:", "")).expanduser()
        source_text = source_path.read_text(encoding="utf-8").strip()
        prompt = build_prompt(family_id.split("::")[0], source_text)
        records.append({
            "family": family_id,
            "source_class": "synthetic",
            "corpus_role": args.role,
            "source_sha256": row["source_sha256"],
            **prompt,
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
        encoding="utf-8")
    print(f"wrote {len(records)} prompts -> {args.out}")
    print(f"  corpus_role: {args.role}  (NOT production-ready training data)")
    if skipped:
        print(f"  skipped {len(skipped)} kind(s):")
        for kind, why in skipped:
            print(f"    {kind}: {', '.join(why)}")


if __name__ == "__main__":
    main()
