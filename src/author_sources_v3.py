"""Author one distinct synthetic source artifact PER FAMILY.

    ./.venv/bin/python src/author_sources_v3.py --out ~/tantular-source-pack-v3

Source pack v2 authored one artifact per (kind, split) — 78 documents shared by
260 families, mean 3.33 and up to 8 families per document. Every family
"resolved" to an artifact, so the inventory reported 260/260 ready, but a
training set inherited each document 3.33x behind an identical prompt and eval
was far narrower than its family count suggested.

v3 authors 260: one per family, all digests distinct.

WHAT MUST NOT COLLAPSE. The split worlds stay substantively different, not
paraphrases of each other. Splits differ by DOMAIN — procurement, access
control, staff training — with their own topics, units, table columns, item
vocabulary and finding shapes. Within a split, families differ by ENTITY AND
QUANTITY: a different organisation unit, owner, dates, counts, amounts and a
different four-row table. So a train document and an eval document are about
different worlds, while two train documents are different cases in one world —
which is the distinction that makes eval mean something.

DETERMINISM. Everything derives from the family's index within its split, in a
fixed order. No randomness, no clock. Re-running produces byte-identical output,
so a digest recorded today still identifies the same artifact next month.

Composition is delegated to author_sources' builders, which are already tested
and produce the document shapes the prompts and checks expect. This module
replaces scenario SELECTION, not document composition — so `preserve` lists
stay derivable from structured fields rather than scraped from prose.

All content is fabricated. No real person, organisation or figure appears.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import author_sources
import splits as splits_module

# --- three domains, one per split; the substantive axis ------------------------
# Pools are sized so that topic x unit x owner exceeds the families each split
# needs (train 172, eval 54, challenge 34), letting every family take a distinct
# combination before quantities are varied at all.
WORLDS = {
    "train": {
        "topics": ["Pengadaan Perangkat Kerja", "Pengadaan Alat Tulis Kantor",
                   "Seleksi Vendor Logistik", "Pengadaan Perangkat Lunak",
                   "Pembaruan Kontrak Pemeliharaan", "Pengadaan Meubel Kantor",
                   "Seleksi Penyedia Jasa Kebersihan", "Pengadaan Kendaraan Operasional"],
        "units": ["Bagian Pengadaan", "Unit Layanan Pengadaan", "Bagian Umum",
                  "Tim Evaluasi Vendor", "Bagian Perencanaan", "Unit Anggaran"],
        "owners": ["Rina Hartanti", "Doni Saputra", "Lestari Wibowo", "Agus Mulyana",
                   "Nadia Kusuma", "Hendra Gunawan", "Fitri Rahmawati", "Yusuf Maulana"],
        "items": ["Alfa Nusa", "Beta Karya", "Gama Teknik", "Delta Prima", "Eka Sarana",
                  "Fajar Mandiri", "Graha Utama", "Harmoni Jaya"],
        "columns": ("Teknis", "Harga", "Layanan"),
        "verdicts": ("Lolos", "Cadangan"),
        "findings": [
            "Pengajuan perangkat turun dari {count2} unit menjadi {count} unit",
            "Anggaran terpakai {amount} dari pagu {amount2}",
            "Empat vendor lolos ambang penilaian teknis",
            "Dua vendor belum melengkapi dokumen legalitas",
            "Jadwal pengiriman disepakati {days} hari kerja setelah kontrak",
        ],
    },
    "eval": {
        "topics": ["Peninjauan Hak Akses Sistem", "Audit Akun Pengguna",
                   "Peninjauan Izin Aplikasi Internal", "Tinjauan Akses Basis Data",
                   "Peninjauan Akun Mitra", "Audit Hak Akses Berkas"],
        "units": ["Unit Keamanan Informasi", "Tim Tata Kelola TI", "Bagian Kepatuhan",
                  "Unit Audit Internal", "Tim Administrasi Sistem"],
        "owners": ["Bayu Prasetyo", "Sinta Permata", "Rizal Anwar", "Maya Oktaviani",
                   "Bagas Setiawan", "Dewi Anggraeni"],
        "items": ["Sistem Keuangan", "Sistem Kepegawaian", "Portal Vendor", "Arsip Digital",
                  "Sistem Persediaan", "Portal Pelaporan", "Basis Data Pelanggan"],
        "columns": ("Akun", "Tinjau", "Nonaktif"),
        "verdicts": ("Baik", "Perlu tinjau"),
        "findings": [
            "Ditemukan {count} akun aktif milik pemegang yang telah berpindah bagian",
            "Akun tidak aktif dibiarkan lebih dari {days} hari",
            "{count2} akun pada sistem mitra belum ditinjau tahun ini",
            "Tidak ditemukan upaya akses tidak sah",
            "Biaya perbaikan sistem pemantauan {amount}",
        ],
    },
    "challenge": {
        "topics": ["Program Pelatihan Staf Operasional", "Program Sertifikasi Teknis",
                   "Pelatihan Layanan Pelanggan", "Program Pengembangan Penyelia",
                   "Pelatihan Keselamatan Kerja"],
        "units": ["Bagian Pengembangan SDM", "Unit Pelatihan", "Bagian Kepegawaian",
                  "Tim Manajemen Talenta"],
        "owners": ["Sari Anggraini", "Irfan Nugroho", "Ratna Puspita", "Adi Wicaksono",
                   "Melati Handayani"],
        "items": ["Pelatihan Arsip", "Pelatihan Keselamatan", "Pelatihan Sistem Baru",
                  "Pelatihan Layanan", "Pelatihan Kepemimpinan", "Pelatihan Anggaran"],
        "columns": ("Peserta", "Lulus", "Ulang"),
        "verdicts": ("Selesai", "Berjalan"),
        "findings": [
            "{count} staf terdaftar, {count2} staf telah menyelesaikan pelatihan",
            "Tingkat penyelesaian keseluruhan {pct} persen",
            "Satu angkatan tertunda {days} hari kerja",
            "Anggaran terpakai {amount} dari {amount2}",
            "Tiga bagian belum mengirimkan daftar peserta",
        ],
    },
}

MONTHS = ["Januari", "Februari", "Maret", "April", "Mei", "Juni",
          "Juli", "Agustus", "September", "Oktober", "November", "Desember"]


def rupiah(value: int) -> str:
    """Indonesian thousands grouping, e.g. 612000000 -> 'Rp 612.000.000'."""
    return "Rp " + f"{value:,}".replace(",", ".")


def scenario_for(index: int, split: str) -> dict:
    """The scenario for the index-th family of a split.

    Mixed-radix over the pools so consecutive families differ in topic first,
    then unit, then owner — consecutive indices never share a document shape.
    Quantities vary independently of the pools, so two families that happened to
    share a topic still differ in every figure.
    """
    w = WORLDS[split]
    topic = w["topics"][index % len(w["topics"])]
    unit = w["units"][(index // len(w["topics"])) % len(w["units"])]
    owner = w["owners"][(index // (len(w["topics"]) * len(w["units"]))) % len(w["owners"])]

    # Quantities: distinct per index, and kept in plausible ranges per field.
    count = 12 + (index * 7) % 180
    count2 = 8 + (index * 11) % 150
    days = 3 + (index * 5) % 55
    pct = 41 + (index * 13) % 58
    amount_v = 25_000_000 + (index * 3_500_000) % 900_000_000
    amount2_v = amount_v + 15_000_000 + (index * 1_250_000) % 200_000_000

    month = index % 12
    day = 1 + (index * 3) % 27
    deadline_day = 1 + (index * 3 + 13) % 27
    deadline_month = (month + (1 if deadline_day <= day else 0)) % 12
    year = 2026

    # Four table rows drawn from the world's item vocabulary, offset by index so
    # different families table different things, with per-row figures.
    names = w["items"]
    items = []
    for r in range(4):
        name = names[(index + r) % len(names)]
        a = 20 + (index * 3 + r * 17) % 200
        b = 2 + (index + r * 5) % 40
        c = 1 + (index * 2 + r * 3) % 25
        score = f"{50 + (index + r * 9) % 49},{(index + r) % 10}"
        verdict = w["verdicts"][(index + r) % len(w["verdicts"])]
        items.append((name, a, b, c, score, verdict))

    fields = {"count": count, "count2": count2, "days": days, "pct": pct,
              "amount": rupiah(amount_v), "amount2": rupiah(amount2_v)}
    findings = [t.format(**fields) for t in w["findings"]]

    return {
        "topic": topic, "unit": unit, "owner": owner,
        "date": f"{day} {MONTHS[month]} {year}",
        "deadline": f"{deadline_day} {MONTHS[deadline_month]} {year}",
        "items": items, "findings": findings, **fields,
    }


def family_index(manifest: dict) -> dict[str, int]:
    """family -> its index within its split, in sorted family order.

    The single definition of that ordering. main() and every consumer that needs
    to recover a family's scenario must agree on it, or a check would be derived
    from a different document than the one the prompt carries.
    """
    per_split: dict[str, list[str]] = {}
    for family in sorted(manifest["assignments"]):
        per_split.setdefault(manifest["assignments"][family], []).append(family)
    return {family: index
            for families in per_split.values()
            for index, family in enumerate(families)}


def scenario_for_family(family: str, manifest: dict) -> dict:
    """The scenario that produced this family's artifact."""
    return scenario_for(family_index(manifest)[family], manifest["assignments"][family])


def build(kind: str, scenario: dict) -> str:
    """Compose the document. Delegates to author_sources' tested builders."""
    if kind in author_sources.BUILDERS:
        return author_sources.BUILDERS[kind](scenario)
    if kind.startswith("router:"):
        return author_sources.router_request(scenario, kind.split(":", 1)[1])
    return author_sources.prose_source(scenario, kind.split(":", 1)[1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    manifest = splits_module.load()
    assignments = manifest["assignments"]
    kinds = manifest["kinds"]

    # Index each family within its split, in sorted family order, so the mapping
    # from family to scenario is stable and inspectable.
    indices = family_index(manifest)

    out = args.out.expanduser()
    digests: dict[str, str] = {}
    written = 0
    for family in sorted(assignments):
        scenario = scenario_for(indices[family], assignments[family])
        text = build(kinds[family], scenario)
        path = out / family.replace("::", "__") / "source.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        digests[family] = hashlib.sha256(text.encode()).hexdigest()
        written += 1

    # --- pool headroom, which only this module can compute ----------------
    # Router artifacts are one sentence parameterised by TOPIC ALONE — no unit,
    # owner or figure appears in them. So a router kind with as many families in
    # a split as that split has topics is distinct only because the index
    # spacing happened to land on every topic. It is currently exactly that
    # tight: 8 train topics against a router kind with 8 train families, margin
    # 0. It passes, and it would stop passing if instances_per_kind rose, splits
    # were reshuffled, or a topic pool were trimmed.
    #
    # Reported at generation because the pack cannot show it: once written,
    # every group reads as "n families, n distinct documents" whether the margin
    # was 0 or 40.
    from collections import Counter
    router_load = Counter((kinds[f].split(":", 1)[0], assignments[f])
                          for f in assignments if kinds[f].startswith("router:"))
    per_kind_split = Counter((kinds[f], assignments[f])
                             for f in assignments if kinds[f].startswith("router:"))
    print("\nrouter topic headroom (router documents vary by topic only):")
    worst = []
    for (kind, split), n in sorted(per_kind_split.items()):
        pool = len(WORLDS[split]["topics"])
        worst.append((pool - n, kind, split, n, pool))
    worst.sort()
    for margin, kind, split, n, pool in worst[:3]:
        flag = "  <-- NO MARGIN" if margin <= 0 else ""
        print(f"  {kind:<22} {split:<10} {n} families vs {pool} topics, "
              f"margin {margin}{flag}")
    if worst and worst[0][0] <= 0:
        print("  Distinctness here rests on index spacing, not on pool size. "
              "Widen the\n  topic pool before raising instances_per_kind or "
              "reshuffling splits.")

    (out / "digests.json").write_text(json.dumps(digests, indent=2, sort_keys=True) + "\n",
                                      encoding="utf-8")
    print(f"wrote {written} artifacts -> {out}")
    print(f"distinct digests: {len(set(digests.values()))}/{written}")
    if len(set(digests.values())) != written:
        sys.exit("digest collision — two families share a document; run the audit")


if __name__ == "__main__":
    main()
