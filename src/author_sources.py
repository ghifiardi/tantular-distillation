"""Author one distinct synthetic source artifact per (kind, split).

    ./.venv/bin/python src/author_sources.py --out ~/tantular-source-pack-v2

26 kinds x 3 splits = 78 artifacts. Each split draws on a DIFFERENT scenario
world, so a train document and an eval document of the same kind are about
different organisations, numbers and events — not paraphrases of one another.
Paraphrases would defeat the point: the model could learn the scenario in
training and recognise it in eval.

  train      procurement and vendor selection
  eval       system access control and account review
  challenge  training programmes and staff development

Content is composed from per-scenario data rather than independently written
prose, and the generator provenance says so. What matters for split integrity
is that no two artifacts share a digest and no scenario spans a split, both of
which are audited before generation.

All content is fabricated. No real person, organisation or figure appears.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

# --- three scenario worlds, one per split -------------------------------------
SCENARIOS = {
    "train": {
        "topic": "Pengadaan Perangkat Kerja",
        "unit": "Bagian Pengadaan",
        "owner": "Rina Hartanti",
        "date": "12 Maret 2026",
        "deadline": "27 Maret 2026",
        "count": 34, "count2": 41, "days": 5,
        "amount": "Rp 612.000.000", "amount2": "Rp 700.000.000",
        "pct": 68,
        "items": [
            ("Alfa Nusa", 85, 72, 90, "82,1", "Lolos"),
            ("Beta Karya", 78, 91, 74, "81,4", "Lolos"),
            ("Gama Teknik", 92, 65, 88, "82,3", "Lolos"),
            ("Delta Prima", 70, 88, 81, "78,9", "Cadangan"),
        ],
        "findings": [
            "Pengajuan perangkat turun dari 41 unit menjadi 34 unit",
            "Anggaran terpakai Rp 612.000.000 dari pagu Rp 700.000.000",
            "Empat vendor lolos ambang penilaian teknis",
            "Dua vendor belum melengkapi dokumen legalitas",
            "Jadwal pengiriman disepakati 5 hari kerja setelah kontrak",
        ],
    },
    "eval": {
        "topic": "Peninjauan Hak Akses Sistem",
        "unit": "Unit Keamanan Informasi",
        "owner": "Bayu Prasetyo",
        "date": "8 Mei 2026",
        "deadline": "22 Mei 2026",
        "count": 17, "count2": 23, "days": 60,
        "amount": "Rp 148.000.000", "amount2": "Rp 200.000.000",
        "pct": 74,
        "items": [
            ("Sistem Keuangan", 120, 17, 8, "86,0", "Perlu tinjau"),
            ("Sistem Kepegawaian", 95, 6, 2, "93,7", "Baik"),
            ("Portal Vendor", 64, 23, 11, "64,1", "Perlu tinjau"),
            ("Arsip Digital", 210, 4, 1, "98,1", "Baik"),
        ],
        "findings": [
            "Ditemukan 17 akun aktif milik pemegang yang telah berpindah bagian",
            "Akun tidak aktif dibiarkan lebih dari 60 hari",
            "23 akun pada Portal Vendor belum ditinjau tahun ini",
            "Tidak ditemukan upaya akses tidak sah",
            "Biaya perbaikan sistem pemantauan Rp 148.000.000",
        ],
    },
    "challenge": {
        "topic": "Program Pelatihan Staf Operasional",
        "unit": "Bagian Pengembangan SDM",
        "owner": "Sari Anggraini",
        "date": "3 Juli 2026",
        "deadline": "18 Juli 2026",
        "count": 128, "count2": 96, "days": 12,
        "amount": "Rp 87.500.000", "amount2": "Rp 120.000.000",
        "pct": 73,
        "items": [
            ("Pelatihan Arsip", 48, 44, 4, "91,7", "Selesai"),
            ("Pelatihan Keselamatan", 36, 28, 8, "77,8", "Berjalan"),
            ("Pelatihan Sistem Baru", 44, 24, 20, "54,5", "Tertunda"),
            ("Pelatihan Layanan", 30, 29, 1, "96,7", "Selesai"),
        ],
        "findings": [
            "128 staf terdaftar, 96 staf telah menyelesaikan pelatihan",
            "Tingkat penyelesaian keseluruhan 73 persen",
            "Pelatihan Sistem Baru tertunda 12 hari kerja",
            "Anggaran terpakai Rp 87.500.000 dari Rp 120.000.000",
            "Tiga bagian belum mengirimkan daftar peserta",
        ],
    },
}

FOOTER = "\n\n(Dokumen sintetis untuk pengujian Tantular. Bukan data nyata.)\n"


def memo(s):
    return (f"MEMO INTERNAL\n\nKepada: Seluruh Kepala Bagian\nDari: {s['unit']}\n"
            f"Tanggal: {s['date']}\nPerihal: {s['topic']}\n\n"
            f"Sehubungan dengan {s['topic'].lower()}, {s['unit']} menyampaikan bahwa "
            f"proses terkait wajib diselesaikan paling lambat {s['deadline']}.\n\n"
            f"Ketentuan:\n"
            + "\n".join(f"{i}. {f}." for i, f in enumerate(s["findings"][:3], 1))
            + f"\n\nPertanyaan disampaikan kepada {s['owner']} di {s['unit']}." + FOOTER)


def email(s):
    return (f"Dari: {s['owner'].split()[0].lower()}@contoh-perusahaan.co.id\n"
            f"Kepada: {s['unit'].lower().replace(' ', '-')}@contoh-perusahaan.co.id\n"
            f"Tanggal: {s['date']}\nPerihal: Tindak lanjut {s['topic'].lower()}\n\n"
            f"Selamat pagi,\n\nMenindaklanjuti pembahasan sebelumnya mengenai "
            f"{s['topic'].lower()}, berikut ringkasan posisi terkini:\n\n"
            + "\n".join(f"- {f}." for f in s["findings"][:3])
            + f"\n\nMohon tanggapan paling lambat {s['deadline']}.\n\nSalam,\n"
            f"{s['owner']}\n{s['unit']}" + FOOTER)


def report(s):
    return (f"LAPORAN {s['topic'].upper()}\nDisusun oleh: {s['unit']}\n"
            f"Tanggal: {s['date']}\nStatus: Final\n\n"
            f"1. Ruang Lingkup\nTinjauan mencakup {s['topic'].lower()} pada seluruh "
            f"bagian operasional.\n\n2. Temuan\n"
            + "\n".join(f"2.{i} {f}." for i, f in enumerate(s["findings"], 1))
            + f"\n\n3. Rekomendasi\n3.1 Selesaikan tindak lanjut paling lambat "
            f"{s['deadline']}.\n3.2 Laporkan perkembangan kepada {s['unit']} setiap "
            f"{s['days']} hari kerja." + FOOTER)


def spreadsheet(s):
    header = "Nama              | Kol A | Kol B | Kol C | Nilai | Status"
    sep = "-" * len(header)
    rows = "\n".join(
        f"{n:<17} | {a:>5} | {b:>5} | {c:>5} | {v:>5} | {st}" for n, a, b, c, v, st in s["items"])
    return (f"Rekapitulasi {s['topic']}\nSumber: lembar \"Rekap\" pada berkas "
            f"{s['topic'].replace(' ', '_')}_{s['date'].split()[-1]}.xlsx\n\n"
            f"{header}\n{sep}\n{rows}\n\n"
            f"Ambang penilaian: 75,0. Pemilik data: {s['owner']}, {s['unit']}." + FOOTER)


def slide(s):
    return (f"Konten slide sintetis — {s['topic']}\n\nJudul slide: Ringkasan {s['topic']}\n\n"
            + "\n".join(f"- {f}" for f in s["findings"])
            + f"\n- Batas tindak lanjut {s['deadline']}"
            + f"\n- Penanggung jawab {s['unit']}" + FOOTER)


def edit_koreksi(s):
    return (f"Teks sumber sintetis:\n\n{s['unit']} telah menyelesaikan peninjuan "
            f"{s['topic'].lower()} pada hari kemis, dan hasil nya sudah di kirim ke "
            f"seluruh bagian. Terdapat {s['count']} berkas yang belum di lengkapi, "
            f"serta {s['count2']} berkas yang salah penulisan nomer urut. Kordinator "
            f"meminta perbaikan di selesaikan paling lambat {s['deadline']}.\n\n"
            f"Permintaan penyuntingan: koreksi.\nPerbaiki ejaan dan tata bahasa. "
            f"Pertahankan seluruh angka, tanggal, dan nama bagian persis seperti sumber.\n")


def edit_perjelas(s):
    return (f"Teks sumber sintetis:\n\nTerkait dengan hal tersebut di atas, sehubungan "
            f"dengan adanya {s['topic'].lower()} yang telah disampaikan sebelumnya, maka "
            f"pihak-pihak terkait diharapkan dapat melakukan langkah sebagaimana mestinya "
            f"dalam rangka mendukung kelancaran proses dimaksud, yang mana hal ini "
            f"berlaku sejak {s['date']}.\n\nPermintaan penyuntingan: perjelas.\n"
            f"Tulis ulang agar dipahami sekali baca. Pertahankan tanggal {s['date']}.\n")


def edit_elaborasi(s):
    return (f"Teks sumber sintetis:\n\n{s['topic']} dimulai {s['date']}. Bagian yang "
            f"belum menanggapi dalam {s['days']} hari kerja menerima pengingat kedua.\n\n"
            f"Permintaan penyuntingan: elaborasi.\nKembangkan menjadi paragraf penjelas "
            f"berdasarkan apa yang sudah tersirat. Jangan menambah tanggal atau angka baru.\n")


def edit_ringkas(s):
    return (f"Teks sumber sintetis:\n\nBerdasarkan tinjauan {s['unit']} pada {s['date']}, "
            + " ".join(f"{f}." for f in s["findings"])
            + f" {s['unit']} merekomendasikan seluruh tindak lanjut diselesaikan paling "
            f"lambat {s['deadline']}.\n\nPermintaan penyuntingan: ringkas_bagian.\n"
            f"Padatkan menjadi TEPAT dua kalimat tanpa menghilangkan satu pun angka.\n")


def edit_istilah(s):
    return (f"Teks sumber sintetis:\n\nTim IT sudah setting sistem dengan enkripsi "
            f"{s['count']}6 bit dan ada firewall-nya, jadi data aman dari hacker. Mereka "
            f"juga rutin backup tiap {s['days']} jam ke server cadangan, dan user yang "
            f"resign langsung di-disable akunnya.\n\nPermintaan penyuntingan: ubah_istilah.\n"
            f"Ganti istilah tidak baku (setting, hacker, backup, user, di-disable) menjadi "
            f"istilah baku bahasa Indonesia. Pertahankan angka persis seperti sumber.\n")


def edit_restruktur(s):
    return (f"Teks sumber sintetis:\n\n" + " ".join(f"{f}." for f in reversed(s["findings"]))
            + f" Batas tindak lanjut {s['deadline']}.\n\n"
            f"Permintaan penyuntingan: restrukturisasi.\nSusun ulang menjadi tiga bagian "
            f"berlabel Capaian, Temuan, dan Rekomendasi. Jangan menambah atau menghapus "
            f"fakta apa pun.\n")


def prose_source(s, task):
    return (f"Teks sumber sintetis untuk tugas {task}:\n\n{s['topic']} dikelola oleh "
            f"{s['unit']} sejak {s['date']}. "
            + " ".join(f"{f}." for f in s["findings"][:4])
            + f" Batas tindak lanjut ditetapkan {s['deadline']}.\n")


def router_request(s, intent):
    requests = {
        "TANYA_DOKUMEN": f"Di dokumen {s['topic'].lower()} yang terbuka, berapa jumlah "
                         f"yang tercatat dan siapa bagian penanggung jawabnya?",
        "EDIT_TEKS": f"Pada paragraf yang saya blok tentang {s['topic'].lower()}, ganti "
                     f"penulisan \"Rp.\" menjadi \"Rp\" dan rapikan spasi ganda.",
        "DRAFT_TEKS": f"Tolong susunkan surat pengantar baru untuk {s['topic'].lower()} "
                      f"kepada pimpinan. Belum ada teksnya, mulai dari nol.",
        "TERJEMAH": f"Bagian ringkasan {s['topic'].lower()} ini perlu versi bahasa Inggris "
                    f"untuk mitra luar negeri. Isinya jangan diubah.",
        "RINGKAS": f"Notulen tentang {s['topic'].lower()} ini terlalu panjang untuk "
                   f"pimpinan. Padatkan jadi satu paragraf saja.",
        "UBAH_NADA": f"Pesan tentang {s['topic'].lower()} ini sudah benar isinya, tapi "
                     f"terlalu santai untuk direksi. Buat lebih resmi.",
        "CEK_AMAN": f"Sebelum lampiran {s['topic'].lower()} ini saya kirim ke vendor, "
                    f"tolong periksa apakah ada data internal yang sebaiknya tidak keluar.",
        "UMUM": f"Sebenarnya kenapa {s['topic'].lower()} perlu ditinjau berkala, dan apa "
                f"risikonya kalau tidak dilakukan?",
    }
    return (f"Permintaan pengguna sintetis:\n\n{requests[intent]}\n")


BUILDERS = {
    "document:memo": memo, "document:email": email, "document:report": report,
    "document:spreadsheet-text": spreadsheet, "document:slide-text": slide,
    "edit:koreksi": edit_koreksi, "edit:perjelas": edit_perjelas,
    "edit:elaborasi": edit_elaborasi, "edit:ringkas_bagian": edit_ringkas,
    "edit:ubah_istilah": edit_istilah, "edit:restrukturisasi": edit_restruktur,
}


def build(kind: str, split: str) -> str:
    s = SCENARIOS[split]
    if kind in BUILDERS:
        return BUILDERS[kind](s)
    if kind.startswith("router:"):
        return router_request(s, kind.split(":", 1)[1])
    return prose_source(s, kind.split(":", 1)[1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--kinds", type=Path, default=None,
                        help="JSON list of kinds; defaults to the split manifest")
    args = parser.parse_args()

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import splits as splits_module
    kinds = sorted(set(splits_module.load()["kinds"].values()))

    out = args.out.expanduser()
    written, digests = 0, {}
    for kind in kinds:
        for split in ("train", "eval", "challenge"):
            path = out / kind / f"{split}.txt"
            path.parent.mkdir(parents=True, exist_ok=True)
            text = build(kind, split)
            path.write_text(text, encoding="utf-8")
            digests[f"{kind}|{split}"] = hashlib.sha256(text.encode()).hexdigest()
            written += 1

    print(f"wrote {written} artifacts -> {out}")
    print(f"distinct digests: {len(set(digests.values()))}/{written}")
    collisions = len(digests) - len(set(digests.values()))
    if collisions:
        print(f"WARNING: {collisions} artifact(s) share content — splits would leak")
    (out / "digests.json").write_text(json.dumps(digests, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
