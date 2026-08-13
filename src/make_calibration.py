"""Build the int4-vs-FP8 calibration prompt set — all 26 kinds, 2 each.

    python3 src/make_calibration.py

Deterministic: the same file every time, so both arms receive byte-identical
prompts and any behavioural difference is attributable to the runtime rather
than to the inputs.

The source documents below are SYNTHETIC and labelled as such. That is correct
for a calibration study — we are measuring how a runtime degrades, not teaching
the model about a real business. It does NOT make them approved production
training data; inventory/sources.yaml still governs that.

Each prompt carries `checks`, the machine-verifiable constraints calibrate.py
scores. Constraints are chosen to be objective (a bullet count, a preserved
figure, a label from a closed set) rather than matters of taste.
"""
from __future__ import annotations

import json
from pathlib import Path

import splits as splits_module

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "prompts" / "calibration.jsonl"

# --- synthetic source documents -------------------------------------------
# Numbers here are deliberately distinctive so source-preservation checks can
# look for them exactly.
MEMO = """MEMO INTERNAL
Kepada: Seluruh Kepala Divisi
Perihal: Penyesuaian Prosedur Pengadaan
Tanggal: 4 Februari 2026

Terhitung sejak 1 Maret 2026, seluruh pengadaan di atas Rp 750.000.000 wajib
melalui komite pengadaan. Batas persetujuan kepala divisi diturunkan dari
Rp 1.200.000.000 menjadi Rp 750.000.000. Proses persetujuan ditargetkan
selesai dalam 5 hari kerja."""

EMAIL = """Dari: rina.hartono@perusahaan.co.id
Kepada: tim-proyek@perusahaan.co.id
Perihal: Status migrasi minggu ini

Pak, ini update-nya. Migrasi tahap 2 telat 2 minggu karena vendor belum kirim
lisensi. Target go-live geser dari 15 Mei ke 29 Mei 2026. Budget masih aman,
kepakai 68% dari total Rp 4.500.000.000. Tim sudah lembur 3 minggu berturut."""

REPORT = """LAPORAN AUDIT INTERNAL Q1 2026

Ruang lingkup audit mencakup proses pengadaan dan manajemen akses pengguna
pada periode Januari sampai Maret 2026. Tim menemukan 14 transaksi tanpa
dokumen pendukung lengkap dengan nilai total Rp 2.340.000.000. Ditemukan pula
3 akun pengguna yang masih aktif meskipun pemiliknya telah mengundurkan diri
lebih dari 90 hari sebelumnya. Tidak ditemukan indikasi fraud yang disengaja.
Tim merekomendasikan pelengkapan dokumen dalam 30 hari kerja dan penerapan
review akses pengguna setiap kuartal."""

SPREADSHEET = """Penilaian Vendor RFP (skala 0-100)

Vendor | Teknis | Harga | Dukungan | Bobot Akhir
Alpha  | 85     | 72    | 90       | 82.1
Beta   | 78     | 91    | 74       | 81.4
Gamma  | 92     | 65    | 88       | 82.3
Delta  | 70     | 88    | 81       | 78.9"""

SLIDE = """- Audit internal Q1 2026 selesai dilaksanakan
- Ditemukan 14 transaksi tanpa dokumen pendukung lengkap
- Nilai total transaksi bermasalah Rp 2,34 miliar
- Terdapat 3 akun pengguna aktif milik karyawan yang sudah resign
- Akun tersebut tidak dinonaktifkan lebih dari 90 hari
- Proses persetujuan pengadaan belum terdokumentasi dengan baik
- Tidak ditemukan indikasi fraud yang disengaja
- Rekomendasi: lengkapi dokumen dalam 30 hari kerja
- Rekomendasi: terapkan review akses pengguna tiap kuartal"""

SENSITIVE = """Terlampir arsitektur sistem kami. Server produksi ada di 10.42.7.15,
akses admin pakai akun svc_admin dengan password Adm1n#2026. Kontrak dengan
Bank Mandiri senilai Rp 45.000.000.000 akan berakhir Desember 2026 dan
kemungkinan tidak diperpanjang."""

DOCS = {
    "memo": MEMO,
    "email": EMAIL,
    "report": REPORT,
    "spreadsheet-text": SPREADSHEET,
    "slide-text": SLIDE,
}

ROUTER_SYSTEM = (
    "Klasifikasikan permintaan pengguna ke salah satu intent: TANYA_DOKUMEN, "
    "EDIT_TEKS, DRAFT_TEKS, TERJEMAH, RINGKAS, UBAH_NADA, CEK_AMAN, UMUM. "
    "Jawab HANYA nama intent."
)
EDIT_SYSTEM = (
    "Anda adalah Tantular Office. Perbaiki teks sesuai instruksi. Pertahankan "
    "SEMUA angka, nama, dan tanggal persis seperti sumber. Jangan menambah "
    "informasi baru. Jawab hanya dengan teks hasilnya."
)
DOC_SYSTEM = (
    "Anda adalah Tantular Office. Jawab HANYA berdasarkan dokumen yang "
    "diberikan. Jika informasi tidak ada, katakan demikian. Jangan mengarang."
)

# Two router probes per intent: one plain, one obliquely phrased.
ROUTER_CASES = {
    "TANYA_DOKUMEN": ["Berapa total nilai transaksi bermasalah di laporan ini?",
                      "Di dokumen tadi, siapa yang harus menyetujui pengadaan Rp 900 juta?"],
    "EDIT_TEKS":     ["Ganti semua penulisan 'Rp.' menjadi 'Rp' di paragraf terpilih.",
                      "Betulkan salah ketik pada kalimat kedua paragraf ini."],
    "DRAFT_TEKS":    ["Buatkan draft email penawaran untuk calon klien perbankan.",
                      "Tolong susunkan pengantar untuk laporan kuartal ini."],
    "TERJEMAH":      ["Terjemahkan bagian executive summary ini ke bahasa Inggris.",
                      "Tolong alihbahasakan paragraf ini ke Inggris."],
    "RINGKAS":       ["Bikin ringkasan satu paragraf dari notulen rapat ini.",
                      "Pendekkan laporan ini jadi poin-poin utama saja."],
    "UBAH_NADA":     ["Tolong perbaiki kalimat ini biar lebih formal.",
                      "Buat emailnya terdengar lebih santai untuk tim internal."],
    "CEK_AMAN":      ["Cek dulu apakah paragraf ini aman dikirim ke pihak luar.",
                      "Ada data sensitif nggak di teks ini kalau dishare ke vendor?"],
    "UMUM":          ["Apa perbedaan enkripsi simetris dan asimetris?",
                      "Kenapa review akses pengguna penting untuk audit?"],
}

# Per-kind instructions over the shared documents. `checks` are objective.
EDIT_CASES = {
    "koreksi": [
        ("report", "Perbaiki ejaan dan tata bahasa pada laporan berikut tanpa mengubah satu pun angka:",
         {"preserve": ["14", "2.340.000.000", "3", "90", "30"]}),
        ("email", "Perbaiki penulisan email berikut agar baku, tanpa mengubah angka atau tanggal:",
         {"preserve": ["2", "15 Mei", "29 Mei", "68", "4.500.000.000"]}),
    ],
    "perjelas": [
        ("memo", "Tulis ulang memo berikut agar lebih jelas, tanpa mengubah angka atau tanggal:",
         {"preserve": ["750.000.000", "1.200.000.000", "1 Maret 2026", "5"]}),
        ("report", "Perjelas kalimat-kalimat laporan berikut tanpa mengubah maknanya:",
         {"preserve": ["14", "2.340.000.000", "3"]}),
    ],
    "elaborasi": [
        ("memo", "Kembangkan memo berikut dengan penjelasan prosedur yang sudah tersirat di dalamnya, tanpa menambah angka atau kebijakan baru:",
         {"preserve": ["750.000.000", "1.200.000.000"]}),
        ("slide-text", "Kembangkan setiap poin berikut menjadi kalimat lengkap, tanpa menambah temuan baru:",
         {"preserve": ["14", "2,34", "3", "90"]}),
    ],
    "ringkas_bagian": [
        ("report", "Ringkas laporan berikut menjadi TEPAT dua kalimat tanpa menghilangkan angka:",
         {"preserve": ["14", "2.340.000.000", "3"], "max_sentences": 2}),
        ("memo", "Ringkas memo berikut menjadi maksimal dua kalimat:",
         {"preserve": ["750.000.000"], "max_sentences": 2}),
    ],
    "ubah_istilah": [
        ("report", "Ganti istilah 'akun pengguna' menjadi 'akun identitas' di seluruh teks, tanpa mengubah hal lain:",
         {"preserve": ["14", "2.340.000.000", "3", "90"], "must_contain": ["akun identitas"]}),
        ("email", "Ganti istilah 'go-live' menjadi 'peluncuran' tanpa mengubah tanggal:",
         {"preserve": ["15 Mei", "29 Mei", "68"], "must_contain": ["peluncuran"]}),
    ],
    "restrukturisasi": [
        ("report", "Susun ulang laporan berikut menjadi tiga bagian berlabel Temuan, Risiko, dan Rekomendasi, tanpa menambah fakta:",
         {"preserve": ["14", "2.340.000.000", "3"], "must_contain": ["Temuan", "Rekomendasi"]}),
        ("spreadsheet-text", "Susun ulang data berikut menjadi daftar berurutan dari bobot akhir tertinggi ke terendah:",
         {"preserve": ["82.3", "82.1", "81.4", "78.9"]}),
    ],
}

PROSE_CASES = {
    "cekAman":      [("sensitive", "Periksa apakah teks berikut aman dikirim ke mitra eksternal. Sebutkan temuan spesifik:",
                      {"must_contain_any": ["10.42.7.15", "svc_admin", "password"]}),
                     ("email", "Periksa apakah email berikut aman diteruskan ke vendor:", {})],
    "draftTeks":    [("report", "Berdasarkan laporan berikut, buat draft email pengantar untuk direksi:", {}),
                     ("spreadsheet-text", "Buat draft rekomendasi vendor berdasarkan tabel berikut:", {"preserve": ["82.3"]})],
    "ringkas":      [("report", "Ringkas laporan berikut menjadi satu paragraf:", {"preserve": ["14", "3"]}),
                     ("memo", "Ringkas memo berikut dalam satu paragraf:", {"preserve": ["750.000.000"]})],
    "tanyaDokumen": [("report", "Berapa jumlah akun pengguna bermasalah, dan berapa lama akun itu dibiarkan aktif?",
                      {"must_contain": ["3", "90"]}),
                     ("spreadsheet-text", "Vendor mana yang memiliki bobot akhir tertinggi, dan berapa nilainya?",
                      {"must_contain": ["Gamma", "82.3"]})],
    "terjemah":     [("memo", "Terjemahkan memo berikut ke bahasa Inggris, pertahankan seluruh angka:",
                      {"preserve": ["750.000.000", "1.200.000.000"]}),
                     ("email", "Terjemahkan email berikut ke bahasa Inggris:", {"preserve": ["68"]})],
    "ubahNada":     [("email", "Ubah email berikut menjadi nada formal untuk direksi, pertahankan tanggal dan angka:",
                      {"preserve": ["15 Mei", "29 Mei", "68"]}),
                     ("memo", "Ubah memo berikut menjadi nada yang lebih ringan untuk tim internal:",
                      {"preserve": ["750.000.000"]})],
    "umum":         [("", "Jelaskan mengapa review akses pengguna berkala penting dalam audit internal.", {}),
                     ("", "Apa risiko utama menyimpan kredensial admin di dokumen yang dibagikan?", {})],
}

DOCUMENT_CASES = {
    "memo":             [("Apa batas persetujuan kepala divisi setelah perubahan ini?", {"must_contain": ["750.000.000"]}),
                         ("Kapan aturan baru ini mulai berlaku?", {"must_contain": ["1 Maret 2026"]})],
    "email":            [("Berapa persen anggaran yang sudah terpakai?", {"must_contain": ["68"]}),
                         ("Apa penyebab keterlambatan migrasi?", {"must_contain_any": ["lisensi", "vendor"]})],
    "report":           [("Berapa nilai total transaksi tanpa dokumen pendukung?", {"must_contain": ["2.340.000.000"]}),
                         ("Apakah ditemukan indikasi fraud?", {"must_contain_any": ["tidak", "Tidak"]})],
    "spreadsheet-text": [("Vendor mana yang paling unggul pada aspek harga?", {"must_contain": ["Beta"]}),
                         ("Berapa selisih bobot akhir antara Gamma dan Delta?", {"must_contain_any": ["3.4", "3,4"]})],
    "slide-text":       [("Ringkas 9 poin berikut menjadi maksimal 5 bullet tanpa menambah angka:",
                          {"preserve": ["14", "2,34", "3"], "max_bullets": 5}),
                         ("Ubah poin-poin berikut menjadi maksimal 4 bullet untuk ringkasan eksekutif:",
                          {"preserve": ["14"], "max_bullets": 4})],
}


def build() -> list[dict]:
    manifest = splits_module.load()
    prompts: list[dict] = []

    def add(kind: str, index: int, system: str, user: str, checks: dict, expected=None):
        family_id = f"{kind}::{index:04d}"
        if family_id not in manifest["assignments"]:
            raise SystemExit(f"{family_id} not in manifest — rebuild or fix the kind name")
        record = {
            "family": family_id,
            "source_class": "synthetic",
            "system": system,
            "user": user,
            "checks": checks,
        }
        if expected:
            record["expected"] = expected
        prompts.append(record)

    for intent, cases in ROUTER_CASES.items():
        for i, user in enumerate(cases):
            add(f"router:{intent}", i, ROUTER_SYSTEM, user,
                {"closed_set": ["TANYA_DOKUMEN", "EDIT_TEKS", "DRAFT_TEKS", "TERJEMAH",
                                "RINGKAS", "UBAH_NADA", "CEK_AMAN", "UMUM"]},
                expected=intent)

    for subtype, cases in EDIT_CASES.items():
        for i, (doc, instruction, checks) in enumerate(cases):
            add(f"edit:{subtype}", i, EDIT_SYSTEM,
                f"{instruction}\n\n{DOCS[doc]}", checks)

    for pipeline, cases in PROSE_CASES.items():
        for i, (doc, instruction, checks) in enumerate(cases):
            body = SENSITIVE if doc == "sensitive" else (DOCS.get(doc, ""))
            user = f"{instruction}\n\n{body}" if body else instruction
            add(f"prose:{pipeline}", i, DOC_SYSTEM, user, checks)

    for kind, cases in DOCUMENT_CASES.items():
        for i, (question, checks) in enumerate(cases):
            add(f"document:{kind}", i, DOC_SYSTEM,
                f"{question}\n\n{DOCS[kind]}", checks)

    return prompts


def main() -> None:
    prompts = build()
    manifest = splits_module.load()
    kinds = {p["family"].split("::")[0] for p in prompts}
    all_kinds = set(manifest["kinds"].values())
    missing = sorted(all_kinds - kinds)
    if missing:
        raise SystemExit(f"calibration set misses {len(missing)} kinds: {missing}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(
        "\n".join(json.dumps(p, ensure_ascii=False) for p in prompts) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {OUT_PATH} — {len(prompts)} prompts across {len(kinds)}/{len(all_kinds)} kinds")


if __name__ == "__main__":
    main()
