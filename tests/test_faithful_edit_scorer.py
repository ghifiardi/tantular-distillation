"""The faithful-editing scorer must pass correct work and fail each defect.

    ./.venv/bin/python -m pytest tests/test_faithful_edit_scorer.py -q

Two halves, and BOTH matter:

  correct answers pass   a scorer that rejects correct work is worse than no
                         scorer, because it teaches people to ignore it. The
                         pilot dry run caught exactly that: a declared date
                         licensed the date but not the bare number inside it.
  each check fires       a check that never fails is decoration. Every property
                         gets a hand-made defect that only it should catch.

Requires node and the add-in, because property 5 uses the product's REAL
parser rather than a copy that could drift.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PY_BIN = str(ROOT / ".venv" / "bin" / "python")
ITEMS = ROOT / "prompts" / "faithful_edit_pilot.v1.jsonl"
ADDIN = (ROOT.parent / "tantular_office_addin" / "src").resolve()

needs_addin = pytest.mark.skipif(
    shutil.which("node") is None or not (ADDIN / "chat" / "editContract.js").is_file(),
    reason="node and the add-in source are required for the real contract parser")


def items() -> dict[str, dict]:
    return {json.loads(l)["id"]: json.loads(l)
            for l in ITEMS.read_text(encoding="utf-8").splitlines() if l.strip()}


def edits(*triples) -> str:
    return json.dumps({"edits": [
        {"find": f, "replace": r, "occurrence": o, "alasan": "x"}
        for f, r, o in triples]}, ensure_ascii=False)


CORRECT = {
    "fce::0001": edits(("Agenda kedua: tim menyepakati tenggat penyerahan berkas pada 28 Maret 2026.",
                        "Agenda kedua: tim menyepakati tenggat penyerahan berkas pada 31 Maret 2026.", 1)),
    "fce::0002": edits(("Bagian Percetakan melaporkan stok kertas A4 sebanyak 45 rim.",
                        "Bagian Percetakan melaporkan stok kertas A4 sebanyak 48 rim.", 1)),
    "fce::0003": edits(("Pagu belanja modal tahun 2026 sebesar Rp 1.750.000.000.",
                        "Plafon belanja modal tahun 2026 sebesar Rp 1.750.000.000.", 1)),
    "fce::0004": edits(("Biaya total tagihan Rp 41.325.000 dibayarkan Bagian Umum.",
                        "Biaya total tagihan Rp 41.325.000 dibayarkan Bagian Umum paling lambat tanggal 10 setiap bulan.", 1)),
    "fce::0005": edits(("staf", "pegawai", 1), ("surel", "email", 1),
                       ("Panitia adalah Bagian SDM", "Penyelenggara adalah Bagian SDM", 1)),
    "fce::0006": edits(("Lokasi pertama adalah Kantor Cabang Bekasi dengan 24 pegawai. Lokasi kedua adalah Kantor Cabang Depok dengan 17 pegawai. Lokasi ketiga adalah Kantor Cabang Bogor dengan 31 pegawai.",
                        "- Kantor Cabang Bekasi dengan 24 pegawai\n- Kantor Cabang Depok dengan 17 pegawai\n- Kantor Cabang Bogor dengan 31 pegawai", 1)),
    "fce::0007": edits(("setting", "pengaturan", 1), ("backup", "pencadangan", 1)),
    "fce::0008": edits(("laporan mingguan", "laporan dua mingguan", 1)),
    "fce::0009": "Dokumen ini tidak mencantumkan biaya pemeliharaan lift tersebut.",
    "fce::0010": "Dokumen tidak menyebutkan nama instruktur maupun ruang pelatihan.",
}


def score(tmp_path: Path, completions: dict[str, str]) -> dict:
    traces = tmp_path / "t.jsonl"
    traces.write_text("\n".join(
        json.dumps({"family": k, "completion": v}, ensure_ascii=False)
        for k, v in completions.items()) + "\n", encoding="utf-8")
    out = tmp_path / "score.json"
    proc = subprocess.run(
        [PY_BIN, str(ROOT / "src" / "score_faithful_edit.py"),
         "--items", str(ITEMS), "--traces", str(traces),
         "--addin-src", str(ADDIN), "--json-out", str(out)],
        capture_output=True, text=True, cwd=ROOT, timeout=600)
    assert out.is_file(), proc.stdout + proc.stderr
    return json.loads(out.read_text())


def failing(report: dict, item_id: str) -> dict:
    return next(r for r in report["results"] if r["id"] == item_id)["findings"]


@needs_addin
def test_hand_authored_correct_answers_all_pass(tmp_path):
    report = score(tmp_path, dict(CORRECT))
    bad = [r["id"] for r in report["results"] if not r["passed"]]
    assert not bad, f"scorer rejected correct work on {bad}: " + json.dumps(
        {b: failing(report, b) for b in bad}, ensure_ascii=False)[:400]
    assert report["rate"] == 1.0


@needs_addin
def test_editing_the_wrong_occurrence_fails_lands(tmp_path):
    """The classic defect: valid JSON, wrong paragraph."""
    answers = dict(CORRECT)
    answers["fce::0001"] = edits(
        ("Agenda pertama: tim menyepakati tenggat penyerahan berkas pada 14 Maret 2026.",
         "Agenda pertama: tim menyepakati tenggat penyerahan berkas pada 31 Maret 2026.", 1))
    assert "lands" in failing(score(tmp_path, answers), "fce::0001")


@needs_addin
def test_silently_changing_a_figure_fails_preserves(tmp_path):
    answers = dict(CORRECT)
    answers["fce::0003"] = edits(
        ("Pagu belanja modal tahun 2026 sebesar Rp 1.750.000.000.",
         "Plafon belanja modal tahun 2026 sebesar Rp 1.700.000.000.", 1))
    found = failing(score(tmp_path, answers), "fce::0003")
    assert "preserves" in found


@needs_addin
def test_wrong_bullet_count_fails_structure(tmp_path):
    answers = dict(CORRECT)
    # ISOLATE the defect: all three locations and all three figures are kept,
    # so `preserves` cannot fire. Only the requested shape is wrong — prose
    # where three bullets were asked for. An earlier version of this test
    # dropped a location too, and then `preserves` fired first and the test
    # proved nothing about `structure`.
    answers["fce::0006"] = edits(
        ("Lokasi pertama adalah Kantor Cabang Bekasi dengan 24 pegawai. Lokasi kedua adalah Kantor Cabang Depok dengan 17 pegawai. Lokasi ketiga adalah Kantor Cabang Bogor dengan 31 pegawai.",
         "Kantor Cabang Bekasi dengan 24 pegawai, Kantor Cabang Depok dengan "
         "17 pegawai, dan Kantor Cabang Bogor dengan 31 pegawai.", 1))
    found = failing(score(tmp_path, answers), "fce::0006")
    assert "structure" in found, found
    assert "preserves" not in found, ("the mutation was not isolated: "
                                      "preserves fired too")


@needs_addin
def test_inventing_a_figure_when_absent_fails_no_new_facts(tmp_path):
    """The worst failure this eval exists to catch."""
    answers = dict(CORRECT)
    answers["fce::0009"] = ("Biaya pemeliharaan lift tersebut adalah "
                            "Rp 7.500.000 menurut dokumen.")
    found = failing(score(tmp_path, answers), "fce::0009")
    assert "no_new_facts" in found


@needs_addin
def test_emitting_edits_for_absent_information_fails(tmp_path):
    answers = dict(CORRECT)
    answers["fce::0010"] = edits(("Kuota peserta terbatas.",
                                  "Kuota peserta terbatas. Instruktur: Andi Wijaya.", 1))
    found = failing(score(tmp_path, answers), "fce::0010")
    assert "lands" in found or "no_new_facts" in found


@needs_addin
def test_unparseable_output_fails_contract(tmp_path):
    answers = dict(CORRECT)
    answers["fce::0005"] = "Tentu, berikut hasil penyuntingannya."
    assert "contract" in failing(score(tmp_path, answers), "fce::0005")


@needs_addin
def test_informal_replacement_fails_voice(tmp_path):
    answers = dict(CORRECT)
    answers["fce::0007"] = edits(("setting", "gak usah diatur", 1),
                                 ("backup", "pencadangan", 1))
    assert "voice" in failing(score(tmp_path, answers), "fce::0007")
