"""The trainer must start from one exact, fully-authorised run freeze.

These tests use the real corpus and promotion outputs but write the freeze
itself under tmp_path. Nothing here trains or calls a model endpoint.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PY = str(ROOT / ".venv" / "bin" / "python")
FREEZER = str(ROOT / "src" / "freeze_training_run.py")
TRAINER = str(ROOT / "src" / "train_qlora.py")
CONFIG = ROOT / "train" / "qlora_9b.yaml"
CORPUS = ROOT / "data" / "v3-candidate" / "traces.r0.jsonl"
PROMOTION = ROOT / "train" / "RUN_MANIFEST.v1-mechanical.json"
WAIVER = ROOT / "calibration" / "INT4_WAIVER.md"


def freeze(out: Path, *, config: Path = CONFIG,
           promotion: Path = PROMOTION, waiver: Path | None = WAIVER):
    cmd = [
        PY, FREEZER,
        "--corpus", str(CORPUS),
        "--config", str(config),
        "--promotion-manifest", str(promotion),
        "--out", str(out),
        "--frozen-at", "2026-08-19T12:00:00+07:00",
        "--write",
    ]
    if waiver is not None:
        cmd += ["--waiver", str(waiver)]
    return subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)


def run_trainer(run_manifest: Path, *, config: Path = CONFIG,
                promotion: Path = PROMOTION):
    return subprocess.run(
        [PY, TRAINER, "--dry-run",
         "--config", str(config),
         "--run-manifest", str(run_manifest),
         "--promotion-manifest", str(promotion)],
        capture_output=True, text=True, cwd=ROOT)


@pytest.fixture
def valid_freeze(tmp_path) -> Path:
    out = tmp_path / "RUN.json"
    proc = freeze(out)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return out


def doctor(source: Path, out: Path, edit) -> Path:
    payload = json.loads(source.read_text(encoding="utf-8"))
    edit(payload)
    out.write_text(json.dumps(payload), encoding="utf-8")
    return out


@pytest.mark.requires_local_corpus
def test_freeze_v4_pins_promotion_outputs_and_failed_gate(valid_freeze):
    payload = json.loads(valid_freeze.read_text())
    assert payload["schema_version"] == 4
    assert payload["promotion_manifest"]["sha256"]
    assert payload["promotion_manifest"]["promoted"]["train"]["traces"] == 136
    assert payload["promotion_manifest"]["promoted"]["eval"]["traces"] == 47
    # The signed waiver authorises proceeding despite this. It is not a pass.
    assert payload["gate"]["exit_code"] == 1
    assert payload["gate"]["verdict"] == "FAILED"
    assert len(payload["gate"]["violations"]) == 1
    assert "quantized teacher" in payload["gate"]["violations"][0]
    assert payload["waiver"]["sha256"]


@pytest.mark.requires_local_corpus
def test_checked_in_v1_freeze_is_current_and_accepted():
    proc = subprocess.run([PY, TRAINER, "--dry-run"],
                          capture_output=True, text=True, cwd=ROOT)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "schema             4" in proc.stdout
    assert "waiver verified" in proc.stdout
    assert "DRY RUN OK" in proc.stdout


@pytest.mark.requires_local_corpus
def test_valid_v4_freeze_allows_complete_cpu_dry_run(valid_freeze):
    proc = run_trainer(valid_freeze)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "waiver verified" in proc.stdout
    assert "DRY RUN OK" in proc.stdout


@pytest.mark.requires_local_corpus
def test_changed_config_is_refused_before_any_gate_or_gpu_work(tmp_path):
    config = tmp_path / "qlora.yaml"
    config.write_bytes(CONFIG.read_bytes())
    manifest = tmp_path / "RUN.json"
    assert freeze(manifest, config=config).returncode == 0
    config.write_text(config.read_text() + "\n# changed after freeze\n")

    proc = run_trainer(manifest, config=config)
    assert proc.returncode == 2
    assert "training config is STALE or changed" in proc.stderr
    assert "=== HELD-OUT VERIFICATION ===" not in proc.stdout


@pytest.mark.requires_local_corpus
def test_changed_promotion_manifest_is_refused(valid_freeze, tmp_path):
    promotion = tmp_path / "promotion.json"
    promotion.write_bytes(PROMOTION.read_bytes())
    manifest = tmp_path / "RUN-promotion.json"
    assert freeze(manifest, promotion=promotion).returncode == 0
    promotion.write_text(promotion.read_text() + "\n")

    proc = run_trainer(manifest, promotion=promotion)
    assert proc.returncode == 2
    assert "promotion manifest is STALE or changed" in proc.stderr


@pytest.mark.requires_local_corpus
def test_doctored_promoted_snapshot_is_refused(valid_freeze, tmp_path):
    manifest = doctor(
        valid_freeze, tmp_path / "doctored.json",
        lambda p: p["promotion_manifest"]["promoted"]["train"].update(
            {"sha256": "0" * 64}),
    )
    proc = run_trainer(manifest)
    assert proc.returncode == 2
    assert "promoted train sha256 disagrees" in proc.stderr


@pytest.mark.requires_local_corpus
def test_failed_gate_without_waiver_is_refused(valid_freeze, tmp_path):
    manifest = doctor(
        valid_freeze, tmp_path / "no-waiver.json",
        lambda p: p.update({"waiver": {"path": None, "sha256": None}}),
    )
    proc = run_trainer(manifest)
    assert proc.returncode == 2
    assert "gate FAILED" in proc.stderr
    assert "no signed waiver" in proc.stderr


@pytest.mark.requires_local_corpus
def test_changed_waiver_is_refused(valid_freeze, tmp_path):
    manifest = doctor(
        valid_freeze, tmp_path / "bad-waiver.json",
        lambda p: p["waiver"].update({"sha256": "0" * 64}),
    )
    proc = run_trainer(manifest)
    assert proc.returncode == 2
    assert "signed waiver is STALE or changed" in proc.stderr


@pytest.mark.requires_local_corpus
def test_gate_verdict_cannot_turn_a_failure_into_a_pass(valid_freeze, tmp_path):
    manifest = doctor(
        valid_freeze, tmp_path / "false-pass.json",
        lambda p: p["gate"].update({"verdict": "passed"}),
    )
    proc = run_trainer(manifest)
    assert proc.returncode == 2
    assert "gate verdict contradicts its exit code" in proc.stderr


@pytest.mark.requires_local_corpus
def test_arbitrary_waiver_cannot_authorize_the_int4_failure(tmp_path):
    other = tmp_path / "OTHER_WAIVER.md"
    other.write_text(WAIVER.read_text())
    proc = freeze(tmp_path / "RUN.json", waiver=other)
    assert proc.returncode != 0
    assert "requires the accepted waiver" in proc.stdout + proc.stderr


@pytest.mark.requires_local_corpus
def test_int4_waiver_cannot_authorize_an_unrelated_corpus_failure(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    rows = [json.loads(line) for line in CORPUS.read_text().splitlines()]
    rows[0]["completion"] = ""
    corpus.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    promotion_payload = json.loads(PROMOTION.read_text())
    import hashlib
    promotion_payload["source_corpus"]["path"] = str(corpus)
    promotion_payload["source_corpus"]["sha256"] = hashlib.sha256(
        corpus.read_bytes()).hexdigest()
    promotion_payload["source_corpus"]["traces"] = len(rows)
    promotion = tmp_path / "promotion.json"
    promotion.write_text(json.dumps(promotion_payload))

    cmd = [
        PY, FREEZER,
        "--corpus", str(corpus),
        "--config", str(CONFIG),
        "--promotion-manifest", str(promotion),
        "--waiver", str(WAIVER),
        "--out", str(tmp_path / "RUN.json"),
        "--frozen-at", "2026-08-19T12:00:00+07:00",
        "--write",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    assert proc.returncode != 0
    assert "waiver does not cover" in proc.stdout + proc.stderr


def test_malformed_run_manifest_fails_closed(tmp_path):
    manifest = tmp_path / "broken.json"
    manifest.write_text("{not json")
    proc = run_trainer(manifest)
    assert proc.returncode == 2
    assert "not readable JSON" in proc.stderr


@pytest.mark.requires_local_corpus
def test_freezer_refuses_failed_gate_without_waiver(tmp_path):
    proc = freeze(tmp_path / "RUN.json", waiver=None)
    assert proc.returncode != 0
    assert "gate FAILED and no --waiver" in proc.stdout + proc.stderr


# --- v4: the corpus audit is recorded, not remembered -----------------------

def test_freeze_records_the_provenance_audit(valid_freeze):
    """Six months on, "why can this checkpoint not claim FP8?" has to be
    answerable from the manifest rather than from anyone's memory."""
    audit = json.loads(valid_freeze.read_text())["provenance_audit"]

    assert audit["fp8_gate"]["status"] == "UNMET"
    assert audit["fp8_gate"]["untrainable_quantizations"] == {"int4_ollama": 260}
    assert audit["source_classes"] == {"synthetic": 260}
    assert audit["license_freshness"]["resolved"][0]["registry_model"] == "muse-glimmer-30b"
    # Freshness is measured against the freeze date, not the clock. This fixture
    # freezes at 2026-08-19 while the registry entry was reviewed on 2026-09-03,
    # so the review is in the FUTURE relative to the freeze.
    assert audit["as_of_date"] == "2026-08-19"
    resolved = audit["license_freshness"]["resolved"][0]
    assert resolved["age_days"] == -15
    assert resolved["status"] == "UNKNOWN", (
        "a review dated after the freeze is a data error, not freshness; a "
        "negative age would otherwise pass `age > max_age` forever")
    assert audit["identity_verification"] == {
        "all_verified": False, "unverified_models": ["muse-glimmer-30b"]}
    assert audit["trainable_as_is"] is False
    # Stated inside the block, not left to be inferred from its absence.
    assert audit["authorizes_training"] is False


def test_the_audit_does_not_change_what_the_gate_decides(valid_freeze):
    """Additive evidence only: the recorded verdict is still the gate's."""
    payload = json.loads(valid_freeze.read_text())
    assert payload["gate"]["exit_code"] == 1
    assert payload["gate"]["verdict"] == "FAILED"
    assert payload["provenance_audit"]["fp8_gate"]["status"] == "UNMET"


def test_an_unparseable_frozen_at_refuses_rather_than_dating_from_the_clock(tmp_path):
    out = tmp_path / "RUN.json"
    proc = subprocess.run(
        [PY, FREEZER, "--corpus", str(CORPUS), "--config", str(CONFIG),
         "--promotion-manifest", str(PROMOTION), "--waiver", str(WAIVER),
         "--out", str(out), "--frozen-at", "sometime in August", "--write"],
        capture_output=True, text=True, cwd=ROOT)
    assert proc.returncode != 0
    assert "ISO date" in proc.stdout + proc.stderr
    assert not out.exists()


def test_the_checked_in_freeze_is_dated_when_it_was_actually_made():
    """A schema-4 manifest written on 2026-09-03 must not claim an August
    freeze. The historical August freeze is preserved separately, byte for
    byte, at train/archive/RUN_MANIFEST.v1.schema-2.json — backdating this one
    to match it would put a false date on a new artifact."""
    payload = json.loads((ROOT / "train" / "RUN_MANIFEST.v1.json").read_text())
    audit = payload["provenance_audit"]

    assert payload["schema_version"] == 4
    assert payload["frozen_at"].startswith("2026-09-03")
    assert audit["as_of_date"] == "2026-09-03"

    resolved = audit["license_freshness"]["resolved"][0]
    assert resolved["reviewed_at"] == "2026-09-03"
    assert resolved["age_days"] == 0, "the licence review is same-day, not -15"
    assert resolved["status"] == "FRESH"

    # Still not trainable, and for the honest reasons.
    assert audit["fp8_gate"]["status"] == "UNMET"
    assert audit["source_classes"] == {"synthetic": 260}
    assert audit["identity_verification"]["all_verified"] is False
    assert audit["trainable_as_is"] is False
    assert audit["authorizes_training"] is False


def test_the_archived_schema_2_freeze_still_holds_the_august_record():
    archived = json.loads(
        (ROOT / "train" / "archive" / "RUN_MANIFEST.v1.schema-2.json").read_text())
    assert archived["schema_version"] == 2
    assert archived["frozen_at"].startswith("2026-08-20")
    assert "provenance_audit" not in archived
    # The failed gate it recorded is preserved, not tidied away.
    assert archived["gate"]["exit_code"] == 1
    assert archived["gate"]["verdict"] == "FAILED"
