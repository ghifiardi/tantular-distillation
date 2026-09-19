"""Gold-set contract (RSI MVP Phase 1, Task 1.1).

The load-bearing property is that this module cannot manufacture the evidence
it validates. Every test below is either "a structurally wrong record is
refused" or "a structurally perfect record is still not production gold",
because the second is the one that keeps a fixture from becoming a measurement.

Offline and deterministic: fixtures only, no network, no model.
"""
from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import gold_set as gs  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures" / "gold"


def rows(name: str) -> list[dict]:
    return gs.parse_jsonl((FIXTURES / name).read_text(encoding="utf-8"))


def valid_rows() -> list[dict]:
    return rows("valid.synthetic.jsonl")


def a_production_record() -> dict:
    """A structurally valid record with real-looking, approved provenance.

    Built in the test, never shipped as data: this is what an approved item
    would look like, used to prove the eligibility check accepts one.
    """
    record = copy.deepcopy(valid_rows()[0])
    record["source_class"] = "internal"
    record["source"] = {
        "kind": "human_authored",
        "author_ref": "reviewer-registry-0001",
        "authored_at": "2026-09-19",
        "approved_by": "reviewer-registry-0002",
        "approved_at": "2026-09-19",
        "evidence_ref": "local-review-record-0001",
    }
    return record


# --- structural validation --------------------------------------------------


def test_every_valid_fixture_row_is_structurally_valid():
    for record in valid_rows():
        assert gs.validate_record(record) == [], record["id"]


def test_fixture_file_covers_all_four_splits():
    splits = {r["split"] for r in valid_rows()}
    assert splits == set(gs.SPLITS)


def test_unknown_split_is_refused():
    for record in rows("invalid_unknown_split.jsonl"):
        problems = gs.validate_record(record)
        assert any("split" in p for p in problems), problems


def test_missing_verifier_is_refused():
    for record in rows("invalid_missing_verifier.jsonl"):
        problems = gs.validate_record(record)
        assert any("verifier" in p for p in problems), problems


def test_non_indonesian_language_is_refused():
    for record in rows("invalid_non_indonesian.jsonl"):
        problems = gs.validate_record(record)
        assert any("language" in p for p in problems), problems


def test_empty_provenance_is_refused():
    for record in rows("invalid_empty_provenance.jsonl"):
        problems = gs.validate_record(record)
        assert any("source" in p for p in problems), problems


def test_judge_item_may_not_score_correctness():
    for record in rows("invalid_judge_correctness.jsonl"):
        problems = gs.validate_record(record)
        assert any("judge" in p for p in problems), problems


def test_judge_item_is_accepted_as_style():
    style = [r for r in valid_rows() if r["verifier"]["type"] == "judge"]
    assert style, "fixture must contain a judge/style row"
    for record in style:
        assert record["score_role"] == "style"
        assert gs.validate_record(record) == []


def test_unknown_top_level_field_is_refused():
    # The false-positive fixture carries `fixture_model_output`, which its own
    # README says is test metadata and must be removed before strict
    # validation. That it is refused while present is the point.
    record = rows("false_positive.synthetic.jsonl")[0]
    assert "fixture_model_output" in record
    problems = gs.validate_record(record)
    assert any("unknown field" in p for p in problems), problems

    record.pop("fixture_model_output")
    assert gs.validate_record(record) == []


def test_wrong_schema_version_is_refused():
    record = copy.deepcopy(valid_rows()[0])
    record["schema_version"] = 2
    assert any("schema_version" in p for p in gs.validate_record(record))


def test_unenumerated_normalization_is_refused():
    record = copy.deepcopy(valid_rows()[0])
    record["expected"]["normalization"] = ["fuzzy_match"]
    problems = gs.validate_record(record)
    assert any("normalization" in p for p in problems), problems


def test_executor_timeout_must_be_bounded():
    record = copy.deepcopy(
        [r for r in valid_rows() if r["verifier"]["type"] == "executor"][0])
    record["expected"]["timeout_seconds"] = 10_000
    assert any("timeout_seconds" in p for p in gs.validate_record(record))


def test_unknown_runner_is_refused():
    record = copy.deepcopy(
        [r for r in valid_rows() if r["verifier"]["type"] == "executor"][0])
    record["expected"]["runner"] = "bash"
    assert any("runner" in p for p in gs.validate_record(record))


def test_empty_correctness_expectation_is_refused():
    record = copy.deepcopy(valid_rows()[0])
    record["expected"] = {}
    assert any("expected" in p for p in gs.validate_record(record))


def test_a_non_object_record_is_refused_without_raising():
    assert gs.validate_record(["not", "an", "object"])
    assert gs.validate_record(None)


def test_invalid_json_line_is_reported_with_its_line_number():
    with pytest.raises(gs.GoldSetError) as error:
        gs.parse_jsonl('{"ok": 1}\nnot json\n')
    assert "line 2" in str(error.value)


# --- production eligibility -------------------------------------------------


def test_synthetic_fixture_is_never_production_gold():
    for record in valid_rows():
        assert gs.validate_record(record) == []
        blockers = gs.production_problems(record)
        assert blockers, f"{record['id']} must be ineligible for production"


def test_an_approved_human_authored_record_is_eligible():
    assert gs.production_problems(a_production_record()) == []


def test_self_approval_is_not_an_independent_review():
    record = a_production_record()
    record["source"]["approved_by"] = record["source"]["author_ref"]
    blockers = gs.production_problems(record)
    assert any("distinct" in b for b in blockers), blockers


def test_unparseable_approval_date_is_refused():
    record = a_production_record()
    record["source"]["approved_at"] = "sometime last week"
    assert any("approved_at" in b for b in gs.production_problems(record))


def test_source_class_synthetic_is_never_production():
    record = a_production_record()
    record["source_class"] = "synthetic"
    assert any("source_class" in b for b in gs.production_problems(record))


def test_missing_source_class_defaults_to_internal_not_synthetic():
    record = a_production_record()
    record.pop("source_class")
    assert gs.validate_record(record) == []
    assert gs.production_problems(record) == []


# --- set-level behaviour ----------------------------------------------------


def test_duplicate_ids_are_refused():
    duplicated = valid_rows()[:1] * 2
    gold = gs.GoldSet(duplicated, source="test")
    assert any("duplicate id" in p for p in gold.problems)
    assert len(gold.valid) == 1


def test_a_fixture_set_has_zero_approved_items():
    gold = gs.GoldSet(valid_rows(), source="test")
    assert gold.problems == []
    assert gold.approved_count == 0
    assert len(gold.ineligible) == len(valid_rows())


def test_gate_on_a_fixture_set_reports_the_blocked_message():
    gold = gs.GoldSet(valid_rows(), source="test")
    reasons = gs.gate(gold)
    assert gs.BLOCKED_MESSAGE in reasons


def test_gate_requires_every_split_once_items_exist():
    record = a_production_record()          # reasoning only
    gold = gs.GoldSet([record], source="test")
    assert gold.approved_count == 1
    reasons = gs.gate(gold)
    assert gs.BLOCKED_MESSAGE not in reasons
    for split in ("instruction_following", "knowledge", "tool_use"):
        assert any(split in r for r in reasons), reasons


def test_gate_refuses_a_judge_only_set():
    record = a_production_record()
    record["verifier"] = {"type": "judge", "config": {}}
    record["expected"] = {"rubric": ["alami"]}
    record["score_role"] = "style"
    gold = gs.GoldSet([record], source="test")
    assert gold.approved_count == 1
    assert any("correctness" in r for r in gs.gate(gold))


def test_a_full_approved_set_passes_the_gate():
    records = []
    for i, split in enumerate(gs.SPLITS, start=1):
        record = a_production_record()
        record["id"] = f"gold::{split}::{i:04d}"
        record["split"] = split
        record["source"]["evidence_ref"] = f"local-review-record-{i:04d}"
        records.append(record)
    gold = gs.GoldSet(records, source="test")
    assert gold.approved_count == len(gs.SPLITS)
    assert gs.gate(gold) == []


# --- the empty data/gold directory -----------------------------------------


def test_the_real_gold_directory_is_empty_and_that_is_correct():
    gold = gs.load(gs.GOLD_DIR)
    assert gold.records == [], "data/gold must contain no items yet"
    assert gold.approved_count == 0
    assert gs.BLOCKED_MESSAGE in gs.gate(gold)


def test_loading_a_missing_directory_refuses():
    with pytest.raises(gs.GoldSetError):
        gs.load(ROOT / "data" / "gold-does-not-exist")


def test_summary_never_claims_gate_ready():
    gold = gs.load(gs.GOLD_DIR)
    assert gold.summary()["gate_ready"] is False


# --- CLI --------------------------------------------------------------------


def run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(ROOT / "src" / "gold_set.py"), *args],
        capture_output=True, text=True, timeout=60,
    )


def test_cli_gate_on_empty_gold_exits_nonzero_with_the_named_message():
    result = run_cli("--gate")
    assert result.returncode != 0
    assert gs.BLOCKED_MESSAGE in result.stderr
    assert "GOLD SET REFUSED" in result.stderr


def test_cli_without_gate_reports_and_succeeds():
    result = run_cli("--json")
    assert result.returncode == 0
    summary = json.loads(result.stdout)
    assert summary["approved_items"] == 0
    assert summary["gate_ready"] is False
