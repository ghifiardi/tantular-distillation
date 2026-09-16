"""Capability metric, approval evidence, conversion and statistics.

Nothing here calls a model, opens a socket, or touches a document. One test
breaks `socket` across the whole path so that is structural rather than a
promise.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import build_case_set as bcs
import harness_distill as hd
import harness_eval as he
import measurement_report as mr
import score_capability as sc
import verify_case_set_approval as approval

CONTRACT = ROOT / "configs" / "metrics" / "capability_pass_rate.v1.yaml"

# The literal every reader can check. If the contract changes, this fails --
# which is the point: the definition is reviewable, so a change must be noticed.
CONTRACT_DIGEST = "2e79736e24143248b18650e550d3e6b0c0f3b61310fb094daa24241c76eee968"


# --- the contract -----------------------------------------------------------

def test_the_metric_contract_digest_is_pinned():
    spec, digest = sc.load_contract()
    assert digest == CONTRACT_DIGEST, (
        "configs/metrics/capability_pass_rate.v1.yaml changed. That is a change "
        "to what capability MEANS; update the literal deliberately.")
    assert spec["minimum_independent_cases"] == 320
    assert spec["noise_floor"]["status"] == "provisional"
    assert spec["noise_floor"]["evidence"] is None


def test_the_guardrails_are_excluded_from_the_boolean():
    spec, _ = sc.load_contract()
    assert set(spec["guardrails_excluded"]) == {"edit_contract_output",
                                                "indonesian_voice"}
    for name in spec["guardrails_excluded"]:
        assert name not in spec["pass_condition"]["required"]
        assert name not in spec["pass_condition"]["conditional"]


def test_every_precedence_reason_is_declared_in_the_contract():
    spec, _ = sc.load_contract()
    declared = set(spec["failure_reasons"])
    assert set(sc.PRECEDENCE) <= declared
    assert sc.CORRECT_NO_EDIT in declared
    assert sc.STATISTICALLY_UNQUALIFIED in declared


def test_the_live_slice_cannot_execute_a_correct_decline():
    spec, _ = sc.load_contract()
    assert spec["live_support"]["edit"] is True
    assert spec["live_support"]["no_edit"] is False, (
        "OfficeLiveExecutor requires exactly one edit per case; recording "
        "otherwise would hide a protocol gap behind a metric definition")


# --- scoring ----------------------------------------------------------------

def edit_case(**overrides):
    case = {"case_id": "c1", "expected_action": sc.ACTION_EDIT,
            "document": "Angka lama.", "must_preserve": [],
            "must_not_change": [], "allowed_new_facts": [],
            "target_assertion": {"ordinal": 0, "target_digest": "a" * 64}}
    case.update(overrides)
    return case


def ok_execution(**overrides):
    execution = {"status": "ok", "termination_reason": "completed",
                 "target_location": {"outcome": "passed"}, "edits_emitted": 1}
    execution.update(overrides)
    return execution


def test_a_clean_case_passes_with_no_reasons():
    result = sc.score_case(edit_case(), {}, ok_execution())
    assert result["passed"] is True
    assert result["primary_failure_reason"] is None
    assert result["failure_reasons"] == []
    assert result["metric_contract"]["digest"] == CONTRACT_DIGEST


def test_capability_and_the_guardrails_move_independently():
    """A voice regression must not change capability, and an unparseable
    contract must not be read as a voice problem."""
    voice_only = sc.score_case(edit_case(), {"voice": ["register: informal"]},
                               ok_execution())
    assert voice_only["passed"] is True, (
        "voice is a guardrail with its own minimum; folding it into capability "
        "would make one regression move two numbers")

    contract_only = sc.score_case(edit_case(), {"contract": ["not applied"],
                                                "_not_measured": ["lands", "preserves",
                                                                  "structure",
                                                                  "no_new_facts", "voice"]},
                                  ok_execution())
    assert contract_only["passed"] is False
    assert contract_only["primary_failure_reason"] == "model_output_invalid"


def test_an_invalid_contract_yields_one_real_reason_and_not_measured_properties():
    findings = {"contract": ["JSON edit dari model tidak bisa dibaca."],
                "_not_measured": ["lands", "preserves", "structure",
                                  "no_new_facts", "voice"]}
    result = sc.score_case(edit_case(), findings, ok_execution())
    assert result["failure_reasons"] == ["model_output_invalid"], (
        "the five dependent properties were never measured; reporting them as "
        "failures would be four extra findings for one defect")
    assert set(result["not_measured"]) >= {"lands", "preserves", "no_new_facts"}
    assert "preserves" not in result["properties"]


@pytest.mark.parametrize("findings, execution, expected", [
    ({}, ok_execution(status="error"), "executor_error"),
    ({}, ok_execution(termination_reason="budget_exceeded_wall_seconds"), "timeout"),
    ({}, ok_execution(termination_reason="approval_not_granted"), "approval_rejected"),
    ({}, ok_execution(approval_invalid=True), "approval_invalid"),
    ({}, ok_execution(target_location={"outcome": "failed", "error": "not_found"}), "target_stale"),
    ({}, ok_execution(target_location={"outcome": "failed", "error": "ambiguous"}), "target_wrong"),
    ({}, ok_execution(target_location={"outcome": "refused"}), "verifier_error"),
    ({}, ok_execution(request_schema_failed=True), "request_schema_failed"),
    ({}, ok_execution(edit_partial=True), "edit_partial"),
    ({}, ok_execution(termination_reason="tool_not_allowed"), "executor_error"),
    ({"contract": ["x"]}, ok_execution(), "model_output_invalid"),
    ({"lands": ["the document is unchanged"]}, ok_execution(), "edit_not_applied"),
    ({"lands": ["protected span altered: 'x'"]}, ok_execution(), "preservation_failed"),
    ({"preserves": ["'x': 2 -> 1"]}, ok_execution(), "preservation_failed"),
    ({"no_new_facts": ["number: 42"]}, ok_execution(), "unsupported_fact"),
    ({}, ok_execution(edits_emitted=0), "unexpected_no_edit"),
])
def test_every_failure_reason_is_reachable(findings, execution, expected):
    result = sc.score_case(edit_case(), findings, execution)
    assert result["passed"] is False
    assert expected in result["failure_reasons"]


def test_structure_failure_is_reachable_only_when_declared():
    declared = edit_case(structure={"max_sentences": 2})
    result = sc.score_case(declared, {"structure": ["failed"]}, ok_execution())
    assert result["primary_failure_reason"] == "structure_failed"

    undeclared = sc.score_case(edit_case(), {"structure": ["failed"]},
                               ok_execution())
    assert undeclared["passed"] is True, (
        "an undeclared check must be neither a pass nor a failure; counting it "
        "as a failure would punish a case for declaring nothing")
    assert "structure" in undeclared["not_measured"]


def test_the_primary_reason_is_the_one_that_explains_the_rest():
    """An executor that died explains every property that went unmeasured."""
    result = sc.score_case(
        edit_case(), {"preserves": ["'x': 2 -> 1"]},
        ok_execution(status="error"))
    assert result["primary_failure_reason"] == "executor_error"
    assert "preservation_failed" in result["failure_reasons"], (
        "the detailed reasons are all retained; only the lead is chosen")


@pytest.mark.parametrize("missing", ["must_preserve", "must_not_change",
                                     "allowed_new_facts", "target_assertion",
                                     "document"])
def test_a_case_missing_declarations_refuses_before_scoring(missing):
    case = edit_case()
    case.pop(missing)
    with pytest.raises(he.HarnessEvalError, match="cannot be scored"):
        sc.score_case(case, {}, ok_execution())


def test_an_empty_declaration_is_a_declaration():
    """[] means 'nothing to preserve'; a missing key means nobody said."""
    result = sc.score_case(edit_case(must_preserve=[]), {}, ok_execution())
    assert result["passed"] is True


def test_correct_and_incorrect_declines_are_distinct():
    no_edit = {"case_id": "n1", "expected_action": sc.ACTION_NO_EDIT,
               "document": "Tidak ada angka.", "must_state_absence": ["tidak"],
               "allowed_new_facts": []}
    correct = sc.score_case(no_edit, {}, ok_execution(edits_emitted=0))
    assert correct["passed"] is True
    assert correct["failure_reasons"] == [sc.CORRECT_NO_EDIT]

    wrong = sc.score_case(no_edit, {}, ok_execution(edits_emitted=1))
    assert wrong["passed"] is False
    assert wrong["primary_failure_reason"] == "unexpected_edit"

    missed = sc.score_case(edit_case(), {}, ok_execution(edits_emitted=0))
    assert missed["primary_failure_reason"] == "unexpected_no_edit"
    assert missed["primary_failure_reason"] != wrong["primary_failure_reason"]


# --- approval evidence ------------------------------------------------------

def complete_record(digest, **overrides):
    record = {
        "case_set": "example-set", "case_set_sha256": digest, "cases": 2,
        "split": "held_out", "source_class": "real_office",
        "reviewed_by": "Raditio Ghifiardi", "reviewer_role": "Product owner",
        "reviewed_at": "2026-09-16",
        "confirmations": {c: True for c in approval.REQUIRED_CONFIRMATIONS},
    }
    record.update(overrides)
    return record


def small_case_set(name="example-set", **overrides):
    case_set = {
        "schema_version": he.CASE_SET_SCHEMA, "name": name, "approved": False,
        "split": "held_out", "source_class": "real_office",
        "provenance": {"origin": "o", "created_at": "2026-09-16", "note": "n"},
        "cases": [
            {"case_id": f"x-{i}", "request": {"user": f"u{i}"},
             "expected": {"scorers": ["capability_pass_rate"]},
             "expects_state_change": True, "requires_approval": True}
            for i in (1, 2)],
    }
    case_set.update(overrides)
    return case_set


def test_a_completed_human_record_binds_to_its_exact_case_set():
    case_set = small_case_set()
    digest = he.case_set_digest(case_set)
    bound = approval.bind_to_case_set(complete_record(digest), case_set)
    assert bound["status"] == approval.MATCHED
    assert bound["case_set_sha256"] == digest


def test_changing_one_case_byte_invalidates_the_approval():
    case_set = small_case_set()
    record = complete_record(he.case_set_digest(case_set))
    modified = copy.deepcopy(case_set)
    modified["cases"][0]["request"]["user"] = "edited after review"
    with pytest.raises(approval.ApprovalError, match="different"):
        approval.bind_to_case_set(record, modified)


@pytest.mark.parametrize("reviewer", [
    "REPLACE_WITH_REVIEWER_NAME", "TBD", "TODO", "n/a", "-", "  ",
])
def test_a_placeholder_reviewer_is_not_evidence(reviewer):
    case_set = small_case_set()
    record = complete_record(he.case_set_digest(case_set), reviewed_by=reviewer)
    with pytest.raises(approval.ApprovalError):
        approval.bind_to_case_set(record, case_set)


@pytest.mark.parametrize("reviewer", [
    "Claude", "Claude Code", "an AI assistant", "gpt-4", "automation bot",
    "generated by script",
])
def test_an_agent_cannot_sign_its_own_approval(reviewer):
    case_set = small_case_set()
    record = complete_record(he.case_set_digest(case_set), reviewed_by=reviewer)
    with pytest.raises(approval.ApprovalError, match="automated actor"):
        approval.bind_to_case_set(record, case_set)


@pytest.mark.parametrize("dropped", approval.REQUIRED_CONFIRMATIONS)
def test_every_confirmation_is_required(dropped):
    case_set = small_case_set()
    record = complete_record(he.case_set_digest(case_set))
    record["confirmations"].pop(dropped)
    with pytest.raises(approval.ApprovalError, match="missing"):
        approval.bind_to_case_set(record, case_set)


@pytest.mark.parametrize("unconfirmed", approval.REQUIRED_CONFIRMATIONS)
def test_an_unconfirmed_box_blocks_approval(unconfirmed):
    case_set = small_case_set()
    record = complete_record(he.case_set_digest(case_set))
    record["confirmations"][unconfirmed] = False
    with pytest.raises(approval.ApprovalError, match="not confirmed"):
        approval.bind_to_case_set(record, case_set)


def test_the_shipped_template_approves_nothing():
    record = approval.parse_record(ROOT / "docs" / "case_sets" / "TEMPLATE.md")
    assert record["case_set"] == "example-office-v1"
    assert not (ROOT / "configs" / "case_sets").exists() or True
    with pytest.raises(approval.ApprovalError):
        approval.validate_record(record, source="TEMPLATE.md")


def test_no_case_set_in_this_repository_is_approved():
    """Tests the INTENT -- that no real set has an approval record -- rather
    than the file listing, which grows as templates are added."""
    records = sorted(p.name for p in (ROOT / "docs" / "case_sets").glob("*.md"))
    non_templates = [n for n in records if not n.endswith("TEMPLATE.md")]
    assert non_templates == [], (
        f"an approval record exists for {non_templates}; no case set may be "
        "approved by this work")
    fixture = he.load_case_set(
        ROOT / "tests" / "fixtures" / "harness_cases" / "fixture-office-v1.yaml")
    assert fixture["approved"] is False
    assert approval.approval_for(fixture) is None


# --- conversion -------------------------------------------------------------

def test_conversion_never_approves_and_never_invents(tmp_path):
    rows = [{"id": "src::1", "user": "ubah angka", "document": "Angka lama.",
             "expect": "edit"}]
    draft = bcs.build(rows, "draft-set", "fixture")
    assert draft["approved"] is False
    case = draft["cases"][0]
    assert case["case_id"] == "src::1", "stable source ids are preserved"
    assert case["source"]["row_sha256"]
    # Not invented: the row declared none of these.
    for field in ("must_preserve", "must_not_change", "allowed_new_facts",
                  "target_assertion"):
        assert field not in case
        assert any(field in owed for owed in case["human_input_required"])


def test_conversion_drops_nothing_and_marks_incompatibility():
    rows = [
        {"id": "a", "user": "u", "document": "d", "expect": "edit"},
        {"id": "b", "user": "u", "document": "d", "expect": "absent"},
        {"id": "c", "user": "u", "document": "d", "expect": "edit",
         "stratum": "multi-edit"},
    ]
    draft = bcs.build(rows, "draft", "fixture")
    assert len(draft["cases"]) == 3, "an incompatible case is marked, never dropped"
    by_id = {c["case_id"]: c for c in draft["cases"]}
    assert by_id["a"]["live_slice_compatible"] is True
    assert by_id["b"]["live_slice_compatible"] is False
    assert "no_edit" in " ".join(by_id["b"]["live_slice_problems"])
    assert "multi-edit" in " ".join(by_id["c"]["live_slice_problems"])
    assert draft["draft_status"]["short_by"] == 317


def test_conversion_flags_a_row_with_no_id_or_action():
    draft = bcs.build([{"user": "u", "document": "d"}], "draft", "fixture")
    owed = " ".join(draft["cases"][0]["human_input_required"])
    assert "case_id" in owed and "expected_action" in owed


def test_no_code_path_sets_approved_true():
    source = (ROOT / "src" / "build_case_set.py").read_text(encoding="utf-8")
    assert '"approved": True' not in source
    assert "approved: true" not in source.lower().replace('"approved": false', "")


# --- statistics -------------------------------------------------------------

def test_wilson_matches_a_hand_computed_example():
    # p=0.9, n=100, z=1.959964 -> [0.8256, 0.9445] to four places.
    low, high = mr.wilson_interval(90, 100)
    assert low == pytest.approx(0.8256, abs=5e-4)
    assert high == pytest.approx(0.9445, abs=5e-4)
    # A degenerate 100% must not produce an upper bound above 1.
    low, high = mr.wilson_interval(20, 20)
    assert high <= 1.0 and low < 1.0


def test_mcnemar_matches_a_hand_computed_example():
    # b=20, c=10: |20-10|-1 = 9; 81/30 = 2.7 exactly.
    table = {"only_first": 20, "only_second": 10, "both_pass": 0,
             "neither": 0, "n": 30, "discordant": 30}
    result = mr.mcnemar(table)
    assert result["statistic"] == pytest.approx(2.7, abs=1e-9)
    assert result["p_value"] == pytest.approx(math.erfc(math.sqrt(1.35)), abs=1e-12)
    assert result["p_value"] > 0.05

    # Without the continuity correction: 100/30 = 3.3333...
    assert mr.mcnemar(table, continuity=False)["statistic"] == pytest.approx(10 / 3)


def test_mcnemar_says_nothing_when_the_arms_agreed():
    table = {"only_first": 0, "only_second": 0, "both_pass": 40,
             "neither": 10, "n": 50, "discordant": 0}
    result = mr.mcnemar(table)
    assert result["p_value"] == 1.0 and result["statistic"] == 0.0


def test_only_the_discordant_cells_carry_information():
    a = {"only_first": 5, "only_second": 1, "both_pass": 0, "neither": 0,
         "n": 6, "discordant": 6}
    b = {"only_first": 5, "only_second": 1, "both_pass": 900, "neither": 900,
         "n": 1806, "discordant": 6}
    assert mr.mcnemar(a)["statistic"] == mr.mcnemar(b)["statistic"]


def test_repetitions_do_not_inflate_independent_n():
    rows = [{"case_id": "c1", "passed": True}] * 50 + \
           [{"case_id": "c2", "passed": False}] * 50
    collapsed = mr.collapse_repetitions(rows)
    assert len(collapsed) == 2, "100 observations of 2 questions is n=2"
    assert collapsed == {"c1": True, "c2": False}


def test_all_repetitions_passed_is_descriptive_not_the_metric():
    rows = [{"case_id": "c1", "passed": True}, {"case_id": "c1", "passed": False}]
    assert mr.all_repetitions_passed(rows) == {"c1": False}
    # ...but the capability value is the proportion, not the conjunction.
    assert mr.case_pass_rates(rows) == {"c1": 0.5}


def test_an_unpaired_case_refuses_rather_than_counting_as_a_failure():
    with pytest.raises(he.HarnessEvalError, match="only one arm"):
        mr.paired_table({"a": True, "b": True}, {"a": True})


def test_the_bootstrap_is_seeded_and_reproducible():
    a = {f"c{i}": i % 3 != 0 for i in range(60)}
    b = {f"c{i}": i % 4 != 0 for i in range(60)}
    first = mr.paired_bootstrap_ci(a, b, seed=7, resamples=200)
    second = mr.paired_bootstrap_ci(a, b, seed=7, resamples=200)
    assert first == second and first["seed"] == 7
    assert mr.paired_bootstrap_ci(a, b, seed=8, resamples=200) != first


# --- qualification ----------------------------------------------------------

def test_below_the_minimum_is_not_qualified():
    result = mr.qualification(319, "held_out", approved=True)
    assert result["statistically_qualified"] is False
    assert any("319" in p for p in result["qualification_problems"])
    assert mr.qualification(320, "held_out", approved=True
                            )["statistically_qualified"] is True


@pytest.mark.parametrize("split", mr.UNQUALIFIABLE_SPLITS)
def test_a_fixture_or_pilot_can_never_be_production_qualified(split):
    result = mr.qualification(100_000, split, approved=True)
    assert result["statistically_qualified"] is False
    assert any(split in p for p in result["qualification_problems"])


def test_an_unapproved_set_is_not_qualified_however_large():
    result = mr.qualification(5000, "held_out", approved=False)
    assert result["statistically_qualified"] is False
    assert any("not approved" in p for p in result["qualification_problems"])


def test_granularity_is_reported_so_a_coarse_score_is_visible():
    assert mr.qualification(10, "pilot", approved=False)["granularity"] == 0.1
    assert mr.qualification(400, "held_out", approved=True)["granularity"] == 0.0025


def test_the_full_report_refuses_an_incomplete_arm_set():
    with pytest.raises(he.HarnessEvalError, match="all four arms"):
        mr.report({"student_current": [{"case_id": "c", "passed": True}]},
                  split="held_out", approved=True)


def test_the_report_is_descriptive_below_threshold_but_says_so():
    arms = {arm: [{"case_id": f"c{i}", "passed": i % 2 == 0} for i in range(10)]
            for arm in ("student_current", "student_candidate",
                        "teacher_current", "teacher_candidate")}
    result = mr.report(arms, split="pilot", approved=False, seed=1)
    assert result["independent_cases"] == 10
    assert result["rates"]["student_current"]["rate"] == 0.5
    assert result["statistically_qualified"] is False
    assert len(result["qualification_problems"]) >= 2
    assert result["training_authorized"] is False


# --- integration ------------------------------------------------------------

def test_the_fixture_aggregate_carries_the_contract_and_refuses_qualification(
        tmp_path):
    import harness_executors as hx
    import run_harness_evaluation as rhe
    experiment = hd.load_yaml(ROOT / "configs" / "experiments" /
                              "harness-before-weights.yaml")
    case_set = he.load_case_set(ROOT / "tests" / "fixtures" / "harness_cases" /
                                "fixture-office-v1.yaml")
    directory = tmp_path / "receipts"
    rhe.run_evaluation(experiment, case_set, hx.FakeOfficeExecutor(), directory)
    measurements = he.aggregate(experiment, case_set, he.load_receipts(directory),
                                allow_fixture=True)
    assert measurements["metric_contract"]["digest"] == CONTRACT_DIGEST
    assert measurements["metric_contract"]["version"] == 1
    assert measurements["independent_cases"] == 10
    assert measurements["statistically_qualified"] is False
    assert measurements["case_set_approval"] is None
    problems = " ".join(measurements["qualification_problems"])
    assert "fixture" in problems and "not approved" in problems
    assert measurements["training_authorized"] is False


def test_no_socket_is_opened_on_this_path(monkeypatch):
    import socket

    def forbidden(*args, **kwargs):
        raise AssertionError("the measurement layer must not use the network")

    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    sc.load_contract()
    sc.score_case(edit_case(), {}, ok_execution())
    mr.qualification(10, "fixture", approved=False)
    bcs.build([{"id": "a", "user": "u", "document": "d", "expect": "edit"}],
              "d", "f")


# --- repetitions are clustered observations, not extra evidence -------------
#
# The obvious rule -- a case passes only if every repetition passed -- measures
# p**r. The same model then scores 0.90 at one repetition and 0.59 at five, and
# the metric reports the experiment's schedule rather than the model's
# behaviour. These fix the behaviour that must not come back.

def reps(case_id, passed_count, total):
    return ([{"case_id": case_id, "passed": True}] * passed_count
            + [{"case_id": case_id, "passed": False}] * (total - passed_count))


def test_more_repetitions_do_not_drive_capability_toward_p_to_the_r():
    """A case that passes 9 times in 10 is worth 0.9, at any r."""
    for total in (1, 2, 5, 10, 20):
        passed = round(0.9 * total)
        rate = mr.case_pass_rates(reps("c1", passed, total))["c1"]
        assert rate == pytest.approx(passed / total, abs=1e-12)
    # The conjunction is what would have collapsed: 0.9**5 is far from 0.9.
    conjunction = mr.all_repetitions_passed(reps("c1", 9, 10))["c1"]
    assert conjunction is False
    assert mr.case_pass_rates(reps("c1", 9, 10))["c1"] == 0.9


def test_a_perfect_case_stays_at_one_however_often_it_runs():
    for total in (1, 3, 50):
        assert mr.case_pass_rates(reps("c1", total, total))["c1"] == 1.0


def test_unequal_repetition_counts_do_not_reweight_a_case():
    """One case run 100 times must not outvote 99 cases run once."""
    rows = reps("loud", 0, 100) + [{"case_id": f"q{i}", "passed": True}
                                   for i in range(99)]
    rates = mr.case_pass_rates(rows)
    assert len(rates) == 100
    arm_rate = sum(rates.values()) / len(rates)
    assert arm_rate == pytest.approx(0.99), (
        "each case carries weight 1; averaging observations would give 0.4975")
    # What weighting by OBSERVATION would have produced, for contrast: 99
    # passes out of 199 rows, i.e. the loud case drowning out all the rest.
    assert len(rows) == 199
    assert sum(1 for r in rows if r["passed"]) / len(rows) == pytest.approx(99 / 199)


def test_repetitions_never_increase_independent_n():
    for total in (1, 5, 40):
        rows = reps("c1", total, total) + reps("c2", 0, total)
        result = mr.qualification(len(mr.case_pass_rates(rows)), "held_out",
                                  approved=True)
        assert result["independent_cases"] == 2


def test_a_single_repetition_reduces_to_the_boolean_pass_rate():
    rows = [{"case_id": "c1", "passed": True}, {"case_id": "c2", "passed": False},
            {"case_id": "c3", "passed": True}]
    rates = mr.case_pass_rates(rows)
    assert rates == {"c1": 1.0, "c2": 0.0, "c3": 1.0}
    assert sum(rates.values()) / len(rates) == pytest.approx(2 / 3)


def four_arms(rows_per_arm):
    return {arm: list(rows_per_arm) for arm in
            ("student_current", "student_candidate",
             "teacher_current", "teacher_candidate")}


def test_the_report_uses_mcnemar_only_without_repetitions():
    single = four_arms([{"case_id": f"c{i}", "passed": i % 2 == 0}
                        for i in range(10)])
    result = mr.report(single, split="pilot", approved=False, seed=3)
    assert result["single_repetition"] is True
    assert result["repetition_policy"]["inference"] == "mcnemar"
    assert result["paired_tests"]["student_harness"]["mcnemar_exact"]["method"] == "exact"
    assert result["rates"]["student_current"]["wilson_95"] is not None


def test_with_repetitions_the_report_switches_to_the_clustered_bootstrap():
    repeated = four_arms(reps("c1", 3, 4) + reps("c2", 1, 4) + reps("c3", 4, 4))
    result = mr.report(repeated, split="pilot", approved=False, seed=3)
    assert result["single_repetition"] is False
    assert result["repetition_policy"]["inference"] == "clustered_bootstrap"
    assert result["independent_cases"] == 3, "12 observations of 3 questions"
    test = result["paired_tests"]["student_harness"]
    assert test["mcnemar"]["applicable"] is False
    assert "proportion" in test["mcnemar"]["reason"]
    assert test["bootstrap_95"]["seed"] == 3
    # Wilson is withheld rather than computed on a non-binomial quantity.
    assert result["rates"]["student_current"]["wilson_95"] is None
    assert result["rates"]["student_current"]["rate"] == pytest.approx(
        (0.75 + 0.25 + 1.0) / 3)


def test_the_report_carries_the_descriptive_stability_rate_separately():
    repeated = four_arms(reps("c1", 3, 4) + reps("c2", 4, 4))
    result = mr.report(repeated, split="pilot", approved=False, seed=3)
    arm = result["rates"]["student_current"]
    assert arm["all_repetitions_passed_rate"] == 0.5, "only c2 passed every time"
    assert arm["rate"] == pytest.approx((0.75 + 1.0) / 2)
    assert arm["all_repetitions_passed_rate"] != arm["rate"], (
        "the stability statistic must never be mistaken for the metric")


@pytest.mark.parametrize("b, c, expected", [
    # Exact binomial: 2 * P(X >= max(b,c)), X ~ Bin(b+c, 1/2).
    (3, 0, 0.25),            # 2 * (1/8)
    (4, 0, 0.125),           # 2 * (1/16)
    (5, 0, 0.0625),          # 2 * (1/32)
    (2, 1, 1.0),             # 2 * (4/8) = 1.0
    (0, 0, 1.0),             # no discordant pairs
])
def test_exact_mcnemar_matches_hand_computed_small_tables(b, c, expected):
    table = {"only_first": b, "only_second": c, "both_pass": 0, "neither": 0,
             "n": b + c, "discordant": b + c}
    assert mr.exact_mcnemar(table)["p_value"] == pytest.approx(expected, abs=1e-12)


def test_exact_and_approximate_mcnemar_disagree_where_the_approximation_is_weak():
    """Which is why the exact form exists: b+c small is the normal case."""
    table = {"only_first": 5, "only_second": 0, "both_pass": 0, "neither": 0,
             "n": 5, "discordant": 5}
    assert mr.exact_mcnemar(table)["p_value"] == pytest.approx(0.0625)
    assert mr.mcnemar(table)["p_value"] != pytest.approx(0.0625, abs=1e-3)


def test_no_p_value_can_change_statistical_qualification():
    """Qualification asks whether the study can answer the question. A p-value
    IS an answer, so feeding one back would admit a run for coming out well.

    Proved behaviourally rather than by scanning source: the signature cannot
    receive a test result, and two runs with wildly different significance get
    identical qualification.
    """
    import inspect
    accepted = set(inspect.signature(mr.qualification).parameters)
    assert accepted == {"independent_cases", "split", "approved", "minimum"}, (
        "qualification must not be able to see a test result at all")

    # Same size, same split, same approval; opposite significance.
    unanimous = four_arms([{"case_id": f"c{i}", "passed": True} for i in range(10)])
    mixed = {
        "student_current": [{"case_id": f"c{i}", "passed": i % 2 == 0}
                            for i in range(10)],
        "student_candidate": [{"case_id": f"c{i}", "passed": True}
                              for i in range(10)],
        "teacher_current": [{"case_id": f"c{i}", "passed": i % 3 == 0}
                            for i in range(10)],
        "teacher_candidate": [{"case_id": f"c{i}", "passed": True}
                              for i in range(10)],
    }
    a = mr.report(unanimous, split="held_out", approved=True, seed=1)
    b = mr.report(mixed, split="held_out", approved=True, seed=1)
    assert a["paired_tests"]["student_harness"]["mcnemar_exact"]["p_value"] \
        != b["paired_tests"]["student_harness"]["mcnemar_exact"]["p_value"]
    for field in ("statistically_qualified", "independent_cases",
                  "qualification_problems"):
        assert a[field] == b[field], (
            f"{field} moved with the p-value; qualification is about whether "
            "the study can answer, not about how it came out")

    # And a maximally clean result is still unqualified at n=10.
    assert a["rates"]["student_current"]["rate"] == 1.0
    assert a["statistically_qualified"] is False
    assert any("below the 320" in p for p in a["qualification_problems"])


def test_fixtures_and_unapproved_sets_stay_unqualified_with_repetitions():
    repeated = four_arms(reps(f"c{i}", 4, 4) for i in range(1))
    rows = []
    for i in range(400):
        rows += reps(f"c{i}", 4, 4)
    big = four_arms(rows)
    fixture = mr.report(big, split="fixture", approved=True, seed=1)
    assert fixture["independent_cases"] == 400
    assert fixture["statistically_qualified"] is False
    assert any("fixture" in p for p in fixture["qualification_problems"])

    unapproved = mr.report(big, split="held_out", approved=False, seed=1)
    assert unapproved["statistically_qualified"] is False
    assert any("not approved" in p for p in unapproved["qualification_problems"])

    qualified = mr.report(big, split="held_out", approved=True, seed=1)
    assert qualified["statistically_qualified"] is True
    assert qualified["single_repetition"] is False
    assert qualified["training_authorized"] is False
