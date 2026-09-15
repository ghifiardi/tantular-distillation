"""Evaluation-controller tests: a score may only arrive as measured evidence.

src/harness_distill.py accepts measurements as plain JSON, so a typed-in pass
rate and a measured one look identical to it. Everything here exists to make
that difference real: the arms must be complete, the case set identical across
them, the receipts bound to a model revision and a harness digest, and the
arithmetic derived from receipts rather than supplied alongside them.

NOTHING HERE touches the network, loads a model, drives Office, or trains. The
executor under test replays declarations from a fixture case set, and one test
asserts that structurally by breaking the socket module for the whole pipeline.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import harness_distill as hd
import harness_eval as he
import harness_executors as hx
import run_harness_evaluation as rhe

EXPERIMENT = ROOT / "configs" / "experiments" / "harness-before-weights.yaml"
FIXTURE_CASES = ROOT / "tests" / "fixtures" / "harness_cases" / "fixture-office-v1.yaml"


@pytest.fixture
def experiment():
    return hd.load_yaml(EXPERIMENT)


@pytest.fixture
def case_set():
    return he.load_case_set(FIXTURE_CASES)


def run_pipeline(experiment, case_set, tmp_path, executor=None, **kwargs):
    """Execute every arm x case with the fake executor and aggregate."""
    executor = executor or hx.FakeOfficeExecutor()
    directory = tmp_path / "receipts"
    rhe.run_evaluation(experiment, case_set, executor, directory)
    receipts = he.load_receipts(directory)
    return receipts, he.aggregate(experiment, case_set, receipts,
                                  allow_fixture=True, **kwargs)


# --- 1 & 2: the fourth arm is load-bearing ----------------------------------

def test_all_four_arms_are_required(experiment):
    assert hd.REQUIRED_ARMS == ("student_current", "student_candidate",
                                "teacher_current", "teacher_candidate")
    experiment["arms"] = ["student_current", "student_candidate",
                          "teacher_current"]
    with pytest.raises(hd.HarnessPlanError, match="teacher_candidate"):
        hd.build_plan(experiment)


def test_evaluate_refuses_measurements_without_the_teacher_candidate_arm(
        experiment, case_set, tmp_path):
    _, measurements = run_pipeline(experiment, case_set, tmp_path)
    del measurements["arms"]["teacher_candidate"]
    with pytest.raises(hd.HarnessPlanError, match="teacher_candidate"):
        hd.evaluate(experiment, measurements)


def test_teacher_candidate_changes_the_reported_interaction(
        experiment, case_set, tmp_path):
    """The whole point of the fourth cell: the same student numbers with a
    different teacher_candidate must not produce the same finding."""
    _, measurements = run_pipeline(experiment, case_set, tmp_path)
    baseline = hd.evaluate(experiment, measurements)
    assert baseline["interaction"]["teacher_harness_gain"] == pytest.approx(0.0)
    assert baseline["interaction"]["interaction"] == pytest.approx(-0.2)

    lifted = copy.deepcopy(measurements)
    lifted["arms"]["teacher_current"]["capability_pass_rate"] = 0.9
    moved = hd.evaluate(experiment, lifted)
    assert moved["interaction"]["teacher_harness_gain"] == pytest.approx(0.1)
    assert moved["interaction"]["interaction"] == pytest.approx(-0.1)
    assert moved["interaction"] != baseline["interaction"]


def test_every_arm_reports_every_configured_metric(experiment, case_set, tmp_path):
    _, measurements = run_pipeline(experiment, case_set, tmp_path)
    expected = {"capability_pass_rate", "indonesian_voice", "edit_contract_output"}
    for arm in hd.REQUIRED_ARMS:
        assert set(measurements["arms"][arm]) == expected, arm
    # A guardrail missing from a TEACHER arm used to be invisible: evaluate()
    # only read guardrails on the student side.
    del measurements["arms"]["teacher_candidate"]["indonesian_voice"]
    with pytest.raises(hd.HarnessPlanError,
                       match="teacher_candidate.indonesian_voice"):
        hd.evaluate(experiment, measurements)


# --- 3: one case set, identically, across all four arms ---------------------

def test_all_arms_execute_the_same_case_ids_and_digest(
        experiment, case_set, tmp_path):
    receipts, measurements = run_pipeline(experiment, case_set, tmp_path)
    digest = he.case_set_digest(case_set)
    assert {r["case_set_digest"] for r in receipts} == {digest}
    ids = set(he.case_ids(case_set))
    for arm in hd.REQUIRED_ARMS:
        assert {r["case_id"] for r in receipts if r["arm"] == arm} == ids
    assert measurements["case_set"]["digest"] == digest


def test_an_arm_answering_a_different_case_set_is_refused(
        experiment, case_set, tmp_path):
    receipts, _ = run_pipeline(experiment, case_set, tmp_path)
    for receipt in receipts:
        if receipt["arm"] == "teacher_candidate":
            receipt["case_set_digest"] = "f" * 64
    with pytest.raises(he.HarnessEvalError, match="did not answer the same"):
        he.aggregate(experiment, case_set, receipts, allow_fixture=True)


def test_an_empty_case_set_is_refused(case_set):
    case_set["cases"] = []
    with pytest.raises(he.HarnessEvalError, match="non-empty cases list"):
        he.validate_case_set(case_set)


def test_duplicate_case_ids_are_refused(case_set):
    case_set["cases"].append(copy.deepcopy(case_set["cases"][0]))
    with pytest.raises(he.HarnessEvalError, match="duplicate case_id"):
        he.validate_case_set(case_set)


def test_a_mutable_case_reference_is_refused(case_set):
    case = case_set["cases"][0]
    case.pop("request")
    case["request_ref"] = {"uri": "s3://bucket/case-1.json", "sha256": "nope"}
    with pytest.raises(he.HarnessEvalError, match="64 lowercase hex"):
        he.validate_case_set(case_set)


def test_a_fixture_set_may_not_declare_itself_approved(case_set):
    case_set["approved"] = True
    with pytest.raises(he.HarnessEvalError, match="fixture split cannot be approved"):
        he.validate_case_set(case_set)


# --- 4 & 5: identity is bound into every receipt ----------------------------

def test_receipts_bind_model_and_harness_identity(experiment, case_set, tmp_path):
    receipts, _ = run_pipeline(experiment, case_set, tmp_path)
    student = hd.load_yaml(ROOT / "configs" / "models" / "qwen35-9b-instruct.yaml")
    teacher = hd.load_yaml(ROOT / "configs" / "models" / "muse-glimmer-30b.yaml")
    current = hd.canonical_digest(hd.load_harness("tantular-office-current"))
    candidate = hd.canonical_digest(hd.load_harness("tantular-office-candidate"))

    by_arm = {}
    for receipt in receipts:
        by_arm.setdefault(receipt["arm"], []).append(receipt)

    for arm, (model, digest) in {
        "student_current": (student, current),
        "student_candidate": (student, candidate),
        "teacher_current": (teacher, current),
        "teacher_candidate": (teacher, candidate),
    }.items():
        for receipt in by_arm[arm]:
            assert receipt["model_id"] == model["model_id"]
            assert receipt["model_revision"] == model["revision"]
            assert receipt["harness_digest"] == digest


def test_receipt_provenance_is_the_existing_canonical_block(
        experiment, case_set, tmp_path):
    """One source of truth: the receipt must carry harness_provenance() output,
    not a second implementation of the same idea."""
    receipts, _ = run_pipeline(experiment, case_set, tmp_path)
    receipt = next(r for r in receipts if r["arm"] == "student_candidate")
    expected = hd.harness_provenance(
        hd.load_harness("tantular-office-candidate"),
        execution_model_registry="qwen35-9b-instruct")
    assert receipt["harness_provenance"] == expected
    assert receipt["harness_digest"] == expected["digest"]
    assert expected["digest"] == hd.canonical_digest(
        hd.load_harness("tantular-office-candidate"))


def test_a_receipt_whose_fields_contradict_its_provenance_is_refused(
        experiment, case_set, tmp_path):
    receipts, _ = run_pipeline(experiment, case_set, tmp_path)
    receipts[0]["harness_digest"] = "a" * 64
    with pytest.raises(he.HarnessEvalError, match="disagrees with"):
        he.validate_receipt(receipts[0])


def test_an_unpinned_model_cannot_be_evaluated(experiment, monkeypatch):
    """A receipt that cannot name the exact checkpoint attributes a score to a
    moving target."""
    real = hd.load_yaml

    def unpinned(path):
        spec = real(path)
        if path.name.startswith("qwen35-9b"):
            spec["revision"] = "REPLACE_WITH_PINNED_HUB_COMMIT"
        return spec

    monkeypatch.setattr(hd, "load_yaml", unpinned)
    monkeypatch.setattr(rhe.hd, "load_yaml", unpinned)
    with pytest.raises(he.HarnessEvalError, match="no pinned revision"):
        rhe.resolve_arms(experiment)


# --- 6: receipts that cannot be counted -------------------------------------

def test_a_missing_arm_case_pair_is_refused(experiment, case_set, tmp_path):
    receipts, _ = run_pipeline(experiment, case_set, tmp_path)
    dropped = [r for r in receipts
               if not (r["arm"] == "teacher_current"
                       and r["case_id"] == "fixture-004")]
    with pytest.raises(he.HarnessEvalError, match="no receipt"):
        he.aggregate(experiment, case_set, dropped, allow_fixture=True)


def test_a_duplicate_receipt_is_refused(experiment, case_set, tmp_path):
    receipts, _ = run_pipeline(experiment, case_set, tmp_path)
    with pytest.raises(he.HarnessEvalError, match="duplicate receipt"):
        he.aggregate(experiment, case_set, receipts + [receipts[0]],
                     allow_fixture=True)


def test_a_malformed_receipt_is_refused(experiment, case_set, tmp_path):
    receipts, _ = run_pipeline(experiment, case_set, tmp_path)
    del receipts[3]["harness_provenance"]
    with pytest.raises(he.HarnessEvalError, match="missing fields"):
        he.aggregate(experiment, case_set, receipts, allow_fixture=True)


def test_a_receipt_edited_after_writing_is_refused(experiment, case_set, tmp_path):
    directory = tmp_path / "receipts"
    rhe.run_evaluation(experiment, case_set, hx.FakeOfficeExecutor(), directory)
    path = directory / "student_current__fixture-001.json"
    payload = json.loads(path.read_text())
    # Flip a real outcome: the receipt still validates, so only the digest it
    # carries can catch the edit.
    assert payload["scores"]["capability_pass_rate"] is True
    payload["scores"]["capability_pass_rate"] = False
    path.write_text(json.dumps(payload))
    with pytest.raises(he.HarnessEvalError, match="modified after it was written"):
        he.load_receipts(directory)


def test_an_ok_receipt_missing_a_declared_scorer_is_refused(
        experiment, case_set, tmp_path):
    receipts, _ = run_pipeline(experiment, case_set, tmp_path)
    receipts[0]["scores"].pop("indonesian_voice")
    with pytest.raises(he.HarnessEvalError, match="records no result"):
        he.aggregate(experiment, case_set, receipts, allow_fixture=True)


# --- 7: prompt identity for a real measurement ------------------------------

def test_unverified_prompt_identity_is_refused_for_a_real_measurement(
        experiment, case_set, tmp_path):
    receipts, _ = run_pipeline(experiment, case_set, tmp_path)
    for receipt in receipts:
        receipt["prompt_verified"] = False
        receipt["prompt_sha256"] = None
        receipt["harness_provenance"]["prompt_verified"] = False
        receipt["harness_provenance"]["prompt_sha256"] = None
        receipt["executor"]["produces_real_measurements"] = True
    approved = copy.deepcopy(case_set)
    approved["approved"] = True
    approved["split"] = "held_out"
    for receipt in receipts:
        receipt["case_set_digest"] = he.case_set_digest(approved)
    with pytest.raises(he.HarnessEvalError, match="unverified prompt identity"):
        he.aggregate(experiment, approved, receipts)


def test_an_unapproved_case_set_cannot_back_a_real_measurement(
        experiment, case_set, tmp_path):
    receipts, _ = run_pipeline(experiment, case_set, tmp_path)
    for receipt in receipts:
        receipt["executor"]["produces_real_measurements"] = True
    with pytest.raises(he.HarnessEvalError, match="not approved"):
        he.aggregate(experiment, case_set, receipts)


# --- 8, 9, 10: harness policy is enforced by the controller -----------------

def _mutate_fixture(case_set, case_id, arm, **changes):
    case = next(c for c in case_set["cases"] if c["case_id"] == case_id)
    case["fixture"]["arms"][arm].update(changes)
    return case_set


def test_a_state_changing_tool_without_approval_is_refused(
        experiment, case_set, tmp_path):
    _mutate_fixture(case_set, "fixture-001", "student_current",
                    tool_calls=[{"tool": "office_edit", "approved": False}])
    receipts, _ = run_pipeline(experiment, case_set, tmp_path)
    receipt = next(r for r in receipts if r["arm"] == "student_current"
                   and r["case_id"] == "fixture-001")
    assert receipt["status"] == he.STATUS_REFUSED
    assert receipt["termination_reason"] == hx.TERMINATION_APPROVAL_MISSING
    assert receipt["scores"] == {}          # a refusal scores nothing


def test_a_tool_outside_the_allowlist_is_refused(experiment, case_set, tmp_path):
    _mutate_fixture(case_set, "fixture-002", "student_candidate",
                    tool_calls=[{"tool": "shell_exec", "approved": True}])
    receipts, _ = run_pipeline(experiment, case_set, tmp_path)
    receipt = next(r for r in receipts if r["arm"] == "student_candidate"
                   and r["case_id"] == "fixture-002")
    assert receipt["status"] == he.STATUS_REFUSED
    assert receipt["termination_reason"] == hx.TERMINATION_TOOL_NOT_ALLOWED


def test_the_step_budget_is_enforced_per_harness(experiment, case_set, tmp_path):
    """The current harness allows 4 steps and the candidate 8. The SAME declared
    execution must therefore be refused on one arm and accepted on the other."""
    for arm in ("student_current", "student_candidate"):
        _mutate_fixture(case_set, "fixture-003", arm, steps=6)
    receipts, _ = run_pipeline(experiment, case_set, tmp_path)
    by = {(r["arm"], r["case_id"]): r for r in receipts}
    refused = by[("student_current", "fixture-003")]
    allowed = by[("student_candidate", "fixture-003")]
    assert refused["status"] == he.STATUS_REFUSED
    assert refused["termination_reason"] == hx.TERMINATION_BUDGET_STEPS
    assert refused["budgets"]["max_steps"] == 4
    assert allowed["status"] == he.STATUS_OK
    assert allowed["budgets"]["max_steps"] == 8


def test_the_wall_clock_budget_is_enforced(experiment, case_set, tmp_path):
    _mutate_fixture(case_set, "fixture-005", "student_current", wall_seconds=400)
    receipts, _ = run_pipeline(experiment, case_set, tmp_path)
    receipt = next(r for r in receipts if r["arm"] == "student_current"
                   and r["case_id"] == "fixture-005")
    assert receipt["termination_reason"] == hx.TERMINATION_BUDGET_WALL


def test_the_repair_budget_is_enforced(experiment, case_set, tmp_path):
    """The current harness declares repair_attempts: 0."""
    _mutate_fixture(case_set, "fixture-006", "student_current", repair_attempts=1)
    receipts, _ = run_pipeline(experiment, case_set, tmp_path)
    receipt = next(r for r in receipts if r["arm"] == "student_current"
                   and r["case_id"] == "fixture-006")
    assert receipt["status"] == he.STATUS_REFUSED
    assert receipt["termination_reason"] == hx.TERMINATION_REPAIR_LIMIT


def test_verifier_and_repair_activity_is_recorded(experiment, case_set, tmp_path):
    receipts, _ = run_pipeline(experiment, case_set, tmp_path)
    receipt = next(r for r in receipts if r["arm"] == "student_candidate"
                   and r["case_id"] == "fixture-001")
    assert [c["check"] for c in receipt["before_action"]] == ["request_schema"]
    assert [c["check"] for c in receipt["after_action"]] == \
        ["edit_contract", "faithful_edit"]
    assert receipt["repair_attempts"] == 1       # candidate harness allows one
    assert receipt["tools_offered"] == ["office_read", "office_edit"]
    assert receipt["approvals"] == [
        {"tool": "office_edit", "granted": True, "by": "fixture"}]


# --- 11: a failure is durable ----------------------------------------------

class ExplodingExecutor:
    """Raises on one case; the controller still owes a record."""

    def identity(self):
        return {"name": "exploding", "version": "0", "kind": "fake",
                "produces_real_measurements": False}

    def execute(self, request):
        if request.case_id == "fixture-002" and request.arm == "teacher_current":
            raise RuntimeError("adapter died mid-case")
        return hx.FakeOfficeExecutor().execute(request)


def test_executor_failure_leaves_a_failure_receipt(experiment, case_set, tmp_path):
    directory = tmp_path / "receipts"
    summary = rhe.run_evaluation(experiment, case_set, ExplodingExecutor(),
                                 directory)
    assert summary["status_counts"]["error"] == 1
    path = directory / "teacher_current__fixture-002.json"
    assert path.is_file(), "a crashed execution must not vanish"
    receipt = json.loads(path.read_text())
    assert receipt["status"] == he.STATUS_ERROR
    assert receipt["termination_reason"] == hx.TERMINATION_EXECUTOR_ERROR
    assert "adapter died mid-case" in " ".join(receipt["problems"])
    assert receipt["scores"] == {}
    assert receipt["training_authorized"] is False


def test_an_errored_case_counts_against_the_denominator(
        experiment, case_set, tmp_path):
    """Dropping a failed case would let a flaky arm score 1.0 on the cases that
    happened to run."""
    directory = tmp_path / "receipts"
    rhe.run_evaluation(experiment, case_set, ExplodingExecutor(), directory)
    measurements = he.aggregate(experiment, case_set,
                                he.load_receipts(directory), allow_fixture=True)
    metrics = measurements["evidence"]["teacher_current"]["metrics"]
    assert metrics["capability_pass_rate"]["denominator"] == 10
    assert metrics["capability_pass_rate"]["numerator"] == 9
    assert metrics["capability_pass_rate"]["error"] == 1
    assert measurements["arms"]["teacher_current"]["capability_pass_rate"] == 0.9


# --- 12 & 13: a replay is not a measurement ---------------------------------

def test_fixture_receipts_cannot_be_aggregated_as_a_real_measurement(
        experiment, case_set, tmp_path):
    directory = tmp_path / "receipts"
    rhe.run_evaluation(experiment, case_set, hx.FakeOfficeExecutor(), directory)
    receipts = he.load_receipts(directory)
    with pytest.raises(he.HarnessEvalError, match="produces_real_measurements"):
        he.aggregate(experiment, case_set, receipts)


def test_a_fixture_artifact_labels_itself(experiment, case_set, tmp_path):
    _, measurements = run_pipeline(experiment, case_set, tmp_path)
    assert measurements["measurement_class"] == "fixture"
    assert measurements["produces_real_measurements"] is False
    assert measurements["case_set"]["approved"] is False


def test_real_and_fixture_receipts_cannot_be_mixed(experiment, case_set, tmp_path):
    receipts, _ = run_pipeline(experiment, case_set, tmp_path)
    receipts[0]["executor"]["produces_real_measurements"] = True
    with pytest.raises(he.HarnessEvalError, match="partly replayed"):
        he.aggregate(experiment, case_set, receipts, allow_fixture=True)


def test_scores_are_derived_from_receipts_not_accepted_alongside_them(
        experiment, case_set, tmp_path):
    """Editing a receipt's per-case outcome must move the aggregate; there is no
    channel for supplying a rate directly."""
    directory = tmp_path / "receipts"
    rhe.run_evaluation(experiment, case_set, hx.FakeOfficeExecutor(), directory)
    receipts = he.load_receipts(directory)
    before = he.aggregate(experiment, case_set, receipts, allow_fixture=True)
    assert before["arms"]["student_current"]["capability_pass_rate"] == 0.6

    for receipt in receipts:
        if receipt["arm"] == "student_current" and receipt["case_id"] == "fixture-007":
            receipt["scores"]["capability_pass_rate"] = True
    after = he.aggregate(experiment, case_set, receipts, allow_fixture=True)
    assert after["arms"]["student_current"]["capability_pass_rate"] == 0.7
    # aggregate() takes no score argument at all: the only inputs are the
    # experiment, the case set and the receipts.
    assert "scores" not in he.aggregate.__code__.co_varnames[
        :he.aggregate.__code__.co_argcount]


def test_the_receipt_set_digest_is_order_independent_but_content_sensitive(
        experiment, case_set, tmp_path):
    receipts, first = run_pipeline(experiment, case_set, tmp_path)
    shuffled = list(reversed(receipts))
    assert he.receipt_set_digest(shuffled) == first["receipt_set_digest"]
    shuffled[0] = copy.deepcopy(shuffled[0])
    shuffled[0]["scores"]["capability_pass_rate"] = \
        not shuffled[0]["scores"]["capability_pass_rate"]
    assert he.receipt_set_digest(shuffled) != first["receipt_set_digest"]


# --- 14: bounded scores -----------------------------------------------------

def test_a_per_case_score_must_be_a_boolean_not_a_rate(experiment, case_set,
                                                       tmp_path):
    receipts, _ = run_pipeline(experiment, case_set, tmp_path)
    receipts[0]["scores"]["capability_pass_rate"] = 1.4
    with pytest.raises(he.HarnessEvalError, match="boolean per-case outcome"):
        he.validate_receipt(receipts[0])


def test_aggregated_scores_stay_inside_the_unit_interval(
        experiment, case_set, tmp_path):
    _, measurements = run_pipeline(experiment, case_set, tmp_path)
    for arm, scores in measurements["arms"].items():
        for metric, value in scores.items():
            assert isinstance(value, float), (arm, metric)
            assert 0.0 <= value <= 1.0, (arm, metric, value)


def test_a_non_numeric_metric_is_refused_by_the_evaluator(experiment, case_set,
                                                          tmp_path):
    _, measurements = run_pipeline(experiment, case_set, tmp_path)
    measurements["arms"]["student_current"]["capability_pass_rate"] = "0.6"
    with pytest.raises(hd.HarnessPlanError, match="must be numeric"):
        hd.evaluate(experiment, measurements)


# --- 15: guardrails still block --------------------------------------------

def test_a_guardrail_regression_blocks_promotion_and_candidacy(
        experiment, case_set, tmp_path):
    """Two voice failures on the candidate arm: 0.8 is below the 0.95 minimum
    and is a regression against the current arm's 1.0."""
    for case_id in ("fixture-009", "fixture-010"):
        _mutate_fixture(case_set, case_id, "student_candidate",
                        scores={"capability_pass_rate": True,
                                "indonesian_voice": False,
                                "edit_contract_output": True})
    _, measurements = run_pipeline(experiment, case_set, tmp_path)
    assert measurements["arms"]["student_candidate"]["indonesian_voice"] == 0.8

    result = hd.evaluate(experiment, measurements)
    assert result["guardrail_failures"]
    assert result["harness_optimization_sufficient"] is False
    assert result["weight_distillation_candidate"] is False
    assert result["training_authorized"] is False


def test_the_configured_thresholds_are_unchanged(experiment):
    decision = experiment["decision"]
    assert decision["target"] == 0.90
    assert decision["minimum_residual_gap"] == 0.05
    assert decision["noise_floor"] == 0.025
    assert decision["guardrails"]["indonesian_voice"] == {
        "minimum": 0.95, "max_regression": 0.00}
    assert decision["guardrails"]["edit_contract_output"] == {
        "minimum": 0.90, "max_regression": 0.00}


# --- 16: training is never authorized --------------------------------------

def test_training_authorized_is_false_in_every_artifact(
        experiment, case_set, tmp_path):
    plan = rhe.plan_evaluation(experiment, case_set)
    receipts, measurements = run_pipeline(experiment, case_set, tmp_path)
    decision = hd.evaluate(experiment, measurements)
    assert plan["training_authorized"] is False
    assert measurements["training_authorized"] is False
    assert decision["training_authorized"] is False
    assert all(r["training_authorized"] is False for r in receipts)
    assert "training_authorized" not in str(decision.get("reasons", "")).lower() \
        or decision["training_authorized"] is False


def test_a_receipt_claiming_training_authorization_is_refused(
        experiment, case_set, tmp_path):
    receipts, _ = run_pipeline(experiment, case_set, tmp_path)
    receipts[0]["training_authorized"] = True
    with pytest.raises(he.HarnessEvalError, match="must be false"):
        he.validate_receipt(receipts[0])


# --- 17: no real execution is reachable from here ---------------------------

def test_the_real_office_executor_fails_closed(experiment, case_set):
    executor = hx.get_executor("office")
    assert executor.identity()["available"] is False
    resolved = rhe.resolve_arms(experiment)["student_current"]
    request = hx.ExecutionRequest(
        case=case_set["cases"][0], arm="student_current",
        model_registry=resolved["model_registry"],
        model_id=resolved["model_id"], model_revision=resolved["model_revision"],
        harness_name=resolved["harness_name"],
        harness_digest=resolved["harness_digest"],
        prompt_sha256=resolved["provenance"]["prompt_sha256"],
        prompt_verified=True, tool_policy={}, verification_policy={},
        budgets={"max_steps": 4, "max_wall_seconds": 300}, run_id="x")
    with pytest.raises(he.HarnessEvalError, match="no Office harness adapter"):
        executor.execute(request)


def test_the_cli_refuses_a_real_executor_without_the_explicit_flag(tmp_path):
    with pytest.raises(SystemExit) as excinfo:
        rhe.main(["run", str(EXPERIMENT), "--cases", str(FIXTURE_CASES),
                  "--executor", "office", "--output", str(tmp_path / "r")])
    assert excinfo.value.code == 2


def test_the_cli_refuses_to_dress_a_fixture_run_as_real(tmp_path):
    with pytest.raises(SystemExit) as excinfo:
        rhe.main(["run", str(EXPERIMENT), "--cases", str(FIXTURE_CASES),
                  "--executor", "fake", "--real",
                  "--output", str(tmp_path / "r")])
    assert excinfo.value.code == 2


def test_the_harnesses_still_refuse_generate_py_attribution():
    """This milestone must not have quietly opted the harnesses in."""
    for name in ("tantular-office-current", "tantular-office-candidate"):
        spec = hd.load_harness(name)
        assert spec["trace_generation"]["supported_by_generate_py"] is False
        assert hd.generation_support(spec), name


def test_the_whole_pipeline_runs_with_networking_broken(
        experiment, case_set, tmp_path, monkeypatch):
    """Structural proof rather than a promise: if anything here opened a socket,
    this test would fail."""
    import socket

    def forbidden(*args, **kwargs):
        raise AssertionError("the evaluation controller must not use the network")

    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    receipts, measurements = run_pipeline(experiment, case_set, tmp_path)
    assert len(receipts) == 40
    assert measurements["receipts"] == 40


# --- the CLI surface --------------------------------------------------------

def test_plan_writes_nothing(experiment, case_set, tmp_path, capsys):
    before = {p: p.stat().st_mtime_ns for p in ROOT.rglob("configs/**/*.yaml")}
    rhe.main(["plan", str(EXPERIMENT), "--cases", str(FIXTURE_CASES)])
    payload = json.loads(capsys.readouterr().out)
    assert payload["executions_planned"] == 40
    assert payload["real_execution_available"] is False
    assert payload["training_authorized"] is False
    after = {p: p.stat().st_mtime_ns for p in ROOT.rglob("configs/**/*.yaml")}
    assert before == after


def test_the_cli_round_trip_feeds_harness_distill(tmp_path, capsys):
    receipts_dir = tmp_path / "receipts"
    measurements = tmp_path / "measurements.json"
    rhe.main(["run", str(EXPERIMENT), "--cases", str(FIXTURE_CASES),
              "--executor", "fake", "--output", str(receipts_dir)])
    capsys.readouterr()
    rhe.main(["aggregate", str(EXPERIMENT), "--cases", str(FIXTURE_CASES),
              "--receipts", str(receipts_dir), "--output", str(measurements),
              "--allow-fixture"])
    capsys.readouterr()
    result = hd.evaluate(hd.load_yaml(EXPERIMENT),
                         json.loads(measurements.read_text()))
    assert result["weight_distillation_candidate"] is True
    assert result["harness_optimization_sufficient"] is False
    assert result["training_authorized"] is False
