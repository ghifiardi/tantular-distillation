"""GEPA -> harness-before-weights evidence (RSI MVP Phase 2, Task 2.2).

GEPA supplies one arm. It does not decide anything, and it may not invent a
guardrail measurement it never took. Both properties are tested here, along
with the acceptance criterion: the four-arm attribution is produced with
`student_candidate` populated from GEPA, and the verdict stays
`training_authorized: false`.

Offline and deterministic: fixtures only.
"""
from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import gepa_evidence as ge  # noqa: E402
import harness_distill as hd  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures" / "gepa"
EXPERIMENT = ROOT / "configs" / "experiments" / "harness-before-weights.yaml"


def load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


@pytest.fixture
def experiment():
    return yaml.safe_load(EXPERIMENT.read_text(encoding="utf-8"))


@pytest.fixture
def measurements():
    return load("measurements_other_arms.json")


# --- the acceptance criterion ----------------------------------------------


def test_four_arm_attribution_with_the_candidate_from_gepa(experiment, measurements):
    verdict = ge.evaluate_with_gepa(
        experiment, measurements, load("gepa_result_sufficient.json"))
    scores = verdict["scores"]
    assert scores["student_current"] == pytest.approx(0.62)
    assert scores["student_candidate"] == pytest.approx(0.91)
    assert scores["teacher_current"] == pytest.approx(0.93)
    assert scores["harness_gain"] == pytest.approx(0.91 - 0.62)
    assert scores["residual_model_gap"] == pytest.approx(0.93 - 0.91)
    assert verdict["candidate_arm_provenance"]["source"] == "gepa"


def test_the_verdict_still_forbids_training(experiment, measurements):
    for name in ("gepa_result_sufficient.json", "gepa_result_insufficient.json"):
        verdict = ge.evaluate_with_gepa(experiment, measurements, load(name))
        assert verdict["training_authorized"] is False


def test_a_sufficient_harness_is_not_a_weight_distillation_candidate(
        experiment, measurements):
    verdict = ge.evaluate_with_gepa(
        experiment, measurements, load("gepa_result_sufficient.json"))
    assert verdict["harness_optimization_sufficient"] is True
    assert verdict["weight_distillation_candidate"] is False
    assert any("closes the capability gap" in r for r in verdict["reasons"])


def test_an_insufficient_harness_leaves_a_residual_gap(experiment, measurements):
    verdict = ge.evaluate_with_gepa(
        experiment, measurements, load("gepa_result_insufficient.json"))
    assert verdict["harness_optimization_sufficient"] is False
    assert verdict["scores"]["residual_model_gap"] == pytest.approx(0.93 - 0.78)
    # Still not authorization: the gate names it a candidate, never a licence.
    assert verdict["training_authorized"] is False


def test_this_module_changes_nothing_the_gate_decides(experiment, measurements):
    """The verdict must equal what harness_distill.evaluate produces alone."""
    result = load("gepa_result_sufficient.json")
    merged = ge.merge_candidate_arm(
        measurements, result,
        metric=experiment["decision"]["capability_metric"],
        guardrails=tuple(experiment["decision"]["guardrails"].keys()))
    direct = hd.evaluate(experiment, merged)
    viage = ge.evaluate_with_gepa(experiment, measurements, result)
    viage.pop("candidate_arm_provenance")
    assert viage == direct


# --- the refusals -----------------------------------------------------------


def test_a_gepa_result_without_guardrail_measurements_is_refused(
        experiment, measurements):
    with pytest.raises(ge.GepaEvidenceError) as error:
        ge.evaluate_with_gepa(experiment, measurements,
                              load("gepa_result_no_guardrails.json"))
    message = str(error.value)
    assert "indonesian_voice" in message
    assert "evidence nobody" in message


def test_guardrail_values_are_never_borrowed_from_the_current_arm(
        experiment, measurements):
    """The failure mode the refusal exists to prevent.

    student_current passes both guardrails. If the bridge copied those across,
    the gate would read "no regression" for a candidate nobody measured.
    """
    result = load("gepa_result_no_guardrails.json")
    with pytest.raises(ge.GepaEvidenceError):
        ge.arm_from_gepa(result, metric="capability_pass_rate",
                         guardrails=("indonesian_voice", "edit_contract_output"))
    # And the current arm's values really were available to borrow.
    assert measurements["arms"]["student_current"]["indonesian_voice"] == 0.96


def test_a_measured_guardrail_regression_reaches_the_gate(experiment, measurements):
    verdict = ge.evaluate_with_gepa(
        experiment, measurements, load("gepa_result_guardrail_regression.json"))
    assert verdict["guardrail_failures"]
    assert any("indonesian_voice" in f for f in verdict["guardrail_failures"])
    assert verdict["harness_optimization_sufficient"] is False


def test_a_result_without_identity_cannot_populate_an_arm(experiment, measurements):
    with pytest.raises(ge.GepaEvidenceError) as error:
        ge.evaluate_with_gepa(experiment, measurements,
                              load("gepa_result_no_identity.json"))
    assert "harness_identity" in str(error.value)


def test_an_existing_candidate_arm_is_not_overwritten(experiment):
    with pytest.raises(ge.GepaEvidenceError) as error:
        ge.evaluate_with_gepa(experiment,
                              load("measurements_candidate_present.json"),
                              load("gepa_result_sufficient.json"))
    assert "refusing to overwrite" in str(error.value)


def test_a_result_claiming_training_authorization_is_refused(experiment,
                                                             measurements):
    result = load("gepa_result_sufficient.json")
    result["training_authorized"] = True
    with pytest.raises(ge.GepaEvidenceError) as error:
        ge.evaluate_with_gepa(experiment, measurements, result)
    assert "training_authorized" in str(error.value)


def test_a_fixture_result_cannot_populate_a_production_arm(
        experiment, measurements):
    result = load("gepa_result_sufficient.json")
    result["fixture"] = True
    with pytest.raises(ge.GepaEvidenceError) as error:
        ge.evaluate_with_gepa(experiment, measurements, result)
    assert "fixture" in str(error.value)
    assert "production experiment arm" in str(error.value)


def test_a_score_outside_zero_to_one_is_refused(experiment, measurements):
    result = load("gepa_result_sufficient.json")
    result["best"]["score"] = 91
    with pytest.raises(ge.GepaEvidenceError) as error:
        ge.evaluate_with_gepa(experiment, measurements, result)
    assert "rate in [0, 1]" in str(error.value)


def test_merging_does_not_mutate_the_callers_measurements(experiment, measurements):
    before = copy.deepcopy(measurements)
    ge.merge_candidate_arm(
        measurements, load("gepa_result_sufficient.json"),
        metric="capability_pass_rate",
        guardrails=("indonesian_voice", "edit_contract_output"))
    assert measurements == before


# --- CLI --------------------------------------------------------------------


def run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(ROOT / "src" / "gepa_evidence.py"), *args],
        capture_output=True, text=True, timeout=60)


def test_cli_emits_the_four_arm_attribution():
    result = run_cli("--experiment", str(EXPERIMENT),
                     "--measurements", str(FIXTURES / "measurements_other_arms.json"),
                     "--gepa", str(FIXTURES / "gepa_result_sufficient.json"))
    assert result.returncode == 0, result.stderr
    verdict = json.loads(result.stdout)
    assert verdict["scores"]["student_candidate"] == pytest.approx(0.91)
    assert verdict["training_authorized"] is False


def test_cli_refuses_a_result_without_guardrails_nonzero():
    result = run_cli("--experiment", str(EXPERIMENT),
                     "--measurements", str(FIXTURES / "measurements_other_arms.json"),
                     "--gepa", str(FIXTURES / "gepa_result_no_guardrails.json"))
    assert result.returncode != 0
    assert "GEPA EVIDENCE REFUSED" in result.stderr
