"""Separation and stability gate (RSI MVP Phase 1, Task 1.3).

The gate's job is to refuse. Most of this file is therefore about what it
refuses and why: a missing split, a missing identity, a teacher in a student
slot, a difference inside the harness's own re-run noise.

Offline and deterministic: fixture measurement files only, no endpoint is ever
contacted.
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
sys.path.insert(0, str(ROOT / "tests" / "fixtures"))

import eval_harness as eh  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures" / "eval"
EXPERIMENT = ROOT / "configs" / "experiments" / "harness-before-weights.yaml"


@pytest.fixture
def use_statistics_double(monkeypatch):
    """The repo's paired statistics — the real module when it is available.

    On a branch carrying `src/measurement_report.py` this yields the REAL
    module and patches nothing, so the gate is exercised against the
    statistics it will actually use in production. Only on a branch without it
    does this fall back to the stand-in under tests/fixtures/, which exists
    solely so the gate's own logic can be tested meanwhile.

    Delete the fallback (and the double) once measurement_report is on every
    branch that runs this suite.
    """
    try:
        import measurement_report as real  # noqa: PLC0415
        return real
    except ImportError:
        import measurement_report_double as double  # noqa: PLC0415
        monkeypatch.setitem(sys.modules, "measurement_report", double)
        return double


@pytest.fixture
def experiment():
    return yaml.safe_load(EXPERIMENT.read_text(encoding="utf-8"))


def load(name: str) -> dict:
    return eh.load_measurement(FIXTURES / name)


def nine() -> dict:
    return load("measurement_9b.json")


def four() -> dict:
    return load("measurement_4b.json")


# --- the statistics dependency ---------------------------------------------


def test_the_gate_refuses_when_the_paired_statistics_are_unavailable(
        monkeypatch, experiment):
    """measurement_report is on origin/main and absent here. The gate must say
    so, not silently substitute a weaker test."""
    monkeypatch.setitem(sys.modules, "measurement_report", None)
    real_import = __import__

    def blocked(name, *args, **kwargs):
        if name == "measurement_report":
            raise ImportError("no module named measurement_report")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", blocked)
    with pytest.raises(eh.EvalHarnessError) as error:
        eh.separation_gate(experiment, nine(), four())
    assert "measurement_report" in str(error.value)
    assert "origin/main" in str(error.value)


def test_the_statistics_agree_with_hand_computed_values(use_statistics_double):
    a = {"c1": True, "c2": True, "c3": False, "c4": False}
    b = {"c1": True, "c2": False, "c3": True, "c4": False}
    table = use_statistics_double.paired_table(a, b)
    assert table == {"both_pass": 1, "only_first": 1, "only_second": 1,
                     "neither": 1, "n": 4, "discordant": 2}
    # b == c, so the corrected statistic is 0 and there is no evidence of a
    # difference. If the real module ever disagrees, this test is the tripwire.
    assert use_statistics_double.mcnemar(table)["statistic"] == 0.0


def test_unpaired_cases_are_refused_rather_than_scored_as_failures(
        use_statistics_double):
    with pytest.raises(Exception) as error:
        use_statistics_double.paired_table({"c1": True}, {"c2": True})
    assert "unpaired" in str(error.value) or "only one arm" in str(error.value)


# --- validation refusals ----------------------------------------------------


def test_a_missing_split_fails_the_gate_with_a_specific_message(
        use_statistics_double, experiment):
    with pytest.raises(eh.EvalHarnessError) as error:
        eh.separation_gate(experiment, load("measurement_9b_missing_split.json"),
                           four())
    assert "tool_use" in str(error.value)
    assert "missing" in str(error.value)


def test_a_missing_model_identity_is_refused(use_statistics_double, experiment):
    with pytest.raises(eh.EvalHarnessError) as error:
        eh.separation_gate(experiment, load("measurement_no_identity.json"), four())
    assert "model_identity" in str(error.value)


def test_a_missing_endpoint_is_refused(use_statistics_double, experiment):
    broken = nine()
    broken["model_identity"].pop("endpoint")
    with pytest.raises(eh.EvalHarnessError) as error:
        eh.separation_gate(experiment, broken, four())
    assert "endpoint" in str(error.value)


def test_a_teacher_in_a_student_slot_is_refused(use_statistics_double, experiment):
    with pytest.raises(eh.EvalHarnessError) as error:
        eh.separation_gate(
            experiment, load("measurement_teacher_in_student_slot.json"), four())
    assert "teacher" in str(error.value).lower()


def test_the_same_model_in_both_arms_is_refused(use_statistics_double, experiment):
    with pytest.raises(eh.EvalHarnessError) as error:
        eh.separation_gate(experiment, nine(), nine())
    assert "same model identity" in str(error.value)


def test_non_boolean_per_item_results_are_refused(use_statistics_double, experiment):
    broken = nine()
    broken["splits"]["reasoning"]["per_item"]["reasoning::000"] = "yes"
    with pytest.raises(eh.EvalHarnessError) as error:
        eh.separation_gate(experiment, broken, four())
    assert "boolean" in str(error.value)


def test_an_unknown_split_is_refused(use_statistics_double, experiment):
    broken = nine()
    broken["splits"]["astrologi"] = broken["splits"]["reasoning"]
    with pytest.raises(eh.EvalHarnessError) as error:
        eh.separation_gate(experiment, broken, four())
    assert "astrologi" in str(error.value)


def test_a_missing_measurement_file_is_refused():
    with pytest.raises(eh.EvalHarnessError):
        eh.load_measurement(FIXTURES / "does_not_exist.json")


def test_a_decision_without_a_noise_floor_is_refused(use_statistics_double,
                                                     experiment):
    broken = copy.deepcopy(experiment)
    broken["decision"].pop("noise_floor")
    with pytest.raises(eh.EvalHarnessError) as error:
        eh.separation_gate(broken, nine(), four())
    assert "noise_floor" in str(error.value)


# --- the decision -----------------------------------------------------------


def test_clear_separation_passes_and_reports_structure(use_statistics_double,
                                                       experiment):
    verdict = eh.separation_gate(experiment, nine(), four())
    assert verdict["passed"] is True
    assert verdict["separated_on_every_split"] is True
    assert verdict["stable_on_every_split"] is True
    assert set(verdict["splits"]) == set(eh.gs.SPLITS)
    row = verdict["splits"]["reasoning"]
    assert row["delta"] == pytest.approx(0.20, abs=1e-9)
    assert row["separated"] is True
    assert "paired" in row and "mcnemar" in row


def test_a_within_noise_difference_fails_the_gate(use_statistics_double,
                                                  experiment):
    verdict = eh.separation_gate(
        experiment, load("measurement_9b_within_noise.json"), four())
    assert verdict["passed"] is False
    assert verdict["separated_on_every_split"] is False
    row = verdict["splits"]["reasoning"]
    assert row["separated"] is False
    assert any("noise floor" in r or "declared effect" in r for r in row["reasons"])


def test_unstable_reruns_fail_the_gate_even_when_separated(use_statistics_double,
                                                           experiment):
    verdict = eh.separation_gate(
        experiment, load("measurement_9b_unstable.json"), four())
    assert verdict["separated_on_every_split"] is True
    assert verdict["stable_on_every_split"] is False
    assert verdict["passed"] is False
    assert any("variation" in r for r in verdict["splits"]["reasoning"]["reasons"])


def test_stability_is_unproven_without_repeated_runs(use_statistics_double,
                                                     experiment):
    single = nine()
    for split in single["splits"].values():
        split.pop("runs")
    verdict = eh.separation_gate(experiment, single, four())
    row = verdict["splits"]["reasoning"]
    assert row["stable"] is None
    assert any("unproven" in r for r in row["reasons"])
    # Unproven is not proven. The gate must not pass on absent evidence.
    assert verdict["passed"] is False


def test_separation_must_hold_on_every_split_not_on_average(use_statistics_double,
                                                            experiment):
    mixed = nine()
    # Make one split indistinguishable while the others stay far ahead.
    mixed["splits"]["knowledge"]["per_item"] = dict(
        four()["splits"]["knowledge"]["per_item"])
    verdict = eh.separation_gate(experiment, mixed, four())
    assert verdict["splits"]["reasoning"]["separated"] is True
    assert verdict["splits"]["knowledge"]["separated"] is False
    assert verdict["passed"] is False


def test_every_verdict_says_training_is_not_authorized(use_statistics_double,
                                                       experiment):
    verdict = eh.separation_gate(experiment, nine(), four())
    assert verdict["training_authorized"] is False
    assert verdict["measured"] is True


def test_the_experiment_file_still_forbids_training(experiment):
    assert experiment["training_authorized"] is False


# --- CLI --------------------------------------------------------------------


def run_cli(*args: str) -> subprocess.CompletedProcess:
    env = {"PYTHONPATH": str(ROOT / "tests" / "fixtures")}
    import os
    full = {**os.environ, **env}
    return subprocess.run(
        [sys.executable, str(ROOT / "src" / "eval_harness.py"), *args],
        capture_output=True, text=True, timeout=120, env=full)


def test_cli_help_lists_the_separation_gate():
    result = run_cli("separation-gate", "--help")
    assert result.returncode == 0
    assert "--stronger" in result.stdout and "--weaker" in result.stdout


def test_cli_refuses_a_missing_split_nonzero():
    result = run_cli("separation-gate",
                     "--stronger", str(FIXTURES / "measurement_9b_missing_split.json"),
                     "--weaker", str(FIXTURES / "measurement_4b.json"))
    assert result.returncode != 0
    assert "SEPARATION GATE REFUSED" in result.stderr


def test_cli_contacts_no_endpoint(use_statistics_double, experiment):
    # Structural, not behavioural: the module must not import an HTTP client.
    source = (ROOT / "src" / "eval_harness.py").read_text(encoding="utf-8")
    for banned in ("import httpx", "import requests", "urllib.request"):
        assert banned not in source, f"{banned} would make this gate non-offline"
