"""Public-suite adapters (RSI MVP Phase 1, Task 1.4).

Report §G: these suites "are for reporting, not for steering the loop". The
tests therefore care about three things — that a supplied result is normalized,
that an absent one is simply absent rather than fabricated, and that nothing
here can reach a gate.

Offline: fixture files only. The adapters never run or download a suite.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import public_suites as ps  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures" / "public_suites"


# --- normalization ----------------------------------------------------------


def test_sea_helm_result_is_normalized():
    result = ps.normalize_sea_helm(
        json.loads((FIXTURES / "sea_helm.json").read_text(encoding="utf-8")),
        source="sea_helm.json")
    assert result["suite"] == "sea_helm"
    assert result["model"] == "fixture-model-9b"
    assert result["scale"] == "0-100"
    assert set(result["pillars"]) == set(ps.SEA_HELM_PILLARS)
    assert result["pillars"]["safety"] == pytest.approx(72.5)


def test_indommlu_result_is_normalized():
    result = ps.normalize_indommlu(
        json.loads((FIXTURES / "indommlu.json").read_text(encoding="utf-8")),
        source="indommlu.json")
    assert result["suite"] == "indommlu"
    assert len(result["subjects"]) == 5
    assert result["subjects"]["bahasa_indonesia"] == pytest.approx(63.5)


def test_a_derived_overall_says_it_is_derived():
    result = ps.normalize_indommlu(
        json.loads((FIXTURES / "indommlu.json").read_text(encoding="utf-8")))
    # The fixture declares no overall, so the adapter derives one and labels it.
    assert result["overall_origin"] == "derived_unweighted_mean"
    assert result["overall"] == pytest.approx((55.0 + 63.5 + 41.2 + 58.1 + 47.8) / 5)


def test_a_reported_overall_is_not_relabelled_as_derived():
    data = json.loads((FIXTURES / "indommlu.json").read_text(encoding="utf-8"))
    data["overall"] = 51.0
    result = ps.normalize_indommlu(data)
    assert result["overall_origin"] == "reported"
    assert result["overall"] == pytest.approx(51.0)


# --- refusals ---------------------------------------------------------------


def test_a_missing_scale_is_refused_rather_than_guessed():
    data = json.loads((FIXTURES / "sea_helm_no_scale.json").read_text(encoding="utf-8"))
    with pytest.raises(ps.PublicSuiteError) as error:
        ps.normalize_sea_helm(data)
    assert "scale" in str(error.value)


def test_an_unknown_pillar_is_refused_not_averaged_in():
    data = json.loads(
        (FIXTURES / "sea_helm_unknown_pillar.json").read_text(encoding="utf-8"))
    with pytest.raises(ps.PublicSuiteError) as error:
        ps.normalize_sea_helm(data)
    assert "astrologi" in str(error.value)


def test_a_malformed_file_refuses_rather_than_reporting_absent():
    # The distinction that matters: absent is fine, broken is not.
    with pytest.raises(ps.PublicSuiteError) as error:
        ps.load_public_suites({"sea_helm": FIXTURES / "malformed.json"})
    assert "invalid JSON" in str(error.value)


def test_a_missing_model_is_refused():
    with pytest.raises(ps.PublicSuiteError):
        ps.normalize_sea_helm({"scale": "0-100", "pillars": {"safety": 1}})


def test_a_non_numeric_score_is_refused():
    with pytest.raises(ps.PublicSuiteError) as error:
        ps.normalize_sea_helm({"model": "m", "scale": "0-1",
                               "pillars": {"safety": "tinggi"}})
    assert "numeric" in str(error.value)


def test_an_out_of_range_score_is_refused():
    with pytest.raises(ps.PublicSuiteError):
        ps.normalize_sea_helm({"model": "m", "scale": "0-100",
                               "pillars": {"safety": 1000}})


def test_an_unknown_suite_is_refused():
    with pytest.raises(ps.PublicSuiteError) as error:
        ps.load_public_suites({"helm_lite": FIXTURES / "sea_helm.json"})
    assert "unknown suite" in str(error.value)


def test_a_nonexistent_supplied_path_is_refused():
    with pytest.raises(ps.PublicSuiteError):
        ps.load_public_suites({"sea_helm": FIXTURES / "does_not_exist.json"})


# --- absence ----------------------------------------------------------------


def test_no_files_supplied_yields_no_results():
    assert ps.load_public_suites({}) == {}
    assert ps.load_public_suites({"sea_helm": None, "indommlu": None}) == {}


def test_an_absent_suite_is_omitted_from_the_report_not_fabricated():
    report = {"gate": "separation", "passed": True}
    attached = ps.attach_public_suites(report, {})
    assert "public_suites" not in attached
    # And nothing was invented in its place.
    assert attached == report


def test_only_the_supplied_suite_appears():
    suites = ps.load_public_suites({"indommlu": FIXTURES / "indommlu.json"})
    attached = ps.attach_public_suites({"gate": "separation"}, suites)
    assert set(attached["public_suites"]["results"]) == {"indommlu"}


# --- separation from the loop ----------------------------------------------


def test_every_result_declares_that_it_does_not_steer_the_loop():
    suites = ps.load_public_suites({
        "sea_helm": FIXTURES / "sea_helm.json",
        "indommlu": FIXTURES / "indommlu.json"})
    assert set(suites) == {"sea_helm", "indommlu"}
    for result in suites.values():
        assert result["steers_loop"] is False
        assert result["role"] == "reporting_and_calibration_only"


def test_the_attached_section_is_separated_and_labelled():
    suites = ps.load_public_suites({"sea_helm": FIXTURES / "sea_helm.json"})
    attached = ps.attach_public_suites({"gate": "separation", "passed": False},
                                       suites)
    section = attached["public_suites"]
    assert section["steers_loop"] is False
    assert "do not steer the loop" in section["note"]
    # The gate's own verdict is untouched.
    assert attached["passed"] is False


def test_attaching_does_not_mutate_the_callers_report():
    report = {"gate": "separation"}
    suites = ps.load_public_suites({"sea_helm": FIXTURES / "sea_helm.json"})
    ps.attach_public_suites(report, suites)
    assert "public_suites" not in report


def test_no_public_suite_score_reaches_a_gate_field():
    suites = ps.load_public_suites({"sea_helm": FIXTURES / "sea_helm.json"})
    attached = ps.attach_public_suites(
        {"gate": "separation", "passed": True, "splits": {}}, suites)
    # The section sits beside the verdict; it adds no split and no rate.
    assert attached["splits"] == {}
    assert set(attached) == {"gate", "passed", "splits", "public_suites"}


def test_the_module_runs_and_downloads_nothing():
    source = (ROOT / "src" / "public_suites.py").read_text(encoding="utf-8")
    for banned in ("import httpx", "import requests", "urllib.request",
                   "subprocess", "git clone", "pip install"):
        assert banned not in source, f"{banned} would violate reporting-only"


# --- CLI --------------------------------------------------------------------


def run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(ROOT / "src" / "public_suites.py"), *args],
        capture_output=True, text=True, timeout=60)


def test_cli_with_no_files_emits_a_report_without_the_section():
    result = run_cli()
    assert result.returncode == 0
    assert "public_suites" not in json.loads(result.stdout)


def test_cli_refuses_a_malformed_file_nonzero():
    result = run_cli("--sea-helm", str(FIXTURES / "malformed.json"))
    assert result.returncode != 0
    assert "PUBLIC SUITE REFUSED" in result.stderr


# --- scale bounds are enforced per declared scale ---------------------------
# Checking against the widest possible range would let 68.41 through a file
# declaring `scale: 0-1` — the exact hundredfold confusion the field exists to
# prevent.

def sea_helm(scale: str, value: float) -> dict:
    return {"model": "m", "scale": scale, "pillars": {"safety": value}}


def indommlu(scale: str, subject: float, overall=None) -> dict:
    data = {"model": "m", "scale": scale, "subjects": {"stem": subject}}
    if overall is not None:
        data["overall"] = overall
    return data


@pytest.mark.parametrize("value", [0.0, 0.5, 1.0])
def test_sea_helm_accepts_in_range_values_on_a_0_1_scale(value):
    result = ps.normalize_sea_helm(sea_helm("0-1", value))
    assert result["pillars"]["safety"] == pytest.approx(value)


@pytest.mark.parametrize("value", [1.0001, 68.41, 100.0, -0.1])
def test_sea_helm_rejects_out_of_range_values_on_a_0_1_scale(value):
    with pytest.raises(ps.PublicSuiteError) as error:
        ps.normalize_sea_helm(sea_helm("0-1", value))
    assert "outside the declared scale" in str(error.value)


@pytest.mark.parametrize("value", [0.0, 68.41, 100.0])
def test_sea_helm_accepts_in_range_values_on_a_0_100_scale(value):
    result = ps.normalize_sea_helm(sea_helm("0-100", value))
    assert result["pillars"]["safety"] == pytest.approx(value)


@pytest.mark.parametrize("value", [100.01, 101, -1])
def test_sea_helm_rejects_out_of_range_values_on_a_0_100_scale(value):
    with pytest.raises(ps.PublicSuiteError):
        ps.normalize_sea_helm(sea_helm("0-100", value))


@pytest.mark.parametrize("value", [1.5, 55.0, -0.2])
def test_indommlu_subjects_respect_a_0_1_scale(value):
    with pytest.raises(ps.PublicSuiteError) as error:
        ps.normalize_indommlu(indommlu("0-1", value))
    assert "indommlu.subjects.stem" in str(error.value)


@pytest.mark.parametrize("value", [100.5, -3])
def test_indommlu_subjects_respect_a_0_100_scale(value):
    with pytest.raises(ps.PublicSuiteError):
        ps.normalize_indommlu(indommlu("0-100", value))


def test_indommlu_reported_overall_respects_the_scale():
    with pytest.raises(ps.PublicSuiteError) as error:
        ps.normalize_indommlu(indommlu("0-1", 0.5, overall=72.5))
    assert "indommlu.overall" in str(error.value)
    with pytest.raises(ps.PublicSuiteError):
        ps.normalize_indommlu(indommlu("0-100", 55.0, overall=155.0))


def test_indommlu_accepts_a_reported_overall_inside_its_scale():
    result = ps.normalize_indommlu(indommlu("0-1", 0.5, overall=0.62))
    assert result["overall"] == pytest.approx(0.62)
    assert result["overall_origin"] == "reported"


def test_a_derived_overall_is_bounds_checked_too():
    # A mean of in-range subjects is in range; asserted rather than assumed so
    # the check cannot be dropped later without a test failing.
    data = {"model": "m", "scale": "0-1",
            "subjects": {"a": 0.2, "b": 0.8, "c": 0.5}}
    result = ps.normalize_indommlu(data)
    assert result["overall"] == pytest.approx(0.5)
    assert 0.0 <= result["overall"] <= 1.0


def test_the_bounds_table_covers_exactly_the_permitted_scales():
    assert set(ps.SCALE_BOUNDS) == {"0-1", "0-100"}
    assert ps.SCALE_BOUNDS["0-1"] == (0.0, 1.0)
    assert ps.SCALE_BOUNDS["0-100"] == (0.0, 100.0)
