"""Verifier tier (RSI MVP Phase 1, Task 1.2).

Two properties carry this file:

* a judge verdict can never become a correctness score;
* a right answer with wrong reasoning is RECORDED as a false positive rather
  than silently accepted.

The second is the error-amplification channel the report names (§04). Offline
and deterministic: no network, no model. The executor runs Python in a
subprocess, which is local and bounded.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import gold_set as gs  # noqa: E402
import verifiers as V  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures" / "gold"


def rows(name: str) -> list[dict]:
    return gs.parse_jsonl((FIXTURES / name).read_text(encoding="utf-8"))


def by_type(vtype: str) -> dict:
    for record in rows("valid.synthetic.jsonl"):
        if record["verifier"]["type"] == vtype:
            return copy.deepcopy(record)
    raise AssertionError(f"no {vtype} fixture")


def conforming_payload(schema: dict) -> dict:
    """A minimal object satisfying `schema`, typed per property.

    Built from the schema rather than hardcoded so the test cannot drift from
    the fixture it is checking.
    """
    sample = {"string": "contoh", "integer": 0, "number": 0,
              "boolean": True, "array": [], "object": {}}
    payload = {}
    for key in schema.get("required", []):
        spec = schema.get("properties", {}).get(key, {})
        payload[key] = sample.get(spec.get("type", "string"), "contoh")
    return payload


def false_positive_record() -> tuple[dict, dict]:
    record = rows("false_positive.synthetic.jsonl")[0]
    output = record.pop("fixture_model_output")
    return record, output


# --- ordering and quarantine ------------------------------------------------


def test_judge_is_last_in_the_trust_order():
    assert V.TRUST_ORDER[-1] == "judge"
    assert "judge" in V.QUARANTINED


def test_every_verifier_type_is_implemented():
    assert set(V.VERIFIERS) == set(gs.VERIFIER_TYPES)


def test_a_judge_result_never_counts_toward_correctness():
    record = by_type("judge")
    result = V.verify(record, "jawaban yang terdengar alami")
    assert result.counts_toward_correctness is False
    assert result.passed is False
    assert "quarantined" in result.detail


def test_a_judge_item_claiming_correctness_is_refused_not_dropped():
    record = by_type("judge")
    record["score_role"] = "correctness"
    with pytest.raises(V.VerifierError) as error:
        V.verify(record, "apa pun")
    assert "correctness" in str(error.value)


def test_verify_refuses_an_invalid_gold_record():
    record = by_type("exact_match")
    record["language"] = "en"
    with pytest.raises(V.VerifierError):
        V.verify(record, {"answer": "42"})


# --- exact_match ------------------------------------------------------------


def test_exact_match_passes_on_the_declared_answer():
    record = by_type("exact_match")
    result = V.verify(record, {"answer": "42", "reasoning": "6 × 7 = 42 kelereng."})
    assert result.passed is True
    assert result.false_positive is False
    assert result.counts_toward_correctness is True


def test_exact_match_fails_on_a_different_answer():
    record = by_type("exact_match")
    result = V.verify(record, {"answer": "41", "reasoning": "6 × 7 = 42"})
    assert result.passed is False


def test_normalization_is_applied():
    record = by_type("exact_match")
    result = V.verify(record, {"answer": "  42  ", "reasoning": "6 × 7 = 42"})
    assert result.passed is True


def test_an_unenumerated_normalization_raises_rather_than_being_skipped():
    with pytest.raises(V.VerifierError):
        V.normalize("x", ["fuzzy"])


# --- the false positive -----------------------------------------------------


def test_right_answer_wrong_reasoning_is_recorded_not_silently_accepted():
    record, output = false_positive_record()
    result = V.verify(record, output)
    # The answer really did match. That is not erased.
    assert result.passed is True
    # And the reasoning claim really was absent. That is not erased either.
    assert result.false_positive is True
    assert result.reasoning_checked is True
    assert "missing claim" in result.detail


def test_a_false_positive_is_visible_in_the_aggregate():
    record, output = false_positive_record()
    clean = by_type("exact_match")
    results = [
        V.verify(record, output),
        V.verify(clean, {"answer": "42", "reasoning": "6 × 7 = 42"}),
    ]
    summary = V.score(results)
    assert summary["passed"] == 2
    assert summary["pass_rate"] == 1.0
    # A pass rate of 1.0 that hides a false positive is the failure mode.
    assert summary["false_positives"] == 1
    assert summary["false_positive_rate"] == 0.5
    assert record["id"] in summary["false_positive_ids"]


def test_missing_reasoning_on_a_required_claims_item_is_a_false_positive():
    record = by_type("exact_match")
    result = V.verify(record, {"answer": "42"})
    assert result.passed is True
    assert result.false_positive is True


def test_a_wrong_answer_is_a_failure_not_a_false_positive():
    record = by_type("exact_match")
    result = V.verify(record, {"answer": "7", "reasoning": "salah"})
    assert result.passed is False
    assert result.false_positive is False


def test_an_item_without_a_reasoning_check_reports_false_positive_as_none():
    record = by_type("knowledge") if False else by_type("exact_match")
    record["expected"].pop("reasoning_check")
    result = V.verify(record, {"answer": "42"})
    assert result.reasoning_checked is False
    assert result.false_positive is None


# --- schema -----------------------------------------------------------------


def test_schema_accepts_a_conforming_object():
    record = by_type("schema")
    payload = conforming_payload(record["expected"]["json_schema"])
    result = V.verify(record, payload)
    assert result.passed is True, result.problems


def test_schema_rejects_a_missing_required_field():
    record = by_type("schema")
    result = V.verify(record, {})
    assert result.passed is False
    assert any("required" in p for p in result.problems)


def test_parsing_is_not_passing():
    # Valid JSON that satisfies nothing the contract requires. This is the
    # distinction check_edit_contract.mjs draws between parse_ok and
    # contract_ok, and the reason schema validation exists at all.
    record = by_type("schema")
    result = V.verify(record, '{"tidak": "relevan"}')
    assert result.passed is False


def test_unparseable_output_reports_parse_failure():
    record = by_type("schema")
    result = V.verify(record, "bukan json")
    assert result.passed is False
    assert any("parse_ok=false" in p for p in result.problems)


def test_additional_properties_false_is_enforced():
    record = by_type("schema")
    schema = record["expected"]["json_schema"]
    if schema.get("additionalProperties") is not False:
        pytest.skip("fixture schema does not close additionalProperties")
    payload = conforming_payload(schema)
    payload["selundupan"] = 1
    result = V.verify(record, payload)
    assert result.passed is False
    assert any("unexpected field" in p for p in result.problems)


def test_an_unsupported_schema_keyword_raises_rather_than_passing():
    record = by_type("schema")
    record["expected"]["json_schema"] = {"type": "object", "minProperties": 1}
    with pytest.raises(V.VerifierError):
        V.verify(record, {})


# --- executor ---------------------------------------------------------------


def test_executor_passes_a_correct_program():
    record = by_type("executor")
    program = "a, b = input().split()\nprint(int(a) + int(b))"
    result = V.verify(record, program)
    assert result.passed is True, result.problems


def test_executor_fails_a_wrong_program():
    record = by_type("executor")
    result = V.verify(record, "print(0)")
    assert result.passed is False
    assert any("stdout" in p for p in result.problems)


def test_executor_reports_a_crash_rather_than_passing():
    record = by_type("executor")
    result = V.verify(record, "raise SystemExit(3)")
    assert result.passed is False
    assert any("exit 3" in p for p in result.problems)


def test_executor_enforces_its_timeout():
    record = by_type("executor")
    record["expected"]["timeout_seconds"] = 1
    result = V.verify(record, "import time\ntime.sleep(30)")
    assert result.passed is False
    assert any("timed out" in p for p in result.problems)


def test_executor_refuses_an_unallowed_runner():
    record = by_type("executor")
    record["expected"]["runner"] = "bash"
    with pytest.raises((V.VerifierError, Exception)):
        V.verify(record, "echo hi")


def test_executor_refuses_an_empty_candidate():
    record = by_type("executor")
    result = V.verify(record, "   ")
    assert result.passed is False


# --- scoring ----------------------------------------------------------------


def test_score_excludes_quarantined_items_from_correctness():
    judge = V.verify(by_type("judge"), "teks gaya")
    exact = V.verify(by_type("exact_match"),
                     {"answer": "42", "reasoning": "6 × 7 = 42"})
    summary = V.score([judge, exact])
    assert summary["items"] == 2
    assert summary["correctness_items"] == 1
    assert summary["quarantined_items"] == 1
    assert summary["pass_rate"] == 1.0


def test_score_of_nothing_is_zero_not_one():
    summary = V.score([])
    assert summary["pass_rate"] == 0.0
    assert summary["false_positive_rate"] == 0.0


def test_every_score_says_training_is_not_authorized():
    assert V.score([])["training_authorized"] is False


def test_a_measurement_is_measured_never_passed():
    # run_gates' distinction: measuring is not the same as passing a gate.
    assert V.score([])["measured"] is True
    assert "passed_gate" not in V.score([])


# --- the executor backend is quarantined ------------------------------------
# The local backend is timeout-bounded and nothing more: no network policy, no
# isolation, no resource limits. It is therefore restricted to records this
# repository authored as fixtures, and a production item must fail closed.

def human_authored_executor() -> dict:
    record = by_type("executor")
    record["source"] = {
        "kind": "human_authored",
        "author_ref": "reviewer-registry-0001",
        "authored_at": "2026-09-19",
        "approved_by": "reviewer-registry-0002",
        "approved_at": "2026-09-19",
        "evidence_ref": "local-review-record-0001",
    }
    record["source_class"] = "internal"
    return record


def test_a_synthetic_fixture_may_use_the_local_backend():
    allowed, why = V.executor_backend_allows(by_type("executor"))
    assert allowed is True
    assert "synthetic fixture" in why


def test_a_human_authored_executor_item_fails_closed():
    record = human_authored_executor()
    # Structurally valid and production-eligible, and still refused.
    assert gs.validate_record(record) == []
    assert gs.production_problems(record) == []
    allowed, why = V.executor_backend_allows(record)
    assert allowed is False
    assert "reviewed sandbox backend is required" in why


def test_verify_refuses_a_production_executor_item_rather_than_running_it():
    record = human_authored_executor()
    with pytest.raises(V.VerifierError) as error:
        V.verify(record, "print('should never run')")
    assert "sandbox backend" in str(error.value)


def test_an_unknown_source_kind_is_refused_by_the_backend():
    record = by_type("executor")
    record["source"] = {"kind": "imported_corpus", "evidence_ref": "x"}
    allowed, _ = V.executor_backend_allows(record)
    assert allowed is False


def test_fixture_code_runs_in_a_disposable_directory_not_the_repo():
    """A fixture that writes a file must not leave it in the repository."""
    record = by_type("executor")
    record["expected"] = {"runner": "python", "timeout_seconds": 5,
                          "tests": [{"stdin": "", "stdout": "ok\n"}]}
    marker = "executor_side_effect_marker.txt"
    program = (f"open({marker!r}, 'w').write('x')\n"
               "import os\n"
               "print('ok')")
    result = V.verify(record, program)
    assert result.passed is True, result.problems
    assert not (ROOT / marker).exists(), "a fixture wrote into the repository"
    assert not (ROOT / "src" / marker).exists()


def test_the_working_directory_is_not_the_repository_root():
    record = by_type("executor")
    record["expected"] = {"runner": "python", "timeout_seconds": 5,
                          "tests": [{"stdin": "", "stdout": "absent\n"}]}
    # src/ exists at the repo root; it must not be visible from the cwd.
    program = ("import os\n"
               "print('present' if os.path.isdir('src') else 'absent')")
    result = V.verify(record, program)
    assert result.passed is True, result.problems


def test_oversized_output_is_a_failure_not_a_truncated_comparison():
    record = by_type("executor")
    record["expected"] = {"runner": "python", "timeout_seconds": 5,
                          "tests": [{"stdin": "", "stdout": "x\n"}]}
    program = f"print('x' * {V.MAX_CAPTURED_CHARS + 100})"
    result = V.verify(record, program)
    assert result.passed is False
    assert any("exceeded" in p for p in result.problems)


def test_the_module_does_not_claim_to_deny_networking():
    source = (ROOT / "src" / "verifiers.py").read_text(encoding="utf-8")
    assert "NO_PROXY" not in source, \
        "proxy variables are not a firewall and must not imply one"
    assert "no_proxy" not in source
    # And it says what it actually is.
    assert "TIMEOUT-BOUNDED, not sandboxed" in source


def test_the_docstring_does_not_promise_a_sandbox():
    import verifiers
    doc = verifiers.__doc__ or ""
    assert "QUARANTINED" in doc
    assert "not sandboxed" in doc
