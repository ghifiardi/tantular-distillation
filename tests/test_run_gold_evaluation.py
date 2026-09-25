"""Gold evaluation runner (RSI next-execution handoff, Stage 2).

The runner produces the measurement JSON that `src/eval_harness.py
separation-gate` consumes. Everything here is offline and deterministic: the
model client is an injected fake, the gold set is a synthetic fixture copied
into a temporary directory, and the model registry is a fixture directory.

What is proven:

* `plan` writes nothing and contacts no client.
* an empty gold directory refuses with the loader's own BLOCKED message.
* fake 9B and 4B runs give distinct, paired measurements the gate accepts.
* repetitions are preserved for stability.
* missing, duplicate and corrupt receipts refuse; mixed identities refuse.
* a teacher identity in a student slot refuses.
* a client or verifier exception becomes an error receipt, not a hole.
* expected answers never appear in receipts or measurements.
* every artifact carries training_authorized: false.
"""
from __future__ import annotations

import copy
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import eval_harness as eh                                  # noqa: E402
import gold_set as gs                                      # noqa: E402
import run_gold_evaluation as rge                          # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures" / "gold_evaluation"
GOLD_FIXTURE = ROOT / "tests" / "fixtures" / "gold" / "valid.synthetic.jsonl"
MODEL_DIR = FIXTURES / "models"
EXPERIMENT = ROOT / "configs" / "experiments" / "harness-before-weights.yaml"

# Strings that live only inside `expected` blocks of the fixture gold set.
SECRETS = ("Bandung", "6 × 7 = 42", "jumlah_butir", "17 25")


@pytest.fixture
def gold_dir(tmp_path) -> Path:
    directory = tmp_path / "gold"
    directory.mkdir()
    shutil.copy(GOLD_FIXTURE, directory / "valid.synthetic.jsonl")
    return directory


@pytest.fixture
def empty_gold(tmp_path) -> Path:
    directory = tmp_path / "gold-empty"
    directory.mkdir()
    return directory


def outcomes(name: str) -> dict:
    return json.loads((FIXTURES / f"fake_outcomes_{name}.json").read_text("utf-8"))


def fake_client(name: str) -> rge.FakeGoldClient:
    return rge.FakeGoldClient(outcomes(name))


class RefusingClient:
    """Fails the test if anything touches it: for refusal-before-call cases."""

    def identity(self) -> dict:
        raise AssertionError("client identity was requested")

    def answer(self, record: dict, decoding: dict) -> str:
        raise AssertionError("client was called")


def run_arm(gold_dir: Path, out: Path, name: str, registry: str,
            repetitions: int = 2, run_id: str = "run-1") -> dict:
    return rge.run_evaluation(
        gold_dir=gold_dir, model_registry=registry, model_dir=MODEL_DIR,
        client=fake_client(name), repetitions=repetitions, output=out,
        run_id=run_id, allow_fixture=True, experiment_path=EXPERIMENT)


def measure(gold_dir: Path, receipts: Path, arm: str) -> dict:
    return rge.aggregate_measurement(
        gold_dir=gold_dir, receipts=receipts, arm=arm, allow_fixture=True,
        experiment_path=EXPERIMENT)


def snapshot(directory: Path) -> dict[str, int]:
    return {str(p): p.stat().st_size for p in directory.rglob("*") if p.is_file()}


# --- plan -------------------------------------------------------------------


def test_plan_writes_nothing_and_contacts_no_client(gold_dir, tmp_path):
    before = snapshot(tmp_path)
    plan = rge.plan_evaluation(
        gold_dir=gold_dir, model_registry="qwen35-9b-fixture", model_dir=MODEL_DIR,
        endpoint="http://127.0.0.1:9/v1", repetitions=2,
        experiment_path=EXPERIMENT, allow_fixture=True)
    assert snapshot(tmp_path) == before
    assert plan["training_authorized"] is False
    assert plan["repetitions"] == 2
    assert plan["model_identity"]["expected"] == "Qwen/Qwen3.5-9B"
    assert plan["endpoint"] == "http://127.0.0.1:9/v1"
    assert set(plan["by_split"]) == set(gs.SPLITS)
    text = json.dumps(plan, ensure_ascii=False)
    for secret in SECRETS:
        assert secret not in text, f"plan exposed {secret!r}"


def test_plan_on_an_empty_gold_set_reports_the_blocked_message(empty_gold):
    plan = rge.plan_evaluation(
        gold_dir=empty_gold, model_registry="qwen35-9b-fixture", model_dir=MODEL_DIR,
        endpoint="http://127.0.0.1:9/v1", repetitions=2,
        experiment_path=EXPERIMENT, allow_fixture=False)
    assert gs.BLOCKED_MESSAGE in plan["blockers"]
    assert plan["executable"] is False


def test_a_fixture_set_is_a_blocker_for_a_real_run(gold_dir):
    plan = rge.plan_evaluation(
        gold_dir=gold_dir, model_registry="qwen35-9b-fixture", model_dir=MODEL_DIR,
        endpoint="http://127.0.0.1:9/v1", repetitions=2,
        experiment_path=EXPERIMENT, allow_fixture=False)
    assert plan["executable"] is False
    assert gs.BLOCKED_MESSAGE in plan["blockers"]


# --- refusals before any client call ---------------------------------------


def test_an_empty_gold_set_refuses_before_any_client_call(empty_gold, tmp_path):
    with pytest.raises(rge.GoldEvaluationError, match=gs.BLOCKED_MESSAGE):
        rge.run_evaluation(
            gold_dir=empty_gold, model_registry="qwen35-9b-fixture",
            model_dir=MODEL_DIR, client=RefusingClient(), repetitions=2,
            output=tmp_path / "out", run_id="run-1", allow_fixture=False,
            experiment_path=EXPERIMENT)
    assert not (tmp_path / "out").exists()


def test_a_fixture_set_without_allow_fixture_refuses_before_any_client_call(
        gold_dir, tmp_path):
    with pytest.raises(rge.GoldEvaluationError, match=gs.BLOCKED_MESSAGE):
        rge.run_evaluation(
            gold_dir=gold_dir, model_registry="qwen35-9b-fixture",
            model_dir=MODEL_DIR, client=RefusingClient(), repetitions=2,
            output=tmp_path / "out", run_id="run-1", allow_fixture=False,
            experiment_path=EXPERIMENT)


def test_a_teacher_identity_in_the_student_slot_refuses(gold_dir, tmp_path):
    with pytest.raises(rge.GoldEvaluationError, match="teacher"):
        rge.run_evaluation(
            gold_dir=gold_dir, model_registry="teacher-fixture",
            model_dir=MODEL_DIR, client=RefusingClient(), repetitions=2,
            output=tmp_path / "out", run_id="run-1", allow_fixture=True,
            experiment_path=EXPERIMENT)


def test_a_served_model_that_is_not_the_expected_one_refuses(gold_dir, tmp_path):
    wrong = outcomes("9b")
    wrong["served"] = ["someone-else/other-model"]
    with pytest.raises(rge.GoldEvaluationError, match="served"):
        rge.run_evaluation(
            gold_dir=gold_dir, model_registry="qwen35-9b-fixture",
            model_dir=MODEL_DIR, client=rge.FakeGoldClient(wrong), repetitions=2,
            output=tmp_path / "out", run_id="run-1", allow_fixture=True,
            experiment_path=EXPERIMENT)
    assert not list((tmp_path / "out").glob("*.json")) if (tmp_path / "out").exists() else True


def test_fewer_than_two_repetitions_is_refused(gold_dir, tmp_path):
    with pytest.raises(rge.GoldEvaluationError, match="repetitions"):
        run_arm(gold_dir, tmp_path / "out", "9b", "qwen35-9b-fixture", repetitions=1)


def test_existing_receipts_are_never_overwritten(gold_dir, tmp_path):
    out = tmp_path / "out"
    run_arm(gold_dir, out, "9b", "qwen35-9b-fixture")
    before = snapshot(out)
    with pytest.raises(rge.GoldEvaluationError, match="exist"):
        run_arm(gold_dir, out, "9b", "qwen35-9b-fixture")
    assert snapshot(out) == before


# --- receipts ---------------------------------------------------------------


def test_run_leaves_one_receipt_per_item_and_repetition(gold_dir, tmp_path):
    out = tmp_path / "out"
    summary = run_arm(gold_dir, out, "9b", "qwen35-9b-fixture", repetitions=2)
    gold = gs.load(gold_dir)
    assert summary["receipts_written"] == len(gold.valid) * 2
    receipts = rge.load_receipts(out)
    keys = {(r["item_id"], r["repetition"]) for r in receipts}
    assert len(keys) == len(receipts)
    assert {r["repetition"] for r in receipts} == {1, 2}
    for receipt in receipts:
        assert receipt["training_authorized"] is False
        assert receipt["model_identity"]["expected"] == "Qwen/Qwen3.5-9B"
        assert receipt["gold_set_digest"] == rge.gold_set_digest(gold)
        assert receipt["status"] in ("ok", "error")
        assert "output" not in receipt and "expected" not in receipt


def test_a_verifier_or_client_exception_becomes_an_error_receipt(gold_dir, tmp_path):
    class Flaky(rge.FakeGoldClient):
        def answer(self, record, decoding):
            if record["id"] == "fixture::knowledge::0001":
                raise RuntimeError("simulated client failure")
            return super().answer(record, decoding)

    out = tmp_path / "out"
    rge.run_evaluation(
        gold_dir=gold_dir, model_registry="qwen35-9b-fixture", model_dir=MODEL_DIR,
        client=Flaky(outcomes("9b")), repetitions=2, output=out, run_id="run-1",
        allow_fixture=True, experiment_path=EXPERIMENT)
    receipts = [r for r in rge.load_receipts(out)
                if r["item_id"] == "fixture::knowledge::0001"]
    assert len(receipts) == 2
    assert all(r["status"] == "error" for r in receipts)
    assert all("simulated client failure" in r["error"] for r in receipts)
    # And the item is still in the measurement, as a failure, not a hole.
    measurement = measure(gold_dir, out, "student_9b")
    assert measurement["splits"]["knowledge"]["per_item"]["fixture::knowledge::0001"] is False
    assert measurement["status_counts"]["error"] == 2


def test_a_reasoning_false_positive_is_recorded_not_erased(gold_dir, tmp_path):
    fp = outcomes("9b")
    fp["outputs"]["fixture::reasoning::0001"] = {
        "answer": "42", "reasoning": "Karena 40 + 2 = 42."}
    out = tmp_path / "out"
    rge.run_evaluation(
        gold_dir=gold_dir, model_registry="qwen35-9b-fixture", model_dir=MODEL_DIR,
        client=rge.FakeGoldClient(fp), repetitions=2, output=out, run_id="run-1",
        allow_fixture=True, experiment_path=EXPERIMENT)
    receipt = [r for r in rge.load_receipts(out)
               if r["item_id"] == "fixture::reasoning::0001"][0]
    assert receipt["verifier_result"]["passed"] is True
    assert receipt["verifier_result"]["false_positive"] is True
    measurement = measure(gold_dir, out, "student_9b")
    assert "fixture::reasoning::0001" in measurement["false_positive_ids"]


# --- measurements -----------------------------------------------------------


def test_fake_9b_and_4b_produce_distinct_paired_measurements_the_gate_accepts(
        gold_dir, tmp_path):
    nine = tmp_path / "9b"
    four = tmp_path / "4b"
    run_arm(gold_dir, nine, "9b", "qwen35-9b-fixture")
    run_arm(gold_dir, four, "4b", "qwen35-4b-fixture")
    m9 = measure(gold_dir, nine, "student_9b")
    m4 = measure(gold_dir, four, "student_4b")

    assert m9["model_identity"]["expected"] != m4["model_identity"]["expected"]
    assert m9["fixture"] is True and m4["fixture"] is True
    for split in gs.SPLITS:
        assert set(m9["splits"][split]["per_item"]) == set(m4["splits"][split]["per_item"])
        assert len(m9["splits"][split]["runs"]) == 2
        assert len(m4["splits"][split]["runs"]) == 2
    assert eh.validate_measurement(m9, label="student_9b") == []
    assert eh.validate_measurement(m4, label="student_4b") == []

    experiment = yaml.safe_load(EXPERIMENT.read_text("utf-8"))
    verdict = eh.separation_gate(experiment, m9, m4)
    assert verdict["passed"] is True
    assert verdict["training_authorized"] is False


def test_the_measurement_is_digested_and_reproducible(gold_dir, tmp_path):
    out = tmp_path / "out"
    run_arm(gold_dir, out, "9b", "qwen35-9b-fixture")
    a = measure(gold_dir, out, "student_9b")
    b = measure(gold_dir, out, "student_9b")
    assert a == b
    assert a["measurement_sha256"] == rge.measurement_digest(a)
    assert a["receipt_set_digest"] == rge.receipt_set_digest(rge.load_receipts(out))


def test_expected_answers_never_appear_in_receipts_or_measurements(gold_dir, tmp_path):
    out = tmp_path / "out"
    run_arm(gold_dir, out, "9b", "qwen35-9b-fixture")
    measurement = measure(gold_dir, out, "student_9b")
    texts = [p.read_text("utf-8") for p in out.glob("*.json")]
    texts.append(json.dumps(measurement, ensure_ascii=False))
    for text in texts:
        for secret in SECRETS:
            assert secret not in text, f"artifact exposed {secret!r}"


def test_every_artifact_says_training_authorized_false(gold_dir, tmp_path):
    out = tmp_path / "out"
    summary = run_arm(gold_dir, out, "9b", "qwen35-9b-fixture")
    assert summary["training_authorized"] is False
    for receipt in rge.load_receipts(out):
        assert receipt["training_authorized"] is False
    assert measure(gold_dir, out, "student_9b")["training_authorized"] is False


# --- aggregation refusals ---------------------------------------------------


def test_a_missing_receipt_refuses_aggregation(gold_dir, tmp_path):
    out = tmp_path / "out"
    run_arm(gold_dir, out, "9b", "qwen35-9b-fixture")
    victim = sorted(out.glob("*.json"))[0]
    victim.unlink()
    with pytest.raises(rge.GoldEvaluationError, match="no receipt"):
        measure(gold_dir, out, "student_9b")


def test_a_duplicate_receipt_refuses_aggregation(gold_dir, tmp_path):
    out = tmp_path / "out"
    run_arm(gold_dir, out, "9b", "qwen35-9b-fixture")
    source = sorted(out.glob("*.json"))[0]
    shutil.copy(source, out / "zz-duplicate.json")
    with pytest.raises(rge.GoldEvaluationError, match="duplicate"):
        measure(gold_dir, out, "student_9b")


def test_a_corrupt_receipt_refuses_aggregation(gold_dir, tmp_path):
    out = tmp_path / "out"
    run_arm(gold_dir, out, "9b", "qwen35-9b-fixture")
    victim = sorted(out.glob("*.json"))[0]
    data = json.loads(victim.read_text("utf-8"))
    data["verifier_result"]["passed"] = not data["verifier_result"]["passed"]
    victim.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(rge.GoldEvaluationError, match="receipt_sha256"):
        measure(gold_dir, out, "student_9b")


def test_receipts_with_mixed_identities_refuse_aggregation(gold_dir, tmp_path):
    out = tmp_path / "out"
    run_arm(gold_dir, out, "9b", "qwen35-9b-fixture", run_id="run-1")
    # A second run of a DIFFERENT model into the same directory.
    run_arm(gold_dir, out, "4b", "qwen35-4b-fixture", run_id="run-2")
    with pytest.raises(rge.GoldEvaluationError, match="mix"):
        measure(gold_dir, out, "student_9b")


def test_receipts_from_a_different_gold_set_refuse_aggregation(gold_dir, tmp_path):
    out = tmp_path / "out"
    run_arm(gold_dir, out, "9b", "qwen35-9b-fixture")
    other = tmp_path / "other-gold"
    other.mkdir()
    rows = [json.loads(l) for l in GOLD_FIXTURE.read_text("utf-8").splitlines() if l.strip()]
    rows[0]["prompt"] = rows[0]["prompt"] + " (diubah)"
    (other / "valid.synthetic.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", "utf-8")
    with pytest.raises(rge.GoldEvaluationError, match="gold_set_digest"):
        measure(other, out, "student_9b")


def test_fixture_receipts_are_refused_as_a_real_measurement(gold_dir, tmp_path):
    out = tmp_path / "out"
    run_arm(gold_dir, out, "9b", "qwen35-9b-fixture")
    # Refused either by the gold gate (a fixture set is not approved) or, if a
    # set were approved, by the fixture-client check. Both are refusals.
    with pytest.raises(rge.GoldEvaluationError, match="fixture|BLOCKED"):
        rge.aggregate_measurement(gold_dir=gold_dir, receipts=out, arm="student_9b",
                                  allow_fixture=False, experiment_path=EXPERIMENT)


def test_a_production_executor_item_is_a_blocker_not_a_failure(tmp_path):
    """A human-authored executor item cannot be run by the timeout-bounded
    backend. The plan must say so, and aggregation must refuse rather than
    score it as a failed item."""
    gold = tmp_path / "gold"
    gold.mkdir()
    rows = [json.loads(l) for l in GOLD_FIXTURE.read_text("utf-8").splitlines() if l.strip()]
    for row in rows:
        row["source_class"] = "internal"
        row["source"] = {"kind": "human_authored", "author_ref": "author-a",
                         "authored_at": "2026-09-20", "approved_by": "reviewer-b",
                         "approved_at": "2026-09-21", "evidence_ref": "review-1"}
    (gold / "items.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", "utf-8")
    plan = rge.plan_evaluation(
        gold_dir=gold, model_registry="qwen35-9b-fixture", model_dir=MODEL_DIR,
        endpoint="http://127.0.0.1:9/v1", repetitions=2,
        experiment_path=EXPERIMENT, allow_fixture=False)
    assert plan["executable"] is False
    assert any("executor" in b for b in plan["blockers"])


# --- CLI --------------------------------------------------------------------


def run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(ROOT / "src" / "run_gold_evaluation.py"), *args],
        capture_output=True, text=True, timeout=120)


def test_cli_plan_help_and_plan_on_the_empty_production_set():
    result = run_cli("plan", "--help")
    assert result.returncode == 0
    assert "--real" not in result.stdout, "plan never takes --real"
    result = run_cli("plan", "--gold-dir", str(ROOT / "data" / "gold"),
                     "--model-registry", "qwen35-9b-instruct",
                     "--endpoint", "http://127.0.0.1:9/v1", "--repetitions", "2")
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan["executable"] is False
    assert gs.BLOCKED_MESSAGE in plan["blockers"]
    assert plan["training_authorized"] is False


def test_cli_run_without_real_or_fake_refuses(gold_dir, tmp_path):
    result = run_cli("run", "--gold-dir", str(gold_dir),
                     "--model-registry", "qwen35-9b-fixture",
                     "--model-dir", str(MODEL_DIR),
                     "--endpoint", "http://127.0.0.1:9/v1",
                     "--output", str(tmp_path / "out"), "--allow-fixture")
    assert result.returncode != 0
    assert "--real" in result.stderr and "--fake-outcomes" in result.stderr
    assert not (tmp_path / "out").exists()


def test_cli_real_run_on_the_empty_production_set_refuses_before_network(tmp_path):
    result = run_cli("run", "--gold-dir", str(ROOT / "data" / "gold"),
                     "--model-registry", "qwen35-9b-instruct",
                     "--endpoint", "http://127.0.0.1:9/v1",
                     "--output", str(tmp_path / "out"), "--real")
    assert result.returncode != 0
    assert gs.BLOCKED_MESSAGE in result.stderr
    assert not (tmp_path / "out").exists()


def test_cli_fake_run_and_aggregate_end_to_end(gold_dir, tmp_path):
    out = tmp_path / "out"
    result = run_cli("run", "--gold-dir", str(gold_dir),
                     "--model-registry", "qwen35-9b-fixture",
                     "--model-dir", str(MODEL_DIR),
                     "--endpoint", "http://127.0.0.1:9/v1",
                     "--output", str(out), "--repetitions", "2",
                     "--fake-outcomes", str(FIXTURES / "fake_outcomes_9b.json"),
                     "--allow-fixture")
    assert result.returncode == 0, result.stderr
    target = tmp_path / "measurement-9b.json"
    result = run_cli("aggregate", "--gold-dir", str(gold_dir),
                     "--receipts", str(out), "--arm", "student_9b",
                     "--output", str(target), "--allow-fixture")
    assert result.returncode == 0, result.stderr
    measurement = json.loads(target.read_text("utf-8"))
    assert measurement["fixture"] is True
    assert measurement["training_authorized"] is False
    # And the output file is never silently overwritten.
    result = run_cli("aggregate", "--gold-dir", str(gold_dir),
                     "--receipts", str(out), "--arm", "student_9b",
                     "--output", str(target), "--allow-fixture")
    assert result.returncode != 0
    assert "exist" in result.stderr
