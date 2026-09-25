"""GEPA runner (RSI next-execution handoff, Stage 3).

`src/run_gepa.py` exposes the tested `gepa.optimize()` controller behind a
bounded CLI. Everything here is offline: a fake client replays declared
outputs per instruction, a fake reflector replays declared mutations, and the
train and held-out sets are synthetic fixtures.

What is proven:

* an unapproved held-out set refuses before any client call;
* `plan` writes nothing;
* a fake client drives a complete iteration through the real verifier layer;
* held-out prompts, ids and expected content never reach the reflector;
* a higher mean with one regressed held-out item is rejected;
* the result can populate the `student_candidate` arm (guardrails aside);
* production harness files are unchanged by a run;
* output is byte-identical across two runs with the same seed;
* every artifact says training_authorized: false.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import gepa                                                # noqa: E402
import gepa_evidence                                       # noqa: E402
import gold_set as gs                                      # noqa: E402
import run_gepa as rg                                      # noqa: E402
import verifiers                                           # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures" / "gepa_run"
MODEL_DIR = ROOT / "tests" / "fixtures" / "gold_evaluation" / "models"
HARNESS_DIR = ROOT / "configs" / "harnesses"
EXPERIMENT = ROOT / "configs" / "experiments" / "harness-before-weights.yaml"
TRAIN = FIXTURES / "train.synthetic.jsonl"

SEED = "Balas hanya dengan JSON kontrak panggilan alat. Jangan menambah fakta baru."
IMPROVED = "Balas hanya dengan JSON panggilan alat yang valid dan jangan menambah teks lain."

# Strings that exist only in the held-out fixture and must never reach the
# proposer: ids, a prompt marker, and a required schema key.
HOLDOUT_SECRETS = ("holdout::tool_use", "RAHASIA-HOLDOUT", "selector")


@pytest.fixture
def holdout_dir(tmp_path) -> Path:
    directory = tmp_path / "holdout"
    directory.mkdir()
    shutil.copy(FIXTURES / "holdout.synthetic.jsonl", directory / "holdout.synthetic.jsonl")
    return directory


def bundle(name: str) -> dict:
    return json.loads((FIXTURES / f"fake_bundle_{name}.json").read_text("utf-8"))


def fixture_records() -> list[dict]:
    return [
        json.loads(line)
        for line in (FIXTURES / "holdout.synthetic.jsonl").read_text(
            "utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n"
                for record in records),
        encoding="utf-8")


def style_record(item_id: str, *, judge: bool) -> dict:
    record = fixture_records()[0]
    record["id"] = item_id
    record["prompt"] = f"STYLE-ONLY {item_id}"
    record["score_role"] = "style"
    if judge:
        record["verifier"] = {"type": "judge", "config": {}}
        record["expected"] = {
            "rubric": ["alami", "ringkas"],
            "calibration_set_ref": "fixture-raters-v1",
        }
    else:
        record["verifier"] = {"type": "exact_match", "config": {}}
        record["expected"] = {"answer": "baik", "normalization": ["trim"]}
    return record


@pytest.fixture
def production_holdout_dir(tmp_path) -> Path:
    """A complete, approved four-split set for fake-vs-production refusal."""
    directory = tmp_path / "production-holdout"
    directory.mkdir()
    source = fixture_records()[0]
    records = []
    for index, split in enumerate(gs.SPLITS, start=1):
        record = json.loads(json.dumps(source))
        record["id"] = f"production::{split}::{index:04d}"
        record["split"] = split
        record["prompt"] = f"Kasus manusia yang disetujui untuk {split}."
        record["source_class"] = "internal"
        record["source"] = {
            "kind": "human_authored",
            "author_ref": "human-author-a",
            "authored_at": "2026-09-25",
            "approved_by": "human-reviewer-b",
            "approved_at": "2026-09-25",
            "evidence_ref": f"local-review::{split}",
        }
        records.append(record)
    write_jsonl(directory / "approved.jsonl", records)
    return directory


def harness_digests() -> dict[str, str]:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(HARNESS_DIR.glob("*"))}


class RefusingClient:
    def identity(self):
        raise AssertionError("client identity was requested")

    def answer(self, record, decoding, instruction):
        raise AssertionError("client was called")


def run(holdout_dir: Path, out: Path, name: str = "improving", *,
        budget: int = 12, seed: int = 0, reflect=None, client=None) -> dict:
    data = bundle(name)
    return rg.run_gepa(
        train_set=TRAIN, holdout_gold=holdout_dir, model_registry="qwen35-9b-fixture",
        model_dir=MODEL_DIR, client=client or rg.FakeGepaClient(data),
        reflect=reflect or rg.SequencedReflector(data["reflections"]),
        harness="tantular-office-candidate", slice_name="tool_use",
        rollout_budget=budget, output=out, allow_fixture=True,
        seed_instruction=SEED, seed=seed, experiment_path=EXPERIMENT)


def snapshot(directory: Path) -> dict[str, int]:
    return {str(p): p.stat().st_size for p in directory.rglob("*") if p.is_file()}


# --- plan -------------------------------------------------------------------


def test_plan_writes_nothing_and_shows_the_bounded_run(holdout_dir, tmp_path):
    before = snapshot(tmp_path)
    plan = rg.plan_gepa(
        train_set=TRAIN, holdout_gold=holdout_dir, model_registry="qwen35-9b-fixture",
        model_dir=MODEL_DIR, endpoint="http://127.0.0.1:9/v1",
        harness="tantular-office-candidate", slice_name="tool_use",
        rollout_budget=12, allow_fixture=True, seed_instruction=SEED,
        experiment_path=EXPERIMENT)
    assert snapshot(tmp_path) == before
    assert plan["slice"] == "tool_use"
    assert plan["rollout_budget"] == 12
    assert plan["train_count"] == 3
    assert plan["holdout_count"] == 3
    assert len(plan["harness_digest"]) == 64
    assert plan["model_identity"]["expected"] == "Qwen/Qwen3.5-9B"
    assert plan["training_authorized"] is False
    text = json.dumps(plan, ensure_ascii=False)
    for secret in HOLDOUT_SECRETS[1:]:
        assert secret not in text


def test_plan_on_the_empty_production_set_is_blocked(tmp_path):
    empty = tmp_path / "gold"
    empty.mkdir()
    plan = rg.plan_gepa(
        train_set=TRAIN, holdout_gold=empty, model_registry="qwen35-9b-fixture",
        model_dir=MODEL_DIR, endpoint="http://127.0.0.1:9/v1",
        harness="tantular-office-candidate", slice_name="tool_use",
        rollout_budget=12, allow_fixture=False, seed_instruction=SEED,
        experiment_path=EXPERIMENT)
    assert plan["executable"] is False
    assert gs.BLOCKED_MESSAGE in plan["blockers"]


# --- refusals before any client call ---------------------------------------


def test_an_unapproved_holdout_refuses_before_any_client_call(holdout_dir, tmp_path):
    with pytest.raises(rg.GepaRunError, match=gs.BLOCKED_MESSAGE):
        rg.run_gepa(
            train_set=TRAIN, holdout_gold=holdout_dir, model_registry="qwen35-9b-fixture",
            model_dir=MODEL_DIR, client=RefusingClient(),
            reflect=rg.SequencedReflector([IMPROVED]),
            harness="tantular-office-candidate", slice_name="tool_use",
            rollout_budget=12, output=tmp_path / "out", allow_fixture=False,
            seed_instruction=SEED, seed=0, experiment_path=EXPERIMENT)
    assert not (tmp_path / "out").exists()


def test_a_fake_client_refuses_approved_production_gold_without_fixture_posture(
        production_holdout_dir, tmp_path):
    data = bundle("improving")
    with pytest.raises(rg.GepaRunError, match="fake client.*fixture posture"):
        rg.run_gepa(
            train_set=TRAIN, holdout_gold=production_holdout_dir,
            model_registry="qwen35-9b-fixture", model_dir=MODEL_DIR,
            client=rg.FakeGepaClient(data),
            reflect=rg.SequencedReflector(data["reflections"]),
            harness="tantular-office-candidate", slice_name="tool_use",
            rollout_budget=12, output=tmp_path / "out", allow_fixture=False,
            seed_instruction=SEED, seed=0, experiment_path=EXPERIMENT)
    assert not (tmp_path / "out").exists()


def test_train_and_holdout_overlap_refuses(holdout_dir, tmp_path):
    with pytest.raises(rg.GepaRunError, match="overlap"):
        rg.run_gepa(
            train_set=holdout_dir / "holdout.synthetic.jsonl", holdout_gold=holdout_dir,
            model_registry="qwen35-9b-fixture", model_dir=MODEL_DIR,
            client=RefusingClient(), reflect=rg.SequencedReflector([IMPROVED]),
            harness="tantular-office-candidate", slice_name="tool_use",
            rollout_budget=12, output=tmp_path / "out", allow_fixture=True,
            seed_instruction=SEED, seed=0, experiment_path=EXPERIMENT)


def test_an_output_inside_the_production_config_tree_refuses(holdout_dir):
    with pytest.raises(rg.GepaRunError, match="isolated"):
        run(holdout_dir, HARNESS_DIR / "candidate-out")


# --- a complete fake iteration ---------------------------------------------


def test_a_fake_client_drives_a_complete_iteration(holdout_dir, tmp_path):
    out = tmp_path / "out"
    result = run(holdout_dir, out)
    assert result["rollouts"] >= 3
    assert result["accepted"], result["rejected"]
    assert result["best"]["instruction"] == IMPROVED
    assert result["best"]["score"] > result["seed"]["score"]
    assert result["training_authorized"] is False
    assert (out / "result.json").is_file()
    assert (out / "candidates.jsonl").is_file()
    assert (out / "restricted" / "per_instance.jsonl").is_file()


def test_candidate_evidence_carries_the_required_fields(holdout_dir, tmp_path):
    out = tmp_path / "out"
    run(holdout_dir, out)
    rows = [json.loads(l) for l in (out / "candidates.jsonl").read_text("utf-8").splitlines()]
    assert rows, "candidate evidence must be written"
    for row in rows:
        for key in ("candidate_id", "parent_id", "instruction_sha256",
                    "base_harness_digest", "candidate_harness_digest",
                    "model_identity", "train_summary", "holdout_decision",
                    "reason", "rollouts", "rollout_budget", "archive_only",
                    "promotable", "training_authorized"):
            assert key in row, key
        assert row["training_authorized"] is False
        assert row["archive_only"] is not row["promotable"]
        assert "scores" not in row and "per_instance" not in row
        assert "instruction" not in row


def test_per_instance_scores_live_in_the_restricted_artifact_only(holdout_dir, tmp_path):
    out = tmp_path / "out"
    run(holdout_dir, out)
    restricted = (out / "restricted" / "per_instance.jsonl").read_text("utf-8")
    assert "holdout::tool_use::0001" in restricted
    public = (out / "candidates.jsonl").read_text("utf-8")
    assert "holdout::tool_use" not in public


def test_public_result_recursively_contains_no_scores_or_holdout_identifiers(
        holdout_dir, tmp_path):
    out = tmp_path / "out"
    run(holdout_dir, out, "regressing")
    public = json.loads((out / "result.json").read_text("utf-8"))

    def inspect(value):
        if isinstance(value, dict):
            assert "scores" not in value
            for key, item in value.items():
                assert not str(key).startswith("holdout::")
                inspect(item)
        elif isinstance(value, list):
            for item in value:
                inspect(item)
        elif isinstance(value, str):
            for secret in HOLDOUT_SECRETS:
                assert secret not in value

    inspect(public)
    restricted = (out / "restricted" / "per_instance.jsonl").read_text("utf-8")
    assert "holdout::tool_use::0001" in restricted


def test_held_out_content_never_reaches_the_reflector(holdout_dir, tmp_path):
    data = bundle("improving")
    reflect = rg.SequencedReflector(data["reflections"])
    run(holdout_dir, tmp_path / "out", reflect=reflect)
    assert reflect.calls, "the reflector must have been consulted"
    for messages in reflect.calls:
        text = json.dumps(messages, ensure_ascii=False)
        for secret in HOLDOUT_SECRETS:
            assert secret not in text, f"reflector saw {secret!r}"
        # Train feedback is allowed and expected.
        assert "train::tool_use" in text


def test_a_higher_mean_with_one_regressed_holdout_item_is_rejected(holdout_dir, tmp_path):
    out = tmp_path / "out"
    result = run(holdout_dir, out, "regressing")
    assert result["accepted"] == []
    assert result["best"]["id"] == "seed"
    assert result["rejected"] and "regresses" in result["rejected"][0]["reason"]
    rows = [json.loads(l) for l in (out / "candidates.jsonl").read_text("utf-8").splitlines()]
    child = [r for r in rows if r["candidate_id"] != "seed"][0]
    assert child["promotable"] is False and child["archive_only"] is True
    assert child["holdout_decision"] == "rejected"


def test_a_fixture_run_cannot_populate_the_student_candidate_arm(
        holdout_dir, tmp_path):
    out = tmp_path / "out"
    run(holdout_dir, out)
    result = json.loads((out / "result.json").read_text("utf-8"))
    with pytest.raises(gepa_evidence.GepaEvidenceError, match="fixture"):
        gepa_evidence.arm_from_gepa(
            result, metric="capability_pass_rate", guardrails=())


def test_production_harness_files_are_unchanged(holdout_dir, tmp_path):
    before = harness_digests()
    run(holdout_dir, tmp_path / "out")
    assert harness_digests() == before


def test_output_is_deterministic_under_a_fixed_seed(holdout_dir, tmp_path):
    run(holdout_dir, tmp_path / "a", seed=7)
    run(holdout_dir, tmp_path / "b", seed=7)
    for name in ("result.json", "candidates.jsonl", "restricted/per_instance.jsonl"):
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()


def test_candidate_evidence_is_append_only(holdout_dir, tmp_path):
    out = tmp_path / "out"
    run(holdout_dir, out)
    with pytest.raises(rg.GepaRunError, match="exist"):
        run(holdout_dir, out)


def test_every_artifact_says_training_authorized_false(holdout_dir, tmp_path):
    out = tmp_path / "out"
    result = run(holdout_dir, out)
    assert result["training_authorized"] is False
    for path in out.rglob("*.json*"):
        for line in path.read_text("utf-8").splitlines():
            if line.strip().startswith("{") and line.strip().endswith("}"):
                assert json.loads(line)["training_authorized"] is False
    assert json.loads((out / "result.json").read_text("utf-8"))["training_authorized"] is False


# --- capability items only --------------------------------------------------


def test_mixed_style_and_correctness_holdout_uses_only_scoreable_items(tmp_path):
    directory = tmp_path / "mixed"
    directory.mkdir()
    records = fixture_records()
    records.extend([
        style_record("holdout::tool_use::style-judge", judge=True),
        style_record("holdout::tool_use::style-exact", judge=False),
    ])
    write_jsonl(directory / "mixed.synthetic.jsonl", records)

    _, selected = rg.holdout_items(
        directory, "tool_use", allow_fixture=True)
    assert {record["id"] for record in selected} == {
        "holdout::tool_use::0001",
        "holdout::tool_use::0002",
        "holdout::tool_use::0003",
    }
    assert all(record["score_role"] == "correctness" for record in selected)
    assert all(record["verifier"]["type"] not in verifiers.QUARANTINED
               for record in selected)


def test_style_only_or_quarantined_holdout_refuses_instead_of_scoring_zero(
        tmp_path):
    directory = tmp_path / "style-only"
    directory.mkdir()
    write_jsonl(directory / "style.synthetic.jsonl", [
        style_record("holdout::tool_use::style-judge", judge=True),
        style_record("holdout::tool_use::style-exact", judge=False),
    ])
    with pytest.raises(
            rg.GepaRunError, match="no scoreable correctness items"):
        rg.holdout_items(directory, "tool_use", allow_fixture=True)


# --- CLI --------------------------------------------------------------------


def run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(ROOT / "src" / "run_gepa.py"), *args],
        capture_output=True, text=True, timeout=120)


def test_cli_plan_help_and_defaults():
    result = run_cli("plan", "--help")
    assert result.returncode == 0
    assert "tool_use" in result.stdout
    assert "12" in result.stdout


def test_cli_plan_on_the_empty_production_set_is_blocked():
    result = run_cli("plan", "--train-set", str(TRAIN),
                     "--holdout-gold", str(ROOT / "data" / "gold"),
                     "--model-registry", "qwen35-9b-instruct",
                     "--endpoint", "http://127.0.0.1:9/v1",
                     "--seed-instruction", SEED)
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan["executable"] is False
    assert gs.BLOCKED_MESSAGE in plan["blockers"]
    assert plan["training_authorized"] is False


def test_cli_real_run_on_the_empty_production_set_refuses_before_network(tmp_path):
    result = run_cli("run", "--train-set", str(TRAIN),
                     "--holdout-gold", str(ROOT / "data" / "gold"),
                     "--model-registry", "qwen35-9b-instruct",
                     "--endpoint", "http://127.0.0.1:9/v1",
                     "--seed-instruction", SEED,
                     "--output", str(tmp_path / "out"), "--real")
    assert result.returncode != 0
    assert gs.BLOCKED_MESSAGE in result.stderr
    assert not (tmp_path / "out").exists()


def test_cli_run_without_real_or_fake_refuses(holdout_dir, tmp_path):
    result = run_cli("run", "--train-set", str(TRAIN),
                     "--holdout-gold", str(holdout_dir),
                     "--model-registry", "qwen35-9b-fixture", "--model-dir", str(MODEL_DIR),
                     "--endpoint", "http://127.0.0.1:9/v1",
                     "--seed-instruction", SEED,
                     "--output", str(tmp_path / "out"), "--allow-fixture")
    assert result.returncode != 0
    assert "--real" in result.stderr and "--fake-bundle" in result.stderr


def test_cli_fake_run_end_to_end(holdout_dir, tmp_path):
    out = tmp_path / "out"
    result = run_cli("run", "--train-set", str(TRAIN),
                     "--holdout-gold", str(holdout_dir),
                     "--model-registry", "qwen35-9b-fixture", "--model-dir", str(MODEL_DIR),
                     "--endpoint", "http://127.0.0.1:9/v1",
                     "--seed-instruction", SEED,
                     "--fake-bundle", str(FIXTURES / "fake_bundle_improving.json"),
                     "--output", str(out), "--allow-fixture")
    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary["training_authorized"] is False
    assert summary["fixture"] is True
    assert (out / "result.json").is_file()


def test_cli_fake_bundle_requires_explicit_allow_fixture(holdout_dir, tmp_path):
    result = run_cli(
        "run", "--train-set", str(TRAIN),
        "--holdout-gold", str(holdout_dir),
        "--model-registry", "qwen35-9b-fixture",
        "--model-dir", str(MODEL_DIR),
        "--endpoint", "http://127.0.0.1:9/v1",
        "--seed-instruction", SEED,
        "--fake-bundle", str(FIXTURES / "fake_bundle_improving.json"),
        "--output", str(tmp_path / "out"))
    assert result.returncode != 0
    assert "--fake-bundle requires explicit --allow-fixture" in result.stderr
    assert not (tmp_path / "out").exists()


def test_gepa_dry_run_is_preserved():
    result = subprocess.run(
        [sys.executable, str(ROOT / "src" / "gepa.py"), "--dry-run"],
        capture_output=True, text=True, timeout=60)
    assert result.returncode == 0
    assert "training_authorized  : false" in result.stdout
