"""Planner tests: every refusal the planner exists for must actually fire.

    ./.venv/bin/python -m pytest tests/test_distill_plan.py -q

The planner's whole value is that it REFUSES. A gate that quietly passes when it
cannot run is worse than no gate, so each case below drives a specific refusal
and asserts it happened — licence freshness, the tokenizer compatibility key,
undeclared hardware, and the mechanical limits of a promoted corpus.

Nothing here reaches the network, loads a model, or reads the real fleet: the
host list is monkeypatched and corpora are written under tmp_path. The model
registry under configs/models/ IS read by the corpus audit, deliberately — that
resolution is part of what is being tested.
"""
from __future__ import annotations

import datetime as _dt
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import distill_plan as dp                                  # noqa: E402

TODAY = _dt.date(2026, 9, 3)


# --- fixtures: the smallest specs that satisfy validate_model ---------------

def license_block(**over) -> dict:
    block = {
        "identifier": "apache-2.0",
        "output_training_permitted": True,
        "reviewed_at": "2026-09-01",
        "recheck_max_age_days": 180,
        "evidence_sha256": "e" * 64,
    }
    block.update(over)
    return block


def teacher(**over) -> dict:
    spec = {
        "model_id": "org/Teacher-30B",
        "role": "teacher",
        "revision": "a" * 40,
        "params": {"total_b": 30.0},
        "tokenizer": {"sha256": "TOK_SHARED"},
        "capabilities": {"tools": True, "logprobs": True},
        "license": license_block(),
    }
    spec.update(over)
    return spec


def student(**over) -> dict:
    spec = {
        "model_id": "org/Student-9B",
        "role": "student",
        "revision": "b" * 40,
        "params": {"total_b": 9.0},
        "tokenizer": {"sha256": "TOK_SHARED"},
        "capabilities": {"tools": True, "logprobs": True},
        "license": license_block(),
        "architecture_profile": "arch",
    }
    spec.update(over)
    return spec


ARCH_OK = {"name": "arch", "mode_c": {"eligible": True, "targets_lm_head": False}}
PLAN_WITH_PAIRS = {"preference": {"scope": ["voice", "faithfulness"]}}
PLAN_NO_PAIRS: dict = {}


# --- licence gate -----------------------------------------------------------

def test_permitted_and_fresh_licence_passes():
    st = dp.license_status(teacher(), TODAY)
    assert st["status"] == "FRESH"
    assert st["problems"] == []
    assert st["age_days"] == 2
    dp.license_gate(teacher(), "t", TODAY)          # must not raise


def test_licence_not_permitted_refuses():
    spec = teacher(license=license_block(output_training_permitted=False))
    assert dp.license_status(spec, TODAY)["status"] == "NOT_PERMITTED"
    with pytest.raises(SystemExit):
        dp.license_gate(spec, "t", TODAY)


def test_licence_older_than_recheck_window_is_stale():
    """Apache 2.0 held across 3.5/3.6 and stopped at the 3.8 flagship. A licence
    reviewed once and trusted forever is the hazard this catches."""
    spec = teacher(license=license_block(reviewed_at="2026-01-01",
                                         recheck_max_age_days=30))
    st = dp.license_status(spec, TODAY)
    assert st["status"] == "STALE"
    assert st["age_days"] == 245 and "limit 30" in st["reason"]
    with pytest.raises(SystemExit):
        dp.license_gate(spec, "t", TODAY)


@pytest.mark.parametrize("lic", [
    license_block(reviewed_at=None),
    license_block(recheck_max_age_days=None),
    license_block(reviewed_at="last Tuesday"),
])
def test_unreviewable_licence_is_unknown_not_fresh(lic):
    spec = teacher(license=lic)
    assert dp.license_status(spec, TODAY)["status"] == "UNKNOWN"
    with pytest.raises(SystemExit):
        dp.license_gate(spec, "t", TODAY)


def test_a_licence_reviewed_in_the_future_is_not_fresh():
    """A future reviewed_at passes `age > max_age` forever. It is a data error,
    not evidence of a recent review — as the v1 freeze showed, where the corpus
    predates the registry entry by 15 days."""
    spec = teacher(license=license_block(reviewed_at="2026-12-25"))
    st = dp.license_status(spec, TODAY)
    assert st["status"] == "UNKNOWN"
    assert st["age_days"] == -113
    assert "is after 2026-09-03" in st["reason"]
    with pytest.raises(SystemExit):
        dp.license_gate(spec, "t", TODAY)


def test_missing_evidence_digest_fails_the_gate():
    """In date, but nothing recorded that the review actually happened."""
    spec = teacher(license=license_block(evidence_sha256=""))
    st = dp.license_status(spec, TODAY)
    assert st["status"] == "FRESH_NO_EVIDENCE"
    assert st["evidence_present"] is False
    with pytest.raises(SystemExit):
        dp.license_gate(spec, "t", TODAY)


# --- compatibility key and mode selection -----------------------------------

def test_matching_compatibility_key_selects_mode_c_under_auto():
    mode, _ = dp.choose_mode("auto", PLAN_WITH_PAIRS, teacher(), "t",
                             student(), ARCH_OK)
    assert dp.tokenizers_compatible(teacher(), student()) is True
    assert mode == dp.MODE_C


def test_mismatched_key_falls_back_to_preference_then_sequence():
    cross = teacher(tokenizer={"sha256": "TOK_OTHER"})
    assert dp.tokenizers_compatible(cross, student()) is False

    mode, notes = dp.choose_mode("auto", PLAN_WITH_PAIRS, cross, "t",
                                 student(), ARCH_OK)
    assert mode == dp.MODE_PREFERENCE
    assert any("compatibility keys differ" in n for n in notes)

    mode, notes = dp.choose_mode("auto", PLAN_NO_PAIRS, cross, "t",
                                 student(), ARCH_OK)
    assert mode == dp.MODE_SEQUENCE


def test_explicit_mode_c_on_a_mismatch_is_refused_never_downgraded():
    """The silent downgrade is the failure mode: an operator who asked for
    token-level KL must not be handed sequence distillation instead."""
    cross = teacher(tokenizer={"sha256": "TOK_OTHER"})
    with pytest.raises(SystemExit):
        dp.choose_mode(dp.MODE_C, PLAN_WITH_PAIRS, cross, "t", student(), ARCH_OK)


def test_explicit_preference_without_pairs_is_refused():
    with pytest.raises(SystemExit):
        dp.choose_mode(dp.MODE_PREFERENCE, PLAN_NO_PAIRS, teacher(), "t",
                       student(), ARCH_OK)


def test_unknown_mode_is_refused():
    with pytest.raises(SystemExit):
        dp.choose_mode("distil-harder", PLAN_WITH_PAIRS, teacher(), "t",
                       student(), ARCH_OK)


# --- Mode C preflight: the non-tokenizer blockers ---------------------------

def test_mode_c_preflight_clean():
    assert dp._mode_c_preflight(teacher(), "t", student(), ARCH_OK) == []


@pytest.mark.parametrize("spec_over,arch,marker", [
    ({"capabilities": {"logprobs": False}}, ARCH_OK, "does not expose logprobs"),
    ({"revision": ""}, ARCH_OK, "revision is not pinned"),
    ({}, None, "no architecture_profile"),
    ({}, {"name": "arch", "mode_c": {"eligible": True, "targets_lm_head": True}},
     "targets lm_head"),
    ({}, {"name": "arch", "mode_c": {"eligible": False}}, "not Mode-C eligible"),
    ({"tokenizer": {}}, ARCH_OK, "tokenizer.sha256 is missing"),
])
def test_mode_c_preflight_blockers(spec_over, arch, marker):
    errs = dp._mode_c_preflight(teacher(**spec_over), "t", student(), arch)
    assert any(marker in e for e in errs), errs
    # auto must fall back rather than run Mode C anyway.
    mode, _ = dp.choose_mode("auto", PLAN_WITH_PAIRS, teacher(**spec_over), "t",
                             student(), arch)
    assert mode != dp.MODE_C


# --- hardware: undeclared capacity is not "probably fine" -------------------

def fleet(*hosts):
    return lambda: [dict(h) for h in hosts]


def test_declared_capacity_above_need_is_servable(monkeypatch):
    monkeypatch.setattr(dp, "_fleet", fleet(
        {"_name": "rented", "gpu_memory_gb": 48, "gpu_count": 1,
         "gpu_memory_utilization": 0.92}))
    hw = dp.hardware_plan(teacher(), ["fp8"])
    attempt = hw["attempts"][0]
    assert attempt["estimated_need_gb"] == pytest.approx(40.5)
    assert [n for n, _ in attempt["eligible_hosts"]] == ["rented"]


def test_host_without_declared_capacity_is_ineligible(monkeypatch):
    """Fail-closed: an undeclared host is unknown, not roomy."""
    monkeypatch.setattr(dp, "_fleet", fleet({"_name": "mystery-box"}))
    hw = dp.hardware_plan(teacher(), ["fp8"])
    assert hw["attempts"][0]["eligible_hosts"] == []


def test_ampere_host_is_excluded_at_fp8_but_not_at_bf16(monkeypatch):
    """ai19's cards are sm_86; FP8 needs >= 8.9. Capacity alone is not enough."""
    monkeypatch.setattr(dp, "_fleet", fleet(
        {"_name": "ampere", "gpu_memory_gb": 80, "gpu_count": 2,
         "supports_fp8": False, "gpu_memory_utilization": 0.90}))
    hw = dp.hardware_plan(teacher(), ["fp8", "bf16"])
    by_precision = {a["precision"]: a["eligible_hosts"] for a in hw["attempts"]}
    assert by_precision["fp8"] == []
    assert [n for n, _ in by_precision["bf16"]] == ["ampere"]


def test_teacher_serving_false_host_is_excluded(monkeypatch):
    monkeypatch.setattr(dp, "_fleet", fleet(
        {"_name": "gateway-box", "gpu_memory_gb": 512, "teacher_serving": False}))
    assert dp.hardware_plan(teacher(), ["fp8"])["attempts"][0]["eligible_hosts"] == []


def test_missing_total_b_refuses_rather_than_guessing(monkeypatch):
    monkeypatch.setattr(dp, "_fleet", fleet({"_name": "rented", "gpu_memory_gb": 48}))
    with pytest.raises(SystemExit):
        dp.hardware_plan(teacher(params={}), ["fp8"])


def test_unknown_vision_tower_marks_the_estimate_incomplete(monkeypatch):
    monkeypatch.setattr(dp, "_fleet", fleet({"_name": "rented", "gpu_memory_gb": 48}))
    spec = teacher(modality={"vision": True}, params={"total_b": 30.0, "vision_b": None})
    assert dp.hardware_plan(spec, ["fp8"])["attempts"][0]["estimate_incomplete"] is True


# --- architecture signature -------------------------------------------------

DENSE_9B = {
    "model_type": "qwen3_5_text",
    "num_hidden_layers": 32,
    "layer_types": ["linear_attention"] * 3 + ["full_attention"],
    "hidden_size": 4096,
    "num_attention_heads": 16,
    "num_key_value_heads": 4,
}


def test_signature_is_stable_for_the_same_shape():
    shuffled = dict(reversed(list(DENSE_9B.items())))
    assert dp.architecture_signature(DENSE_9B) == dp.architecture_signature(dict(DENSE_9B))
    assert dp.architecture_signature(shuffled) == dp.architecture_signature(DENSE_9B)


@pytest.mark.parametrize("field,value", [
    ("num_hidden_layers", 48),
    ("layer_types", ["full_attention"] * 4),
    ("model_type", "qwen3_5_moe_text"),
    ("hidden_size", 5120),
    ("num_key_value_heads", 8),
])
def test_signature_changes_when_the_shape_changes(field, value):
    changed = dict(DENSE_9B, **{field: value})
    assert dp.architecture_signature(changed) != dp.architecture_signature(DENSE_9B)


# --- provenance audit -------------------------------------------------------

def corpus(tmp_path: Path, *rows: dict) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "traces.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


def trace(**over) -> dict:
    row = {
        "family": "f1",
        "source_class": "real",
        "provenance": {"teacher": "muse-glimmer", "repo": "muse-glimmer:30b",
                       "quantization": "fp8", "host": "rented-48gb",
                       "license": "apache-2.0"},
    }
    row.update(over)
    return row


def prov(**over) -> dict:
    p = dict(trace()["provenance"])
    p.update(over)
    return {"provenance": p}


def test_int4_traces_leave_the_fp8_gate_unmet(tmp_path):
    path = corpus(tmp_path, trace(**prov(quantization="int4_ollama",
                                         host="ai19-ollama")))
    report = dp.audit_corpus(path, TODAY)
    assert report["fp8_gate"]["status"] == "UNMET"
    assert report["fp8_gate"]["untrainable_quantizations"] == {"int4_ollama": 1}
    # The signed waiver names exactly this pair, so nothing is uncovered...
    assert report["fp8_gate"]["uncovered_by_waiver"] == {}
    # ...but the waiver authorises proceeding despite the failure, not a pass.
    assert report["trainable_as_is"] is False


def test_gateway_traces_are_outside_the_signed_waiver(tmp_path):
    """A `remote` trace through the third-party gateway is also int4, but the
    waiver names int4_ollama@ai19-ollama only."""
    path = corpus(tmp_path, trace(**prov(quantization="remote", host="gateway")))
    report = dp.audit_corpus(path, TODAY)
    assert report["fp8_gate"]["status"] == "UNMET"
    assert report["fp8_gate"]["uncovered_by_waiver"] == {"remote@gateway": 1}


def test_synthetic_corpus_supports_no_real_office_claim(tmp_path):
    path = corpus(tmp_path, trace(source_class="synthetic"), trace())
    report = dp.audit_corpus(path, TODAY)
    assert report["real_office_claim"] == {"supported": False, "synthetic_traces": 1}
    assert any("synthetic" in limit for limit in report["limits"])


def test_unverified_identity_alone_blocks_trainable_as_is(tmp_path):
    """FP8, real sources, a resolved teacher and a fresh licence — everything
    passes EXCEPT the tokenizer/template digests, which are still placeholders.
    That alone must be disqualifying: the compatibility key those digests carry
    is what chose the distillation mode, and it was never checked against real
    files."""
    path = corpus(tmp_path, trace())
    report = dp.audit_corpus(path, TODAY, teacher_overrides=["muse-glimmer-30b"])

    assert report["fp8_gate"]["status"] == "MET"
    assert report["real_office_claim"]["supported"] is True
    assert report["license_freshness"]["resolved"][0]["status"] == "FRESH"
    assert report["license_freshness"]["unresolved_corpus_teachers"] == []

    assert report["identity_verification"] == {
        "all_verified": False, "unverified_models": ["muse-glimmer-30b"]}
    assert report["trainable_as_is"] is False
    # The warning stays, and is the only thing left to fix.
    assert report["limits"] == [
        "Registry model 'muse-glimmer-30b' digests_verified is not true; "
        "tokenizer/template sha256 unverified."
    ]


def test_the_same_corpus_is_trainable_once_identity_is_verified(tmp_path, monkeypatch):
    """The positive control. Without it, `trainable_as_is: false` could be a
    constant rather than a verdict. The registry file is NOT edited: no
    placeholder digest is invented to make a test pass."""
    verified = dict(dp._load("models", "muse-glimmer-30b"))
    verified["digests_verified"] = True
    monkeypatch.setattr(dp, "_resolve_teacher_specs",
                        lambda teachers, overrides: ({"muse-glimmer-30b": verified}, []))

    report = dp.audit_corpus(corpus(tmp_path, trace()), TODAY)
    assert report["identity_verification"] == {
        "all_verified": True, "unverified_models": []}
    assert report["limits"] == []
    assert report["trainable_as_is"] is True


def test_verified_identity_does_not_rescue_any_other_failure(tmp_path, monkeypatch):
    """Identity is one requirement among several, not an override."""
    verified = dict(dp._load("models", "muse-glimmer-30b"))
    verified["digests_verified"] = True
    monkeypatch.setattr(dp, "_resolve_teacher_specs",
                        lambda teachers, overrides: ({"muse-glimmer-30b": verified}, []))

    quantized = dp.audit_corpus(
        corpus(tmp_path / "a", trace(**prov(quantization="int4_ollama",
                                            host="ai19-ollama"))), TODAY)
    assert quantized["identity_verification"]["all_verified"] is True
    assert quantized["trainable_as_is"] is False

    synthetic = dp.audit_corpus(
        corpus(tmp_path / "b", trace(source_class="synthetic")), TODAY)
    assert synthetic["identity_verification"]["all_verified"] is True
    assert synthetic["trainable_as_is"] is False


def test_identity_is_unverified_when_no_teacher_resolves(tmp_path):
    """Fail-closed: nothing resolved is not the same as nothing wrong."""
    report = dp.audit_corpus(
        corpus(tmp_path, trace(**prov(teacher="some-unregistered-model"))), TODAY)
    assert report["identity_verification"]["all_verified"] is False
    assert report["trainable_as_is"] is False


def test_unregistered_teacher_leaves_licence_freshness_unknown(tmp_path):
    path = corpus(tmp_path, trace(**prov(teacher="some-unregistered-model")))
    report = dp.audit_corpus(path, TODAY)
    assert report["license_freshness"]["unresolved_corpus_teachers"] == \
        ["some-unregistered-model"]
    assert any("Could not resolve" in limit for limit in report["limits"])
    assert report["trainable_as_is"] is False


def test_stale_teacher_licence_surfaces_in_the_audit(tmp_path):
    path = corpus(tmp_path, trace())
    far_future = _dt.date(2036, 1, 1)
    report = dp.audit_corpus(path, far_future, teacher_overrides=["muse-glimmer-30b"])
    assert report["license_freshness"]["resolved"][0]["status"] == "STALE"
    assert report["trainable_as_is"] is False


def test_empty_or_missing_corpus_refuses(tmp_path):
    with pytest.raises(SystemExit):
        dp.audit_corpus(tmp_path / "nope.jsonl", TODAY)
    blank = tmp_path / "blank.jsonl"
    blank.write_text("\n\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        dp.audit_corpus(blank, TODAY)


def test_audit_never_authorizes_training(tmp_path):
    report = dp.audit_corpus(corpus(tmp_path, trace()), TODAY)
    assert report["authorizes_training"] is False


# --- the signature must read the shape where Qwen3.5 actually puts it -------

def nested(**text_over) -> dict:
    """A unified VL config in the shape Qwen3.5 really ships."""
    text = dict(DENSE_9B, model_type="qwen3_5_text", **text_over)
    return {
        "model_type": "qwen3_5",
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "image_token_id": 151655,
        "text_config": text,
        "vision_config": {"model_type": "qwen3_5_vl", "depth": 27,
                          "hidden_size": 1152, "out_hidden_size": 4096},
    }


def test_nested_vl_config_signature_reflects_the_text_stack():
    """Regression: reading only the top level of a Qwen3.5 config yields
    all-null, so a 32-layer dense student and a 48-layer MoE would share one
    signature and the train-time check would enforce nothing."""
    dense = nested()
    moe = nested(num_hidden_layers=48, num_experts=256, num_experts_per_tok=8)
    assert dp.architecture_signature(dense) != dp.architecture_signature(moe)

    described = dp.describe_architecture(dense)
    assert described["num_hidden_layers"] == 32
    assert described["text_model_type"] == "qwen3_5_text"
    assert described["model_type"] == "qwen3_5"


def test_a_swapped_vision_tower_changes_the_signature():
    other = nested()
    other["vision_config"] = dict(other["vision_config"], depth=32)
    assert dp.architecture_signature(other) != dp.architecture_signature(nested())


def test_a_shapeless_config_refuses_rather_than_digesting_nulls():
    with pytest.raises(SystemExit):
        dp.architecture_signature({"model_type": "qwen3_5"})
    with pytest.raises(SystemExit):
        dp.architecture_signature({"model_type": "qwen3_5", "text_config": {}})


# --- registry <-> serving config: one checkpoint, two files -----------------

def registered(**over) -> dict:
    spec = student(serving_config="office-student-9b")
    spec["model_id"] = "Qwen/Qwen3.5-9B"
    spec["tokenizer"] = {"model_id": "Qwen/Qwen3.5-9B", "sha256": "TOK"}
    spec.update(over)
    return spec


def serving(**over) -> dict:
    cfg = {
        "registry_model": "qwen35-9b-instruct",
        "served_model_name": "Qwen/Qwen3.5-9B",
        "license": "apache-2.0",
        "repos": {"bf16": "Qwen/Qwen3.5-9B"},
        "tokenizer": "Qwen/Qwen3.5-9B",
    }
    cfg.update(over)
    return cfg


def test_the_shipped_pairs_actually_reconcile():
    """The real files, not fixtures: configs/models/*.yaml and their
    configs/teachers/*.yaml counterparts must agree today."""
    for name in ("qwen35-9b-instruct", "muse-glimmer-30b"):
        spec = dp._load("models", name)
        assert spec.get("serving_config"), f"{name} declares no serving_config"
        assert dp.serving_mismatches(spec, name) == []


def test_a_one_sided_link_is_caught(monkeypatch):
    monkeypatch.setattr(dp, "_serving_config",
                        lambda _: serving(registry_model="something-else"))
    problems = dp.serving_mismatches(registered(), "qwen35-9b-instruct")
    assert any("must be mutual" in p for p in problems)


def test_a_base_checkpoint_in_the_serving_repos_is_caught(monkeypatch):
    """The hazard src/model_ids.py exists for: Qwen3.5-9B and Qwen3.5-9B-Base
    are one suffix apart, and train/BASE_VS_INSTRUCT.md says they are not
    interchangeable. Gates serving Base while the plan targets instruct would
    otherwise look fine."""
    monkeypatch.setattr(dp, "_serving_config",
                        lambda _: serving(repos={"bf16": "Qwen/Qwen3.5-9B-Base"}))
    problems = dp.serving_mismatches(registered(), "qwen35-9b-instruct")
    assert any("repos.bf16" in p for p in problems)


def test_a_disagreeing_licence_identifier_is_caught(monkeypatch):
    monkeypatch.setattr(dp, "_serving_config", lambda _: serving(license="nvidia-oml"))
    problems = dp.serving_mismatches(registered(), "qwen35-9b-instruct")
    assert any("license" in p for p in problems)


def test_a_different_tokenizer_repo_is_caught(monkeypatch):
    """prompt_sha256 in the corpus was computed over rendered prompts; a
    different tokenizer repo silently changes what was measured."""
    monkeypatch.setattr(dp, "_serving_config",
                        lambda _: serving(tokenizer="Qwen/Qwen3.5-9B-Base"))
    assert any("tokenizer" in p
               for p in dp.serving_mismatches(registered(), "qwen35-9b-instruct"))


def test_a_local_serving_alias_is_not_an_identity_claim(monkeypatch):
    """`served_model_name: muse-glimmer` is what the server registers, not a
    repo id. Only a repo-shaped name is compared."""
    monkeypatch.setattr(dp, "_serving_config",
                        lambda _: serving(served_model_name="muse-glimmer"))
    assert dp.serving_mismatches(registered(), "qwen35-9b-instruct") == []


def test_a_serving_config_that_does_not_exist_is_caught(monkeypatch):
    monkeypatch.setattr(dp, "_serving_config", lambda _: None)
    assert any("names no file" in p
               for p in dp.serving_mismatches(registered(), "qwen35-9b-instruct"))


def test_an_unlinked_registry_entry_is_not_an_error():
    """qwen35-122b-a10b is an illustrative entry with no serving config. The
    plan warns; it does not invent a reconciliation."""
    assert dp.serving_mismatches(student(serving_config=None), "x") == []


def test_reconcile_gate_refuses_on_drift(monkeypatch):
    monkeypatch.setattr(dp, "_serving_config",
                        lambda _: serving(repos={"bf16": "Qwen/Qwen3.5-9B-Base"}))
    with pytest.raises(SystemExit):
        dp.reconcile_gate(registered(), "qwen35-9b-instruct")


# --- --dry-run: a mapping, not an execution ---------------------------------

def repo_state() -> dict:
    """Every tracked file the pipeline could plausibly write to."""
    state = {}
    for folder in ("data", "train", "configs", "adapters"):
        base = ROOT / folder
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            if path.is_file():
                stat = path.stat()
                state[str(path)] = (stat.st_size, stat.st_mtime_ns)
    return state


def test_dry_run_writes_nothing_and_calls_nothing():
    before = repo_state()
    proc = subprocess.run(
        [str(ROOT / ".venv" / "bin" / "python"), str(ROOT / "src" / "distill_plan.py"),
         "plan", "office-v2-sequence", "--today", "2026-09-03", "--dry-run"],
        capture_output=True, text=True, cwd=ROOT)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "DRY RUN" in proc.stdout
    assert repo_state() == before


def test_dry_run_prints_no_training_command():
    """Execution stays behind train/TRAINING_BLOCKED.md. A pasteable train
    command in the output is how a blocked run becomes an accidental one."""
    lines = dp.dry_run_sequence("p", dp.MODE_SEQUENCE, teacher(), "t",
                                student(), "rented-48gb")
    commands = [l for l in lines if l.startswith("./.venv")]
    assert commands, "the sequence should name real tools"
    assert not any("train_qlora" in l or "train_tinker" in l for l in commands)
    assert any("TRAINING_BLOCKED.md is controlling" in l for l in lines)


def test_dry_run_names_the_serving_configs_not_the_registry_names():
    """src/generate.py and src/run_gates.py resolve configs/teachers/ names;
    handing them a registry name would fail at the first command."""
    lines = dp.dry_run_sequence("p", dp.MODE_SEQUENCE,
                                teacher(serving_config="muse-glimmer"), "muse-glimmer-30b",
                                student(serving_config="office-student-9b"), "h")
    text = "\n".join(lines)
    assert "--teacher muse-glimmer \\" in text
    assert "--teacher office-student-9b" in text
    assert "muse-glimmer-30b" not in text


# --- one audit, every limit ---------------------------------------------------
#
# Model quant, source_class, FP8 status, licence freshness AND harness
# attribution in one report, so nobody has to remember to run a second command
# to discover that a corpus cannot support the claim being made of it.

def harness_block(**over) -> dict:
    value = {
        "schema_version": 1, "name": "h", "status": "candidate",
        "digest": "a" * 64, "compatible_registry_models": ["muse-glimmer-30b"],
        "execution_model_registry": "muse-glimmer-30b",
        "prompt_sha256": "b" * 64, "prompt_verified": True,
        "tool_policy_digest": "c" * 64, "verification_policy_digest": "d" * 64,
    }
    value.update(over)
    return value


DECLARED = {"required": True, "attributed": True, "name": "h", "digest": "a" * 64,
            "prompt_verified": True, "prompt_sha256": "b" * 64,
            "execution_model_registry": "muse-glimmer-30b"}


def declaring_pass(tmp_path: Path, records: list[dict], declared=DECLARED) -> Path:
    import hashlib
    directory = tmp_path / "pass"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "traces.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    (directory / "MANIFEST.json").write_text(json.dumps({
        "files": {path.name: {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}},
        "harness": declared}), encoding="utf-8")
    return path


@pytest.mark.requires_local_corpus
def test_the_legacy_corpus_audit_reports_no_attribution():
    report = dp.audit_corpus(ROOT / "data" / "promoted" / "train.jsonl", TODAY)
    assert report["harness"] == {"required": False, "attributed": False,
                                 "coverage": 0.0}
    assert report["authorizes_training"] is False
    # No limit line: an unattributed corpus is the norm today, and a complaint
    # on every legacy audit would drown the limits that are actionable.
    assert not any("harness" in limit for limit in report["limits"])


@pytest.mark.requires_local_corpus
def test_the_audit_keeps_every_pre_existing_key():
    """The harness block is additive. A reshaped report would break every reader
    that predates it."""
    report = dp.audit_corpus(ROOT / "data" / "promoted" / "train.jsonl", TODAY)
    for key in ("corpus", "records", "teachers", "repos", "quantizations", "hosts",
                "licenses_recorded", "source_classes", "fp8_gate",
                "real_office_claim", "license_freshness", "identity_verification",
                "limits", "trainable_as_is", "authorizes_training"):
        assert key in report, key


def test_a_declared_harness_aware_corpus_reports_full_coverage(tmp_path):
    path = declaring_pass(tmp_path, [trace(**{"harness_provenance": harness_block()}),
                                     trace(**{"harness_provenance": harness_block()})])
    report = dp.audit_corpus(path, TODAY, teacher_overrides=["muse-glimmer-30b"])
    assert report["harness"]["required"] is True
    assert report["harness"]["coverage"] == 1.0
    assert report["harness"]["observed"] == DECLARED
    assert "disagreement" not in report["harness"]
    assert not any("harness" in limit for limit in report["limits"])


def test_a_declared_but_under_attributed_corpus_reports_the_disagreement(tmp_path):
    """Not merely "0% covered": the corpus claims to be something it is not, and
    saying so is more useful than either number alone."""
    path = declaring_pass(tmp_path, [trace(**{"harness_provenance": harness_block()}),
                                     trace()])
    report = dp.audit_corpus(path, TODAY, teacher_overrides=["muse-glimmer-30b"])
    assert report["harness"]["required"] is True
    assert report["harness"]["coverage"] == 0.5
    assert "not attributed" in report["harness"]["disagreement"]
    assert any("harness attribution is incoherent" in limit
               for limit in report["limits"])
    assert report["trainable_as_is"] is False


def test_a_legacy_declaration_over_attributed_traces_is_a_contradiction(tmp_path):
    """No adjacent manifest means legacy. Traces carrying attribution then
    contradict the declaration, and the audit must say so rather than reporting
    attributed: true as though it were fine."""
    path = tmp_path / "traces.jsonl"
    path.write_text("\n".join(json.dumps(trace(**{"harness_provenance": harness_block()}))
                              for _ in range(2)) + "\n", encoding="utf-8")
    report = dp.audit_corpus(path, TODAY, teacher_overrides=["muse-glimmer-30b"])
    assert report["harness"]["required"] is False
    assert "harness-aware" in report["harness"]["disagreement"]
    assert any("incoherent" in limit for limit in report["limits"])
    assert report["trainable_as_is"] is False


def test_incomplete_required_attribution_blocks_trainable_as_is(tmp_path, monkeypatch):
    """Everything else clean — fp8, real sources, verified identity — and it is
    still not trainable, because a corpus whose attribution is incoherent is a
    corpus nobody can characterise."""
    verified = dict(dp._load("models", "muse-glimmer-30b"))
    verified["digests_verified"] = True
    monkeypatch.setattr(dp, "_resolve_teacher_specs",
                        lambda teachers, overrides: ({"muse-glimmer-30b": verified}, []))
    path = declaring_pass(tmp_path, [trace(**{"harness_provenance": harness_block()}),
                                     trace()])
    report = dp.audit_corpus(path, TODAY)
    assert report["fp8_gate"]["status"] == "MET"
    assert report["identity_verification"]["all_verified"] is True
    assert report["harness"]["disagreement"]
    assert report["trainable_as_is"] is False, "harness incoherence must block it"
