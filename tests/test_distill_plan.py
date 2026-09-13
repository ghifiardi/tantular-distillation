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
import yaml

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


RECEIPT_DIGEST = "9a" * 32
RECEIPT_REVISION = "e8" * 20            # a complete 40-char lowercase commit
MUSE_REGISTRY = "muse-glimmer-30b"
MUSE_MODEL_ID = "meta-models/Muse-Glimmer-30B"


def receipt(digest: str = RECEIPT_DIGEST, registry_model: str = MUSE_REGISTRY,
            model_id: str = MUSE_MODEL_ID, revision: str = RECEIPT_REVISION,
            field: str = "manifest_digest") -> dict:
    """A complete execution-artifact receipt: a digest BOUND to the registry
    entry it claims to be a receipt for. A digest alone is 64 characters that
    could belong to any artifact."""
    return {"registry_model": registry_model, "model_id": model_id,
            "revision": revision, field: digest}


def bound_spec(**over) -> dict:
    """The registry entry that the default receipt binds to: same model_id,
    same pinned revision, digests verified."""
    spec = dict(yaml.safe_load(
        (ROOT / "configs" / "models" / "muse-glimmer-30b.yaml").read_text()))
    spec["revision"] = RECEIPT_REVISION
    spec["digests_verified"] = True
    spec.update(over)
    return spec


def trace_with_receipt(digest: str = RECEIPT_DIGEST, **over) -> dict:
    """A trace that names the immutable artifact which produced it.

    Deliberately NOT the default. Recording an execution-artifact receipt is a
    new, mandatory provenance requirement, and burying it in the generic
    fixture would let a future test pass because its shared fixture quietly had
    stronger provenance than the scenario it claims to describe. Every positive
    control that needs identity to be complete opts in here, visibly.
    """
    row = trace(**over)
    row["provenance"] = dict(row["provenance"], execution_artifact=receipt(digest))
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


def test_unverified_identity_alone_blocks_trainable_as_is(tmp_path, monkeypatch):
    """FP8, real sources, a resolved teacher and a fresh licence — everything
    passes EXCEPT the tokenizer/template digests, which are still placeholders.
    That alone must be disqualifying: the compatibility key those digests carry
    is what chose the distillation mode, and it was never checked against real
    files.

    To isolate that one component the registry must be PINNED but UNVERIFIED:
    a receipt cannot bind to a placeholder revision, so leaving the revision
    unpinned would fail execution_artifact_ready too and the test would no
    longer be about the digests.
    """
    monkeypatch.setattr(
        dp, "_resolve_teacher_specs",
        lambda teachers, overrides: (
            {MUSE_REGISTRY: bound_spec(digests_verified=False)}, []))
    # opts in to a receipt so the ONLY thing left unverified is the registry
    path = corpus(tmp_path, trace_with_receipt())
    report = dp.audit_corpus(path, TODAY, teacher_overrides=["muse-glimmer-30b"])

    assert report["fp8_gate"]["status"] == "MET"
    assert report["real_office_claim"]["supported"] is True
    assert report["license_freshness"]["resolved"][0]["status"] == "FRESH"
    assert report["license_freshness"]["unresolved_corpus_teachers"] == []

    assert report["identity_verification"] == {
        "all_verified": False,
        "registry_identity_ready": False,   # the only thing left to fix
        "execution_artifact_ready": True,   # these traces name their artifact
        "unverified_models": ["muse-glimmer-30b"]}
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
    # bound_spec pins a revision as well as verifying digests: a receipt cannot
    # bind to a placeholder, so verifying digests alone is not enough here
    verified = bound_spec()
    monkeypatch.setattr(dp, "_resolve_teacher_specs",
                        lambda teachers, overrides: ({"muse-glimmer-30b": verified}, []))

    # opts in explicitly: this control needs identity COMPLETE, which means the
    # traces must name their artifact as well as the registry being verified
    report = dp.audit_corpus(corpus(tmp_path, trace_with_receipt()), TODAY)
    assert report["identity_verification"] == {
        "all_verified": True,
        "registry_identity_ready": True,
        "execution_artifact_ready": True,
        "unverified_models": []}
    assert report["limits"] == []
    assert report["trainable_as_is"] is True


def test_verified_identity_does_not_rescue_any_other_failure(tmp_path, monkeypatch):
    """Identity is one requirement among several, not an override."""
    # bound_spec pins a revision as well as verifying digests: a receipt cannot
    # bind to a placeholder, so verifying digests alone is not enough here
    verified = bound_spec()
    monkeypatch.setattr(dp, "_resolve_teacher_specs",
                        lambda teachers, overrides: ({"muse-glimmer-30b": verified}, []))

    quantized = dp.audit_corpus(
        corpus(tmp_path / "a", trace_with_receipt(**prov(quantization="int4_ollama",
                                            host="ai19-ollama"))), TODAY)
    assert quantized["identity_verification"]["all_verified"] is True
    assert quantized["trainable_as_is"] is False

    synthetic = dp.audit_corpus(
        corpus(tmp_path / "b", trace_with_receipt(source_class="synthetic")), TODAY)
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
    # bound_spec pins a revision as well as verifying digests: a receipt cannot
    # bind to a placeholder, so verifying digests alone is not enough here
    verified = bound_spec()
    monkeypatch.setattr(dp, "_resolve_teacher_specs",
                        lambda teachers, overrides: ({"muse-glimmer-30b": verified}, []))
    path = declaring_pass(tmp_path, [trace_with_receipt(**{"harness_provenance": harness_block()}),
                                     trace_with_receipt()])
    report = dp.audit_corpus(path, TODAY)
    assert report["fp8_gate"]["status"] == "MET"
    assert report["identity_verification"]["all_verified"] is True
    assert report["harness"]["disagreement"]
    assert report["trainable_as_is"] is False, "harness incoherence must block it"


# --- the legacy verdict must keep its historical REASONS ---------------------

@pytest.mark.requires_local_corpus
def test_the_legacy_corpus_is_blocked_by_exactly_its_historical_reasons():
    """`trainable_as_is: false` is not evidence that nothing changed.

    A new blocker can appear while an old one is fixed and the verdict never
    moves. Harness readiness in particular must contribute NOTHING here: a
    legacy corpus declares no harness and carries no attribution, so it is
    coherent, and its refusal must still come from quantization, synthetic
    sources and unverified identity.
    """
    report = dp.audit_corpus(ROOT / "data" / "promoted" / "train.jsonl", TODAY)

    assert report["readiness"] == {
        "fp8_ready": False,               # int4_ollama traces
        "source_ready": False,            # 136/136 synthetic
        "registry_identity_ready": False, # digests_verified is not true
        "execution_artifact_ready": False,# only a mutable Ollama tag recorded
        "identity_ready": False,          # the conjunction of the two above
        "license_ready": True,            # apache-2.0, reviewed and in date
        "harness_ready": True,            # no declaration, no attribution: coherent
    }
    assert report["trainable_as_is"] is False

    assert report["harness"] == {"required": False, "attributed": False,
                                 "coverage": 0.0}
    assert "disagreement" not in report["harness"]
    assert not any("harness" in limit for limit in report["limits"])


@pytest.mark.requires_local_corpus
def test_trainable_as_is_is_exactly_the_conjunction_of_readiness():
    """Derived, not computed alongside — so the block and the verdict cannot
    drift apart and disagree about why."""
    report = dp.audit_corpus(ROOT / "data" / "promoted" / "train.jsonl", TODAY)
    assert report["trainable_as_is"] == all(report["readiness"].values())


def test_a_harness_disagreement_shows_up_as_a_named_blocker(tmp_path):
    """The corollary: when harness readiness IS the blocker, it says so rather
    than hiding inside a single false."""
    path = declaring_pass(tmp_path, [trace(**{"harness_provenance": harness_block()}),
                                     trace()])
    report = dp.audit_corpus(path, TODAY, teacher_overrides=["muse-glimmer-30b"])
    assert report["readiness"]["harness_ready"] is False
    assert report["readiness"]["fp8_ready"] is True
    assert report["trainable_as_is"] is False


# --- identity is two claims, not one ----------------------------------------
#
# Conflating them let a corpus inherit a verification it had no right to.
# Qualifying the CURRENT registry entry says today's canonical checkpoint is
# pinned; it says nothing about which artifact generated traces months ago. The
# legacy corpus records only repo: "muse-glimmer:30b" -- an Ollama tag that can
# be re-pointed -- so nothing in it identifies the bytes that ran.

def test_identity_ready_is_exactly_the_conjunction_of_its_two_components(tmp_path):
    """Derived, not computed alongside, so the parts and the verdict cannot
    drift apart. Synthetic corpus: this property holds regardless of inputs and
    should run wherever CI runs."""
    path = declaring_pass(tmp_path, [trace(), trace()])
    report = dp.audit_corpus(path, TODAY, teacher_overrides=["muse-glimmer-30b"])
    r = report["readiness"]
    assert r["identity_ready"] == (
        r["registry_identity_ready"] and r["execution_artifact_ready"])
    assert report["trainable_as_is"] == all(r.values())


def test_a_verified_registry_cannot_retroactively_upgrade_the_legacy_corpus(
        tmp_path, monkeypatch):
    """The defect this split exists to prevent, made reachable on purpose.

    With the teacher's digests_verified true, registry_identity_ready becomes
    true -- and identity_ready must STILL be false, because the traces name no
    artifact. Before the split, this flipped identity_ready to true and marked a
    corpus generated by an unpinned tag as identity-verified.
    """
    spec = dict(yaml.safe_load(
        (ROOT / "configs" / "models" / "muse-glimmer-30b.yaml").read_text()))
    spec["digests_verified"] = True
    monkeypatch.setattr(dp, "_resolve_teacher_specs",
                        lambda names, overrides: ({"muse-glimmer-30b": spec}, []))

    # plain trace(): a mutable tag and nothing else, like the legacy corpus
    path = declaring_pass(tmp_path, [trace(), trace()])
    report = dp.audit_corpus(path, TODAY, teacher_overrides=["muse-glimmer-30b"])
    r = report["readiness"]

    # the premise really did flip -- otherwise this test proves nothing
    assert r["registry_identity_ready"] is True
    # ...and the verdict still refuses
    assert r["execution_artifact_ready"] is False
    assert r["identity_ready"] is False
    assert report["trainable_as_is"] is False
    assert report["authorizes_training"] is False


def rows_with(artifact, n=1):
    """n traces whose provenance carries exactly this execution_artifact value.
    The sentinel `_ABSENT` omits the key entirely."""
    out = []
    for _ in range(n):
        prov = {"repo": "muse-glimmer:30b", "host": "ai19-ollama",
                "runtime": "ollama", "template_sha256": "a" * 64}
        if artifact is not _ABSENT:
            prov["execution_artifact"] = artifact
        out.append({"provenance": prov})
    return out


_ABSENT = object()
# hex LETTERS on purpose: "9"*64 is unchanged by .upper(), so a digest of
# digits alone cannot test the lowercase rule.
GOOD = "9a" * 32
OTHER = "8b" * 32


@pytest.mark.parametrize("label, artifact, ready, valid, missing, malformed", [
    # absent is MISSING -- honestly incomplete, not a false claim
    ("absent",             _ABSENT,                                False, 0, 1, 0),
    # every present-but-unreadable shape is MALFORMED, never absence
    ("null",               None,                                   False, 0, 0, 1),
    ("empty mapping",      {},                                     False, 0, 0, 1),
    ("scalar",             "sha256-ish",                           False, 0, 0, 1),
    ("list",               [],                                     False, 0, 0, 1),
    ("list with content",  [{"manifest_digest": GOOD}],            False, 0, 0, 1),
    ("unknown key only",   {"artifact": GOOD},                     False, 0, 0, 1),
    ("template only",      {"template_sha256": GOOD},              False, 0, 0, 1),
    # the fail-open case: a valid first field must not mask a malformed second
    ("valid then malformed", receipt() | {"blob_digest": "nope"},  False, 0, 0, 1),
    ("malformed then valid", {"blob_digest": "nope"} | receipt(),  False, 0, 0, 1),
    # two well-formed digests are two claims; nothing can say which is authoritative
    ("two valid fields",   receipt() | {"blob_digest": GOOD},      False, 0, 0, 1),
    ("three valid fields", receipt() | {"blob_digest": GOOD,
                                        "model_digest": GOOD},     False, 0, 0, 1),
    # digest shape
    ("uppercase",          receipt(digest=GOOD.upper()),           False, 0, 0, 1),
    ("sha256: prefix",     receipt(digest="sha256:" + GOOD),       False, 0, 0, 1),
    ("short",              receipt(digest=GOOD[:63]),              False, 0, 0, 1),
    ("long",               receipt(digest=GOOD + "0"),             False, 0, 0, 1),
    ("non-hex",            receipt(digest="z" * 64),               False, 0, 0, 1),
    ("integer",            receipt(digest=1234),                   False, 0, 0, 1),
    ("null digest",        receipt(digest=None),                   False, 0, 0, 1),
    # a digest with no binding is 64 characters belonging to nothing
    ("digest, no binding", {"manifest_digest": GOOD},              False, 0, 0, 1),
    ("no registry_model",  {k: v for k, v in receipt().items()
                            if k != "registry_model"},             False, 0, 0, 1),
    ("no model_id",        {k: v for k, v in receipt().items()
                            if k != "model_id"},                   False, 0, 0, 1),
    ("no revision",        {k: v for k, v in receipt().items()
                            if k != "revision"},                   False, 0, 0, 1),
    ("placeholder revision", receipt(revision="REPLACE_WITH_PINNED_HUB_COMMIT"),
                                                                   False, 0, 0, 1),
    ("short revision",     receipt(revision="e8" * 10),            False, 0, 0, 1),
    ("uppercase revision", receipt(revision=("e8" * 20).upper()),  False, 0, 0, 1),
    ("empty registry_model", receipt(registry_model=""),           False, 0, 0, 1),
    # the only accepted shapes: a digest BOUND to a registry entry
    ("bound manifest",     receipt(),                              True,  1, 0, 0),
    ("bound blob",         receipt(field="blob_digest"),           True,  1, 0, 0),
    ("bound model",        receipt(field="model_digest"),          True,  1, 0, 0),
    ("bound plus extras",  receipt() | {"note": "x"},              True,  1, 0, 0),
])
def test_the_execution_artifact_state_table(label, artifact, ready, valid,
                                            missing, malformed):
    """Absent and malformed are different states, and a present-but-unreadable
    receipt must never read as ordinary absence -- the fail-open bug this
    repository already fixed once for harness provenance."""
    result = dp.execution_artifact_receipts(rows_with(artifact))
    assert result["ready"] is ready, label
    assert result["valid_receipts"] == valid, label
    assert result["missing_receipts"] == missing, label
    assert result["malformed_receipts"] == malformed, label


def test_a_mutable_serving_alias_is_not_an_execution_artifact_receipt():
    """repo/host/runtime are aliases. An Ollama tag can be re-pointed at other
    bytes, so it cannot say what produced a trace -- and its absence is
    MISSING, not malformed."""
    result = dp.execution_artifact_receipts(rows_with(_ABSENT))
    assert result["ready"] is False
    assert result["missing_receipts"] == 1
    assert result["malformed_receipts"] == 0
    assert "mutable serving aliases" in result["reason"]


def test_template_sha256_is_not_accepted_as_an_artifact_receipt():
    """It pins the rendered interface -- what the model was fed -- not the
    model. Both matter; they are not the same claim. Note the alias rows all
    carry template_sha256 and are still missing a receipt."""
    assert dp.execution_artifact_receipts(rows_with(_ABSENT))["valid_receipts"] == 0
    assert dp.execution_artifact_receipts(
        rows_with({"template_sha256": GOOD}))["malformed_receipts"] == 1


def test_a_complete_uniform_receipt_is_ready():
    result = dp.execution_artifact_receipts(rows_with(receipt(), n=3))
    assert result["ready"] is True
    assert (result["valid_receipts"], result["missing_receipts"],
            result["malformed_receipts"]) == (3, 0, 0)
    assert result["distinct_receipts"] == [{
        "registry_model": MUSE_REGISTRY, "model_id": MUSE_MODEL_ID,
        "revision": RECEIPT_REVISION, "digest_field": "manifest_digest",
        "digest": GOOD}]


def test_a_partially_receipted_corpus_is_not_ready():
    """Half a corpus naming its artifact cannot say what produced the rest."""
    result = dp.execution_artifact_receipts(
        rows_with(receipt()) + rows_with(_ABSENT))
    assert result["ready"] is False
    assert (result["valid_receipts"], result["missing_receipts"]) == (1, 1)
    assert "1/2" in result["reason"]


def test_one_malformed_receipt_disqualifies_an_otherwise_complete_corpus():
    """Not skipped, not outvoted by its neighbours."""
    result = dp.execution_artifact_receipts(
        rows_with(receipt(), n=9) + rows_with(receipt(digest="nope")))
    assert result["ready"] is False
    assert result["malformed_receipts"] == 1
    assert "malformed" in result["reason"]


def test_two_distinct_artifacts_in_one_corpus_are_not_ready():
    result = dp.execution_artifact_receipts(
        rows_with(receipt()) + rows_with(receipt(digest=OTHER)))
    assert result["ready"] is False
    assert "2 distinct execution artifacts" in result["reason"]


def test_the_same_digest_under_different_fields_is_still_two_artifacts():
    """manifest_digest:X and blob_digest:X are different claims about where the
    identity came from, so they are not interchangeable."""
    result = dp.execution_artifact_receipts(
        rows_with(receipt()) + rows_with(receipt(field="blob_digest")))
    assert result["ready"] is False
    assert len(result["distinct_receipts"]) == 2


@pytest.mark.requires_local_corpus
def test_the_legacy_corpus_is_missing_receipts_not_malformed():
    """The distinction matters: 136 traces that never recorded an artifact are
    honestly incomplete. Reading them as malformed would accuse the generator
    of writing something unreadable, and reading malformed as missing would let
    a broken receipt pass as an honest gap."""
    report = dp.audit_corpus(ROOT / "data" / "promoted" / "train.jsonl", TODAY)
    artifact = report["execution_artifact"]
    assert artifact["traces"] == 136
    assert artifact["valid_receipts"] == 0
    assert artifact["missing_receipts"] == 136
    assert artifact["malformed_receipts"] == 0
    assert artifact["ready"] is False
    assert "mutable serving aliases" in artifact["reason"]


@pytest.mark.parametrize("artifact", [
    _ABSENT, None, {}, "scalar", [], {"unknown": 1},
    receipt(), receipt() | {"blob_digest": "bad"}, receipt(digest=GOOD.upper()),
])
def test_the_three_counts_partition_the_corpus(artifact):
    """valid + missing + malformed == traces, always.

    This is what makes `malformed_receipts == 0` safe to state in the ready
    rule even though it is redundant today: each trace lands in exactly one
    bucket, so a malformed trace can never also be counted valid. A refactor
    that let one trace contribute twice would break this partition first, and
    the redundant clause would then start doing real work.
    """
    result = dp.execution_artifact_receipts(rows_with(artifact, n=4))
    assert (result["valid_receipts"] + result["missing_receipts"]
            + result["malformed_receipts"]) == result["traces"] == 4


def test_ready_is_exactly_the_documented_four_part_rule():
    """ready == traces > 0 and valid == traces and malformed == 0
       and exactly one distinct receipt."""
    for artifact, n in [(_ABSENT, 2), (receipt(), 2),
                        (receipt(digest="bad"), 2), (None, 1), ({}, 3)]:
        r = dp.execution_artifact_receipts(rows_with(artifact, n=n))
        assert r["ready"] == (r["traces"] > 0
                              and r["valid_receipts"] == r["traces"]
                              and r["malformed_receipts"] == 0
                              and len(r["distinct_receipts"]) == 1)
    mixed = rows_with(receipt()) + rows_with(receipt(digest=OTHER))
    r = dp.execution_artifact_receipts(mixed)
    assert r["ready"] is False and len(r["distinct_receipts"]) == 2


# --- the receipt must BIND to the resolved registry entry -------------------
#
# A digest alone is 64 characters that could belong to any artifact. Without
# these comparisons, a Qwen artifact's receipt would satisfy a verified Muse
# registry entry: the corpus would be "identity verified" about a checkpoint
# that never produced it. The registry VALIDATES a receipt; it never
# manufactures one, so every value compared here came from the traces.

def audit_with(tmp_path, artifact, spec=None, n=2):
    rows = []
    for _ in range(n):
        row = trace()
        row["provenance"] = dict(row["provenance"], execution_artifact=artifact)
        rows.append(row)
    path = declaring_pass(tmp_path, rows)
    specs = {MUSE_REGISTRY: spec if spec is not None else bound_spec()}
    import unittest.mock as _mock
    with _mock.patch.object(dp, "_resolve_teacher_specs",
                            lambda teachers, overrides: (specs, [])):
        return dp.audit_corpus(path, TODAY, teacher_overrides=[MUSE_REGISTRY])


def test_a_receipt_bound_to_the_resolved_registry_entry_is_ready(tmp_path):
    """The positive control, with the binding made explicit: the receipt names
    the same registry entry, model_id and pinned revision the registry does."""
    report = audit_with(tmp_path, receipt())
    assert report["execution_artifact"]["bound"] is True
    assert report["execution_artifact"]["binding_problems"] == []
    assert report["readiness"]["execution_artifact_ready"] is True
    assert report["readiness"]["registry_identity_ready"] is True
    assert report["readiness"]["identity_ready"] is True


@pytest.mark.parametrize("label, artifact", [
    ("wrong registry model", receipt(registry_model="qwen35-9b-instruct")),
    ("wrong model id",       receipt(model_id="Qwen/Qwen3.5-9B")),
    ("wrong revision",       receipt(revision="a4" * 20)),
])
def test_a_valid_digest_bound_to_the_wrong_entry_refuses(tmp_path, label, artifact):
    """Each of these is a well-formed receipt. What disqualifies it is that it
    describes a different checkpoint from the one the registry resolved."""
    report = audit_with(tmp_path, artifact)
    ea = report["execution_artifact"]
    assert ea["ready"] is False, label
    assert ea["bound"] is False, label
    assert ea["binding_problems"], label
    assert report["readiness"]["identity_ready"] is False, label
    # the registry side is fine -- only the binding failed
    assert report["readiness"]["registry_identity_ready"] is True, label


def test_a_receipt_cannot_bind_to_a_registry_with_a_placeholder_revision(tmp_path):
    """Today's real state. A placeholder is not a revision, so nothing can be
    bound to it -- which is why PR C must pin the parent before any corpus can
    claim a bound receipt."""
    spec = bound_spec(revision="REPLACE_WITH_PINNED_HUB_COMMIT")
    report = audit_with(tmp_path, receipt(), spec=spec)
    ea = report["execution_artifact"]
    assert ea["ready"] is False
    assert any("placeholder" in p for p in ea["binding_problems"])
    assert report["readiness"]["identity_ready"] is False


def test_two_traces_swapping_receipts_between_teachers_refuse(tmp_path):
    """A corpus whose traces name different artifacts cannot say what produced
    it, whichever registry entries those artifacts belong to."""
    rows = []
    for artifact in (receipt(), receipt(registry_model="qwen35-9b-instruct",
                                        model_id="Qwen/Qwen3.5-9B")):
        row = trace()
        row["provenance"] = dict(row["provenance"], execution_artifact=artifact)
        rows.append(row)
    path = declaring_pass(tmp_path, rows)
    import unittest.mock as _mock
    with _mock.patch.object(dp, "_resolve_teacher_specs",
                            lambda teachers, overrides: ({MUSE_REGISTRY: bound_spec()}, [])):
        report = dp.audit_corpus(path, TODAY, teacher_overrides=[MUSE_REGISTRY])
    ea = report["execution_artifact"]
    assert ea["ready"] is False
    assert len(ea["distinct_receipts"]) == 2
    assert report["readiness"]["identity_ready"] is False


def test_a_receipt_naming_an_unresolved_teacher_refuses(tmp_path):
    """The receipt must name a teacher this corpus actually resolved, not any
    entry that happens to exist in the registry."""
    report = audit_with(tmp_path, receipt(registry_model="not-a-teacher"))
    ea = report["execution_artifact"]
    assert ea["ready"] is False
    assert any("not a resolved teacher" in p for p in ea["binding_problems"])


# --- the identity is structured, so delimiters are inert --------------------

@pytest.mark.parametrize("hostile", [
    "a|b", "|", "a|b|c|d|e|f", "muse|glimmer|30b",
    '{"json":"like"}', "with spaces", "quote\"inside", "tab\there",
    "newline\nhere", "unicode-\u00e9\u00e8", "\\backslash",
])
def test_delimiters_in_binding_values_cannot_shift_field_boundaries(hostile):
    """These values come from the traces. A delimited identity string would
    split into the wrong number of fields and crash the binder or silently
    move a value into the wrong slot; escaping would only relocate the bug."""
    result = dp.execution_artifact_receipts(
        rows_with(receipt(registry_model=hostile), n=2))
    assert result["ready"] is True
    assert result["malformed_receipts"] == 0
    identity = result["distinct_receipts"][0]
    assert isinstance(identity, dict)
    assert identity["registry_model"] == hostile        # verbatim, unsplit
    assert identity["model_id"] == MUSE_MODEL_ID        # boundary intact
    assert identity["revision"] == RECEIPT_REVISION
    assert identity["digest"] == RECEIPT_DIGEST


def test_a_hostile_binding_value_refuses_at_the_binder_without_crashing(tmp_path):
    """It must be REFUSED, not merely survive parsing: no registry entry is
    named 'a|b'."""
    report = audit_with(tmp_path, receipt(registry_model="a|b"))
    ea = report["execution_artifact"]
    assert ea["ready"] is False
    assert any("not a resolved teacher" in p for p in ea["binding_problems"])


# --- one corpus, one canonical teacher --------------------------------------

def test_a_corpus_mixing_teacher_identities_cannot_be_bound(tmp_path):
    """Per-trace binding, enforced at the corpus level.

    Two traces, two different declared teachers resolving to two different
    registry models, both carrying the SAME otherwise-valid receipt. Parsing is
    ready -- one distinct receipt, nothing malformed -- so the refusal comes
    specifically from the corpus mixing teacher identities, not from the
    digests differing. Without this check the Qwen-attributed trace would never
    be compared with its own registry entry.
    """
    rows = []
    for teacher in ("muse-glimmer", "qwen"):
        row = trace()
        row["provenance"] = dict(row["provenance"], teacher=teacher,
                                 execution_artifact=receipt())
        rows.append(row)
    path = declaring_pass(tmp_path, rows)

    # the receipts themselves are fine: one distinct identity, none malformed
    parsed = dp.execution_artifact_receipts(rows)
    assert parsed["ready"] is True
    assert len(parsed["distinct_receipts"]) == 1
    assert parsed["malformed_receipts"] == 0

    two_teachers = {MUSE_REGISTRY: bound_spec(),
                    "qwen35-9b-instruct": bound_spec(
                        model_id="Qwen/Qwen3.5-9B")}
    import unittest.mock as _mock
    with _mock.patch.object(dp, "_resolve_teacher_specs",
                            lambda teachers, overrides: (two_teachers, [])):
        report = dp.audit_corpus(path, TODAY)

    ea = report["execution_artifact"]
    assert ea["ready"] is False
    assert ea["bound"] is False
    assert any("mixes teacher identities" in p for p in ea["binding_problems"])
    assert report["readiness"]["execution_artifact_ready"] is False
    assert report["readiness"]["identity_ready"] is False


def test_a_single_teacher_corpus_with_a_matching_receipt_still_binds(tmp_path):
    """The control for the rule above: one canonical teacher binds normally."""
    report = audit_with(tmp_path, receipt())
    assert report["execution_artifact"]["bound"] is True


def test_an_empty_corpus_is_not_ready():
    result = dp.execution_artifact_receipts([])
    assert result["ready"] is False
    assert result["reason"] == "no traces"
