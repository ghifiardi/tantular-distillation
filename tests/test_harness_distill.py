from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import harness_distill as hd


def safe_harness(**overrides):
    value = {
        "schema_version": 1,
        "name": "safe",
        "status": "candidate",
        "model_contract": {
            "protocol": "openai_chat",
            "compatible_registry_models": ["student", "teacher"],
        },
        "prompts": {
            "system": {"path": "prompt.txt", "sha256": None, "verified": False}
        },
        "tools": {
            "allow": ["read", "edit"],
            "state_change_requires_approval": True,
        },
        "memory": {
            "active_context": "bounded",
            "ephemeral_execution": "sandbox",
            "durable_task_state": "append_only",
            "product_memory": "disabled",
        },
        "execution": {
            "isolation": "sandbox",
            "network": "denied_by_default",
            "max_steps": 8,
            "max_wall_seconds": 300,
        },
        "verification": {
            "before_action": ["schema"],
            "after_action": ["contract"],
            "repair_attempts": 1,
        },
        "mutation": {
            "production_self_modify": False,
            "candidate_workspace_only": True,
            "evaluator_mutation_allowed": False,
            "human_approval_required": True,
        },
    }
    for key, item in overrides.items():
        value[key] = item
    return value


def experiment():
    return {
        "schema_version": 1,
        "name": "test",
        "student_model": "student",
        "teacher_model": "teacher",
        "current_harness": "current",
        "candidate_harness": "candidate",
        "arms": [
            "student_current",
            "student_candidate",
            "teacher_current",
            "teacher_candidate",
        ],
        "decision": {
            "capability_metric": "capability",
            "higher_is_better": True,
            "target": 0.90,
            "minimum_residual_gap": 0.05,
            "noise_floor": 0.025,
            "guardrails": {
                "voice": {"minimum": 0.95, "max_regression": 0.0},
                "contract": {"minimum": 0.90, "max_regression": 0.0},
            },
        },
        "training_authorized": False,
    }


def measurements(current=0.70, candidate=0.82, teacher=0.95):
    return {
        "arms": {
            "student_current": {
                "capability": current,
                "voice": 0.95,
                "contract": 0.92,
            },
            "student_candidate": {
                "capability": candidate,
                "voice": 0.95,
                "contract": 0.92,
            },
            "teacher_current": {
                "capability": teacher,
                "voice": 0.80,
                "contract": 0.80,
            },
            "teacher_candidate": {
                "capability": teacher,
                "voice": 0.80,
                "contract": 0.80,
            },
        }
    }


@pytest.fixture
def harness_loader(monkeypatch):
    specs = {"current": safe_harness(), "candidate": safe_harness(name="candidate")}
    monkeypatch.setattr(hd, "load_harness", lambda name: specs[name])
    return specs


def test_digest_is_stable_and_ignores_self_declared_digest():
    first = safe_harness()
    second = dict(first, digest="fabricated")
    assert hd.canonical_digest(first) == hd.canonical_digest(second)


@pytest.mark.parametrize(
    ("section", "key", "unsafe"),
    [
        ("mutation", "production_self_modify", True),
        ("mutation", "candidate_workspace_only", False),
        ("mutation", "evaluator_mutation_allowed", True),
        ("mutation", "human_approval_required", False),
        ("tools", "state_change_requires_approval", False),
    ],
)
def test_unsafe_harness_is_refused(section, key, unsafe):
    spec = safe_harness()
    spec[section][key] = unsafe
    with pytest.raises(hd.HarnessPlanError):
        hd.validate_harness(spec)


def test_unverified_prompt_is_a_warning_not_authorization():
    warnings = hd.validate_harness(safe_harness())
    assert warnings == ["system prompt identity is unverified"]


def test_plan_requires_factorial_arms(harness_loader):
    cfg = experiment()
    cfg["arms"] = ["student_current", "student_candidate"]
    with pytest.raises(hd.HarnessPlanError, match="teacher_current"):
        hd.build_plan(cfg)


def test_harness_that_closes_gap_blocks_weight_distillation(harness_loader):
    result = hd.evaluate(experiment(), measurements(candidate=0.91))
    assert result["harness_optimization_sufficient"] is True
    assert result["weight_distillation_candidate"] is False
    assert "candidate harness closes" in " ".join(result["reasons"])
    assert result["training_authorized"] is False


def test_residual_gap_can_make_weight_distillation_a_candidate(harness_loader):
    result = hd.evaluate(experiment(), measurements())
    assert result["harness_optimization_sufficient"] is False
    assert result["weight_distillation_candidate"] is True
    assert result["scores"]["harness_gain"] == pytest.approx(0.12)
    assert result["scores"]["residual_model_gap"] == pytest.approx(0.13)
    assert result["training_authorized"] is False


def test_no_observed_gap_means_no_distillation(harness_loader):
    result = hd.evaluate(experiment(), measurements(current=0.91, candidate=0.92))
    assert result["weight_distillation_candidate"] is False
    assert "already meets" in " ".join(result["reasons"])


def test_teacher_must_solve_the_gap(harness_loader):
    result = hd.evaluate(experiment(), measurements(teacher=0.85))
    assert result["weight_distillation_candidate"] is False
    assert "teacher_current does not meet" in " ".join(result["reasons"])


def test_guardrail_regression_blocks_candidate(harness_loader):
    data = measurements()
    data["arms"]["student_candidate"]["voice"] = 0.94
    result = hd.evaluate(experiment(), data)
    assert result["weight_distillation_candidate"] is False
    assert result["guardrail_failures"]


def test_trace_audit_reports_missing_harness_attribution(tmp_path):
    path = tmp_path / "traces.jsonl"
    rows = [
        {"provenance": {"teacher": "muse"}, "completion": "a"},
        {
            "provenance": {"teacher": "muse"},
            "harness_provenance": {"digest": "abc"},
            "completion": "b",
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    result = hd.audit_traces(path)
    assert result["model_attributed"] == 2
    assert result["harness_attributed"] == 1
    assert result["harness_coverage"] == 0.5
    assert result["distillation_attribution_ready"] is False
    assert result["training_authorized"] is False


def test_repository_draft_plan_loads():
    result = hd.build_plan(
        hd.load_yaml(ROOT / "configs" / "experiments" / "harness-before-weights.yaml")
    )
    assert result["required_arms"] == [
        "student_current",
        "student_candidate",
        "teacher_current",
        "teacher_candidate",
    ]
    assert result["training_authorized"] is False



# --- harness_provenance: the block stamped onto every harness-aware trace ----
#
# One source of truth. The audit already showed why it matters: the promoted
# corpus is 136 traces, 136 model-attributed, 0 harness-attributed — so nothing
# in it can tell whether a result came from the model or from the scaffold
# around it.

def test_provenance_is_deterministic_and_matches_the_plan_digest():
    """A digest computed here and a digest computed by `plan` must agree, or a
    trace and the experiment that reads it describe different harnesses."""
    spec = safe_harness()
    first = hd.harness_provenance(spec, execution_model_registry="student")
    assert first == hd.harness_provenance(dict(spec), execution_model_registry="student")
    assert first["digest"] == hd.canonical_digest(spec)


def test_provenance_records_the_identity_fields_a_reader_needs():
    block = hd.harness_provenance(safe_harness(), execution_model_registry="student")
    assert block["name"] == "safe"
    assert block["status"] == "candidate"
    assert block["execution_model_registry"] == "student"
    assert block["schema_version"] == 1
    # Tool and verification policy are digested separately: a trace should say
    # which policy produced it without carrying the whole policy.
    assert hd.SHA256_RE.match(block["tool_policy_digest"])
    assert hd.SHA256_RE.match(block["verification_policy_digest"])


def test_an_unverified_prompt_is_recorded_as_unverified_not_guessed():
    """The one thing this must never do is invent a prompt hash. An unverified
    harness says so, and `src/verify_harness_identity.py` is the only thing that
    may fill it in."""
    block = hd.harness_provenance(safe_harness(), execution_model_registry="student")
    assert block["prompt_sha256"] is None
    assert block["prompt_verified"] is False


def test_a_verified_prompt_is_carried_through():
    spec = safe_harness(prompts={"system": {"path": "p.txt", "sha256": "a" * 64,
                                            "verified": True}})
    block = hd.harness_provenance(spec, execution_model_registry="student")
    assert block["prompt_sha256"] == "a" * 64
    assert block["prompt_verified"] is True


def test_a_claimed_verification_without_a_digest_is_not_believed():
    """`verified: true` with no sha256 is a contradiction; fail closed."""
    spec = safe_harness(prompts={"system": {"path": "p.txt", "sha256": None,
                                            "verified": True}})
    assert hd.harness_provenance(spec, execution_model_registry="student")["prompt_verified"] is False


def test_changing_the_policy_changes_the_digest():
    """The digest has to be load-bearing: two harnesses that differ in what the
    agent may do must not share one."""
    base = hd.harness_provenance(safe_harness(), execution_model_registry="student")
    wider = hd.harness_provenance(
        safe_harness(tools={"allow": ["read", "edit", "shell"],
                            "state_change_requires_approval": True}),
        execution_model_registry="student")
    assert wider["digest"] != base["digest"]
    assert wider["tool_policy_digest"] != base["tool_policy_digest"]
    assert wider["verification_policy_digest"] == base["verification_policy_digest"]


def test_provenance_refuses_an_unsafe_harness():
    """Nothing may stamp attribution for a harness that would not be allowed to
    run: the block would then be evidence for a trace that should not exist."""
    with pytest.raises(hd.HarnessPlanError):
        hd.harness_provenance(
            safe_harness(mutation={"production_self_modify": True,
                                   "candidate_workspace_only": True,
                                   "evaluator_mutation_allowed": False,
                                   "human_approval_required": True}),
            execution_model_registry="student")


def test_the_shipped_draft_harnesses_are_unverified_today():
    """Documented state, not aspiration: neither draft harness has had its
    system prompt hashed, so both must say so. Delete this when they are
    verified against the real add-in."""
    for name in ("tantular-office-current", "tantular-office-candidate"):
        block = hd.harness_provenance(hd.load_harness(name),
                                      execution_model_registry="qwen35-9b-instruct")
        assert block["prompt_verified"] is False, name
        assert block["prompt_sha256"] is None, name


# --- the harness is model-COMPATIBLE, not model-bound ------------------------
#
# The whole point of the four-arm design is running ONE harness against a
# student and a teacher. A harness pinned to a single registry model would make
# the teacher_current and teacher_candidate arms impossible to express, so the
# contract lists what it is compatible with and the trace records what actually
# ran.

def test_one_harness_serves_both_factorial_model_arms():
    spec = safe_harness()
    student = hd.harness_provenance(spec, execution_model_registry="student")
    teacher = hd.harness_provenance(spec, execution_model_registry="teacher")

    # Same harness, so the same harness digest — that is what makes the arms
    # comparable at all.
    assert student["digest"] == teacher["digest"]
    # But the traces say which model produced them.
    assert student["execution_model_registry"] == "student"
    assert teacher["execution_model_registry"] == "teacher"
    assert student["compatible_registry_models"] == ["student", "teacher"]


def test_an_incompatible_execution_model_is_refused():
    """Recording a trace as harness-attributed while the harness was never
    declared compatible with the model would make the attribution a guess."""
    with pytest.raises(hd.HarnessPlanError) as exc:
        hd.harness_provenance(safe_harness(), execution_model_registry="some-other")
    assert "not compatible" in str(exc.value)


def test_the_execution_model_must_be_stated():
    with pytest.raises(hd.HarnessPlanError):
        hd.harness_provenance(safe_harness(), execution_model_registry="")


def test_a_harness_declaring_no_compatible_models_is_refused():
    with pytest.raises(hd.HarnessPlanError):
        hd.harness_provenance(
            safe_harness(model_contract={"protocol": "openai_chat",
                                         "compatible_registry_models": []}),
            execution_model_registry="student")


def test_widening_compatibility_changes_the_harness_digest():
    """Compatibility is part of the harness definition, so changing it is a
    different harness — not a free-form annotation."""
    base = hd.harness_provenance(safe_harness(), execution_model_registry="student")
    wider = hd.harness_provenance(
        safe_harness(model_contract={"protocol": "openai_chat",
                                     "compatible_registry_models":
                                         ["student", "teacher", "third"]}),
        execution_model_registry="student")
    assert wider["digest"] != base["digest"]


def test_the_shipped_harnesses_cover_both_planned_arms():
    """The real specs must admit the student AND the teacher, or the planned
    experiment cannot run."""
    for name in ("tantular-office-current", "tantular-office-candidate"):
        models = (hd.load_harness(name)["model_contract"]
                  ["compatible_registry_models"])
        assert "qwen35-9b-instruct" in models, name
        assert "muse-glimmer-30b" in models, name


# --- one attribution summary, used by every consumer -------------------------
#
# The pass manifest, promotion, the freeze, the trainer, the audit and the
# corpus gate all need the same answer to "is this corpus harness-attributed,
# and by what?". Six implementations of that question would be six ways for them
# to disagree, and the disagreement would only surface as a corpus nobody can
# explain.

def trace(**over):
    block = {
        "schema_version": 1, "name": "h", "status": "candidate",
        "digest": "a" * 64,
        "compatible_registry_models": ["m1", "m2"],
        "execution_model_registry": "m1",
        "prompt_sha256": "b" * 64, "prompt_verified": True,
        "tool_policy_digest": "c" * 64,
        "verification_policy_digest": "d" * 64,
    }
    block.update(over)
    return {"family": "f", "completion": "x", "harness_provenance": block}


def test_uniform_attribution_produces_one_pinned_summary():
    summary = hd.summarize_harness_attribution([trace(), trace()], required=True)
    assert summary == {
        "required": True, "attributed": True, "name": "h", "digest": "a" * 64,
        "prompt_verified": True, "prompt_sha256": "b" * 64,
        "execution_model_registry": "m1",
    }


def test_unattributed_legacy_traces_stay_valid():
    legacy = [{"family": "f", "provenance": {"teacher": "t"}}] * 3
    assert hd.summarize_harness_attribution(legacy, required=False) == {
        "required": False, "attributed": False, "name": None, "digest": None,
        "prompt_verified": False, "prompt_sha256": None,
        "execution_model_registry": None,
    }


def test_missing_attribution_under_required_refuses():
    with pytest.raises(hd.HarnessPlanError) as exc:
        hd.summarize_harness_attribution([trace(), {"family": "f"}], required=True)
    assert "not attributed" in str(exc.value)


def test_mixed_harness_digests_refuse():
    with pytest.raises(hd.HarnessPlanError) as exc:
        hd.summarize_harness_attribution([trace(), trace(digest="e" * 64)],
                                         required=True)
    assert "mixes 2 harness digests" in str(exc.value)


def test_mixed_prompt_hashes_refuse():
    """Same harness digest, different prompt identity: the harness definition
    did not change but what the model was told did."""
    with pytest.raises(hd.HarnessPlanError) as exc:
        hd.summarize_harness_attribution([trace(), trace(prompt_sha256="f" * 64)],
                                         required=True)
    assert "prompt" in str(exc.value)


def test_an_unverified_prompt_under_required_refuses():
    with pytest.raises(hd.HarnessPlanError) as exc:
        hd.summarize_harness_attribution(
            [trace(prompt_verified=False, prompt_sha256=None)], required=True)
    assert "unverified" in str(exc.value)


def test_mixed_execution_models_refuse():
    """The factorial arms are SEPARATE generation passes. One file holding both
    is the confound the design exists to remove."""
    with pytest.raises(hd.HarnessPlanError) as exc:
        hd.summarize_harness_attribution(
            [trace(), trace(execution_model_registry="m2")], required=True)
    assert "execution model" in str(exc.value)


def test_attributed_traces_under_legacy_mode_refuse():
    """Not a valid legacy corpus and not a declared harness-aware one. Silently
    accepting it is how the declaration gets lost."""
    with pytest.raises(hd.HarnessPlanError) as exc:
        hd.summarize_harness_attribution([trace()], required=False)
    assert "harness-aware" in str(exc.value)


def test_a_partially_attributed_legacy_file_refuses():
    with pytest.raises(hd.HarnessPlanError):
        hd.summarize_harness_attribution(
            [trace(), {"family": "f"}], required=False)


def test_an_empty_corpus_refuses():
    with pytest.raises(hd.HarnessPlanError):
        hd.summarize_harness_attribution([], required=True)

# --- audit_traces: every attribution state, named -----------------------------
#
# The real 136-trace corpus exercises exactly one of these — all keys absent —
# so it cannot stand as evidence for the malformed-versus-absent distinction.
# That is the finding-2 mistake (presence versus truthiness) one module over, so
# each state is pinned with its own fixture.

VALID_BLOCK = {
    "schema_version": 1, "name": "h", "status": "candidate", "digest": "a" * 64,
    "compatible_registry_models": ["m1"], "execution_model_registry": "m1",
    "prompt_sha256": "b" * 64, "prompt_verified": True,
    "tool_policy_digest": "c" * 64, "verification_policy_digest": "d" * 64,
}


def corpus_of(tmp_path: Path, *rows: dict) -> Path:
    path = tmp_path / "traces.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


def absent() -> dict:
    return {"family": "f", "provenance": {"teacher": "t"}}


def with_block(value) -> dict:
    return {"family": "f", "provenance": {"teacher": "t"},
            "harness_provenance": value}


@pytest.mark.parametrize("rows,attributed,malformed,ready", [
    # the key is absent: an ordinary legacy trace
    ([absent(), absent()],                          0, 0, False),
    # present but empty: asserted attribution, then said nothing
    ([with_block({})],                              0, 1, False),
    ([with_block(None)],                            0, 1, False),
    # present but not a mapping at all
    ([with_block("a-string")],                      0, 1, False),
    ([with_block(["a", "list"])],                   0, 1, False),
    ([with_block({"digest": ""})],                  0, 1, False),
    # a real block
    ([with_block(VALID_BLOCK)],                     1, 0, True),
    ([with_block(VALID_BLOCK), with_block(VALID_BLOCK)], 2, 0, True),
    # partial: some attributed, some not
    ([with_block(VALID_BLOCK), absent()],           1, 0, False),
    # partial AND malformed
    ([with_block(VALID_BLOCK), with_block({})],     1, 1, False),
])
def test_audit_traces_reports_each_attribution_state(tmp_path, rows, attributed,
                                                     malformed, ready):
    report = hd.audit_traces(corpus_of(tmp_path, *rows))
    assert report["harness_attributed"] == attributed
    assert report["harness_malformed"] == malformed
    assert report["distillation_attribution_ready"] is ready
    if malformed:
        assert any("MALFORMED" in limit for limit in report["limits"])
    else:
        assert not any("MALFORMED" in limit for limit in report["limits"])


@pytest.mark.parametrize("rows", [
    [absent()], [with_block({})], [with_block(VALID_BLOCK)],
    [with_block(VALID_BLOCK), absent()],
    [with_block(VALID_BLOCK), with_block({})],
])
def test_attribution_readiness_is_exactly_its_definition(tmp_path, rows):
    """Stated directly, so a matching boolean cannot hide a moved reason."""
    report = hd.audit_traces(corpus_of(tmp_path, *rows))
    assert report["distillation_attribution_ready"] == (
        report["traces"] > 0
        and report["harness_attributed"] == report["traces"]
        and report["harness_malformed"] == 0
    )


@pytest.mark.requires_local_corpus
def test_the_legacy_corpus_is_absent_not_malformed():
    """The real corpus, pinned by all three numbers rather than the verdict: it
    is not attribution-ready because it carries NO attribution, not because its
    attribution is broken."""
    report = hd.audit_traces(ROOT / "data" / "promoted" / "train.jsonl")
    assert report["traces"] == 136
    assert report["harness_attributed"] == 0
    assert report["harness_malformed"] == 0
    assert report["harness_coverage"] == 0.0
    assert report["distillation_attribution_ready"] is False
    assert not any("MALFORMED" in limit for limit in report["limits"])
