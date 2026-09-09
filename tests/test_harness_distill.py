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
        "model_contract": {"registry_model": "student", "prompt_format": "chat"},
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

