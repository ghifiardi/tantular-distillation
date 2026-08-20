"""CPU-only guards for v1 endpoint plumbing and LoRA target coverage."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import config
import train_qlora


def test_host_base_url_environment_override(monkeypatch):
    monkeypatch.setenv("HOST_BASE_URL", "http://student.internal:8020/v1")
    resolved = config.resolve("office-student-9b", "student-serve")
    assert resolved["HOST_BASE_URL"] == "http://student.internal:8020/v1"


def test_student_host_without_runtime_address_stays_unset(monkeypatch):
    monkeypatch.delenv("HOST_BASE_URL", raising=False)
    resolved = config.resolve("office-student-9b", "student-serve")
    assert resolved["HOST_BASE_URL"] == ""


class FakeModel:
    def __init__(self, module_names=(), parameter_names=()):
        self.module_names = list(module_names)
        self.parameter_names = list(parameter_names)

    def named_modules(self):
        return [(name, object()) for name in self.module_names]

    def named_parameters(self):
        return [(name, SimpleNamespace(requires_grad=True, numel=lambda: 10))
                for name in self.parameter_names]

    def parameters(self):
        return [parameter for _, parameter in self.named_parameters()]


def test_target_coverage_accepts_hybrid_projection_names():
    model = FakeModel(module_names=[
        "model.layers.0.linear_attn.in_proj_qkv",
        "model.layers.0.linear_attn.in_proj_z",
        "model.layers.0.linear_attn.out_proj",
    ])
    got = train_qlora.verify_target_module_coverage(
        model, ["in_proj_qkv", "in_proj_z", "out_proj"])
    assert got == {"in_proj_qkv": 1, "in_proj_z": 1, "out_proj": 1}


def test_target_coverage_fails_when_a_name_matches_nothing():
    model = FakeModel(module_names=["model.layers.0.linear_attn.out_proj"])
    with pytest.raises(SystemExit) as exc:
        train_qlora.verify_target_module_coverage(
            model, ["in_proj_qkv", "out_proj"])
    assert exc.value.code == 2


def test_attached_coverage_requires_trainable_lora_for_every_target():
    model = FakeModel(parameter_names=[
        "base.model.layers.0.linear_attn.in_proj_qkv.lora_A.default.weight",
        "base.model.layers.0.linear_attn.in_proj_qkv.lora_B.default.weight",
        "base.model.layers.0.linear_attn.out_proj.lora_A.default.weight",
        "base.model.layers.0.linear_attn.out_proj.lora_B.default.weight",
    ])
    counts, trainable = train_qlora.verify_attached_lora_coverage(
        model, ["in_proj_qkv", "out_proj"])
    assert counts == {"in_proj_qkv": 2, "out_proj": 2}
    assert trainable == 40

