"""Key conversion must change names and nothing else.

    ./.venv/bin/python -m pytest tests/test_adapter_conversion.py -q

The converter exists because a one-segment path difference made vLLM serve base
answers under an adapter's id. A converter that quietly drops or alters tensors
would be a worse version of the same class of bug, so these check the values as
well as the names.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PY_BIN = str(ROOT / ".venv" / "bin" / "python")
CONVERT = str(ROOT / "src" / "convert_adapter_for_vllm.py")

sys.path.insert(0, str(ROOT / "src"))
import convert_adapter_for_vllm as conv

# The key logic is pure string work and must be checkable on a laptop; only the
# round-trip tests need torch, which is a GPU-host dependency. Gating the whole
# module on torch would mean the rule that actually caused the bug is never
# checked where it is written.
import importlib.util

needs_torch = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="torch is a GPU-host dependency")


def make_adapter(path: Path, n_layers: int = 2) -> Path:
    import torch
    from safetensors.torch import save_file
    path.mkdir(parents=True, exist_ok=True)
    (path / "adapter_config.json").write_text(json.dumps(
        {"peft_type": "LORA", "r": 8,
         "base_model_name_or_path": "Qwen/Qwen3.5-9B"}))
    tensors = {}
    for i in range(n_layers):
        base = f"base_model.model.model.layers.{i}.mlp.down_proj"
        tensors[f"{base}.lora_A.weight"] = torch.randn(8, 16)
        tensors[f"{base}.lora_B.weight"] = torch.randn(16, 8)
    save_file(tensors, str(path / "adapter_model.safetensors"))
    return path


def test_key_gets_the_language_model_segment():
    got = conv.convert_key("base_model.model.model.layers.0.mlp.down_proj.lora_A.weight")
    assert got == ("base_model.model.model.language_model.layers.0.mlp."
                   "down_proj.lora_A.weight")


def test_conversion_is_idempotent():
    once = conv.convert_key("base_model.model.model.layers.0.mlp.down_proj.lora_A.weight")
    assert conv.convert_key(once) == once


def test_unrelated_keys_are_left_alone():
    for key in ("lm_head.weight", "base_model.model.lm_head.weight"):
        assert conv.convert_key(key) == key


@needs_torch
def test_values_survive_conversion(tmp_path):
    from safetensors.torch import load_file
    src = make_adapter(tmp_path / "in")
    dst = tmp_path / "out"
    before = load_file(str(src / "adapter_model.safetensors"))
    proc = subprocess.run([PY_BIN, CONVERT, str(src), str(dst)],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    after = load_file(str(dst / "adapter_model.safetensors"))

    assert len(after) == len(before)
    for key, value in before.items():
        assert after[conv.convert_key(key)].equal(value), f"{key} changed"
    assert all("language_model." in k for k in after)


@needs_torch
def test_refuses_to_overwrite_without_force(tmp_path):
    src = make_adapter(tmp_path / "in")
    dst = tmp_path / "out"
    dst.mkdir()
    proc = subprocess.run([PY_BIN, CONVERT, str(src), str(dst)],
                          capture_output=True, text=True)
    assert proc.returncode != 0
    assert "--force" in proc.stdout + proc.stderr


@needs_torch
def test_refuses_an_adapter_it_does_not_understand(tmp_path):
    """Converting nothing must fail loudly, not report success."""
    import torch
    from safetensors.torch import save_file
    src = tmp_path / "weird"
    src.mkdir()
    (src / "adapter_config.json").write_text("{}")
    save_file({"some.other.layout.lora_A.weight": torch.randn(4, 4)},
              str(src / "adapter_model.safetensors"))
    proc = subprocess.run([PY_BIN, CONVERT, str(src), str(tmp_path / "o")],
                          capture_output=True, text=True)
    assert proc.returncode != 0
    assert "nothing was rewritten" in proc.stdout + proc.stderr
