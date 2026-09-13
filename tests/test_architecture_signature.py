"""The trainer must refuse a checkpoint that is not the shape it was written for.

    ./.venv/bin/python -m pytest tests/test_architecture_signature.py -q

train/qlora_9b.yaml's target_modules list is written for ONE layout: 32 layers,
8 of them full-attention, with transformers' SPLIT in_proj names. Against a
different checkpoint the same list matches zero modules in most layers and the
run still trains — producing an adapter that looks fine and learned almost
nothing. Coverage checks catch a target that matches nothing anywhere; they do
not catch a target that matches somewhere in the wrong model.

No model is loaded: the check is driven with plain dicts.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import distill_plan as dp                                    # noqa: E402
import train_qlora as tq                                     # noqa: E402

DENSE_9B = {
    "model_type": "qwen3_5",
    "text_config": {
        "model_type": "qwen3_5_text",
        "num_hidden_layers": 32,
        "layer_types": (["linear_attention"] * 3 + ["full_attention"]) * 8,
        "hidden_size": 4096,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
    },
    "vision_config": {"model_type": "qwen3_5_vl", "depth": 27,
                      "hidden_size": 1152, "out_hidden_size": 4096},
}
MOE = {
    "model_type": "qwen3_5_moe",
    "text_config": dict(DENSE_9B["text_config"], model_type="qwen3_5_moe_text",
                        num_hidden_layers=48, num_experts=256,
                        num_experts_per_tok=8),
    "vision_config": DENSE_9B["vision_config"],
}


def profile(signature: str) -> dict:
    return {"name": "qwen35-hybrid-dense-9b", "signature": signature}


def test_matching_signature_proceeds(capsys):
    pinned = dp.architecture_signature(DENSE_9B)
    assert tq.check_architecture_signature(DENSE_9B, profile(pinned), "p") == pinned
    assert "matches" in capsys.readouterr().out


def test_a_different_checkpoint_aborts(capsys):
    pinned = dp.architecture_signature(DENSE_9B)
    with pytest.raises(SystemExit):
        tq.check_architecture_signature(MOE, profile(pinned), "p")
    assert "ARCHITECTURE MISMATCH" in capsys.readouterr().err


@pytest.mark.parametrize("field,value", [
    ("num_hidden_layers", 48),
    ("layer_types", ["full_attention"] * 32),
    ("hidden_size", 5120),
    ("num_key_value_heads", 8),
])
def test_each_load_bearing_field_is_detected(field, value):
    pinned = dp.architecture_signature(DENSE_9B)
    changed = dict(DENSE_9B,
                   text_config=dict(DENSE_9B["text_config"], **{field: value}))
    with pytest.raises(SystemExit):
        tq.check_architecture_signature(changed, profile(pinned), "p")


def test_an_unpinned_signature_refuses_and_does_not_self_pin(capsys):
    """A placeholder must abort, and must NOT be filled from whatever loaded —
    a signature copied off the model in front of you checks nothing."""
    placeholder = profile("ARCH_SIGNATURE_QWEN35_9B")
    with pytest.raises(SystemExit):
        tq.check_architecture_signature(DENSE_9B, placeholder, "p")
    out = capsys.readouterr().err
    assert "no pinned signature" in out
    assert "arch-signature" in out
    assert placeholder["signature"] == "ARCH_SIGNATURE_QWEN35_9B"


def test_a_shapeless_config_refuses_before_any_comparison():
    with pytest.raises(SystemExit):
        tq.check_architecture_signature({"model_type": "qwen3_5"},
                                        profile("a" * 64), "p")


# --- the wiring: the shipped config must name a real profile ----------------

def test_the_shipped_training_config_names_a_real_profile():
    config = yaml.safe_load((ROOT / "train" / "qlora_9b.yaml").read_text())
    name, loaded = tq.load_architecture_profile(config)
    assert name == "qwen35-hybrid-dense-9b"
    assert loaded["expected"]["layers"] == 32


def test_a_config_without_a_profile_is_refused():
    with pytest.raises(SystemExit):
        tq.load_architecture_profile({"base_model": "Qwen/Qwen3.5-9B"})


# Measured from Qwen/Qwen3.5-9B at commit
# c202236235762e1c871ad0ccb60c8ee5ba337b9a, whose config.json is
# d0883072e01861ed0b2d47be3c16c36a8e81c224c7ffaa310c6558fb3f932b05.
INSTRUCT_9B_SIGNATURE = \
    "52a3c9e2c895c0dc3d81abbb24e6463256cf3869f6e7343974f773a3f002096d"


def test_the_9b_profile_pins_the_measured_instruct_signature():
    """Replaces "the signature is still a placeholder".

    That was true while no instruct snapshot existed locally, and the trainer
    aborting before LoRA attached was the fail-closed outcome. The snapshot has
    now been acquired metadata-only and the signature recomputed from its own
    config.json, so the honest assertion is the exact pin.

    This value also equals the one measured from Qwen3.5-9B-Base, which ships a
    byte-identical config.json. That is shared architecture SHAPE, not shared
    checkpoint identity — the signature deliberately says nothing about
    weights, instruction tuning or chat-template behaviour. What makes it
    pinnable is the provenance recorded above, never the digits.
    """
    loaded = yaml.safe_load(
        (ROOT / "configs" / "architectures" / "qwen35-hybrid-dense-9b.yaml").read_text())
    assert tq.SIGNATURE_RE.match(str(loaded["signature"])), "must be a real digest"
    assert loaded["signature"] == INSTRUCT_9B_SIGNATURE
