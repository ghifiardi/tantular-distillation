"""Guard the TRL/transformers arguments the trainer passes.

    ./.venv/bin/python -m pytest tests/test_trainer_api.py -q

train_qlora.py builds SFTConfig(**kwargs) with a literal dict. If a pinned
library renames or removes one of those arguments, the failure appears inside a
paid run, minutes after the model finished loading. transformers 5.x already did
this once: it REMOVED warmup_ratio and folded it into warmup_steps, which now
takes a float in [0, 1) meaning a ratio.

This does not need the libraries installed — it reads the kwarg names out of the
source and checks them against a list that is updated deliberately when a pin
moves. A silent rename is what it is for.
"""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Verified against transformers 5.15.0 / trl 1.10.0 on 2026-08-19 by reading
# TrainingArguments and SFTConfig out of the published wheels.
KNOWN_GOOD = {
    "output_dir", "max_length", "num_train_epochs",
    "per_device_train_batch_size", "gradient_accumulation_steps",
    "learning_rate", "lr_scheduler_type", "warmup_steps",
    "gradient_checkpointing", "bf16", "logging_steps", "save_steps",
    "report_to", "max_steps", "save_strategy", "eval_strategy",
}
# Removed or renamed by a pinned library. Passing any of these raises TypeError.
REMOVED = {"warmup_ratio", "max_seq_length", "tokenizer", "evaluation_strategy"}


def sft_config_kwargs() -> set[str]:
    tree = ast.parse((ROOT / "src" / "train_qlora.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "build_sft_trainer":
            for sub in ast.walk(node):
                if (isinstance(sub, ast.Assign)
                        and any(getattr(t, "id", "") == "kwargs" for t in sub.targets)
                        and isinstance(sub.value, ast.Dict)):
                    keys = {k.value for k in sub.value.keys
                            if isinstance(k, ast.Constant)}
                    # kwargs.update({...}) inside the same function
                    for call in ast.walk(node):
                        if (isinstance(call, ast.Call)
                                and getattr(call.func, "attr", "") == "update"
                                and call.args and isinstance(call.args[0], ast.Dict)):
                            keys |= {k.value for k in call.args[0].keys
                                     if isinstance(k, ast.Constant)}
                    return keys
    raise AssertionError("build_sft_trainer or its kwargs dict was not found")


def test_no_removed_arguments_are_passed():
    passed = sft_config_kwargs()
    dead = passed & REMOVED
    assert not dead, (
        f"train_qlora.py passes {sorted(dead)} to SFTConfig. transformers 5.x "
        "removed warmup_ratio (use warmup_steps, which accepts a ratio) and TRL "
        "1.x renamed max_seq_length -> max_length and tokenizer -> "
        "processing_class. Each raises TypeError after the model has loaded.")


def test_every_argument_is_one_we_verified():
    passed = sft_config_kwargs()
    unknown = passed - KNOWN_GOOD
    assert not unknown, (
        f"train_qlora.py passes {sorted(unknown)}, which was never checked "
        "against transformers 5.15.0 / trl 1.10.0. Verify it exists on "
        "TrainingArguments or SFTConfig and add it to KNOWN_GOOD.")


def test_warmup_is_still_configured():
    """The ratio must survive the rename — dropping it would silently change
    the schedule rather than fail."""
    assert "warmup_steps" in sft_config_kwargs()
    assert "warmup_ratio" in (ROOT / "train" / "qlora_9b.yaml").read_text()


def test_trainer_uses_the_trl_1x_constructor():
    src = (ROOT / "src" / "train_qlora.py").read_text()
    assert "processing_class=" in src, "TRL 1.x renamed tokenizer -> processing_class"
    assert "tokenizer=" not in src
