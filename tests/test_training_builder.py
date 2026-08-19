"""CPU-only regression guards for the shared TRL construction path."""
from __future__ import annotations

import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import smoke_train
import train_qlora


class Capture:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)


class FakeTrainer(Capture):
    pass


def fake_training_modules(monkeypatch):
    monkeypatch.setitem(sys.modules, "peft",
                        SimpleNamespace(LoraConfig=Capture))
    monkeypatch.setitem(sys.modules, "trl",
                        SimpleNamespace(SFTConfig=Capture,
                                        SFTTrainer=FakeTrainer))


def test_smoke_and_v1_share_the_same_sft_builder(monkeypatch, tmp_path):
    fake_training_modules(monkeypatch)
    config = yaml.safe_load((ROOT / "train" / "qlora_9b.yaml").read_text())

    trainer = train_qlora.build_sft_trainer(
        config, "model", "train", "eval", "tokenizer",
        tmp_path / "checkpoints", max_steps=1, max_length=512, smoke=True)

    assert trainer.kwargs["model"] == "model"
    assert trainer.kwargs["processing_class"] == "tokenizer"
    assert trainer.kwargs["peft_config"].r == 32
    args = trainer.kwargs["args"]
    assert args.max_steps == 1
    assert args.max_length == 512
    assert args.gradient_accumulation_steps == 1
    assert args.save_strategy == "no"
    assert args.eval_strategy == "no"
    assert args.gradient_checkpointing is False


def test_smoke_executes_builder_and_trainer_train():
    source = inspect.getsource(smoke_train.main)
    assert "build_sft_trainer(" in source
    assert "trainer.train()" in source
    assert "trl_max_steps" in source

