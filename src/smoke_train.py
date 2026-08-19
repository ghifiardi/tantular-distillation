"""Bounded GPU smoke test for the training path. 2-4 examples, minutes, no v1 run.

    # on the pod, after installing requirements-train.txt
    ./.venv/bin/python src/smoke_train.py --out ~/tantular-runs/smoke

Steps 1-4 of the seven-step first rental. Steps 5-7 (serve the adapter under a
distinct id, prove it generates, stop the pod) are shell and live in
calibration/TRAINING_SMOKE_TEST.md, because they need a second process.

WHY THIS EXISTS. src/train_qlora.py has never executed a single training step.
Its GPU imports are lazy so every other check works on a laptop, which means the
first time the model loads, LoRA attaches, or a checkpoint is written will be
inside a paid v1 run — and the failure modes there are library API drift and
quantization config, both of which are minutes to find and hours to find late.
The pins in requirements-train.txt are known-consistent by their own metadata
but have never been run together.

WHAT THIS DELIBERATELY DOES NOT DO:

  it does not train        4 examples, 1 optimizer step, tiny max_seq_len. The
                           adapter it writes is garbage and is not a v1 artifact.
  it does not use the      the smoke corpus is generated here, in Indonesian, and
  promoted corpus          never touches data/promoted. A smoke run must not be
                           able to consume held-out material by accident.
  it does not gate         no thresholds, no promotion, no RUN.json.

STOP CONDITIONS are the steps themselves: any failure aborts with a non-zero
exit and the pod should be stopped. A pass authorises nothing beyond
regenerating the schema-v2 freeze and asking for the v1 decision.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXPECTED_VERSIONS = {
    "torch": "2.13.0",
    "transformers": "5.15.0",
    "tokenizers": "0.22.2",
    "trl": "1.10.0",
    "peft": "0.20.0",
    "datasets": "5.0.1",
    "accelerate": "1.14.0",
    "bitsandbytes": "0.50.1",
    "safetensors": "0.8.0",
    "vllm": "0.27.1",
}

# Four short Indonesian office examples, written here and used nowhere else.
SMOKE_EXAMPLES = [
    {"user": "Ringkas paragraf ini menjadi satu kalimat: Rapat anggaran "
             "ditunda ke pekan depan karena data realisasi belum lengkap.",
     "assistant": "Rapat anggaran ditunda ke pekan depan karena data "
                  "realisasi belum lengkap."},
    {"user": "Ubah kalimat ini menjadi lebih formal: tolong cek lagi angkanya ya.",
     "assistant": "Mohon periksa kembali angka tersebut."},
    {"user": "Sebutkan dua hal yang perlu dilampirkan pada permohonan cuti.",
     "assistant": "Formulir permohonan cuti yang telah ditandatangani dan "
                  "surat persetujuan atasan langsung."},
    {"user": "Terjemahkan ke bahasa Inggris: laporan keuangan triwulan pertama.",
     "assistant": "The first-quarter financial report."},
]


def step(n: int, title: str) -> None:
    print(f"\n=== [{n}/4] {title} ===", flush=True)


def die(msg: str, code: int = 1) -> None:
    print(f"\nSMOKE TEST FAILED: {msg}", file=sys.stderr)
    print("Stop the pod. Do not proceed to the v1 training run.", file=sys.stderr)
    sys.exit(code)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="train/qlora_9b.yaml")
    p.add_argument("--out", type=Path, default=Path("~/tantular-runs/smoke"))
    p.add_argument("--max-seq-len", type=int, default=512)
    args = p.parse_args()

    import yaml
    cfg = yaml.safe_load((ROOT / args.config).read_text())
    base_model = cfg["base_model"]
    out = args.out.expanduser()
    out.mkdir(parents=True, exist_ok=True)
    report = {"base_model": base_model, "steps": {}, "versions": {}}
    t0 = time.time()

    # --- 1. CUDA and dependency verification --------------------------------
    step(1, "CUDA and dependency versions")
    try:
        import accelerate, bitsandbytes, datasets, peft, safetensors
        import torch, transformers, trl, vllm
    except ImportError as e:
        die(f"a pinned dependency is missing: {e}\n"
            "Install requirements-train.txt on the pod.")
    import tokenizers
    versions = {m.__name__: getattr(m, "__version__", "?") for m in
                (torch, transformers, tokenizers, trl, peft, datasets,
                 accelerate, bitsandbytes, safetensors, vllm)}
    report["versions"] = versions
    for name, ver in versions.items():
        print(f"  {name:<14} {ver}")
    mismatched = {
        name: (EXPECTED_VERSIONS[name], versions.get(name))
        for name in EXPECTED_VERSIONS
        if versions.get(name) != EXPECTED_VERSIONS[name]
    }
    if mismatched:
        die("pinned versions did not resolve exactly: "
            + ", ".join(f"{name} expected {wanted}, got {got}"
                        for name, (wanted, got) in mismatched.items()))
    if not torch.cuda.is_available():
        die("torch reports no CUDA device. Wrong image, wrong driver, or a "
            "CPU pod.")
    dev = torch.cuda.get_device_properties(0)
    print(f"  device         {dev.name}, cc {dev.major}.{dev.minor}, "
          f"{dev.total_memory / 1e9:.0f} GB")
    if dev.major < 8:
        die(f"compute capability {dev.major}.{dev.minor} is pre-Ampere; bf16 "
            "and bitsandbytes NF4 both need 8.0+.")
    report["steps"]["1_cuda"] = {"gpu": dev.name,
                                 "compute_capability": f"{dev.major}.{dev.minor}",
                                 "total_memory_gb": round(dev.total_memory / 1e9, 1)}

    # --- 2. NF4 load ---------------------------------------------------------
    step(2, f"load {base_model} in NF4")
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    quant = BitsAndBytesConfig(
        load_in_4bit=cfg.get("load_in_4bit", True),
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16)
    try:
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        model = AutoModelForCausalLM.from_pretrained(
            base_model, quantization_config=quant, dtype=torch.bfloat16,
            device_map={"": 0})
    except Exception as e:
        die(f"model load failed: {type(e).__name__}: {e}")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    loaded_quant = getattr(model.config, "quantization_config", None)
    print(f"  loaded, quantization_config: {type(loaded_quant).__name__}")
    if loaded_quant is None:
        die("the model loaded WITHOUT a quantization config — this is not NF4, "
            "and a 9B in bf16 will not train in the memory budget QLoRA assumes.")
    report["steps"]["2_nf4_load"] = {"quantization_config": type(loaded_quant).__name__,
                                     "seconds": round(time.time() - t0, 1)}

    # --- 3. Exact TRL path + non-zero-gradient proof ------------------------
    step(3, "construct the real TRL trainer and take one optimizer step")
    from datasets import Dataset
    from train_qlora import build_sft_trainer

    smoke_ds = Dataset.from_list([{"messages": [
        {"role": "user", "content": ex["user"]},
        {"role": "assistant", "content": ex["assistant"]},
    ]} for ex in SMOKE_EXAMPLES])
    try:
        trainer = build_sft_trainer(
            cfg, model, smoke_ds, smoke_ds, tokenizer,
            out / "checkpoints", max_steps=1,
            max_length=args.max_seq_len, smoke=True)
        model = trainer.model
    except Exception as e:
        die(f"TRL SFTConfig/SFTTrainer construction failed: "
            f"{type(e).__name__}: {e}")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  trainable {trainable:,} / {total:,} ({100 * trainable / total:.3f}%)")
    if trainable == 0:
        die("no trainable parameters after attaching LoRA — nothing would learn.")

    texts = [tokenizer.apply_chat_template(
        [{"role": "user", "content": ex["user"]},
         {"role": "assistant", "content": ex["assistant"]}],
        tokenize=False, add_generation_prompt=False)
        for ex in SMOKE_EXAMPLES]
    batch = tokenizer(texts, return_tensors="pt", padding=True, truncation=True,
                      max_length=args.max_seq_len).to("cuda")
    batch["labels"] = batch["input_ids"].clone()

    model.train()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-4)
    try:
        loss = model(**batch).loss
        loss.backward()
        grad_norm = sum(p.grad.norm().item() ** 2 for p in model.parameters()
                        if p.requires_grad and p.grad is not None) ** 0.5
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    except Exception as e:
        die(f"forward/backward failed: {type(e).__name__}: {e}")
    print(f"  loss {loss.item():.4f}   grad norm {grad_norm:.4f}")
    if not math.isfinite(loss.item()):
        die("loss is non-finite on the first step — the compute dtype or the "
            "quantization config is wrong.")
    if not math.isfinite(grad_norm) or grad_norm == 0.0:
        die(f"gradient norm is invalid ({grad_norm}): the optimizer step changed "
            "nothing or diverged, so training cannot proceed.")
    try:
        trl_result = trainer.train()
    except Exception as e:
        die(f"TRL trainer.train() failed: {type(e).__name__}: {e}")
    trl_metrics = {
        key: value for key, value in (trl_result.metrics or {}).items()
        if isinstance(value, (int, float, str, bool)) or value is None
    }
    print(f"  TRL trainer.train completed: {trl_metrics}")
    report["steps"]["3_lora_step"] = {
        "trainable_params": trainable,
        "manual_loss": round(loss.item(), 4),
        "manual_grad_norm": round(grad_norm, 4),
        "trl_max_steps": 1,
        "trl_metrics": trl_metrics,
    }

    # --- 4. save and reload --------------------------------------------------
    step(4, "save the adapter and reload it")
    adapter_dir = out / "adapter"
    try:
        trainer.save_model(str(adapter_dir))
        tokenizer.save_pretrained(adapter_dir)
    except Exception as e:
        die(f"adapter save failed: {type(e).__name__}: {e}")
    files = sorted(p.name for p in adapter_dir.iterdir())
    print(f"  wrote {adapter_dir}: {', '.join(files)}")
    # The same two files run_gates.verify_adapter_served() requires. If the
    # trainer writes something that shape does not match, the after gates would
    # refuse the real adapter — better to discover that here.
    if not (adapter_dir / "adapter_config.json").is_file():
        die("no adapter_config.json — run_gates would refuse this directory.")
    if not any((adapter_dir / f).is_file() for f in
               ("adapter_model.safetensors", "adapter_model.bin")):
        die("no adapter weights were written.")
    try:
        from peft import PeftConfig
        reloaded = PeftConfig.from_pretrained(adapter_dir)
    except Exception as e:
        die(f"adapter reload failed: {type(e).__name__}: {e}")
    if reloaded.base_model_name_or_path != base_model:
        die(f"the saved adapter names base {reloaded.base_model_name_or_path!r}, "
            f"not {base_model!r}. Serving it against the configured base would "
            "be a mismatch.")
    print(f"  reloaded: r={reloaded.r}, base={reloaded.base_model_name_or_path}")
    report["steps"]["4_save_reload"] = {"path": str(adapter_dir), "files": files,
                                        "r": reloaded.r,
                                        "base": reloaded.base_model_name_or_path}

    report["elapsed_seconds"] = round(time.time() - t0, 1)
    report["authorises"] = ("regenerating the schema-v2 freeze and asking for the "
                            "v1 decision — nothing else")
    (out / "SMOKE.json").write_text(json.dumps(report, indent=2) + "\n",
                                    encoding="utf-8")
    print(f"\nwrote {out / 'SMOKE.json'}  ({report['elapsed_seconds']}s)")
    print("\nSTEPS 1-4 PASSED. Continue with steps 5-7 in "
          "calibration/TRAINING_SMOKE_TEST.md:\n"
          "  serve the adapter under a distinct id, prove that id generates, "
          "stop the pod.")
    print("\nThis authorises NO training run. That is a separate decision.")


if __name__ == "__main__":
    main()
