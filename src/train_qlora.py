"""QLoRA trainer for the Office student, gated before and after.

    # verify everything without a GPU and without training
    ./.venv/bin/python src/train_qlora.py --dry-run

    # the real run (requires an explicit decision and GPU dependencies)
    ./.venv/bin/python src/train_qlora.py --run-dir ~/tantular-runs/v1 \
        --confirm-run-v1

Reads train/qlora_9b.yaml and trains ONLY from data/promoted/{train,eval}.jsonl.

WHAT THIS REFUSES TO DO, and why each refusal exists:

  corpus not matching its manifest   RUN_MANIFEST.v1-mechanical.json records the
                                     sha256 of every promoted file. A corpus
                                     that changed after promotion was never
                                     reviewed, so training on it would attribute
                                     a reviewed provenance to unreviewed data.

  eval prompts inside the corpus     The voice and edit gates are held out. If
                                     an eval item reached training, its gate
                                     measures recall, not capability, and would
                                     report a passing adapter that learned the
                                     test.

  a gate that cannot execute         Stopping is the point. A run that trains
                                     while a gate is unrunnable produces an
                                     adapter nobody checked, and the report
                                     still looks complete.

  writing inside Git                 Checkpoints and logs are large and are not
                                     source. The run directory must be outside
                                     the repository or gitignored, and this
                                     verifies which rather than trusting it.

  training without --confirm-run-v1  The decision to run v1 is a human one.

GPU dependencies are imported LAZILY, inside the training step, so --dry-run
works on a laptop with nothing installed. That is deliberate: every check that
can be made without a GPU should be makeable without one.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.exit("pyyaml is required: pip install -r requirements.txt")

ROOT = Path(__file__).resolve().parent.parent
PY = str(ROOT / ".venv" / "bin" / "python")


def digest_file(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def die(msg: str, code: int = 2) -> None:
    print(f"\nTRAINER ABORTED: {msg}", file=sys.stderr)
    sys.exit(code)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in
            path.read_text(encoding="utf-8").splitlines() if l.strip()]


# --- preflight checks -------------------------------------------------------

def check_corpus_matches_manifest(manifest_path: Path) -> dict:
    if not manifest_path.is_file():
        die(f"promotion manifest missing: {manifest_path}\n"
            "Run src/promote_corpus.py first — training must not read a corpus "
            "whose promotion was never recorded.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    print("=== CORPUS vs PROMOTION MANIFEST ===")
    for split, entry in manifest["promoted"].items():
        path = ROOT / entry["path"]
        if not path.is_file():
            die(f"promoted {split} missing: {path}")
        actual = digest_file(path)
        if actual != entry["sha256"]:
            die(f"promoted {split} CHANGED since promotion.\n"
                f"  manifest {entry['sha256'][:16]}…\n"
                f"  on disk  {actual[:16]}…\n"
                "The corpus that was reviewed is not the corpus on disk.")
        rows = load_jsonl(path)
        if len(rows) != entry["traces"]:
            die(f"promoted {split} has {len(rows)} traces, manifest says {entry['traces']}")
        print(f"  {split:<6} {len(rows):>4} traces  {actual[:16]}…  OK")

    # The manifest's own split assignment must still be the one that governed
    # generation; a reseeded manifest would move families across boundaries.
    sys.path.insert(0, str(ROOT / "src"))
    import splits as splits_module
    live = splits_module.load()["fingerprint"]
    recorded = manifest["splits"]["fingerprint"]
    if live != recorded:
        die(f"split fingerprint changed: manifest {recorded}, live {live}.\n"
            "Families may now sit in different splits than when the corpus was "
            "generated, which would leak train into eval.")
    print(f"  split fingerprint {recorded}  OK")
    return manifest


def check_eval_held_out(config: dict) -> dict:
    """No gate's eval item may appear in the training corpus."""
    print("\n=== HELD-OUT VERIFICATION ===")
    train_rows = load_jsonl(ROOT / "data/promoted/train.jsonl")
    eval_rows = load_jsonl(ROOT / "data/promoted/eval.jsonl")
    corpus_users = {(r.get("user") or "").strip() for r in train_rows + eval_rows}
    corpus_ids = {r.get("family") for r in train_rows + eval_rows}

    report = {}
    for gate in config.get("eval_gates", []):
        src = gate.get("source", "")
        if not src.endswith(".jsonl"):
            continue                      # model-independent gate, no prompts
        path = ROOT / src
        if not path.is_file():
            die(f"gate '{gate['name']}' source missing: {path}")
        items = load_jsonl(path)
        by_text = [i for i in items if (i.get("user") or "").strip() in corpus_users]
        by_id = [i for i in items if i.get("id") in corpus_ids]
        report[gate["name"]] = {"items": len(items),
                                "in_corpus_by_prompt": len(by_text),
                                "in_corpus_by_id": len(by_id)}
        status = "HELD OUT" if not (by_text or by_id) else "LEAKED"
        print(f"  {gate['name']:<22} {len(items):>3} items   {status}")
        if by_text or by_id:
            die(f"gate '{gate['name']}' has {len(by_text) + len(by_id)} item(s) "
                "inside the training corpus. Its gate would measure recall of the "
                "test, not capability.")
    return report


def check_run_dir_outside_git(run_dir: Path) -> str:
    """Checkpoints and logs must not land in version control."""
    print("\n=== RUN DIRECTORY ===")
    run_dir = run_dir.expanduser()
    try:
        rel = run_dir.resolve().relative_to(ROOT.resolve())
    except ValueError:
        print(f"  {run_dir}  outside the repository  OK")
        return "outside-repo"
    proc = subprocess.run(["git", "check-ignore", "-q", str(rel)],
                          cwd=ROOT, capture_output=True)
    if proc.returncode != 0:
        die(f"run directory {rel} is inside the repo and NOT gitignored.\n"
            "Checkpoints and logs are not source. Choose a path outside the "
            "repository, or add it to .gitignore.")
    print(f"  {rel}  inside repo but gitignored  OK")
    return "gitignored"


def run_gates(stage: str, out: Path, adapter: Path | None, args) -> dict:
    cmd = [PY, str(ROOT / "src" / "run_gates.py"), "run",
           "--stage", stage, "--host", args.host, "--teacher", args.teacher,
           "--out", str(out)]
    if adapter:
        cmd += ["--adapter", str(adapter)]
    print(f"\n=== GATES [{stage}] ===")
    proc = subprocess.run(cmd, cwd=ROOT)
    if proc.returncode == 2:
        die(f"a gate could not be executed at stage '{stage}'. Not training past "
            "an unrunnable gate: the adapter would be unchecked and the report "
            "would still look complete.")
    if not out.is_file():
        die(f"gate run produced no report at {out}")
    return json.loads(out.read_text())


# --- the training step ------------------------------------------------------

def train(config: dict, run_dir: Path, args) -> Path:
    """Imports GPU dependencies lazily so every check above works without them."""
    try:
        import torch  # noqa: F401
        from datasets import Dataset  # noqa: F401
        from peft import LoraConfig  # noqa: F401
        from transformers import (AutoModelForCausalLM, AutoTokenizer,  # noqa: F401
                                  BitsAndBytesConfig)
        from trl import SFTConfig, SFTTrainer  # noqa: F401
    except ImportError as e:
        die(f"training dependencies are not installed: {e}\n"
            "These are GPU-host only and deliberately commented out of "
            "requirements.txt. Install them on the training host, not here.", code=3)

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    lora, tr, out = config["lora"], config["training"], config["output"]
    adapter_dir = run_dir / Path(out["adapter_dir"]).name
    tok = AutoTokenizer.from_pretrained(config["base_model"])

    def to_messages(rows):
        return Dataset.from_list([{"messages": [
            {"role": "system", "content": r.get("system", "")},
            {"role": "user", "content": r.get("user", "")},
            {"role": "assistant", "content": r.get("completion", "")}]} for r in rows])

    train_ds = to_messages(load_jsonl(ROOT / "data/promoted/train.jsonl"))
    eval_ds = to_messages(load_jsonl(ROOT / "data/promoted/eval.jsonl"))

    model = AutoModelForCausalLM.from_pretrained(
        config["base_model"], device_map="auto", dtype=torch.bfloat16,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=config.get("load_in_4bit", True),
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True))

    trainer = SFTTrainer(
        model=model, train_dataset=train_ds, eval_dataset=eval_ds,
        peft_config=LoraConfig(
            r=lora["r"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"],
            target_modules=lora["target_modules"], task_type="CAUSAL_LM"),
        args=SFTConfig(
            output_dir=str(run_dir / "checkpoints"),
            max_length=config.get("max_seq_length", 32768),
            num_train_epochs=tr["num_train_epochs"],
            per_device_train_batch_size=tr["per_device_train_batch_size"],
            gradient_accumulation_steps=tr["gradient_accumulation_steps"],
            learning_rate=float(tr["learning_rate"]),
            lr_scheduler_type=tr["lr_scheduler_type"],
            warmup_ratio=tr["warmup_ratio"],
            gradient_checkpointing=tr["gradient_checkpointing"],
            bf16=tr["bf16"], logging_steps=tr["logging_steps"],
            save_steps=tr["save_steps"], report_to=[]))
    trainer.train()
    trainer.save_model(str(adapter_dir))
    tok.save_pretrained(str(adapter_dir))
    return adapter_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="train/qlora_9b.yaml")
    parser.add_argument("--manifest", default="train/RUN_MANIFEST.v1-mechanical.json")
    parser.add_argument("--run-dir", default="~/tantular-runs/v1")
    parser.add_argument("--host", default="ai19-ollama")
    parser.add_argument("--teacher", default="muse-glimmer")
    parser.add_argument("--dry-run", action="store_true",
                        help="verify everything and stop before gates or training")
    parser.add_argument("--confirm-run-v1", action="store_true",
                        help="the explicit decision to train; without it nothing trains")
    args = parser.parse_args()

    config_path = ROOT / args.config
    if not config_path.is_file():
        die(f"config missing: {config_path}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_dir = Path(args.run_dir).expanduser()

    print(f"config    {args.config}  {digest_file(config_path)[:16]}…")
    print(f"base      {config['base_model']}\n")

    manifest = check_corpus_matches_manifest(ROOT / args.manifest)
    held_out = check_eval_held_out(config)
    run_dir_kind = check_run_dir_outside_git(run_dir)

    gate_names = [g["name"] for g in config.get("eval_gates", [])]
    print(f"\n=== GATES DECLARED ===\n  {', '.join(gate_names)}")

    if args.dry_run:
        print("\nDRY RUN OK — configuration, corpus integrity, held-out status and "
              "run directory all verified.\nNothing was generated, trained, or "
              "written. Gates were NOT executed: that needs a served model.")
        return

    if not args.confirm_run_v1:
        die("refusing to train without --confirm-run-v1.\n"
            "Every check above can pass and the decision to spend a GPU on v1 is "
            "still a human one.", code=1)

    run_dir.mkdir(parents=True, exist_ok=True)
    before = run_gates("before", run_dir / "gates.before.json", None, args)
    adapter = train(config, run_dir, args)
    after = run_gates("after", run_dir / "gates.after.json", adapter, args)

    proc = subprocess.run(
        [PY, str(ROOT / "src" / "run_gates.py"), "compare",
         "--before", str(run_dir / "gates.before.json"),
         "--after", str(run_dir / "gates.after.json")], cwd=ROOT)

    (run_dir / "RUN.json").write_text(json.dumps({
        "config": {"path": args.config, "sha256": digest_file(config_path)},
        "corpus_manifest": {"path": args.manifest,
                            "sha256": digest_file(ROOT / args.manifest)},
        "held_out_verification": held_out,
        "run_dir_kind": run_dir_kind,
        "adapter": str(adapter),
        "gates_before": before["gates"], "gates_after": after["gates"],
        "promote_adapter": proc.returncode == 0,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {run_dir / 'RUN.json'}")
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
