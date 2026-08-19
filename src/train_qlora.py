"""QLoRA trainer for the Office student, gated before and after.

    # verify everything without a GPU and without training
    ./.venv/bin/python src/train_qlora.py --dry-run

    # the real run (requires an explicit decision and GPU dependencies)
    ./.venv/bin/python src/train_qlora.py --run-dir ~/tantular-runs/v1 \
        --confirm-run-v1

Reads train/qlora_9b.yaml and trains ONLY from data/promoted/{train,eval}.jsonl.

WHAT THIS REFUSES TO DO, and why each refusal exists:

  stale or incomplete run freeze     RUN_MANIFEST.v1.json pins the config,
                                     candidate corpus, mechanical promotion
                                     manifest, promoted train/eval bytes, corpus
                                     gate result, and signed waiver. Any missing
                                     field or changed digest aborts before gates
                                     or GPU imports.

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
RUN_MANIFEST_SCHEMA_VERSION = 2
REQUIRED_INT4_WAIVER = ROOT / "calibration" / "INT4_WAIVER.md"


def digest_file(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def die(msg: str, code: int = 2) -> None:
    print(f"\nTRAINER ABORTED: {msg}", file=sys.stderr)
    sys.exit(code)


def load_jsonl(path: Path) -> list[dict]:
    try:
        return [json.loads(l) for l in
                path.read_text(encoding="utf-8").splitlines() if l.strip()]
    except (OSError, json.JSONDecodeError) as e:
        die(f"invalid JSONL at {path}: {e}")


def load_json(path: Path, label: str) -> dict:
    if not path.is_file():
        die(f"{label} missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        die(f"{label} is not readable JSON: {path}: {e}")
    if not isinstance(payload, dict):
        die(f"{label} must contain a JSON object: {path}")
    return payload


def rooted(path_value: str, label: str) -> Path:
    if not isinstance(path_value, str) or not path_value.strip():
        die(f"{label} path is missing from the run freeze")
    path = Path(path_value)
    return path if path.is_absolute() else ROOT / path


def require_digest(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        die(f"{label} missing: {path}")
    actual = digest_file(path)
    if not isinstance(expected, str) or actual != expected:
        die(f"{label} is STALE or changed.\n"
            f"  freeze  {expected}\n"
            f"  on disk {actual}\n"
            "Re-run src/freeze_training_run.py after all intended changes.")
    return actual


# --- preflight checks -------------------------------------------------------

def check_corpus_matches_manifest(manifest_path: Path) -> dict:
    """Validate the mechanical promotion manifest and promoted output bytes."""
    manifest = load_json(manifest_path, "promotion manifest")
    print("=== CORPUS vs PROMOTION MANIFEST ===")
    try:
        promoted = manifest["promoted"]
        recorded = manifest["splits"]["fingerprint"]
    except (KeyError, TypeError):
        die("promotion manifest is malformed: expected promoted and "
            "splits.fingerprint")
    for split in ("train", "eval"):
        entry = promoted.get(split)
        if not isinstance(entry, dict):
            die(f"promotion manifest has no promoted.{split} object")
        path = rooted(entry.get("path"), f"promoted {split}")
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
    if live != recorded:
        die(f"split fingerprint changed: manifest {recorded}, live {live}.\n"
            "Families may now sit in different splits than when the corpus was "
            "generated, which would leak train into eval.")
    print(f"  split fingerprint {recorded}  OK")
    return manifest


def run_corpus_gate(corpus: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [PY, str(ROOT / "src" / "verify_corpus.py"), str(corpus), "--gate"],
        capture_output=True, text=True, cwd=ROOT)


def corpus_gate_errors(corpus: Path) -> list[str]:
    sys.path.insert(0, str(ROOT / "src"))
    import splits as splits_module
    import verify_corpus

    manifest = splits_module.load()
    records = verify_corpus.load_corpus([corpus])
    return verify_corpus.check(records, manifest, gate=True)


def check_run_freeze(run_manifest_path: Path, config_path: Path,
                     promotion_manifest_path: Path) -> dict:
    """Enforce the exact, versioned freeze before any endpoint or GPU work.

    A waiver does not turn the int4 gate into a pass. The expected shape is a
    recorded non-zero gate result plus an unchanged signed waiver; both are
    verified independently here.
    """
    freeze = load_json(run_manifest_path, "run manifest")
    version = freeze.get("schema_version")
    if version != RUN_MANIFEST_SCHEMA_VERSION:
        die(
            f"run manifest schema is stale or unsupported: {version!r}; expected "
            f"{RUN_MANIFEST_SCHEMA_VERSION}.\n"
            "This freeze predates promotion-manifest enforcement. Re-run "
            "src/freeze_training_run.py after the training smoke test."
        )

    try:
        config_entry = freeze["training_config"]
        corpus_entry = freeze["corpus"]
        promotion_entry = freeze["promotion_manifest"]
        gate_entry = freeze["gate"]
        waiver_entry = freeze["waiver"]
    except (KeyError, TypeError) as e:
        die(f"run manifest is incomplete: missing or malformed {e}")

    frozen_config = rooted(config_entry.get("path"), "training config")
    if frozen_config.resolve() != config_path.resolve():
        die("run manifest names a different training config.\n"
            f"  freeze  {frozen_config}\n"
            f"  command {config_path}")
    require_digest(config_path, config_entry.get("sha256"), "training config")

    corpus = rooted(corpus_entry.get("path"), "source corpus")
    require_digest(corpus, corpus_entry.get("sha256"), "source corpus")
    rows = load_jsonl(corpus)
    families = len({row.get("family") for row in rows})
    if len(rows) != corpus_entry.get("traces") or families != corpus_entry.get("families"):
        die("source corpus counts do not match the run freeze.\n"
            f"  freeze traces/families {corpus_entry.get('traces')}/"
            f"{corpus_entry.get('families')}\n"
            f"  actual traces/families {len(rows)}/{families}")

    frozen_promotion = rooted(promotion_entry.get("path"), "promotion manifest")
    if frozen_promotion.resolve() != promotion_manifest_path.resolve():
        die("run manifest names a different promotion manifest.\n"
            f"  freeze  {frozen_promotion}\n"
            f"  command {promotion_manifest_path}")
    require_digest(promotion_manifest_path, promotion_entry.get("sha256"),
                   "promotion manifest")
    promotion = check_corpus_matches_manifest(promotion_manifest_path)

    try:
        promoted_snapshot = promotion_entry["promoted"]
        promotion_source = promotion["source_corpus"]
        promotion_split = promotion["splits"]["fingerprint"]
    except (KeyError, TypeError) as e:
        die(f"promotion pin in run manifest is incomplete: {e}")
    if promotion_source.get("sha256") != corpus_entry.get("sha256"):
        die("promotion manifest source corpus does not match the frozen corpus")
    if promotion_entry.get("source_corpus_sha256") != corpus_entry.get("sha256"):
        die("run manifest promotion pin names a different source corpus")
    frozen_split = corpus_entry.get("provenance", {}).get("split_fingerprint")
    if not frozen_split or promotion_split != frozen_split or \
            promotion_entry.get("split_fingerprint") != frozen_split:
        die("split fingerprint disagrees between corpus freeze and promotion "
            f"manifest: corpus={frozen_split!r}, promotion={promotion_split!r}, "
            f"pin={promotion_entry.get('split_fingerprint')!r}")

    for split in ("train", "eval"):
        live = promotion["promoted"].get(split)
        pinned = promoted_snapshot.get(split) if isinstance(promoted_snapshot, dict) else None
        if not isinstance(live, dict) or not isinstance(pinned, dict):
            die(f"run manifest lacks the promoted {split} snapshot")
        for field in ("path", "sha256", "traces"):
            if pinned.get(field) != live.get(field):
                die(f"promoted {split} {field} disagrees between the run freeze "
                    "and promotion manifest")

    current_gate = run_corpus_gate(corpus)
    current_errors = corpus_gate_errors(corpus)
    recorded_exit = gate_entry.get("exit_code")
    recorded_verdict = gate_entry.get("verdict")
    if not isinstance(recorded_exit, int):
        die("run manifest gate.exit_code is missing or not an integer")
    expected_verdict = "FAILED" if recorded_exit else "passed"
    if recorded_verdict != expected_verdict:
        die("run manifest gate verdict contradicts its exit code: "
            f"exit={recorded_exit}, verdict={recorded_verdict!r}")
    if current_gate.returncode != recorded_exit:
        die("corpus gate result changed since the run was frozen.\n"
            f"  freeze exit {recorded_exit}\n"
            f"  current exit {current_gate.returncode}\n"
            "Refresh the freeze; do not reinterpret an old authorization.")
    recorded_errors = gate_entry.get("violations")
    if not isinstance(recorded_errors, list):
        die("run manifest gate.violations is missing or not a list")
    if current_errors != recorded_errors:
        die("corpus gate violations changed since the run was frozen.\n"
            f"  freeze  {recorded_errors}\n"
            f"  current {current_errors}")
    if bool(current_errors) != bool(recorded_exit):
        die("run manifest gate exit code disagrees with its violation list")

    waiver_path_value = waiver_entry.get("path")
    waiver_sha = waiver_entry.get("sha256")
    if recorded_exit != 0:
        if not waiver_path_value or not waiver_sha:
            die("the corpus gate FAILED but the run manifest has no signed waiver")
        waiver = rooted(waiver_path_value, "waiver")
        if waiver.resolve() != REQUIRED_INT4_WAIVER.resolve():
            die("the failed int4 gate is not authorized by an arbitrary waiver.\n"
                f"  required {REQUIRED_INT4_WAIVER}\n"
                f"  freeze   {waiver}")
        import verify_corpus
        if not verify_corpus.waiver_covers(current_errors):
            die("the signed int4 waiver does not cover the current gate "
                "violations:\n  " + "\n  ".join(current_errors))
        require_digest(waiver, waiver_sha, "signed waiver")
        waiver_status = {"path": str(waiver), "sha256": waiver_sha,
                         "authorises_failed_gate": True}
    else:
        if waiver_path_value or waiver_sha:
            if not waiver_path_value or not waiver_sha:
                die("waiver pin is incomplete: path and sha256 must appear together")
            waiver = rooted(waiver_path_value, "waiver")
            require_digest(waiver, waiver_sha, "waiver")
        waiver_status = {"path": waiver_path_value, "sha256": waiver_sha,
                         "authorises_failed_gate": False}

    print("\n=== RUN FREEZE ===")
    print(f"  schema             {version}")
    print(f"  run manifest       {digest_file(run_manifest_path)[:16]}…")
    print(f"  config             {config_entry['sha256'][:16]}…")
    print(f"  source corpus      {corpus_entry['sha256'][:16]}…")
    print(f"  promotion manifest {promotion_entry['sha256'][:16]}…")
    print(f"  corpus gate        exit {recorded_exit} "
          f"({'waiver verified' if recorded_exit else 'passed'})")
    return {
        "manifest": freeze,
        "manifest_path": str(run_manifest_path),
        "manifest_sha256": digest_file(run_manifest_path),
        "corpus": corpus,
        "promotion": promotion,
        "waiver": waiver_status,
    }


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


def run_gates(stage: str, out: Path, adapter: Path | None, args, expect: str) -> dict:
    cmd = [PY, str(ROOT / "src" / "run_gates.py"), "run",
           "--stage", stage, "--host", args.student_host,
           "--teacher", args.student_model, "--expect-model", expect,
           "--out", str(out)]
    if adapter:
        # The adapter must be SERVED under this id before the after gates run,
        # or they measure the base model:
        #   ./scripts/serve_student.sh office-student-9b student-serve \
        #       <adapter> <id>
        cmd += ["--adapter", str(adapter),
                "--adapter-model-id", args.adapter_model_id]
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
    parser.add_argument("--run-manifest", default="train/RUN_MANIFEST.v1.json",
                        help="complete versioned freeze: config, corpus, promotion, "
                             "gate verdict, and waiver")
    parser.add_argument("--promotion-manifest", "--manifest",
                        dest="promotion_manifest",
                        default="train/RUN_MANIFEST.v1-mechanical.json",
                        help="mechanical promotion decision; --manifest is retained "
                             "as a compatibility alias")
    parser.add_argument("--run-dir", default="~/tantular-runs/v1")
    # The gates must measure the STUDENT the config names. These previously
    # defaulted to the teacher (muse-glimmer @ ai19-ollama), which would have made
    # the "before" run measure Muse Glimmer int4 and turned the before/after
    # comparison into teacher-vs-adapter. No defaults now: the serving host must
    # be stated, and its identity is verified against config["base_model"].
    parser.add_argument("--student-host", default=None,
                        help="host serving the BASE STUDENT named in the config")
    parser.add_argument("--student-model", default=None,
                        help="model key for that host")
    parser.add_argument("--adapter-model-id", default="tantular-office-9b-v1",
                        help="model id the endpoint must register the trained "
                             "adapter under, distinct from the base")
    parser.add_argument("--train-host", default=None,
                        help="host config for the machine that will TRAIN. Must "
                             "declare training_allowed: true, and must not be "
                             "ai19, which is production-serving.")
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

    run_manifest_path = rooted(args.run_manifest, "run manifest argument")
    promotion_manifest_path = rooted(args.promotion_manifest,
                                     "promotion manifest argument")
    freeze = check_run_freeze(run_manifest_path, config_path,
                              promotion_manifest_path)
    manifest = freeze["promotion"]
    held_out = check_eval_held_out(config)
    run_dir_kind = check_run_dir_outside_git(run_dir)

    gate_names = [g["name"] for g in config.get("eval_gates", [])]
    print(f"\n=== GATES DECLARED ===\n  {', '.join(gate_names)}")

    base_model = config["base_model"]
    print(f"\n=== TRAINING HOST ===")
    # Serving and training are separate machines on purpose: a training run on
    # the box serving the gates would perturb the very endpoint being measured,
    # and ai19 — the obvious box to reach for — is in production.
    if not args.train_host:
        print("  NOT DECLARED — no --train-host given.")
        print("  Declare where training will run. It is checked twice: the host "
              "config must say")
        print("  training_allowed: true, and this machine's own hostname must "
              "not belong to a")
        print("  forbidden host (so --train-host rented-48gb typed on ai19 "
              "still refuses).")
        train_host_ok = False
    else:
        sys.path.insert(0, str(ROOT / "src"))
        from config import training_guard
        training_guard(args.train_host)          # exits on violation
        print(f"  {args.train_host}  training_allowed  OK")
        train_host_ok = True

    print(f"\n=== STUDENT ENDPOINT ===")
    if not (args.student_host and args.student_model):
        print(f"  config base_model : {base_model}")
        print("  NOT CONFIGURED — no --student-host/--student-model given.")
        print("  The gates need an endpoint serving the STUDENT, not the teacher.")
        print("  Serve it with:")
        print("    configs/teachers/office-student-9b.yaml  (bf16, tokenizer pinned)")
        print("    configs/hosts/student-serve.yaml         (a rental, NOT ai19)")
        print("  then pass --student-host student-serve "
              "--student-model office-student-9b.")
        student_ready = False
    else:
        print(f"  {args.student_model} @ {args.student_host}, expecting {base_model}")
        student_ready = True

    if args.dry_run:
        print("\nDRY RUN OK — configuration, corpus integrity, held-out status and "
              "run directory all verified.\nNothing was generated, trained, or "
              "written. Gates were NOT executed: that needs a served model.")
        if not train_host_ok:
            print("\nSTILL BLOCKED: no training host declared. Pass --train-host "
                  "rented-48gb (or another\nhost whose config declares "
                  "training_allowed: true). ai19 is refused by design.")
        if not student_ready:
            print("\nSTILL BLOCKED: no student endpoint. Training cannot start "
                  "until an endpoint serves\n" + f"  {base_model}\n"
                  "and --student-host/--student-model point at it. Pointing the "
                  "gates at the teacher\nwould measure the wrong model entirely.")
        return

    if not train_host_ok:
        die("no --train-host declared. Where a training run happens is a decision "
            "that must be stated and checked, not inherited from whichever "
            "machine the command was typed on.")
    if not student_ready:
        die("no student endpoint configured. The gates would otherwise be pointed "
            "at whatever is default — the teacher — and the before/after "
            "comparison would be meaningless.")
    if not args.confirm_run_v1:
        die("refusing to train without --confirm-run-v1.\n"
            "Every check above can pass and the decision to spend a GPU on v1 is "
            "still a human one.", code=1)

    run_dir.mkdir(parents=True, exist_ok=True)
    before = run_gates("before", run_dir / "gates.before.json", None, args, base_model)
    # A base model with no Indonesian office training is EXPECTED to sit under
    # 0.95. That is recorded as the starting point, not treated as a pass and not
    # treated as a stop; the same thresholds apply unchanged after training.
    if before.get("below_target"):
        print(f"\n  baseline below target on: {', '.join(before['below_target'])} "
              "— recorded, continuing.")
    adapter = train(config, run_dir, args)
    print(f"\n=== SERVE THE ADAPTER BEFORE THE AFTER GATES ===")
    print(f"  On {args.student_host}, restart the endpoint with the adapter "
          "registered:\n"
          f"    ./scripts/serve_student.sh {args.student_model} "
          f"{args.student_host} \\\n        {adapter} {args.adapter_model_id}")
    input_msg = ("  Press Enter once /v1/models lists "
                 f"{args.adapter_model_id}, or Ctrl-C to stop. ")
    try:
        input(input_msg)
    except EOFError:
        print("\n  (no tty — continuing; the after gates will refuse if the "
              "adapter is not served)")
    after = run_gates("after", run_dir / "gates.after.json", adapter, args, base_model)

    proc = subprocess.run(
        [PY, str(ROOT / "src" / "run_gates.py"), "compare",
         "--before", str(run_dir / "gates.before.json"),
         "--after", str(run_dir / "gates.after.json")], cwd=ROOT)

    (run_dir / "RUN.json").write_text(json.dumps({
        "config": {"path": args.config, "sha256": digest_file(config_path)},
        "run_manifest": {"path": args.run_manifest,
                         "sha256": freeze["manifest_sha256"]},
        "promotion_manifest": {"path": args.promotion_manifest,
                               "sha256": digest_file(promotion_manifest_path)},
        "waiver": freeze["waiver"],
        "held_out_verification": held_out,
        "run_dir_kind": run_dir_kind,
        "adapter": str(adapter),
        "gates_before": before["gates"], "gates_after": after["gates"],
        "baseline_verdict": before.get("verdict"),
        "baseline_below_target": before.get("below_target", []),
        "after_verdict": after.get("verdict"),
        "adapter_model_id": args.adapter_model_id,
        "adapter_evaluated": after.get("adapter", {}).get("evaluated"),
        # Promotion needs the absolute thresholds AND no regression; `compare`
        # is the only thing that checks both, so its exit code is the answer.
        "promote_adapter": proc.returncode == 0,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {run_dir / 'RUN.json'}")
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
