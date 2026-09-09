"""CPU-only guards for the opt-in Tinker training backend."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
PY = str(ROOT / ".venv" / "bin" / "python")
RUNNER = str(ROOT / "src" / "train_tinker_sft.py")
EXPORTER = str(ROOT / "src" / "export_tinker_weights.py")
FREEZER = str(ROOT / "src" / "freeze_training_run.py")
CONFIG = ROOT / "train" / "tinker_sft_9b.yaml"
PREVIEW = ROOT / "train" / "RUN_MANIFEST.tinker-sft-v1.preview.json"
PROMOTION = ROOT / "train" / "RUN_MANIFEST.v1-mechanical.json"
CORPUS = ROOT / "data" / "v3-candidate" / "traces.r0.jsonl"
WAIVER = ROOT / "calibration" / "INT4_WAIVER.md"

sys.path.insert(0, str(ROOT / "src"))
import run_gates
import tinker_payload
import train_tinker_sft


def first_train_row() -> dict:
    line = next(
        line
        for line in (ROOT / "data" / "promoted" / "train.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    )
    return json.loads(line)


def test_default_invocation_is_local_only_and_blocked_by_decision():
    proc = subprocess.run(
        [PY, RUNNER],
        capture_output=True,
        text=True,
        cwd=ROOT,
        env={"PATH": str(Path(PY).parent)},
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "DRY RUN OK" in proc.stdout
    assert "Nothing was written, imported from Tinker, uploaded, or trained" in proc.stdout
    assert "EXECUTION BLOCKED" in proc.stdout


def test_execute_with_preview_manifest_refuses_before_dependencies_or_key():
    proc = subprocess.run(
        [
            PY,
            RUNNER,
            "--execute",
            "--train-host",
            "tinker",
            "--egress-approval",
            "TEST-ONLY",
            "--max-cost-usd",
            "10",
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
        env={"PATH": str(Path(PY).parent)},
    )
    assert proc.returncode == 1
    assert "preview-only" in proc.stderr
    assert "not installed" not in proc.stderr
    assert "TINKER_API_KEY" not in proc.stderr


@pytest.mark.parametrize("source_class", ["local_real", "Synthetic", None])
def test_non_synthetic_or_unclassified_row_is_rejected(source_class):
    row = first_train_row()
    if source_class is None:
        row.pop("source_class")
    else:
        row["source_class"] = source_class
    with pytest.raises(tinker_payload.PayloadError, match="exactly 'synthetic'"):
        tinker_payload.render_rows([row], split="train")


def test_rendered_payload_shape_and_sidecar_bind_the_same_bytes():
    row = first_train_row()
    upload, audit, summary = tinker_payload.render_rows([row], split="train")
    tinker_payload.verify_upload_against_audit(upload, audit)

    payload = json.loads(upload)
    assert set(payload) == {"messages"}
    assert [message["role"] for message in payload["messages"]] == [
        "system",
        "user",
        "assistant",
    ]
    assert payload["messages"][0]["content"] == row["system"]
    assert payload["messages"][1]["content"] == row["user"]
    assert payload["messages"][2]["content"] == row["completion"]
    sidecar = json.loads(audit)
    assert sidecar["family"] == row["family"]
    assert sidecar["source_class"] == "synthetic"
    assert sidecar["upload_line_sha256"]
    assert summary["rows"] == 1


def test_train_and_eval_payloads_remain_separate_and_disjoint():
    rendered = tinker_payload.render_files(
        ROOT / "data" / "promoted" / "train.jsonl",
        ROOT / "data" / "promoted" / "eval.jsonl",
    )
    assert rendered["train"]["rows"] == 136
    assert rendered["eval"]["rows"] == 47
    assert not (
        set(rendered["train"]["families"]) & set(rendered["eval"]["families"])
    )
    assert rendered["train"]["sha256"] != rendered["eval"]["sha256"]


def test_checked_in_preview_is_schema_7_and_cannot_authorize_execution():
    manifest = json.loads(PREVIEW.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 7
    assert manifest["backend"]["name"] == "tinker"
    assert manifest["backend"]["base_model"] == "Qwen/Qwen3.5-9B-Base"
    assert manifest["backend"]["renderer"] == "role_colon"
    assert manifest["backend"]["checkpoint_label"] == "trained_unvalidated"
    assert manifest["execution_authorized"] is False
    assert manifest["preview"] is True
    assert manifest["before_baseline"] is None
    assert manifest["training_authorization"] is None


def test_local_qlora_manifest_remains_schema_6():
    manifest = json.loads(
        (ROOT / "train" / "RUN_MANIFEST.v1.json").read_text(encoding="utf-8")
    )
    assert manifest["schema_version"] == 6


def test_nonpreview_tinker_freeze_requires_baseline_and_authorization(tmp_path):
    proc = subprocess.run(
        [
            PY,
            FREEZER,
            "--corpus",
            str(CORPUS),
            "--config",
            str(CONFIG),
            "--promotion-manifest",
            str(PROMOTION),
            "--waiver",
            str(WAIVER),
            "--backend",
            "tinker",
            "--out",
            str(tmp_path / "RUN.json"),
            "--frozen-at",
            "2026-08-29T12:00:00+07:00",
            "--write",
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert proc.returncode != 0
    assert "--baseline-report and --authorization are required" in (
        proc.stdout + proc.stderr
    )


def test_preview_is_an_explicit_flag_not_a_filename_convention(tmp_path):
    out = tmp_path / "ordinary-name.json"
    proc = subprocess.run(
        [
            PY,
            FREEZER,
            "--corpus",
            str(CORPUS),
            "--config",
            str(CONFIG),
            "--promotion-manifest",
            str(PROMOTION),
            "--waiver",
            str(WAIVER),
            "--backend",
            "tinker",
            "--preview",
            "--out",
            str(out),
            "--frozen-at",
            "2026-08-29T12:00:00+07:00",
            "--write",
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    manifest = json.loads(out.read_text())
    assert manifest["preview"] is True
    assert manifest["execution_authorized"] is False


def test_shared_gate_config_is_digest_pinned():
    effective, shared = run_gates.load_gate_config(CONFIG)
    assert effective["base_model"] == "Qwen/Qwen3.5-9B-Base"
    assert len(effective["eval_gates"]) == 3
    assert shared is not None
    assert shared["sha256"] == yaml.safe_load(CONFIG.read_text())["evaluation"][
        "config_sha256"
    ]
    for gate in effective["eval_gates"]:
        assert gate["stop_sequences"] == ["\n\nUser:"]


def test_tinker_config_does_not_copy_local_peft_targets():
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert config["lora"] == {"rank": 32}
    assert config["renderer"] == "role_colon"
    assert config["output"]["checkpoint_label"] == "trained_unvalidated"
    assert "target_modules" not in CONFIG.read_text(encoding="utf-8")
    assert config["serving"]["chat_template"] == "templates/role_colon.jinja"


def test_render_only_writes_no_promotable_adapter_shape(tmp_path):
    out = tmp_path / "payload"
    proc = subprocess.run(
        [PY, RUNNER, "--render-only", str(out)],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (out / "train.conversations.jsonl").is_file()
    assert (out / "eval.conversations.jsonl").is_file()
    assert not (out / "adapter_config.json").exists()
    assert not list(out.glob("*.safetensors"))


def test_exporter_refuses_a_forged_minimal_run_record(tmp_path):
    """QA finding: a hand-typed record with an arbitrary tinker:// path.

    It carries the right status strings and the current config digest, and
    nothing else. The export path must re-walk the authorization chain.
    """
    import hashlib

    run_record = tmp_path / "RUN.json"
    run_record.write_text(json.dumps({
        "status": "trained_unvalidated",
        "checkpoint_label": "trained_unvalidated",
        "config": {"sha256": hashlib.sha256(CONFIG.read_bytes()).hexdigest()},
        "checkpoint": {
            "final": True,
            "sampler_path": "tinker://anything-i-typed/sampler_weights/final",
            "state_path": "tinker://anything-i-typed/weights/final",
        },
    }))
    output = tmp_path / "export"
    proc = subprocess.run(
        [PY, EXPORTER, "--run-record", str(run_record),
         "--output-dir", str(output)],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "does not pin a schema-v3 run manifest" in proc.stderr
    assert "DRY RUN OK" not in proc.stdout
    assert not output.exists()


def test_exporter_refuses_a_record_pinning_the_preview_manifest(tmp_path):
    """The checked-in manifest is preview-only and never authorized a run."""
    run_record = genuine_run_record(tmp_path / "RUN.json")
    proc = subprocess.run(
        [PY, EXPORTER, "--run-record", str(run_record),
         "--output-dir", str(tmp_path / "export")],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "never authorized execution" in proc.stderr


@pytest.mark.parametrize("mutation,expected", [
    ({"payload": {"train": {"upload_sha256": "00" * 32},
                  "eval": {"upload_sha256": "22" * 32}}},
     "does not match the frozen upload"),
    ({"packages": {}}, "has no packages"),
    ({"cookbook_api_surface": None}, "has no cookbook_api_surface"),
    ({"checkpoint_evidence": {"exists": False, "rows": []}},
     "without checkpoints.jsonl"),
    ({"checkpoint": {"final": True, "sampler_path": "tinker://other/s",
                     "state_path": "tinker://other/w"}},
     "does not match the final row"),
])
def test_exporter_refuses_records_that_do_not_hang_together(
    monkeypatch, tmp_path, capsys, mutation, expected
):
    import export_tinker_weights

    stub_authorized_freeze(monkeypatch)
    record = genuine_run_record(tmp_path / "RUN.json", **mutation)
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    with pytest.raises(SystemExit) as exc:
        export_tinker_weights.load_record(record, config, CONFIG)
    assert exc.value.code == 2
    assert expected in capsys.readouterr().err


def test_a_genuine_record_passes_the_export_dry_run(monkeypatch, tmp_path, capsys):
    """The control: the refusals above are specific, not a blanket block."""
    import export_tinker_weights

    stub_authorized_freeze(monkeypatch)
    record = genuine_run_record(tmp_path / "RUN.json")
    output = tmp_path / "export"
    output.mkdir()                       # a prior dry run must not block a retry
    monkeypatch.setattr(sys, "argv", [
        "export_tinker_weights.py",
        "--run-record", str(record),
        "--output-dir", str(output),
    ])
    export_tinker_weights.main()
    out = capsys.readouterr().out
    assert "DRY RUN OK" in out
    assert "resulting status: peft_unvalidated" in out


def test_exporter_rejects_any_record_that_claims_promotion(tmp_path):
    run_record = tmp_path / "RUN.json"
    run_record.write_text(json.dumps({
        "status": "promoted",
        "checkpoint_label": "trained_unvalidated",
        "checkpoint": {
            "final": True,
            "sampler_path": "tinker://test-run/sampler_weights/final",
        },
    }))
    proc = subprocess.run(
        [
            PY,
            EXPORTER,
            "--run-record",
            str(run_record),
            "--output-dir",
            str(tmp_path / "export"),
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert proc.returncode == 2
    assert "status must be exactly 'trained_unvalidated'" in proc.stderr


def test_authorized_manifest_revalidates_baseline_contents(monkeypatch, tmp_path):
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    config_path = tmp_path / "config.yaml"
    authorization = tmp_path / "TRAINING_JUSTIFIED.md"
    authorization.write_text("approved for test only")
    config["authorization"]["required_document"] = str(authorization)
    config_path.write_text(yaml.safe_dump(config))

    baseline = tmp_path / "before.json"
    baseline.write_text(json.dumps({
        "stage": "before",
        "note": "not a real gate report",
    }))
    import hashlib
    baseline_pin = {
        "path": str(baseline),
        "sha256": hashlib.sha256(baseline.read_bytes()).hexdigest(),
    }
    auth_pin = {
        "path": str(authorization),
        "sha256": hashlib.sha256(authorization.read_bytes()).hexdigest(),
    }
    backend = {"name": "tinker"}
    monkeypatch.setattr(
        train_tinker_sft.train_qlora,
        "check_run_freeze",
        lambda *a, **k: {"manifest": {
            "backend": backend,
            "execution_authorized": True,
            "preview": False,
            "before_baseline": baseline_pin,
            "training_authorization": auth_pin,
        }},
    )
    monkeypatch.setattr(
        train_tinker_sft.freeze_training_run,
        "tinker_backend_snapshot",
        lambda path: backend,
    )
    with pytest.raises(SystemExit):
        train_tinker_sft.check_schema_v3_freeze(
            tmp_path / "manifest.json",
            config_path,
            PROMOTION,
            config,
        )


def test_pricing_bound_includes_every_scheduled_eval_pass():
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    rendered = tinker_payload.render_files(
        ROOT / "data" / "promoted" / "train.jsonl",
        ROOT / "data" / "promoted" / "eval.jsonl",
    )
    pricing = train_tinker_sft.pricing_summary(config, rendered)
    # 68 steps / eval_every 17 = 4 scheduled passes, plus one for a step-zero
    # or post-final evaluation. See test_pricing_eval_bound_covers_a_step_zero…
    assert pricing["eval_passes"] == 5
    assert pricing["worst_case_train_tokens"] == 136 * 2 * 4096
    assert pricing["worst_case_eval_tokens"] == 47 * 5 * 4096
    assert pricing["worst_case_total_tokens"] == (
        pricing["worst_case_train_tokens"] + pricing["worst_case_eval_tokens"]
    )


def test_role_colon_template_is_passed_explicitly_to_vllm():
    teacher = yaml.safe_load(
        (ROOT / "configs" / "teachers" / "office-student-9b-base.yaml")
        .read_text(encoding="utf-8")
    )
    assert teacher["chat_template"] == "templates/role_colon.jinja"
    template = (ROOT / teacher["chat_template"]).read_text(encoding="utf-8")
    assert "add_generation_prompt" in template
    assert "Assistant:" in template
    script = (ROOT / "scripts" / "serve_student.sh").read_text(encoding="utf-8")
    assert "--chat-template" in script
    assert "TEACHER_CHAT_TEMPLATE" in script
    import config as config_module
    resolved = config_module.resolve("office-student-9b-base", "student-serve")
    assert resolved["TEACHER_CHAT_TEMPLATE"] == "templates/role_colon.jinja"


def test_checkpoint_evidence_preserves_raw_rows_before_validation(tmp_path):
    log = tmp_path / "tinker"
    log.mkdir()
    checkpoint_file = log / "checkpoints.jsonl"
    checkpoint_file.write_text(
        '{"state_path":"tinker://state/one"}\nnot-json\n',
        encoding="utf-8",
    )
    evidence = train_tinker_sft.checkpoint_evidence(log)
    assert evidence["exists"] is True
    assert len(evidence["raw_lines"]) == 2
    assert evidence["rows"] == [{"state_path": "tinker://state/one"}]
    assert evidence["parse_errors"][0]["line"] == 2
    with pytest.raises(SystemExit):
        train_tinker_sft.validate_final_checkpoint(evidence)


def test_renderer_token_extraction_failure_is_a_controlled_abort(
    monkeypatch, capsys
):
    renderers = ModuleType("tinker_cookbook.renderers")
    tokenizer_utils = ModuleType("tinker_cookbook.tokenizer_utils")

    class BrokenInput:
        length = 3

        @staticmethod
        def to_ints():
            raise AttributeError("simulated API drift")

    class Weights:
        @staticmethod
        def tolist():
            return [0.0, 1.0, 1.0]

    class Renderer:
        @staticmethod
        def build_supervised_example(*args, **kwargs):
            return BrokenInput(), Weights()

    renderers.TrainOnWhat = SimpleNamespace(ALL_ASSISTANT_MESSAGES="assistant")
    renderers.get_renderer = lambda *args, **kwargs: Renderer()
    tokenizer_utils.get_tokenizer = lambda *args, **kwargs: object()
    monkeypatch.setitem(sys.modules, "tinker_cookbook", ModuleType("tinker_cookbook"))
    monkeypatch.setitem(sys.modules, "tinker_cookbook.renderers", renderers)
    monkeypatch.setitem(
        sys.modules, "tinker_cookbook.tokenizer_utils", tokenizer_utils
    )

    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    one = {
        "upload": (
            json.dumps({"messages": [
                {"role": "system", "content": "s"},
                {"role": "user", "content": "u"},
                {"role": "assistant", "content": "a"},
            ]}) + "\n"
        ).encode(),
        "rows": 1,
    }
    with pytest.raises(SystemExit) as exc:
        train_tinker_sft.exact_token_validation(
            config, {"train": one, "eval": one}
        )
    assert exc.value.code == 2
    assert "cannot extract renderer tokens" in capsys.readouterr().err


# --- regression guards added after independent review -----------------------
#
# Every test below pins a defect that was present in the reviewed candidate.
# None of them contacts Tinker, reads TINKER_API_KEY, or downloads weights.

TEMPLATE = ROOT / "templates" / "role_colon.jinja"


def render_serving_template(messages, *, bos_token, add_generation_prompt=True):
    """Render exactly the way transformers compiles a chat template."""
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    return env.from_string(TEMPLATE.read_text(encoding="utf-8")).render(
        messages=messages,
        add_generation_prompt=add_generation_prompt,
        bos_token=bos_token,
    )


def test_serving_template_never_emits_a_literal_none_for_a_missing_bos():
    """Qwen tokenizers commonly have bos_token=None; Jinja renders that as 'None'."""
    messages = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "U"},
    ]
    without_bos = render_serving_template(messages, bos_token=None)
    assert "None" not in without_bos
    assert without_bos == "System: S\n\nUser: U\n\nAssistant:"
    assert render_serving_template(messages, bos_token="") == without_bos
    with_bos = render_serving_template(messages, bos_token="<|bos|>")
    assert with_bos == "<|bos|>" + without_bos


def test_serving_template_generation_prompt_has_no_trailing_whitespace():
    """A stray newline after 'Assistant:' would not match the trained prefix."""
    rendered = render_serving_template(
        [{"role": "user", "content": "U"}], bos_token=None
    )
    assert rendered.endswith("Assistant:")
    assert not rendered.endswith("Assistant:\n")


def test_serving_template_round_trips_the_assistant_turn():
    rendered = render_serving_template(
        [
            {"role": "user", "content": "U"},
            {"role": "assistant", "content": "A"},
        ],
        bos_token=None,
        add_generation_prompt=False,
    )
    assert rendered == "User: U\n\nAssistant: A\n\n"


def test_print_token_report_tolerates_the_real_report_shape(capsys):
    """The execute path used to crash here with a TypeError before confirmation."""
    report = {
        "renderer": "role_colon",
        "serving_chat_template": "templates/role_colon.jinja",
        "serving_chat_template_sha256": "ab" * 32,
        "train": {"rows": 2, "min_tokens": 10, "max_tokens": 20,
                  "mean_tokens": 15.0, "total_tokens": 30},
        "eval": {"rows": 1, "min_tokens": 11, "max_tokens": 11,
                 "mean_tokens": 11.0, "total_tokens": 11},
    }
    train_tinker_sft.print_token_report(report)
    out = capsys.readouterr().out
    assert "train 10..20 tokens, 30 total" in out
    assert "eval  11..11 tokens, 11 total" in out
    assert "role_colon" not in out


def test_render_only_still_runs_a_requested_renderer_verification(tmp_path):
    """--verify-renderer must not be silently skipped by --render-only."""
    proc = subprocess.run(
        [PY, RUNNER, "--render-only", str(tmp_path / "payload"),
         "--verify-renderer"],
        capture_output=True, text=True, cwd=ROOT,
    )
    # The dedicated Tinker environment is deliberately absent here, so the
    # verification must ABORT with the missing-package exit code rather than
    # print DRY RUN OK as though parity had been proven.
    assert proc.returncode == 3, proc.stdout + proc.stderr
    assert "is not installed" in proc.stderr
    assert "DRY RUN OK" not in proc.stdout
    assert "wrote local payload" not in proc.stdout


def test_pricing_eval_bound_covers_a_step_zero_and_final_evaluation():
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    rendered = tinker_payload.render_files(
        ROOT / "data" / "promoted" / "train.jsonl",
        ROOT / "data" / "promoted" / "eval.jsonl",
    )
    pricing = train_tinker_sft.pricing_summary(config, rendered)
    scheduled = config["training"]["max_steps"] // config["training"]["eval_every"]
    assert pricing["eval_passes"] > scheduled
    assert pricing["worst_case_eval_tokens"] == 47 * pricing["eval_passes"] * 4096


# --- durable failure evidence on the authorized execute path ----------------
#
# These drive main() with every external dependency replaced. No Tinker client
# is constructed, no real credential is read (a placeholder value is injected),
# and execute_training is a local stub.

def authorized_execute(monkeypatch, tmp_path, execute_impl, *, argv_extra=()):
    """Run main() --execute with all cloud and cost gates satisfied locally."""
    import config as config_module

    manifest = {
        "execution_authorized": True,
        "preview": False,
        "before_baseline": {"path": "x", "sha256": "y"},
        "training_authorization": {"path": "z", "sha256": "w"},
    }
    monkeypatch.setattr(
        train_tinker_sft, "check_schema_v3_freeze",
        lambda *a, **k: {"manifest": manifest},
    )
    monkeypatch.setattr(config_module, "training_guard", lambda host: {})
    monkeypatch.setattr(
        train_tinker_sft, "assert_package_versions",
        lambda cfg: {"tinker": "0.26.1", "tinker-cookbook": "0.5.5"},
    )
    monkeypatch.setattr(
        train_tinker_sft, "assert_cookbook_api",
        lambda: {"tinker_cookbook.supervised.train:Config": ["max_steps"]},
    )
    monkeypatch.setattr(
        train_tinker_sft, "exact_token_validation",
        lambda cfg, rendered: {
            "renderer": "role_colon",
            "serving_chat_template": "templates/role_colon.jinja",
            "serving_chat_template_sha256": "ab" * 32,
            "train": {"rows": 136, "min_tokens": 80, "max_tokens": 500,
                      "mean_tokens": 250.0, "total_tokens": 34000},
            "eval": {"rows": 47, "min_tokens": 80, "max_tokens": 500,
                     "mean_tokens": 250.0, "total_tokens": 12000},
        },
    )
    monkeypatch.setattr(train_tinker_sft, "require_confirmation", lambda: None)
    # A placeholder, never a real credential: the runner only checks presence.
    monkeypatch.setenv("TINKER_API_KEY", "placeholder-not-a-real-key")

    async def fake_execute(cfg, payload_paths, log_path):
        execute_impl(log_path)

    monkeypatch.setattr(train_tinker_sft, "execute_training", fake_execute)

    run_dir = tmp_path / "run"
    monkeypatch.setattr(sys, "argv", [
        "train_tinker_sft.py", "--execute",
        "--run-dir", str(run_dir),
        "--train-host", "tinker",
        "--egress-approval", "TEST-EGRESS-REF",
        "--max-cost-usd", "1000",
        *argv_extra,
    ])
    return run_dir


def test_training_exception_leaves_durable_evidence(monkeypatch, tmp_path):
    def blow_up(log_path):
        log_path.mkdir(parents=True, exist_ok=True)
        (log_path / "checkpoints.jsonl").write_text(
            json.dumps({"step": 17, "state_path": "tinker://s/17"}) + "\n",
            encoding="utf-8",
        )
        raise RuntimeError("simulated Tinker outage")

    run_dir = authorized_execute(monkeypatch, tmp_path, blow_up)
    with pytest.raises(SystemExit) as exc:
        train_tinker_sft.main()
    assert exc.value.code == 2

    record = json.loads((run_dir / "RUN.json").read_text())
    assert record["status"] == "training_failed_or_interrupted"
    assert record["error"]["type"] == "RuntimeError"
    assert "simulated Tinker outage" in record["error"]["message"]
    # The partial checkpoint the paid run did produce must survive.
    assert record["checkpoint_evidence"]["rows"][0]["state_path"] == "tinker://s/17"
    assert "checkpoint" not in record
    assert (run_dir / "RUN.request.json").is_file()


def test_missing_checkpoint_file_never_becomes_trained_unvalidated(
    monkeypatch, tmp_path
):
    run_dir = authorized_execute(monkeypatch, tmp_path, lambda log_path: None)
    with pytest.raises(SystemExit) as exc:
        train_tinker_sft.main()
    assert exc.value.code == 2
    record = json.loads((run_dir / "RUN.json").read_text())
    assert record["status"] == "training_returned_checkpoint_unverified"
    assert record["checkpoint_evidence"]["exists"] is False


@pytest.mark.parametrize("final_row", [
    {"step": 68, "state_path": "tinker://s/68", "sampler_path": "tinker://p/68"},
    {"step": 68, "final": True, "state_path": "tinker://s/68"},
    {"step": 68, "final": True, "sampler_path": "tinker://p/68"},
])
def test_incomplete_final_checkpoint_is_not_trained_unvalidated(
    monkeypatch, tmp_path, final_row
):
    def write(log_path):
        log_path.mkdir(parents=True, exist_ok=True)
        (log_path / "checkpoints.jsonl").write_text(
            json.dumps(final_row) + "\n", encoding="utf-8"
        )

    run_dir = authorized_execute(monkeypatch, tmp_path, write)
    with pytest.raises(SystemExit):
        train_tinker_sft.main()
    record = json.loads((run_dir / "RUN.json").read_text())
    assert record["status"] != "trained_unvalidated"
    assert record["checkpoint_evidence"]["rows"] == [final_row]


def test_only_a_validated_final_checkpoint_produces_trained_unvalidated(
    monkeypatch, tmp_path, capsys
):
    final = {"step": 68, "final": True,
             "state_path": "tinker://s/68", "sampler_path": "tinker://p/68"}

    def write(log_path):
        log_path.mkdir(parents=True, exist_ok=True)
        (log_path / "checkpoints.jsonl").write_text(
            json.dumps({"step": 34, "state_path": "tinker://s/34"}) + "\n"
            + json.dumps(final) + "\n",
            encoding="utf-8",
        )

    run_dir = authorized_execute(monkeypatch, tmp_path, write)
    train_tinker_sft.main()
    record = json.loads((run_dir / "RUN.json").read_text())
    assert record["status"] == "trained_unvalidated"
    assert record["checkpoint"] == final
    assert len(record["checkpoint_evidence"]["rows"]) == 2
    out = capsys.readouterr().out
    assert "NOT VALIDATED, NOT PROMOTABLE" in out
    # The token report is printed on this path; it used to raise TypeError.
    assert "train 80..500 tokens" in out


# --- static PEFT/vLLM export compatibility ---------------------------------
#
# torch and safetensors are GPU-host dependencies and are deliberately absent
# from the CPU environment, so the tensor reader is stubbed. No real weights
# are downloaded or read.

class FakeScalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class FakeTensor:
    def __init__(self, magnitude):
        self.magnitude = magnitude

    def abs(self):
        return self

    def max(self):
        return FakeScalar(self.magnitude)


SERVING_PREFIX = "base_model.model.model.language_model.layers.0.self_attn"
TRAINING_PREFIX = "base_model.model.model.layers.0.self_attn"


def write_fake_adapter(directory: Path, *, targets, keys, base_model,
                       magnitude=0.5):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "adapter_config.json").write_text(json.dumps({
        "base_model_name_or_path": base_model,
        "target_modules": targets,
    }), encoding="utf-8")
    (directory / "adapter_model.safetensors").write_text("stub", encoding="utf-8")
    return {key: FakeTensor(magnitude) for key in keys}


def install_fake_safetensors(monkeypatch, tensors):
    safetensors = ModuleType("safetensors")
    torch_module = ModuleType("safetensors.torch")
    torch_module.load_file = lambda path: tensors
    safetensors.torch = torch_module
    monkeypatch.setitem(sys.modules, "safetensors", safetensors)
    monkeypatch.setitem(sys.modules, "safetensors.torch", torch_module)


def compatibility_of(monkeypatch, tmp_path, **kwargs):
    import export_tinker_weights

    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    kwargs.setdefault("base_model", config["base_model"])
    peft = tmp_path / "peft"
    tensors = write_fake_adapter(peft, **kwargs)
    install_fake_safetensors(monkeypatch, tensors)
    return export_tinker_weights.inspect_peft_compatibility(peft, config)


def test_export_accepts_a_servable_qwen35_adapter(monkeypatch, tmp_path):
    result = compatibility_of(
        monkeypatch, tmp_path,
        targets=["q_proj", "gate_proj"],
        keys=[f"{SERVING_PREFIX}.q_proj.lora_A.weight",
              f"{SERVING_PREFIX}.q_proj.lora_B.weight"],
    )
    assert result["compatible"] is True
    assert result["errors"] == []
    assert result["nonzero_lora_b_tensors"] == 1
    assert result["language_model_prefix"] is True


def test_export_rejects_targets_the_qualified_vllm_path_cannot_serve(
    monkeypatch, tmp_path
):
    """train/qlora_9b.yaml excludes in_proj_a/in_proj_b for this exact reason."""
    result = compatibility_of(
        monkeypatch, tmp_path,
        targets=["q_proj", "in_proj_a", "in_proj_b"],
        keys=[f"{SERVING_PREFIX}.q_proj.lora_A.weight",
              f"{SERVING_PREFIX}.q_proj.lora_B.weight"],
    )
    assert result["compatible"] is False
    assert result["forbidden_target_modules"] == ["in_proj_a", "in_proj_b"]
    assert any("unsupported by the qualified vLLM path" in e
               for e in result["errors"])


def test_export_rejects_the_training_layout_without_language_model_prefix(
    monkeypatch, tmp_path
):
    """The measured Qwen3.5 mismatch: vLLM binds nothing and serves the base."""
    result = compatibility_of(
        monkeypatch, tmp_path,
        targets=["q_proj"],
        keys=[f"{TRAINING_PREFIX}.q_proj.lora_A.weight",
              f"{TRAINING_PREFIX}.q_proj.lora_B.weight"],
    )
    assert result["compatible"] is False
    assert result["language_model_prefix"] is False
    assert any("language_model prefix" in e for e in result["errors"])


def test_export_rejects_an_all_zero_untrained_adapter(monkeypatch, tmp_path):
    result = compatibility_of(
        monkeypatch, tmp_path,
        targets=["q_proj"],
        keys=[f"{SERVING_PREFIX}.q_proj.lora_A.weight",
              f"{SERVING_PREFIX}.q_proj.lora_B.weight"],
        magnitude=0.0,
    )
    assert result["compatible"] is False
    assert any("every lora_B tensor is zero" in e for e in result["errors"])


def test_export_rejects_an_adapter_naming_another_base_model(
    monkeypatch, tmp_path
):
    result = compatibility_of(
        monkeypatch, tmp_path,
        targets=["q_proj"],
        keys=[f"{SERVING_PREFIX}.q_proj.lora_A.weight",
              f"{SERVING_PREFIX}.q_proj.lora_B.weight"],
        base_model="Qwen/Qwen3.5-9B",
    )
    assert result["compatible"] is False
    assert any("different base model" in e for e in result["errors"])


FINAL_CHECKPOINT = {
    "step": 68,
    "final": True,
    "sampler_path": "tinker://test-run/sampler_weights/final",
    "state_path": "tinker://test-run/weights/final",
}

# What check_schema_v3_freeze returns for an authorized run. The exporter calls
# the real helper; these tests stub only that one call, so everything the
# exporter checks ITSELF (pins, payload digests, provenance, evidence) is
# exercised for real.
AUTHORIZED_MANIFEST = {
    "execution_authorized": True,
    "preview": False,
    "training_authorization": {"path": "train/TRAINING_JUSTIFIED.md",
                               "sha256": "aa" * 32},
    "before_baseline": {"path": "before.json", "sha256": "bb" * 32},
    "backend": {"payload": {
        "train": {"upload_sha256": "11" * 32},
        "eval": {"upload_sha256": "22" * 32},
    }},
}


def genuine_run_record(path: Path, **overrides) -> Path:
    """Exactly the shape train_tinker_sft.main() writes on success."""
    import hashlib

    def pin(target: Path) -> dict:
        return {"path": str(target),
                "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}

    record = {
        "status": "trained_unvalidated",
        "checkpoint_label": "trained_unvalidated",
        "config": pin(CONFIG),
        "run_manifest": pin(PREVIEW),
        "promotion_manifest": pin(PROMOTION),
        "payload": {
            "train": {"upload_sha256": "11" * 32},
            "eval": {"upload_sha256": "22" * 32},
        },
        "packages": {"tinker": "0.26.1", "tinker-cookbook": "0.5.5"},
        "cookbook_api_surface": {"tinker_cookbook.supervised.train:Config": []},
        "tokens": {"train": {"total_tokens": 31196}},
        "pricing": {"worst_case_estimate_usd": 3.04},
        "checkpoint_evidence": {
            "exists": True,
            "rows": [{"step": 34, "state_path": "tinker://s/34"}, FINAL_CHECKPOINT],
            "raw_lines": ["{}", "{}"],
        },
        "checkpoint": dict(FINAL_CHECKPOINT),
    }
    record.update(overrides)
    path.write_text(json.dumps(record), encoding="utf-8")
    return path


def stub_authorized_freeze(monkeypatch, manifest=None):
    import export_tinker_weights

    monkeypatch.setattr(
        export_tinker_weights.train_tinker_sft, "check_schema_v3_freeze",
        lambda *a, **k: {"manifest": manifest or AUTHORIZED_MANIFEST},
    )


def run_export(monkeypatch, tmp_path, *, build, tensors=None, download=None):
    """Drive export main() with a stubbed cookbook. No download ever happens."""
    import export_tinker_weights

    monkeypatch.setattr(
        export_tinker_weights.train_tinker_sft, "assert_package_versions",
        lambda cfg: {"tinker": "0.26.1", "tinker-cookbook": "0.5.5"},
    )
    monkeypatch.setattr(
        export_tinker_weights.train_tinker_sft, "assert_cookbook_api",
        lambda: {"tinker_cookbook.weights:build_lora_adapter": ["base_model"]},
    )
    stub_authorized_freeze(monkeypatch)
    monkeypatch.setattr(export_tinker_weights, "require_confirmation", lambda: None)
    monkeypatch.setenv("TINKER_API_KEY", "placeholder-not-a-real-key")

    output = tmp_path / "export"
    cookbook = ModuleType("tinker_cookbook")
    weights = ModuleType("tinker_cookbook.weights")

    def default_download(*, tinker_path, output_dir):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        (Path(output_dir) / "weights.bin").write_text("stub", encoding="utf-8")
        return output_dir

    weights.download = download or default_download
    weights.build_lora_adapter = build
    cookbook.weights = weights
    monkeypatch.setitem(sys.modules, "tinker_cookbook", cookbook)
    monkeypatch.setitem(sys.modules, "tinker_cookbook.weights", weights)
    if tensors is not None:
        install_fake_safetensors(monkeypatch, tensors)

    monkeypatch.setattr(sys, "argv", [
        "export_tinker_weights.py",
        "--run-record", str(genuine_run_record(tmp_path / "RUN.json")),
        "--output-dir", str(output),
        "--execute",
    ])
    return export_tinker_weights, output


def test_incompatible_export_is_recorded_and_never_called_unvalidated(
    monkeypatch, tmp_path
):
    holder = {}

    def build(*, base_model, adapter_path, output_path):
        holder["tensors"] = write_fake_adapter(
            Path(output_path),
            targets=["in_proj_a"],
            keys=[f"{TRAINING_PREFIX}.in_proj_a.lora_A.weight",
                  f"{TRAINING_PREFIX}.in_proj_a.lora_B.weight"],
            base_model=base_model,
        )
        install_fake_safetensors(monkeypatch, holder["tensors"])

    module, output = run_export(monkeypatch, tmp_path, build=build)
    with pytest.raises(SystemExit) as exc:
        module.main()
    assert exc.value.code == 2

    record = json.loads((output / "EXPORT.json").read_text())
    assert record["status"] == "peft_incompatible"
    assert record["status"] != "peft_unvalidated"
    assert record["static_vllm_compatibility"]["compatible"] is False
    assert record["_not_promotable"]


def test_compatible_export_is_peft_unvalidated_and_still_not_promotable(
    monkeypatch, tmp_path
):
    def build(*, base_model, adapter_path, output_path):
        tensors = write_fake_adapter(
            Path(output_path),
            targets=["q_proj"],
            keys=[f"{SERVING_PREFIX}.q_proj.lora_A.weight",
                  f"{SERVING_PREFIX}.q_proj.lora_B.weight"],
            base_model=base_model,
        )
        install_fake_safetensors(monkeypatch, tensors)

    module, output = run_export(monkeypatch, tmp_path, build=build)
    module.main()
    record = json.loads((output / "EXPORT.json").read_text())
    assert record["status"] == "peft_unvalidated"
    assert "after gates" in record["_not_promotable"]


def test_failed_export_leaves_durable_evidence(monkeypatch, tmp_path):
    def build(*, base_model, adapter_path, output_path):
        raise RuntimeError("simulated conversion failure")

    module, output = run_export(monkeypatch, tmp_path, build=build)
    with pytest.raises(SystemExit) as exc:
        module.main()
    assert exc.value.code == 2
    record = json.loads((output / "EXPORT.json").read_text())
    assert record["status"] == "export_failed"
    assert record["error"]["type"] == "RuntimeError"
    assert "simulated conversion failure" in record["error"]["message"]
    # The bytes the paid download did fetch are still described.
    assert record["raw_adapter"]["sha256"]


# --- manifest safety: an authorized manifest must earn its authorization ----

def authorized_fixture(tmp_path, *, baseline_body, manifest_overrides=None,
                       authorization_path=None):
    """Build a tmp config plus the manifest an operator would have to forge."""
    import hashlib

    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    authorization = authorization_path or (tmp_path / "TRAINING_JUSTIFIED.md")
    if authorization_path is None:
        authorization.write_text("reviewed for test only", encoding="utf-8")
    config["authorization"]["required_document"] = str(
        tmp_path / "TRAINING_JUSTIFIED.md"
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))

    baseline = tmp_path / "before.json"
    baseline.write_text(json.dumps(baseline_body(config_path)), encoding="utf-8")

    def pin(path):
        return {"path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    manifest = {
        "execution_authorized": True,
        "preview": False,
        "before_baseline": pin(baseline),
        "training_authorization": pin(authorization),
    }
    manifest.update(manifest_overrides or {})
    return config, config_path, manifest


def install_freeze_stubs(monkeypatch, manifest):
    backend = {"name": "tinker"}
    manifest.setdefault("backend", backend)
    monkeypatch.setattr(
        train_tinker_sft.train_qlora, "check_run_freeze",
        lambda *a, **k: {"manifest": manifest},
    )
    monkeypatch.setattr(
        train_tinker_sft.freeze_training_run, "tinker_backend_snapshot",
        lambda path: manifest["backend"],
    )


def gate_result(spec, base_model, **overrides):
    """One gate result exactly as run_gates.cmd_run would have written it."""
    result = {
        "name": spec["name"],
        "rate": 0.42,
        "threshold": spec["min_pass_rate"],
        "items": spec.get("items"),
        "passed": False,
        "model_dependent": bool(spec.get("model_dependent")),
    }
    if result["model_dependent"]:
        result["generated_by_model_id"] = base_model
        result["from_fixtures"] = False
        result["stop_sequences"] = list(spec.get("stop_sequences") or [])
    result.update(overrides)
    return result


def valid_baseline_body(config_path, gates=None):
    """Exactly the report shape baseline_snapshot() accepts.

    Built from the DECLARED gates so the fixture cannot drift into being an
    incomplete evaluation that the checks then have to bless.
    """
    import hashlib
    import freeze_training_run

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    evaluation = config["evaluation"]
    base = config["base_model"]
    declared = freeze_training_run.declared_gate_specs(config_path)
    return {
        "stage": "before",
        "measured": True,
        "config": {
            "path": str(config_path),
            "sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        },
        "shared_evaluation_config": {
            "path": str(ROOT / evaluation["config"]),
            "sha256": evaluation["config_sha256"],
        },
        "model": {
            "teacher": evaluation["required_model_config"],
            "host": evaluation["required_host_config"],
            "expected": base,
            "identity": {"expected": base, "served": [base], "endpoint": "http://x/v1"},
        },
        "gates": (
            [gate_result(spec, base) for spec in declared]
            if gates is None else gates(declared, base)
        ),
    }


def refuse(monkeypatch, tmp_path, capsys, **kwargs):
    """Return every refusal message, from both die() and sys.exit(str)."""
    config, config_path, manifest = authorized_fixture(tmp_path, **kwargs)
    install_freeze_stubs(monkeypatch, manifest)
    with pytest.raises(SystemExit) as exc:
        train_tinker_sft.check_schema_v3_freeze(
            tmp_path / "manifest.json", config_path, PROMOTION, config
        )
    code = exc.value.code
    return capsys.readouterr().err + ("" if isinstance(code, int) else str(code))


def test_forged_baseline_with_a_consistent_digest_is_still_refused(
    monkeypatch, tmp_path, capsys
):
    """The original attack: hand-edit the manifest and point at any file.

    The digest pin is internally consistent, so only re-running the validation
    on the execution path can catch it.
    """
    message = refuse(
        monkeypatch, tmp_path, capsys,
        baseline_body=lambda _: {"stage": "before", "note": "not a gate report"},
    )
    assert "before-baseline report is not valid for Tinker execution" in message
    assert "measured is not true" in message
    assert "report contains no gate results" in message


def test_a_valid_looking_baseline_for_the_wrong_model_is_refused(
    monkeypatch, tmp_path, capsys
):
    def wrong_model(config_path):
        body = valid_baseline_body(config_path)
        body["model"]["expected"] = "Qwen/Qwen3.5-9B"
        body["model"]["identity"]["expected"] = "Qwen/Qwen3.5-9B"
        body["model"]["identity"]["served"] = ["Qwen/Qwen3.5-9B"]
        return body

    message = refuse(monkeypatch, tmp_path, capsys, baseline_body=wrong_model)
    assert "not 'Qwen/Qwen3.5-9B-Base'" in message


def test_a_fixture_generated_baseline_cannot_authorize_execution(
    monkeypatch, tmp_path, capsys
):
    def fixtures(config_path):
        body = valid_baseline_body(config_path)
        for gate in body["gates"]:
            if gate["model_dependent"]:
                gate["from_fixtures"] = True
        return body

    message = refuse(monkeypatch, tmp_path, capsys, baseline_body=fixtures)
    assert "a fixture run never touched the model" in message
    assert "indonesian_voice" in message and "edit_contract_output" in message


def test_a_tampered_pinned_snapshot_is_refused(monkeypatch, tmp_path, capsys):
    """A real report, but the manifest's recorded snapshot was edited."""
    config, config_path, manifest = authorized_fixture(
        tmp_path, baseline_body=valid_baseline_body
    )
    live = train_tinker_sft.freeze_training_run.baseline_snapshot(
        Path(manifest["before_baseline"]["path"]), config_path,
        config["base_model"],
    )
    manifest["before_baseline"] = dict(
        live, gates=[dict(live["gates"][0], rate=0.0)]
    )
    install_freeze_stubs(monkeypatch, manifest)
    with pytest.raises(SystemExit):
        train_tinker_sft.check_schema_v3_freeze(
            tmp_path / "manifest.json", config_path, PROMOTION, config
        )
    assert "no longer match the validated snapshot" in capsys.readouterr().err


def test_a_genuinely_valid_baseline_and_authorization_is_accepted(
    monkeypatch, tmp_path
):
    """The control: the refusals above are specific, not a blanket block."""
    config, config_path, manifest = authorized_fixture(
        tmp_path, baseline_body=valid_baseline_body
    )
    manifest["before_baseline"] = \
        train_tinker_sft.freeze_training_run.baseline_snapshot(
            Path(manifest["before_baseline"]["path"]), config_path,
            config["base_model"],
        )
    install_freeze_stubs(monkeypatch, manifest)
    result = train_tinker_sft.check_schema_v3_freeze(
        tmp_path / "manifest.json", config_path, PROMOTION, config
    )
    assert result["manifest"]["execution_authorized"] is True


def test_preview_true_always_blocks_even_when_authorized(
    monkeypatch, tmp_path, capsys
):
    message = refuse(
        monkeypatch, tmp_path, capsys,
        baseline_body=valid_baseline_body,
        manifest_overrides={"preview": True},
    )
    assert "must explicitly set preview=false" in message


def test_a_missing_preview_field_blocks_an_authorized_manifest(
    monkeypatch, tmp_path, capsys
):
    """preview must be explicitly false; absence is not consent."""
    config, config_path, manifest = authorized_fixture(
        tmp_path, baseline_body=valid_baseline_body
    )
    del manifest["preview"]
    install_freeze_stubs(monkeypatch, manifest)
    with pytest.raises(SystemExit):
        train_tinker_sft.check_schema_v3_freeze(
            tmp_path / "manifest.json", config_path, PROMOTION, config
        )
    assert "must explicitly set preview=false" in capsys.readouterr().err


def test_authorization_must_be_the_exact_reviewed_document(
    monkeypatch, tmp_path, capsys
):
    elsewhere = tmp_path / "SOMEWHERE_ELSE.md"
    elsewhere.write_text("self-signed", encoding="utf-8")
    (tmp_path / "TRAINING_JUSTIFIED.md").write_text("real one", encoding="utf-8")
    config, config_path, manifest = authorized_fixture(
        tmp_path, baseline_body=valid_baseline_body,
        authorization_path=elsewhere,
    )
    # A genuinely validated baseline, so the refusal below is about the
    # authorization document and nothing else.
    manifest["before_baseline"] = \
        train_tinker_sft.freeze_training_run.baseline_snapshot(
            Path(manifest["before_baseline"]["path"]), config_path,
            config["base_model"],
        )
    install_freeze_stubs(monkeypatch, manifest)
    with pytest.raises(SystemExit):
        train_tinker_sft.check_schema_v3_freeze(
            tmp_path / "manifest.json", config_path, PROMOTION, config
        )
    assert "authorization must be the reviewed document" in capsys.readouterr().err


# --- RoleColon stop sequences must reach the live generator -----------------

def test_stop_args_rejects_a_malformed_stop_sequence_list():
    assert run_gates.stop_args({}) == []
    assert run_gates.stop_args({"stop_sequences": ["\n\nUser:"]}) == [
        "--stop", "\n\nUser:",
    ]
    with pytest.raises(SystemExit):
        run_gates.stop_args({"stop_sequences": ["", "x"]})
    with pytest.raises(SystemExit):
        run_gates.stop_args({"stop_sequences": "\n\nUser:"})


def test_voice_gate_passes_the_configured_stop_sequence_to_the_generator(
    monkeypatch, tmp_path
):
    captured = {}

    def fake_run(cmd, *a, **k):
        captured["cmd"] = cmd
        return SimpleNamespace(returncode=1, stdout="", stderr="stopped here")

    monkeypatch.setattr(run_gates.subprocess, "run", fake_run)
    spec = {
        "name": "indonesian_voice",
        "source": "prompts/voice_eval.v1.jsonl",
        "scorer": "src/score_voice.py",
        "stop_sequences": ["\n\nUser:"],
    }
    args = SimpleNamespace(
        teacher="office-student-9b-base", host="student-serve",
        stage="before", expect_model="Qwen/Qwen3.5-9B-Base",
        adapter_model_id="unused", traces=None, edit_traces=None,
    )
    with pytest.raises(SystemExit):
        run_gates.gate_indonesian_voice(spec, "before", args, tmp_path)
    cmd = captured["cmd"]
    assert "--stop" in cmd
    assert cmd[cmd.index("--stop") + 1] == "\n\nUser:"
    assert cmd[cmd.index("--model-id") + 1] == "Qwen/Qwen3.5-9B-Base"


def test_both_model_dependent_gates_build_their_stop_arguments():
    """The edit gate needs the add-in checkout, so assert its call site."""
    import inspect
    for gate in (run_gates.gate_indonesian_voice,
                 run_gates.gate_edit_contract_output):
        assert "*stop_args(spec)" in inspect.getsource(gate)


def test_the_local_qlora_gates_are_left_without_stop_overrides():
    """Ordinary Qwen3.5-9B runs must keep the generator's default stops."""
    qlora, shared = run_gates.load_gate_config(ROOT / "train" / "qlora_9b.yaml")
    assert shared is None
    for gate in qlora["eval_gates"]:
        assert "stop_sequences" not in gate
        assert run_gates.stop_args(gate) == []


# --- pinned cookbook API surface -------------------------------------------
#
# tinker/tinker-cookbook are not installed in the CPU environment, so these
# use stub modules. They assert the drift check itself, not the real packages.

def install_cookbook_stubs(monkeypatch, *, config_params=None, drop=()):
    params = config_params or [
        "log_path", "model_name", "recipe_name", "renderer_name",
        "dataset_builder", "learning_rate", "lr_schedule", "num_epochs",
        "lora_rank", "save_every", "eval_every", "max_steps",
    ]

    def make(name, names):
        source = f"def {name}({', '.join(f'{n}=None' for n in names)}): pass"
        namespace = {}
        exec(source, namespace)                       # noqa: S102 - test stub
        return namespace[name]

    modules = {
        "tinker_cookbook": ModuleType("tinker_cookbook"),
        "tinker_cookbook.supervised": ModuleType("tinker_cookbook.supervised"),
        "tinker_cookbook.supervised.train":
            ModuleType("tinker_cookbook.supervised.train"),
        "tinker_cookbook.supervised.data":
            ModuleType("tinker_cookbook.supervised.data"),
        "tinker_cookbook.supervised.types":
            ModuleType("tinker_cookbook.supervised.types"),
        "tinker_cookbook.weights": ModuleType("tinker_cookbook.weights"),
    }
    modules["tinker_cookbook.supervised.train"].Config = make("Config", params)
    modules["tinker_cookbook.supervised.data"].FromConversationFileBuilder = make(
        "FromConversationFileBuilder",
        ["common_config", "file_path", "test_size", "shuffle_seed"],
    )
    modules["tinker_cookbook.supervised.types"].ChatDatasetBuilderCommonConfig = \
        make("ChatDatasetBuilderCommonConfig",
             ["model_name_for_tokenizer", "renderer_name", "max_length",
              "batch_size", "train_on_what"])
    modules["tinker_cookbook.weights"].download = make(
        "download", ["tinker_path", "output_dir"])
    modules["tinker_cookbook.weights"].build_lora_adapter = make(
        "build_lora_adapter", ["base_model", "adapter_path", "output_path"])
    for name in drop:
        module_name, _, attribute = name.partition(":")
        delattr(modules[module_name], attribute)
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)


def test_cookbook_api_surface_is_accepted_when_it_matches(monkeypatch):
    install_cookbook_stubs(monkeypatch)
    surface = train_tinker_sft.assert_cookbook_api()
    assert "max_steps" in surface["tinker_cookbook.supervised.train:Config"]
    assert surface["tinker_cookbook.weights:build_lora_adapter"] == [
        "adapter_path", "base_model", "output_path",
    ]


def test_a_renamed_config_field_is_a_local_abort_not_a_paid_failure(
    monkeypatch, capsys
):
    """The warmup_ratio lesson: drift must surface before the API client."""
    install_cookbook_stubs(monkeypatch, config_params=[
        "log_path", "model_name", "recipe_name", "renderer_name",
        "dataset_builder", "learning_rate", "lr_schedule", "num_epochs",
        "lora_rank", "save_every", "eval_every", "max_train_steps",
    ])
    with pytest.raises(SystemExit) as exc:
        train_tinker_sft.assert_cookbook_api()
    assert exc.value.code == 3
    assert "does not accept max_steps" in capsys.readouterr().err


def test_a_removed_weights_helper_is_a_local_abort(monkeypatch, capsys):
    install_cookbook_stubs(
        monkeypatch, drop=("tinker_cookbook.weights:build_lora_adapter",)
    )
    with pytest.raises(SystemExit) as exc:
        train_tinker_sft.assert_cookbook_api()
    assert exc.value.code == 3
    assert "has no build_lora_adapter" in capsys.readouterr().err


def test_serve_script_is_valid_bash_and_orders_the_template_before_lora():
    script = ROOT / "scripts" / "serve_student.sh"
    assert subprocess.run(["bash", "-n", str(script)]).returncode == 0
    text = script.read_text(encoding="utf-8")
    # A missing template must refuse rather than start vLLM with the
    # tokenizer's own (possibly absent) chat template.
    assert "REFUSING: configured chat template is missing" in text
    exec_line = text.split("exec \"$VLLM_BIN\" serve")[1]
    assert exec_line.index("CHAT_TEMPLATE_ARGS") < exec_line.index("LORA_ARGS")


def test_empty_chat_template_expands_to_no_arguments():
    """The `${arr[@]+...}` idiom must add nothing under `set -u`."""
    probe = (
        'set -euo pipefail\n'
        'CHAT_TEMPLATE_ARGS=()\n'
        'set -- --dtype bfloat16 '
        '"${CHAT_TEMPLATE_ARGS[@]+"${CHAT_TEMPLATE_ARGS[@]}"}" --port 8020\n'
        'echo "$#"\n'
    )
    empty = subprocess.run(["bash", "-c", probe], capture_output=True, text=True)
    assert empty.stdout.strip() == "4", empty.stderr

    probe_filled = probe.replace(
        "CHAT_TEMPLATE_ARGS=()",
        'CHAT_TEMPLATE_ARGS=(--chat-template "/a b/role_colon.jinja")',
    )
    filled = subprocess.run(
        ["bash", "-c", probe_filled], capture_output=True, text=True
    )
    assert filled.stdout.strip() == "6", filled.stderr


# --- apply_chat_template return shapes -------------------------------------

def test_normalize_token_ids_handles_a_batchencoding_like_mapping():
    """transformers 5.x returns a UserDict subclass, not a dict.

    `isinstance(value, dict)` misses it and iteration yields the KEYS, which
    made the parity check abort on every row with
    "invalid literal for int() with base 10: 'input_ids'" — a check that could
    never pass rather than a check that passed.
    """
    from collections import UserDict

    class BatchEncoding(UserDict):
        @property
        def input_ids(self):
            return self.data["input_ids"]

    assert not isinstance(BatchEncoding({"input_ids": [1, 2]}), dict)
    assert train_tinker_sft.normalize_token_ids(
        BatchEncoding({"input_ids": [1, 2, 3]})
    ) == [1, 2, 3]


@pytest.mark.parametrize("value,expected", [
    ([1, 2, 3], [1, 2, 3]),
    ([[1, 2, 3]], [1, 2, 3]),
    ({"input_ids": [4, 5]}, [4, 5]),
    ({"input_ids": [[4, 5]]}, [4, 5]),
])
def test_normalize_token_ids_accepts_every_documented_shape(value, expected):
    assert train_tinker_sft.normalize_token_ids(value) == expected


@pytest.mark.parametrize("value", [
    {"attention_mask": [1, 1]},
    [[1, 2], [3, 4]],
])
def test_normalize_token_ids_refuses_shapes_it_cannot_interpret(value):
    with pytest.raises(ValueError):
        train_tinker_sft.normalize_token_ids(value)


def test_serving_template_matches_the_documented_role_colon_format():
    """Ground truth: tinker_cookbook/renderers/role_colon.py.

    header = role.capitalize() + ":", output = " " + content + "\\n\\n",
    and BOS is omitted entirely when tokenizer.bos_token is None.
    """
    rendered = render_serving_template(
        [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}],
        bos_token=None,
    )
    assert rendered == "System: S\n\nUser: U\n\nAssistant:"


def test_configured_stop_sequence_matches_the_renderer_contract():
    """RoleColonRenderer.get_stop_sequences() returns exactly ['\\n\\nUser:']."""
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert config["evaluation"]["stop_sequences"] == ["\n\nUser:"]
    teacher = yaml.safe_load(
        (ROOT / "configs" / "teachers" / "office-student-9b-base.yaml")
        .read_text(encoding="utf-8")
    )
    assert teacher["sampling"]["stop"] == ["\n\nUser:"]


@pytest.mark.skipif(
    __import__("importlib.util", fromlist=["util"]).find_spec("tinker_cookbook")
    is None,
    reason="pinned Tinker client environment not installed (.venv-tinker)",
)
def test_real_renderer_parity_against_the_pinned_cookbook():
    """Runs only in .venv-tinker. Reads no credential and makes no API call.

    Executed in a SUBPROCESS with the Hub offline variables set, for two
    reasons. huggingface_hub latches them at import time, so setting them
    inside an already-imported process is unreliable; and this makes the test
    exercise byte-for-byte the same offline path the runbook documents, rather
    than a nearby one that happens to work because a cache was warm.

    A cold cache is a SKIP, not a failure: the tokenizer download is a
    first-run step, and a restricted-network machine has nothing to prove here.
    """
    import os

    script = (
        "import json, sys\n"
        "sys.path.insert(0, 'src')\n"
        "import yaml, train_tinker_sft\n"
        "from tinker_cookbook.renderers import get_renderer\n"
        "from tinker_cookbook.tokenizer_utils import get_tokenizer\n"
        "config = yaml.safe_load(open('train/tinker_sft_9b.yaml'))\n"
        "tok = get_tokenizer(config['base_model'])\n"
        "r = get_renderer(config['renderer'], tok)\n"
        "messages = [{'role': 'system', 'content': 'S'},\n"
        "            {'role': 'user', 'content': 'U'}]\n"
        "tinker_ids = train_tinker_sft.normalize_token_ids(\n"
        "    r.build_generation_prompt(messages).to_ints())\n"
        "served_ids = train_tinker_sft.normalize_token_ids(\n"
        "    tok.apply_chat_template(messages, tokenize=True,\n"
        "        add_generation_prompt=True,\n"
        "        chat_template=open('templates/role_colon.jinja').read()))\n"
        "print(json.dumps({'stops': r.get_stop_sequences(),\n"
        "                  'tinker': tinker_ids, 'served': served_ids}))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, cwd=ROOT,
        env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
             "TINKER_API_KEY": ""},
    )
    if proc.returncode != 0:
        combined = proc.stdout + proc.stderr
        if any(marker in combined for marker in (
                "offline", "OfflineMode", "connect", "Connection",
                "not found in the", "LocalEntryNotFound")):
            pytest.skip(
                "Qwen/Qwen3.5-9B-Base tokenizer is not in the local Hugging "
                "Face cache; run --verify-renderer once with network first"
            )
        pytest.fail(combined)

    result = json.loads(proc.stdout.strip().splitlines()[-1])
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert result["stops"] == config["evaluation"]["stop_sequences"]
    assert result["tinker"] == result["served"]


# --- baseline gate completeness --------------------------------------------
#
# A non-empty gate list is not an evaluation. Each case below is a report that
# passes every per-gate check while leaving part of the contract unmeasured.

def test_a_one_gate_baseline_cannot_authorize_training(
    monkeypatch, tmp_path, capsys
):
    """Reproduces the QA finding: only indonesian_voice was accepted."""
    def only_voice(declared, base):
        return [gate_result(spec, base) for spec in declared
                if spec["name"] == "indonesian_voice"]

    message = refuse(
        monkeypatch, tmp_path, capsys,
        baseline_body=lambda path: valid_baseline_body(path, gates=only_voice),
    )
    assert "missing declared gate(s)" in message
    assert "edit_contract_output" in message
    assert "office_json_contract" in message


def test_a_duplicated_gate_result_is_refused(monkeypatch, tmp_path, capsys):
    def duplicated(declared, base):
        gates = [gate_result(spec, base) for spec in declared]
        return gates + [gates[0]]

    message = refuse(
        monkeypatch, tmp_path, capsys,
        baseline_body=lambda path: valid_baseline_body(path, gates=duplicated),
    )
    assert "duplicate gate result(s)" in message


def test_an_undeclared_gate_is_refused(monkeypatch, tmp_path, capsys):
    def extra(declared, base):
        gates = [gate_result(spec, base) for spec in declared]
        gates.append(gate_result(
            {"name": "invented_gate", "min_pass_rate": 0.5,
             "model_dependent": False},
            base,
        ))
        return gates

    message = refuse(
        monkeypatch, tmp_path, capsys,
        baseline_body=lambda path: valid_baseline_body(path, gates=extra),
    )
    assert "undeclared gate(s): invented_gate" in message


def test_a_lowered_threshold_in_the_baseline_is_refused(
    monkeypatch, tmp_path, capsys
):
    """The bar cannot be moved by the report that will be compared against."""
    def lowered(declared, base):
        gates = [gate_result(spec, base) for spec in declared]
        gates[-1]["threshold"] = 0.10
        return gates

    message = refuse(
        monkeypatch, tmp_path, capsys,
        baseline_body=lambda path: valid_baseline_body(path, gates=lowered),
    )
    assert "is not the configured" in message


def test_a_wrong_item_count_in_the_baseline_is_refused(
    monkeypatch, tmp_path, capsys
):
    def short(declared, base):
        gates = [gate_result(spec, base) for spec in declared]
        for gate in gates:
            if gate["name"] == "indonesian_voice":
                gate["items"] = 5
        return gates

    message = refuse(
        monkeypatch, tmp_path, capsys,
        baseline_body=lambda path: valid_baseline_body(path, gates=short),
    )
    assert "measured 5 items, config declares 40" in message


def test_a_baseline_without_the_configured_stop_sequences_is_refused(
    monkeypatch, tmp_path, capsys
):
    def no_stops(declared, base):
        gates = [gate_result(spec, base) for spec in declared]
        for gate in gates:
            if gate["model_dependent"]:
                gate["stop_sequences"] = []
        return gates

    message = refuse(
        monkeypatch, tmp_path, capsys,
        baseline_body=lambda path: valid_baseline_body(path, gates=no_stops),
    )
    assert "stop sequences" in message
    assert "not the configured" in message


def test_a_flipped_model_dependent_flag_is_refused(monkeypatch, tmp_path, capsys):
    def flipped(declared, base):
        gates = [gate_result(spec, base) for spec in declared]
        for gate in gates:
            if gate["name"] == "indonesian_voice":
                gate["model_dependent"] = False
        return gates

    message = refuse(
        monkeypatch, tmp_path, capsys,
        baseline_body=lambda path: valid_baseline_body(path, gates=flipped),
    )
    assert "model_dependent is False" in message


def test_the_accepted_snapshot_retains_full_gate_metadata(tmp_path):
    """A snapshot that strips items and stops cannot prove what was measured."""
    import freeze_training_run

    config, config_path, manifest = authorized_fixture(
        tmp_path, baseline_body=valid_baseline_body
    )
    snapshot = freeze_training_run.baseline_snapshot(
        Path(manifest["before_baseline"]["path"]), config_path,
        config["base_model"],
    )
    assert [g["name"] for g in snapshot["gates"]] == [
        "edit_contract_output", "indonesian_voice", "office_json_contract",
    ]
    voice = next(g for g in snapshot["gates"] if g["name"] == "indonesian_voice")
    assert voice["items"] == 40
    assert voice["threshold"] == 0.95
    assert voice["stop_sequences"] == ["\n\nUser:"]
    assert voice["from_fixtures"] is False
    assert voice["generated_by_model_id"] == "Qwen/Qwen3.5-9B-Base"


# --- offline verification, runtime pinning, node-suite serialisation --------

def test_offline_flag_sets_the_hub_variables_before_any_loader_runs(
    monkeypatch, tmp_path
):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    seen = {}

    def record(config, rendered):
        import os
        seen["HF_HUB_OFFLINE"] = os.environ.get("HF_HUB_OFFLINE")
        seen["TRANSFORMERS_OFFLINE"] = os.environ.get("TRANSFORMERS_OFFLINE")
        return {"train": {"rows": 1, "min_tokens": 1, "max_tokens": 1,
                          "mean_tokens": 1.0, "total_tokens": 1},
                "eval": {"rows": 1, "min_tokens": 1, "max_tokens": 1,
                         "mean_tokens": 1.0, "total_tokens": 1},
                "serving_chat_template_sha256": "cc" * 32}

    monkeypatch.setattr(train_tinker_sft, "assert_package_versions", lambda c: {})
    monkeypatch.setattr(train_tinker_sft, "assert_cookbook_api", lambda: {})
    monkeypatch.setattr(train_tinker_sft, "exact_token_validation", record)
    monkeypatch.setattr(sys, "argv", [
        "train_tinker_sft.py", "--verify-renderer", "--offline",
    ])
    train_tinker_sft.main()
    assert seen == {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}


def test_a_tokenizer_load_failure_is_a_controlled_abort(monkeypatch, capsys):
    """A cold cache with no network must not produce a raw traceback."""
    renderers = ModuleType("tinker_cookbook.renderers")
    tokenizer_utils = ModuleType("tinker_cookbook.tokenizer_utils")
    renderers.TrainOnWhat = SimpleNamespace(ALL_ASSISTANT_MESSAGES="assistant")
    renderers.get_renderer = lambda *a, **k: object()

    def offline_boom(*args, **kwargs):
        raise ConnectionError("simulated restricted networking")

    tokenizer_utils.get_tokenizer = offline_boom
    monkeypatch.setitem(sys.modules, "tinker_cookbook", ModuleType("tinker_cookbook"))
    monkeypatch.setitem(sys.modules, "tinker_cookbook.renderers", renderers)
    monkeypatch.setitem(
        sys.modules, "tinker_cookbook.tokenizer_utils", tokenizer_utils
    )
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    with pytest.raises(SystemExit) as exc:
        train_tinker_sft.exact_token_validation(config, {})
    assert exc.value.code == 3
    err = capsys.readouterr().err
    assert "cannot load the tokenizer" in err
    assert "--offline" in err
    assert "Traceback" not in err


def test_load_bearing_runtime_versions_are_pinned_and_checked():
    """tinker + tinker-cookbook alone do not determine the token ids."""
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    runtime = config["runtime_packages"]
    for package in ("transformers", "tokenizers", "torch", "datasets",
                    "huggingface-hub", "chz", "tml-renderers", "safetensors"):
        assert runtime.get(package), f"{package} is not pinned"
    # pip freeze normalises separators, so compare on the normalised name.
    locked = {}
    for line in (ROOT / "requirements-tinker.lock").read_text(
        encoding="utf-8"
    ).splitlines():
        if "==" in line and not line.startswith("#"):
            name, _, version = line.partition("==")
            locked[name.strip().replace("-", "_").lower()] = version.strip()
    for package, version in dict(runtime, **config["packages"]).items():
        key = package.replace("-", "_").lower()
        assert locked.get(key) == version, (
            f"{package} is {version} in the config but "
            f"{locked.get(key)} in requirements-tinker.lock"
        )


def test_a_runtime_version_mismatch_aborts_before_any_client(monkeypatch, capsys):
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    installed = dict(config["packages"], **config["runtime_packages"])
    installed["transformers"] = "4.0.0"
    monkeypatch.setattr(
        train_tinker_sft.importlib.metadata, "version",
        lambda name: installed[name],
    )
    with pytest.raises(SystemExit) as exc:
        train_tinker_sft.assert_package_versions(config)
    assert exc.value.code == 3
    assert "transformers version mismatch" in capsys.readouterr().err


def test_the_frozen_manifest_pins_the_runtime_environment():
    manifest = json.loads(PREVIEW.read_text(encoding="utf-8"))
    runtime = manifest["backend"]["runtime_packages"]
    assert runtime["transformers"] == "5.5.4"
    assert runtime["tokenizers"] == "0.22.2"
    assert manifest["backend"]["packages"]["tinker"] == "0.26.1"
