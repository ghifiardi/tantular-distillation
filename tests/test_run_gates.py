"""Gate-runner tests: it must pass, fail, and fail CLOSED.

    ./.venv/bin/python -m pytest tests/test_run_gates.py -q

A gate runner is only worth having if a missing gate is louder than a passing
one. These cover the three cases that matter:

  pass            every gate meets its threshold -> exit 0
  fail            a gate misses its threshold    -> exit 1
  missing runner  a declared gate has no implementation, a source is absent, or
                  a scorer is gone -> exit 2, never a silent skip

No model is called: the voice gate is driven with pre-generated traces via
--traces, so these run anywhere and cost nothing.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
PY = str(ROOT / ".venv" / "bin" / "python")
RUNNER = str(ROOT / "src" / "run_gates.py")
ITEMS = ROOT / "prompts" / "voice_eval.v1.jsonl"

CLEAN = "Dokumen tersebut menyatakan ketentuan yang berlaku beserta tanggalnya."
DIRTY = "Baik Kak, dokumennya udah nyebutin ketentuan itu."


def write_traces(path: Path, n_bad: int) -> Path:
    items = [json.loads(l) for l in ITEMS.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = [{"family": it["id"], "completion": DIRTY if i < n_bad else CLEAN}
            for i, it in enumerate(items)]
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                    encoding="utf-8")
    return path


# --- edit_contract_output: the model-dependent gate -------------------------

EDIT_ITEMS = ROOT / "prompts" / "edit_contract_eval.v1.jsonl"


def write_edit_traces(path: Path, mode: str) -> Path:
    """mode: good | unparseable | find-absent | missing-one"""
    items = [json.loads(l) for l in EDIT_ITEMS.read_text(encoding="utf-8").splitlines()
             if l.strip()]
    rows = []
    for i, it in enumerate(items):
        if mode == "missing-one" and i == 0:
            continue                      # omit an answer entirely
        doc = it["document"]
        word = doc.split()[0]             # a token guaranteed to be in the document
        if mode == "unparseable" and i < 3:
            completion = "Tentu, berikut hasil penyuntingannya."
        elif mode == "find-absent" and i < 3:
            completion = json.dumps({"edits": [
                {"find": "kalimat yang tidak ada di dokumen", "replace": "x",
                 "occurrence": 1}]}, ensure_ascii=False)
        else:
            completion = json.dumps({"edits": [
                {"find": word, "replace": word.upper(), "occurrence": 1}]},
                ensure_ascii=False)
        rows.append({"family": it["id"], "completion": completion})
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                    encoding="utf-8")
    return path


ADAPTER_ID = "tantular-office-9b-v1"


def make_adapter(path: Path) -> Path:
    """A directory that looks like a real PEFT adapter.

    The fixtures used to be a directory with one placeholder file, which is
    exactly what the gates would accept when --adapter was only hashed. It is
    not accepted now: an adapter must carry adapter_config.json and weights.
    """
    path.mkdir(parents=True, exist_ok=True)
    (path / "adapter_config.json").write_text(json.dumps({
        "peft_type": "LORA", "r": 32, "lora_alpha": 64,
        "base_model_name_or_path": "Qwen/Qwen3.5-9B"}))
    (path / "adapter_model.safetensors").write_bytes(b"placeholder weights")
    return path


def run(config: Path, out: Path, traces: Path, stage: str = "before", adapter=None,
        adapter_model_id: str | None = ADAPTER_ID):
    """Drive every gate from fixtures. The edit gate needs its own traces; without
    them the runner would try to reach a model, which is correct behaviour but not
    what these tests are exercising."""
    edit = write_edit_traces(out.parent / f"edit-{out.stem}.jsonl", "good")
    cmd = [PY, RUNNER, "run", "--config", str(config), "--stage", stage,
           "--traces", str(traces), "--edit-traces", str(edit), "--out", str(out)]
    if adapter:
        cmd += ["--adapter", str(adapter)]
    if stage == "after" and adapter_model_id:
        cmd += ["--adapter-model-id", adapter_model_id]
    return subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)


@pytest.fixture
def config(tmp_path):
    """The real config, copied so a test can corrupt it safely."""
    cfg = yaml.safe_load((ROOT / "train" / "qlora_9b.yaml").read_text())
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_pass_when_every_gate_meets_threshold(config, tmp_path):
    traces = write_traces(tmp_path / "t.jsonl", n_bad=0)
    proc = run(config, tmp_path / "before.json", traces)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    report = json.loads((tmp_path / "before.json").read_text())
    assert report["all_passed"] is True
    assert {g["name"] for g in report["gates"]} == {
        "indonesian_voice", "office_json_contract", "edit_contract_output"}


def test_fail_when_a_gate_misses_threshold(config, tmp_path):
    """At --stage after, missing the threshold is a failure. (At --stage before
    it is a recorded baseline; see the stage-semantics tests below.)"""
    adapter = make_adapter(tmp_path / "adapter")
    # 3 bad answers -> 37/40 = 0.925 < 0.95
    traces = write_traces(tmp_path / "t.jsonl", n_bad=3)
    proc = run(config, tmp_path / "after.json", traces, stage="after",
               adapter=adapter)
    assert proc.returncode == 1, proc.stdout + proc.stderr
    report = json.loads((tmp_path / "after.json").read_text())
    voice = next(g for g in report["gates"] if g["name"] == "indonesian_voice")
    assert voice["passed"] is False and voice["rate"] < 0.95
    assert voice["status"] == "FAIL"
    assert report["verdict"] == "FAIL"


def test_missing_runner_fails_closed(config, tmp_path):
    """A gate declared in config with no implementation must ABORT, not skip."""
    cfg = yaml.safe_load(config.read_text())
    cfg["eval_gates"].append({"name": "gate_that_does_not_exist",
                              "source": "prompts", "min_pass_rate": 0.9})
    config.write_text(yaml.safe_dump(cfg))
    traces = write_traces(tmp_path / "t.jsonl", n_bad=0)
    proc = run(config, tmp_path / "before.json", traces)
    assert proc.returncode == 2, "a gate with no runner must abort, not skip"
    assert "no runner" in (proc.stdout + proc.stderr)


def test_missing_eval_source_fails_closed(config, tmp_path):
    cfg = yaml.safe_load(config.read_text())
    for g in cfg["eval_gates"]:
        if g["name"] == "indonesian_voice":
            g["source"] = "prompts/does_not_exist.jsonl"
    config.write_text(yaml.safe_dump(cfg))
    traces = write_traces(tmp_path / "t.jsonl", n_bad=0)
    proc = run(config, tmp_path / "before.json", traces)
    assert proc.returncode == 2
    assert "source missing" in (proc.stdout + proc.stderr)


def test_after_stage_requires_an_adapter(config, tmp_path):
    """An 'after' run with no adapter would re-measure the base model and
    report it as the trained one."""
    traces = write_traces(tmp_path / "t.jsonl", n_bad=0)
    proc = run(config, tmp_path / "after.json", traces, stage="after")
    assert proc.returncode == 2
    assert "requires --adapter" in (proc.stdout + proc.stderr)


def test_compare_detects_regression(config, tmp_path):
    adapter = make_adapter(tmp_path / "adapter")
    before = tmp_path / "before.json"; after = tmp_path / "after.json"
    run(config, before, write_traces(tmp_path / "b.jsonl", 0))
    run(config, after, write_traces(tmp_path / "a.jsonl", 3), stage="after",
        adapter=adapter)
    proc = subprocess.run([PY, RUNNER, "compare", "--before", str(before),
                           "--after", str(after)], capture_output=True, text=True,
                          cwd=ROOT)
    assert proc.returncode == 1
    assert "DO NOT PROMOTE" in proc.stdout
    assert "REGRESSED" in proc.stdout


def test_compare_refuses_mismatched_configs(config, tmp_path):
    """Comparing runs from different configs would blame the adapter for a
    config change."""
    adapter = make_adapter(tmp_path / "adapter")
    before = tmp_path / "before.json"; after = tmp_path / "after.json"
    run(config, before, write_traces(tmp_path / "b.jsonl", 0))
    run(config, after, write_traces(tmp_path / "a.jsonl", 0), stage="after",
        adapter=adapter)
    doctored = json.loads(before.read_text())
    doctored["config"]["sha256"] = "0" * 64
    before.write_text(json.dumps(doctored))
    proc = subprocess.run([PY, RUNNER, "compare", "--before", str(before),
                           "--after", str(after)], capture_output=True, text=True,
                          cwd=ROOT)
    assert proc.returncode == 2
    assert "DIFFERENT configs" in (proc.stdout + proc.stderr)


def test_model_independent_gate_is_labelled(config, tmp_path):
    """office_json_contract cannot detect an adapter regression; the report must
    say so rather than let 'same' read as evidence."""
    traces = write_traces(tmp_path / "t.jsonl", n_bad=0)
    run(config, tmp_path / "before.json", traces)
    report = json.loads((tmp_path / "before.json").read_text())
    contract = next(g for g in report["gates"] if g["name"] == "office_json_contract")
    assert contract["model_dependent"] is False
    assert "CANNOT detect" in contract["_model_independent_note"]


def run_edit(config: Path, out: Path, voice_traces: Path, edit_traces: Path,
             stage: str = "after", adapter=None, adapter_model_id=ADAPTER_ID):
    """Defaults to --stage after: these tests are about whether the edit gate
    SCORES correctly, and only the after stage turns a low score into exit 1.
    At --stage before a low score is a recorded baseline (see stage semantics)."""
    if stage == "after" and adapter is None:
        adapter = make_adapter(out.parent / f"adapter-{out.stem}")
    cmd = [PY, RUNNER, "run", "--config", str(config), "--stage", stage,
           "--traces", str(voice_traces), "--edit-traces", str(edit_traces),
           "--out", str(out)]
    if adapter:
        cmd += ["--adapter", str(adapter)]
    if stage == "after" and adapter_model_id:
        cmd += ["--adapter-model-id", adapter_model_id]
    return subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)


def test_edit_gate_passes_on_valid_contract(config, tmp_path):
    proc = run_edit(config, tmp_path / "b.json",
                    write_traces(tmp_path / "v.jsonl", 0),
                    write_edit_traces(tmp_path / "e.jsonl", "good"))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    report = json.loads((tmp_path / "b.json").read_text())
    gate = next(g for g in report["gates"] if g["name"] == "edit_contract_output")
    assert gate["model_dependent"] is True
    assert gate["passed"] is True and gate["rate"] == 1.0
    assert gate["breakdown"]["contract_ok"] == gate["items"]


def test_edit_gate_fails_when_output_is_not_json(config, tmp_path):
    proc = run_edit(config, tmp_path / "b.json",
                    write_traces(tmp_path / "v.jsonl", 0),
                    write_edit_traces(tmp_path / "e.jsonl", "unparseable"))
    assert proc.returncode == 1, proc.stdout + proc.stderr
    gate = next(g for g in json.loads((tmp_path / "b.json").read_text())["gates"]
                if g["name"] == "edit_contract_output")
    assert gate["passed"] is False
    assert gate["breakdown"]["parse_ok"] < gate["items"]


def test_edit_gate_fails_when_json_parses_but_find_is_absent(config, tmp_path):
    """The case parse_ok alone would miss: valid JSON, useless edits."""
    proc = run_edit(config, tmp_path / "b.json",
                    write_traces(tmp_path / "v.jsonl", 0),
                    write_edit_traces(tmp_path / "e.jsonl", "find-absent"))
    assert proc.returncode == 1, proc.stdout + proc.stderr
    gate = next(g for g in json.loads((tmp_path / "b.json").read_text())["gates"]
                if g["name"] == "edit_contract_output")
    assert gate["breakdown"]["parse_ok"] == gate["items"], "all should parse"
    assert gate["breakdown"]["contract_ok"] < gate["items"], "but not resolve"
    assert gate["passed"] is False


def test_edit_gate_missing_output_fails_closed(config, tmp_path):
    proc = run_edit(config, tmp_path / "b.json",
                    write_traces(tmp_path / "v.jsonl", 0),
                    write_edit_traces(tmp_path / "e.jsonl", "missing-one"))
    assert proc.returncode == 2, "a missing model output must abort, not skip"
    assert "no model output" in (proc.stdout + proc.stderr)


def test_edit_gate_missing_parser_fails_closed(config, tmp_path):
    cfg = yaml.safe_load(config.read_text())
    for g in cfg["eval_gates"]:
        if g["name"] == "edit_contract_output":
            g["addin_src"] = "../does_not_exist/src"
    config.write_text(yaml.safe_dump(cfg))
    proc = run_edit(config, tmp_path / "b.json",
                    write_traces(tmp_path / "v.jsonl", 0),
                    write_edit_traces(tmp_path / "e.jsonl", "good"))
    assert proc.returncode == 2
    assert "add-in parser missing" in (proc.stdout + proc.stderr)


def test_both_contract_gates_coexist(config, tmp_path):
    """The model-independent gate is retained, not replaced."""
    run_edit(config, tmp_path / "b.json",
             write_traces(tmp_path / "v.jsonl", 0),
             write_edit_traces(tmp_path / "e.jsonl", "good"))
    gates = {g["name"]: g for g in json.loads((tmp_path / "b.json").read_text())["gates"]}
    assert gates["office_json_contract"]["model_dependent"] is False
    assert gates["edit_contract_output"]["model_dependent"] is True


# --- model identity: the gates must measure the STUDENT ---------------------
#
# The runner previously defaulted --host/--teacher to the TEACHER
# (muse-glimmer @ ai19-ollama). A `before` run with defaults would have measured
# Muse Glimmer int4 as the student baseline, making the before/after comparison
# teacher-vs-adapter — a complete run, plausible numbers, and no meaning at all.
# Nothing downstream could detect that, so it is checked here.

def test_live_run_requires_expect_model(config, tmp_path):
    """Without fixtures the gates call an endpoint; identity must be asserted."""
    proc = subprocess.run(
        [PY, RUNNER, "run", "--config", str(config), "--stage", "before",
         "--host", "ai19-ollama", "--teacher", "muse-glimmer",
         "--out", str(tmp_path / "g.json")],
        capture_output=True, text=True, cwd=ROOT)
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "--expect-model is required" in proc.stderr


def _identity(monkeypatch, expect, served):
    sys.path.insert(0, str(ROOT / "src"))
    import run_gates
    import config as config_module

    class Resp:
        @staticmethod
        def json():
            return {"data": [{"id": s} for s in served]}

    monkeypatch.setattr(config_module, "resolve",
                        lambda t, h: {"HOST_BASE_URL": "http://x/v1",
                                      "HOST_QUANTIZATION": "none"})
    import httpx
    monkeypatch.setattr(httpx, "get", lambda *a, **k: Resp())
    return run_gates.verify_served_model(expect, "somehost", "somemodel")


def test_identity_accepts_the_configured_student(monkeypatch):
    got = _identity(monkeypatch, "Qwen/Qwen3.5-9B",
                    ["Qwen/Qwen3.5-9B"])
    assert got["expected"] == "Qwen/Qwen3.5-9B"


def test_identity_rejects_the_teacher_endpoint(monkeypatch):
    """Pointing the student gates at Muse Glimmer must abort, not proceed."""
    with pytest.raises(SystemExit) as e:
        _identity(monkeypatch, "Qwen/Qwen3.5-9B", ["muse-glimmer:30b"])
    assert e.value.code == 2


# --- training host: ai19 is production, and must refuse to train ------------
#
# ai19 backs the openai.ina17.com gateway and a face_ai_service. Its config used
# to say `role: training`, which is exactly the kind of stale note someone acts
# on at 2am. The role is now production-serving, but a comment is not a guard —
# these are.

def _guard(host, machine):
    sys.path.insert(0, str(ROOT / "src"))
    import config as config_module
    return config_module.training_guard(host, hostname=machine)


def test_training_allowed_on_a_declared_rental():
    assert _guard("rented-48gb", "pod-7f3a")["gpu"]


def test_ai19_refuses_to_train():
    with pytest.raises(SystemExit) as e:
        _guard("ai19", "pod-7f3a")
    assert "not declared as a training host" in str(e.value)


def test_a_silent_host_refuses_to_train():
    """Serving hosts are not training hosts by default; silence is a refusal."""
    with pytest.raises(SystemExit):
        _guard("student-serve", "pod-7f3a")


def test_declaring_a_rental_does_not_help_when_running_ON_ai19():
    """The machine check is independent of what was declared."""
    with pytest.raises(SystemExit) as e:
        _guard("rented-48gb", "ai19.internal")
    assert "this machine" in str(e.value)


def test_trainer_refuses_ai19_end_to_end(tmp_path):
    run_manifest = tmp_path / "RUN.json"
    freeze = subprocess.run(
        [PY, str(ROOT / "src" / "freeze_training_run.py"),
         "--corpus", "data/v3-candidate/traces.r0.jsonl",
         "--config", "train/qlora_9b.yaml",
         "--promotion-manifest", "train/RUN_MANIFEST.v1-mechanical.json",
         "--waiver", "calibration/INT4_WAIVER.md",
         "--out", str(run_manifest),
         "--frozen-at", "2026-08-19T12:00:00+07:00", "--write"],
        capture_output=True, text=True, cwd=ROOT)
    assert freeze.returncode == 0, freeze.stdout + freeze.stderr
    proc = subprocess.run(
        [PY, str(ROOT / "src" / "train_qlora.py"), "--dry-run",
         "--run-manifest", str(run_manifest), "--train-host", "ai19"],
        capture_output=True, text=True, cwd=ROOT)
    assert proc.returncode != 0
    assert "not declared as a training host" in proc.stdout + proc.stderr


# --- stage semantics --------------------------------------------------------
#
# Decided 2026-08-19. The base student was never trained on Indonesian office
# work, so its voice score will sit well under 0.95 — the bar for a PROMOTED
# adapter. Failing `before` on that would mean the run can never start, and the
# obvious escape is to lower the threshold, which destroys the one number that
# matters after training.
#
# So: a low baseline is MEASURED, not passed. "measured" and "passed" stay
# separate words in the report, the threshold is untouched, and infrastructure
# or identity failures still exit 2 at both stages.

def _adapter(tmp_path, name="adapter"):
    return make_adapter(tmp_path / name)


def test_before_low_baseline_is_measured_not_failed(config, tmp_path):
    """Case 1: baseline below target, but fully measured -> the run continues."""
    # 12 bad answers -> 28/40 = 0.70, far under 0.95
    traces = write_traces(tmp_path / "b.jsonl", n_bad=12)
    proc = run(config, tmp_path / "before.json", traces, stage="before")
    assert proc.returncode == 0, proc.stdout + proc.stderr

    report = json.loads((tmp_path / "before.json").read_text())
    voice = next(g for g in report["gates"] if g["name"] == "indonesian_voice")

    # measured, and every gate produced a rate
    assert report["measured"] is True
    assert len(report["gates"]) == 3
    assert voice["rate"] == pytest.approx(0.70)

    # but NOT a pass, and never promotable
    assert voice["passed"] is False
    assert voice["status"] == "BASELINE_BELOW_TARGET"
    assert report["all_passed"] is False
    assert report["verdict"] == "BASELINE_MEASURED_BELOW_TARGET"
    assert report["promotable"] is False
    assert "indonesian_voice" in report["below_target"]
    assert "This is NOT a pass" in proc.stdout

    # the threshold itself is untouched
    assert voice["threshold"] == 0.95


def test_before_still_exits_2_on_infrastructure_failure(config, tmp_path):
    """A tolerant threshold must not become a tolerant runner."""
    cfg = yaml.safe_load(config.read_text())
    for g in cfg["eval_gates"]:
        if g["name"] == "indonesian_voice":
            g["scorer"] = "src/scorer_that_does_not_exist.py"
    config.write_text(yaml.safe_dump(cfg))
    proc = run(config, tmp_path / "before.json", write_traces(tmp_path / "b.jsonl", 0))
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "scorer missing" in proc.stderr


def test_improved_but_under_threshold_is_not_promotable(config, tmp_path):
    """Case 2: the adapter improves a lot and still misses 0.95.

    Improvement is progress, not a product. This is the case most likely to be
    argued away in the moment, so it is pinned here.
    """
    before, after = tmp_path / "before.json", tmp_path / "after.json"
    run(config, before, write_traces(tmp_path / "b.jsonl", 12))      # 0.700
    run(config, after, write_traces(tmp_path / "a.jsonl", 4),        # 0.900
        stage="after", adapter=_adapter(tmp_path))

    assert json.loads(after.read_text())["verdict"] == "FAIL"
    out = tmp_path / "cmp.json"
    proc = subprocess.run([PY, RUNNER, "compare", "--before", str(before),
                           "--after", str(after), "--json-out", str(out)],
                          capture_output=True, text=True, cwd=ROOT)
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "DO NOT PROMOTE" in proc.stdout
    assert "Improvement is not promotion" in proc.stdout

    cmp = json.loads(out.read_text())
    assert cmp["promotable"] is False
    assert cmp["regressed"] == []                       # it went UP
    assert cmp["below_threshold_after"] == ["indonesian_voice"]
    assert cmp["gates"]["indonesian_voice"]["before"] == pytest.approx(0.70)
    assert cmp["gates"]["indonesian_voice"]["after"] == pytest.approx(0.90)


def test_reaching_threshold_without_regression_is_promotable(config, tmp_path):
    """Case 3: 0.95 met, nothing regressed -> promotable."""
    before, after = tmp_path / "before.json", tmp_path / "after.json"
    run(config, before, write_traces(tmp_path / "b.jsonl", 12))      # 0.700
    run(config, after, write_traces(tmp_path / "a.jsonl", 0),        # 1.000
        stage="after", adapter=_adapter(tmp_path))

    after_report = json.loads(after.read_text())
    assert after_report["verdict"] == "PASS"
    assert all(g["status"] == "PASS" for g in after_report["gates"])

    out = tmp_path / "cmp.json"
    proc = subprocess.run([PY, RUNNER, "compare", "--before", str(before),
                           "--after", str(after), "--json-out", str(out)],
                          capture_output=True, text=True, cwd=ROOT)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "PROMOTABLE" in proc.stdout
    cmp = json.loads(out.read_text())
    assert cmp["promotable"] is True
    assert cmp["regressed"] == [] and cmp["below_threshold_after"] == []
    # the low baseline is reported, not held against the adapter
    assert cmp["baseline_below_target"] == ["indonesian_voice"]


def test_compare_refuses_stages_the_wrong_way_round(config, tmp_path):
    before, after = tmp_path / "before.json", tmp_path / "after.json"
    run(config, before, write_traces(tmp_path / "b.jsonl", 0))
    run(config, after, write_traces(tmp_path / "a.jsonl", 0), stage="after",
        adapter=_adapter(tmp_path))
    proc = subprocess.run([PY, RUNNER, "compare", "--before", str(after),
                           "--after", str(before)], capture_output=True, text=True,
                          cwd=ROOT)
    assert proc.returncode == 2
    assert "wrong way round" in (proc.stdout + proc.stderr)


# --- the after stage must actually evaluate the ADAPTER ----------------------
#
# Until 2026-08-19 --adapter was only existence-checked and hashed: the gates
# then generated from the same model id as the baseline. Every after run would
# have re-measured the BASE model, recorded the adapter's digest beside it, and
# reported a complete before/after comparison of a model against itself. The
# digest made it look verified, which is worse than recording nothing.
#
# These are the four ways that can happen, each of which must now fail.

def test_adapter_path_that_is_not_an_adapter_is_refused(config, tmp_path):
    """(1) a directory that hashes is not an adapter."""
    fake = tmp_path / "not-an-adapter"; fake.mkdir()
    (fake / "weights.bin").write_bytes(b"placeholder")
    proc = run(config, tmp_path / "after.json", write_traces(tmp_path / "t.jsonl", 0),
               stage="after", adapter=fake)
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "not a LoRA adapter directory" in proc.stderr


def test_adapter_config_without_weights_is_refused(config, tmp_path):
    """(1b) config present, nothing to load."""
    half = tmp_path / "half"; half.mkdir()
    (half / "adapter_config.json").write_text("{}")
    proc = run(config, tmp_path / "after.json", write_traces(tmp_path / "t.jsonl", 0),
               stage="after", adapter=half)
    assert proc.returncode == 2
    assert "no weights" in proc.stderr


def test_after_without_an_adapter_model_id_is_refused(config, tmp_path):
    """(2) no id means the gates would request the base model id."""
    proc = run(config, tmp_path / "after.json", write_traces(tmp_path / "t.jsonl", 0),
               stage="after", adapter=make_adapter(tmp_path / "a"),
               adapter_model_id=None)
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "--adapter-model-id" in proc.stderr


def test_adapter_model_id_equal_to_the_base_is_refused(config, tmp_path):
    """(2b) vLLM serves the LoRA alongside the base; asking for the base id
    returns base answers from the same process."""
    cfg = yaml.safe_load(config.read_text())
    base = cfg["base_model"]
    proc = run(config, tmp_path / "after.json", write_traces(tmp_path / "t.jsonl", 0),
               stage="after", adapter=make_adapter(tmp_path / "a"),
               adapter_model_id=base)
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "is a BASE model id/alias" in proc.stderr


def test_adapter_model_id_equal_to_the_served_base_alias_is_refused(config, tmp_path):
    """The server registers the base as both the repo id and office-student-9b.

    /v1/models therefore already lists the short alias before any LoRA is
    loaded. Accepting that alias as the adapter id would recreate the original
    defect: the loaded check passes and generation still addresses the base.
    """
    proc = run(config, tmp_path / "after.json", write_traces(tmp_path / "t.jsonl", 0),
               stage="after", adapter=make_adapter(tmp_path / "a"),
               adapter_model_id="office-student-9b")
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "is a BASE model id/alias" in proc.stderr


def test_report_records_which_model_id_produced_the_answers(config, tmp_path):
    """The positive case: the report must be able to prove what it measured."""
    proc = run(config, tmp_path / "after.json", write_traces(tmp_path / "t.jsonl", 0),
               stage="after", adapter=make_adapter(tmp_path / "a"))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    report = json.loads((tmp_path / "after.json").read_text())
    assert report["adapter"]["evaluated"] is True
    assert report["adapter"]["model_id"] == ADAPTER_ID
    for g in report["gates"]:
        if g["model_dependent"]:
            assert g["generated_by_model_id"] == ADAPTER_ID


def test_compare_refuses_an_after_report_generated_from_the_base(config, tmp_path):
    """(3) the endpoint never served the adapter, so the answers are the base's.

    Simulated by doctoring the report, because the honest version of this needs
    a live endpoint — and a doctored report is exactly what a stale or
    hand-edited artifact looks like.
    """
    before, after = tmp_path / "before.json", tmp_path / "after.json"
    run(config, before, write_traces(tmp_path / "b.jsonl", 3))
    run(config, after, write_traces(tmp_path / "a.jsonl", 0), stage="after",
        adapter=make_adapter(tmp_path / "a"))
    doctored = json.loads(after.read_text())
    base = doctored["model"]["expected"] or doctored["model"]["teacher"]
    for g in doctored["gates"]:
        if g["model_dependent"]:
            g["generated_by_model_id"] = base
    after.write_text(json.dumps(doctored))
    proc = subprocess.run([PY, RUNNER, "compare", "--before", str(before),
                           "--after", str(after)], capture_output=True, text=True,
                          cwd=ROOT)
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "not from the adapter id" in proc.stderr


def test_compare_refuses_an_after_report_with_only_a_digest(config, tmp_path):
    """(4) the shape of every report produced before this check existed."""
    before, after = tmp_path / "before.json", tmp_path / "after.json"
    run(config, before, write_traces(tmp_path / "b.jsonl", 3))
    run(config, after, write_traces(tmp_path / "a.jsonl", 0), stage="after",
        adapter=make_adapter(tmp_path / "a"))
    doctored = json.loads(after.read_text())
    doctored["adapter"] = {"path": doctored["adapter"]["path"],
                           "sha256": doctored["adapter"]["sha256"]}
    after.write_text(json.dumps(doctored))
    proc = subprocess.run([PY, RUNNER, "compare", "--before", str(before),
                           "--after", str(after)], capture_output=True, text=True,
                          cwd=ROOT)
    assert proc.returncode == 2
    assert "no adapter identity" in proc.stderr


def test_missing_adapter_directory_is_refused(config, tmp_path):
    """(4b) --adapter pointing at nothing."""
    proc = run(config, tmp_path / "after.json", write_traces(tmp_path / "t.jsonl", 0),
               stage="after", adapter=tmp_path / "does-not-exist")
    assert proc.returncode == 2
    assert "missing" in proc.stderr


def test_endpoint_not_serving_the_adapter_is_refused(monkeypatch, tmp_path):
    """(3b) the live check: /v1/models does not list the adapter id."""
    sys.path.insert(0, str(ROOT / "src"))
    import run_gates
    import config as config_module

    class Resp:
        @staticmethod
        def json():
            return {"data": [{"id": "Qwen/Qwen3.5-9B"}]}   # base only

    monkeypatch.setattr(config_module, "resolve",
                        lambda t, h: {"HOST_BASE_URL": "http://x/v1"})
    import httpx
    monkeypatch.setattr(httpx, "get", lambda *a, **k: Resp())
    with pytest.raises(SystemExit) as e:
        run_gates.verify_adapter_served(
            make_adapter(tmp_path / "a"), ADAPTER_ID,
            "Qwen/Qwen3.5-9B", "student-serve", "office-student-9b",
            live=True)
    assert e.value.code == 2


def test_endpoint_serving_the_adapter_is_accepted(monkeypatch, tmp_path):
    sys.path.insert(0, str(ROOT / "src"))
    import run_gates
    import config as config_module

    class Resp:
        @staticmethod
        def json():
            return {"data": [{"id": "Qwen/Qwen3.5-9B"},
                             {"id": ADAPTER_ID}]}

    monkeypatch.setattr(config_module, "resolve",
                        lambda t, h: {"HOST_BASE_URL": "http://x/v1"})
    import httpx
    monkeypatch.setattr(httpx, "get", lambda *a, **k: Resp())

    # Serving the id is not applying the LoRA, so the live path now probes both
    # ids for logprobs. Give the adapter a shifted distribution.
    class Probe:
        def __init__(self, lp): self._lp = lp
        def json(self):
            return {"choices": [{"logprobs": {"token_logprobs": self._lp}}]}

    monkeypatch.setattr(httpx, "post", lambda url, timeout=None, json=None:
                        Probe([-0.5, -1.25] if json["model"] == ADAPTER_ID
                              else [-0.5, -1.30]))
    got = run_gates.verify_adapter_served(
        make_adapter(tmp_path / "a"), ADAPTER_ID, "Qwen/Qwen3.5-9B",
        "student-serve", "office-student-9b", live=True)
    assert got["model_id"] == ADAPTER_ID and ADAPTER_ID in got["served"]
    assert got["effect"]["max_logprob_delta"] > 0


def test_a_served_but_inert_adapter_is_refused(monkeypatch, tmp_path):
    """The smoke-rental endpoint, end to end: listed, answering, inert."""
    sys.path.insert(0, str(ROOT / "src"))
    import run_gates
    import config as config_module

    class Models:
        @staticmethod
        def json():
            return {"data": [{"id": "Qwen/Qwen3.5-9B"}, {"id": ADAPTER_ID}]}

    class Probe:
        @staticmethod
        def json():
            return {"choices": [{"logprobs": {"token_logprobs": [-0.5, -1.25]}}]}

    monkeypatch.setattr(config_module, "resolve",
                        lambda t, h: {"HOST_BASE_URL": "http://x/v1"})
    import httpx
    monkeypatch.setattr(httpx, "get", lambda *a, **k: Models())
    monkeypatch.setattr(httpx, "post", lambda *a, **k: Probe())
    with pytest.raises(SystemExit) as e:
        run_gates.verify_adapter_served(
            make_adapter(tmp_path / "b"), ADAPTER_ID, "Qwen/Qwen3.5-9B",
            "student-serve", "office-student-9b", live=True)
    assert e.value.code == 2


# --- being served is not being applied --------------------------------------
#
# 2026-08-19, on the smoke rental: vLLM loaded a genuinely-trained adapter,
# logged "Loaded new LoRA adapter", bound it to nothing because the module paths
# did not match the served architecture, and answered every request with
# base-model output. /v1/models listed the adapter id throughout.
#
# verify_adapter_served() passed that endpoint. It checked the directory was a
# real adapter and the id was distinct and listed — all true, all irrelevant to
# whether the LoRA does anything. These cover the check that closes the gap.

def _probe(monkeypatch, adapter_lp, base_lp):
    sys.path.insert(0, str(ROOT / "src"))
    import run_gates

    class Resp:
        def __init__(self, lp): self._lp = lp
        def json(self):
            return {"choices": [{"logprobs": {"token_logprobs": self._lp}}]}

    def post(url, timeout=None, json=None):
        return Resp(adapter_lp if json["model"] == "adapter-id" else base_lp)

    import httpx
    monkeypatch.setattr(httpx, "post", post)
    return run_gates.adapter_changes_the_distribution(
        "http://x/v1", "adapter-id", "base-id")


def test_identical_logprobs_are_refused(monkeypatch):
    """The exact smoke-rental failure: served, answering, and inert."""
    lp = [-0.5, -1.25, -0.125, -2.0]
    with pytest.raises(SystemExit) as e:
        _probe(monkeypatch, lp, list(lp))
    assert e.value.code == 2


def test_a_shifted_distribution_passes(monkeypatch):
    """Identical TEXT with different logprobs is a pass — a small adapter can
    decode the same string while still being applied."""
    got = _probe(monkeypatch, [-0.5, -1.25, -0.125, -2.0],
                 [-0.5, -1.25, -0.1251, -2.0])
    assert got["max_logprob_delta"] > 0
    assert got["probe_tokens"] == 4


def test_missing_logprobs_fail_closed(monkeypatch):
    """No logprobs means no proof; that must abort rather than pass quietly."""
    with pytest.raises(SystemExit) as e:
        _probe(monkeypatch, [], [])
    assert e.value.code == 2


# --- eval prompts are not corpus --------------------------------------------
#
# The v1 run aborted at the first live gate: generate_normalized assigns every
# prompt a split from the manifest, and held-out eval prompts belong to no
# family by design, so it died with "family '' is not in the split manifest".
#
# It went unnoticed because every other test drives the gates from --traces
# fixtures. The gates had never generated against a live model. These cover the
# flag that separates the two cases.

def test_gates_pass_eval_prompts_flag_to_the_generator():
    """The gate runner must mark eval generation as eval, or it aborts live."""
    src = (ROOT / "src" / "run_gates.py").read_text()
    assert src.count('"--eval-prompts"') == 2, (
        "both model-dependent gates must pass --eval-prompts; without it a live "
        "run dies on split-manifest assignment")


def test_eval_prompts_have_no_family():
    """The premise of the flag: these items are held out, not corpus."""
    for name in ("voice_eval.v1.jsonl", "edit_contract_eval.v1.jsonl"):
        rows = [json.loads(l) for l in
                (ROOT / "prompts" / name).read_text(encoding="utf-8").splitlines()
                if l.strip()]
        assert rows, f"{name} is empty"
        assert not [r for r in rows if r.get("family")], (
            f"{name} carries a family; it would then be governed by the split "
            "manifest and could collide with training data")


def test_eval_prompts_flag_refuses_corpus_prompts(tmp_path):
    """The flag must not become a way to bypass split rules for real corpus."""
    corpus_like = tmp_path / "corpus.jsonl"
    corpus_like.write_text(json.dumps(
        {"id": "x", "family": "document:email::0001", "user": "halo"}) + "\n",
        encoding="utf-8")
    proc = subprocess.run(
        [PY, str(ROOT / "src" / "generate_normalized.py"),
         "--teacher", "office-student-9b", "--host", "student-serve",
         "--eval-prompts", "--prompts", str(corpus_like),
         "--out", str(tmp_path / "out.jsonl")],
        capture_output=True, text=True, cwd=ROOT,
        # A resolvable address so the run reaches the family check; nothing is
        # ever sent, because the refusal happens before the first request.
        env={**os.environ, "HOST_BASE_URL": "http://127.0.0.1:1/v1"})
    assert proc.returncode != 0
    assert "carry a family" in proc.stdout + proc.stderr


def test_eval_prompts_are_classified_synthetic():
    """The egress guard defaults unclassified prompts to `internal`, which a
    rented (external) endpoint refuses — correctly, since it cannot tell real
    Office material from authored material. These eval sets were authored for
    the purpose and reuse nothing from the corpus, so they carry the
    classification explicitly rather than relying on an --egress-approval
    waiver, which would suppress the check rather than answer it."""
    for name in ("voice_eval.v1.jsonl", "edit_contract_eval.v1.jsonl"):
        rows = [json.loads(l) for l in
                (ROOT / "prompts" / name).read_text(encoding="utf-8").splitlines()
                if l.strip()]
        classes = {r.get("source_class") for r in rows}
        assert classes == {"synthetic"}, (
            f"{name} has source_class {classes}; every eval prompt must be "
            "classified, or a live gate run against an external host aborts")


def test_eval_prompts_get_their_id_as_join_key(tmp_path):
    """Eval items have no corpus family, but everything downstream joins on one.

    The v1 run's third abort was KeyError: 'family' while building the trace
    record. The id becomes the join key — the scorers already match on family
    OR id, so nothing they see changes.
    """
    items = tmp_path / "eval.jsonl"
    items.write_text(json.dumps({"id": "edit::0001", "user": "halo",
                                 "source_class": "synthetic"}) + "\n",
                     encoding="utf-8")
    proc = subprocess.run(
        [PY, str(ROOT / "src" / "generate_normalized.py"),
         "--teacher", "office-student-9b", "--host", "student-serve",
         "--eval-prompts", "--prompts", str(items),
         "--out", str(tmp_path / "out.jsonl")],
        capture_output=True, text=True, cwd=ROOT,
        env={**os.environ, "HOST_BASE_URL": "http://127.0.0.1:1/v1"})
    combined = proc.stdout + proc.stderr
    # It cannot reach a real endpoint here; what matters is that it got PAST
    # split assignment and trace construction to the connection attempt.
    assert "family" not in combined or "KeyError" not in combined, combined[:400]


def test_eval_prompts_require_an_id(tmp_path):
    items = tmp_path / "noid.jsonl"
    items.write_text(json.dumps({"user": "halo"}) + "\n", encoding="utf-8")
    proc = subprocess.run(
        [PY, str(ROOT / "src" / "generate_normalized.py"),
         "--teacher", "office-student-9b", "--host", "student-serve",
         "--eval-prompts", "--prompts", str(items),
         "--out", str(tmp_path / "out.jsonl")],
        capture_output=True, text=True, cwd=ROOT,
        env={**os.environ, "HOST_BASE_URL": "http://127.0.0.1:1/v1"})
    assert proc.returncode != 0
    assert "have no id" in proc.stdout + proc.stderr
