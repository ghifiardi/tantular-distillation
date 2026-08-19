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


def run(config: Path, out: Path, traces: Path, stage: str = "before", adapter=None):
    """Drive every gate from fixtures. The edit gate needs its own traces; without
    them the runner would try to reach a model, which is correct behaviour but not
    what these tests are exercising."""
    edit = write_edit_traces(out.parent / f"edit-{out.stem}.jsonl", "good")
    cmd = [PY, RUNNER, "run", "--config", str(config), "--stage", stage,
           "--traces", str(traces), "--edit-traces", str(edit), "--out", str(out)]
    if adapter:
        cmd += ["--adapter", str(adapter)]
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
    # 3 bad answers -> 37/40 = 0.925 < 0.95
    traces = write_traces(tmp_path / "t.jsonl", n_bad=3)
    proc = run(config, tmp_path / "before.json", traces)
    assert proc.returncode == 1, proc.stdout + proc.stderr
    report = json.loads((tmp_path / "before.json").read_text())
    voice = next(g for g in report["gates"] if g["name"] == "indonesian_voice")
    assert voice["passed"] is False and voice["rate"] < 0.95


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
    adapter = tmp_path / "adapter"; adapter.mkdir()
    (adapter / "weights.bin").write_bytes(b"placeholder")
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
    adapter = tmp_path / "adapter"; adapter.mkdir()
    (adapter / "w.bin").write_bytes(b"x")
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
             stage: str = "before", adapter=None):
    cmd = [PY, RUNNER, "run", "--config", str(config), "--stage", stage,
           "--traces", str(voice_traces), "--edit-traces", str(edit_traces),
           "--out", str(out)]
    if adapter:
        cmd += ["--adapter", str(adapter)]
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
    got = _identity(monkeypatch, "Qwen/Qwen3.5-9B-Instruct",
                    ["Qwen/Qwen3.5-9B-Instruct"])
    assert got["expected"] == "Qwen/Qwen3.5-9B-Instruct"


def test_identity_rejects_the_teacher_endpoint(monkeypatch):
    """Pointing the student gates at Muse Glimmer must abort, not proceed."""
    with pytest.raises(SystemExit) as e:
        _identity(monkeypatch, "Qwen/Qwen3.5-9B-Instruct", ["muse-glimmer:30b"])
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


def test_trainer_refuses_ai19_end_to_end():
    proc = subprocess.run(
        [PY, str(ROOT / "src" / "train_qlora.py"), "--dry-run", "--train-host", "ai19"],
        capture_output=True, text=True, cwd=ROOT)
    assert proc.returncode != 0
    assert "not declared as a training host" in proc.stdout + proc.stderr
