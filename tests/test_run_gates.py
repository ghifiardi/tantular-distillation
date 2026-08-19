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


def run(config: Path, out: Path, traces: Path, stage: str = "before", adapter=None):
    cmd = [PY, RUNNER, "run", "--config", str(config), "--stage", stage,
           "--traces", str(traces), "--out", str(out)]
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
    assert {g["name"] for g in report["gates"]} == {"indonesian_voice",
                                                    "office_json_contract"}


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
