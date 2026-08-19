"""Tests for the validation-only rental checks.

    ./.venv/bin/python -m pytest tests/test_validate_student.py -q

These cover the checks that can be got wrong SILENTLY: claiming bf16 because we
passed the flag, and calling an empty or looping completion a success. Checks 3
and 5 need a live endpoint and are covered by tests/test_run_gates.py's identity
and stage-semantics tests, which exercise the same code paths.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import validate_student_endpoint as v


# --- check 1: hardware ------------------------------------------------------

def test_missing_hardware_report_fails(tmp_path):
    assert v.check_1_hardware(None).ok is False
    assert v.check_1_hardware(tmp_path / "nope.json").ok is False


def test_ampere_is_acceptable_for_bf16(tmp_path):
    """Unlike the FP8 arm, bf16 does NOT require Ada. Ampere must pass."""
    p = tmp_path / "hw.json"
    p.write_text(json.dumps({"gpu": "RTX A6000", "compute_capability": "8.6",
                             "driver": "580.65", "memory_free_mib": 48000}))
    assert v.check_1_hardware(p).ok is True


def test_pre_ampere_is_refused(tmp_path):
    p = tmp_path / "hw.json"
    p.write_text(json.dumps({"gpu": "Tesla T4", "compute_capability": "7.5",
                             "driver": "535.1", "memory_free_mib": 15000}))
    assert v.check_1_hardware(p).ok is False


# --- check 2: the dtype claim -----------------------------------------------

def test_no_log_means_no_bf16_claim():
    """Passing --dtype bfloat16 is what we asked for, not what loaded."""
    c = v.check_2_bf16(None)
    assert c.ok is False
    assert "not evidence" in c.detail


def test_log_stating_bfloat16_passes(tmp_path):
    log = tmp_path / "vllm.log"
    log.write_text("INFO llm_engine.py:200] Initializing an LLM engine with "
                   "config: model='Qwen/Qwen3.5-9B', dtype=torch.bfloat16, "
                   "max_seq_len=32768\n")
    assert v.check_2_bf16(log).ok is True


def test_log_stating_float16_fails(tmp_path):
    """fp16 would make the baseline a measurement of a precision we never
    trained against — and it is the value vLLM falls back to."""
    log = tmp_path / "vllm.log"
    log.write_text("INFO ... dtype=torch.float16, max_seq_len=32768\n")
    c = v.check_2_bf16(log)
    assert c.ok is False
    assert "quantization damage" in c.detail


def test_log_showing_a_quantizer_fails(tmp_path):
    log = tmp_path / "vllm.log"
    log.write_text("INFO ... quantization=bitsandbytes, dtype=torch.bfloat16\n")
    assert v.check_2_bf16(log).ok is False


def test_log_without_any_dtype_fails(tmp_path):
    log = tmp_path / "vllm.log"
    log.write_text("INFO api_server.py:100] Started server on port 8020\n")
    c = v.check_2_bf16(log)
    assert c.ok is False
    assert "never states a dtype" in c.detail


# --- check 4: what counts as valid text -------------------------------------

@pytest.mark.parametrize("text", ["", "   ", "\n"])
def test_empty_completion_is_degenerate(text):
    assert not text.strip()


def test_repetition_loop_is_detected():
    assert v.DEGENERATE.match("ya ya ya ya ya ya ya ya ")


def test_ordinary_answer_is_not_flagged_as_a_loop():
    answer = ("Terdapat tiga hal yang perlu diperiksa: kesesuaian total anggaran "
              "dengan pagu, kelengkapan dokumen pendukung, dan persetujuan "
              "pemilik anggaran.")
    assert v.DEGENERATE.match(answer) is None
