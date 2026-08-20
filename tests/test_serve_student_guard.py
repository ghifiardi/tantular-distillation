"""The serve_student.sh duplicate-launch guard.

A second `serve_student.sh` launched while the first is still loading was the
concrete failure on the endpoint pod (2026-08-20): the second vLLM lost the
race for VRAM, died, and interleaved its traceback into the first engine's log,
making a healthy server look crashed. The guard refuses to start when the
target port is already listening, unless SERVE_STUDENT_FORCE=1 is set.

These tests exercise the guard's control flow with a stubbed port probe so they
need no real GPU, no vLLM, and no socket bind (which CI sandboxes often forbid).
"""
from __future__ import annotations

import os
import re
import stat
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "serve_student.sh"


def _guard_block() -> str:
    text = SCRIPT.read_text(encoding="utf-8")
    start = text.index("# Refuse to start a SECOND server")
    end = text.index("# NOTE on architecture", start)
    return text[start:end].rstrip() + "\n"


def _harness(tmp_path: Path, probe_exit: int) -> tuple[list[str], dict[str, str]]:
    probe = tmp_path / "py"
    probe.write_text(f"#!/usr/bin/env bash\nexit {probe_exit}\n", encoding="utf-8")
    probe.chmod(probe.stat().st_mode | stat.S_IEXEC)

    runner = tmp_path / "run.sh"
    runner.write_text(
        "set -euo pipefail\n"
        f'PYTHON_BIN="{probe}"\n'
        "TEACHER_PORT=8020\n"
        f"{_guard_block()}\n"
        'echo "REACHED_EXEC"\n',
        encoding="utf-8",
    )
    return ["bash", str(runner)], os.environ.copy()


def test_guard_allows_when_port_is_free(tmp_path: Path) -> None:
    cmd, env = _harness(tmp_path, probe_exit=1)  # 1 == nothing listening
    result = subprocess.run(cmd, env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert "REACHED_EXEC" in result.stdout


def test_guard_refuses_when_port_is_busy(tmp_path: Path) -> None:
    cmd, env = _harness(tmp_path, probe_exit=0)  # 0 == something listening
    result = subprocess.run(cmd, env=env, text=True, capture_output=True)
    assert result.returncode == 3
    assert "REACHED_EXEC" not in result.stdout
    assert re.search(r"port 8020 is already accepting", result.stderr)


def test_force_overrides_busy_port(tmp_path: Path) -> None:
    cmd, env = _harness(tmp_path, probe_exit=0)
    env["SERVE_STUDENT_FORCE"] = "1"
    result = subprocess.run(cmd, env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert "REACHED_EXEC" in result.stdout
