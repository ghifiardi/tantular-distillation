from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "verify_student_host.sh"


def _write_executable(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)


def _fake_gpu_tools(tmp_path: Path) -> dict[str, str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()

    _write_executable(
        bin_dir / "nvidia-smi",
        """#!/usr/bin/env bash
case "$*" in
  *"--query-gpu=name"*) printf '%s\\n' 'NVIDIA RTX A6000' ;;
  *"--query-gpu=compute_cap"*) printf '%s\\n' '8.6' ;;
  *"--query-gpu=memory.total"*) printf '%s\\n' '49140' ;;
  *"--query-gpu=memory.free"*) printf '%s\\n' '49000' ;;
  *"--query-gpu=driver_version"*) printf '%s\\n' "${FAKE_DRIVER:-570.211.01}" ;;
  *"--query-gpu=utilization.gpu"*) printf '%s\\n' "${FAKE_UTIL:-0}" ;;
  *"--query-gpu=memory.used"*) printf '%s\\n' "${FAKE_USED:-10}" ;;
  *) printf '%s\\n' \
       '| NVIDIA-SMI 570.211.01 Driver Version: 570.211.01 CUDA Version: 12.8 |' ;;
esac
""",
    )
    _write_executable(
        bin_dir / "python3",
        """#!/usr/bin/env bash
cat >/dev/null
if [[ "${FAKE_CUINIT_RC:-0}" == 0 ]]; then
  printf '%s\\n' \
    '  cuInit(0) -> 0 (CUDA_SUCCESS); the driver reports 1 device(s)'
else
  printf '%s\\n' \
    '  cuInit(0) returned 46 (CUDA_ERROR_DEVICES_UNAVAILABLE)'
fi
exit "${FAKE_CUINIT_RC:-0}"
""",
    )
    _write_executable(bin_dir / "sleep", "#!/usr/bin/env bash\nexit 0\n")

    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"
    return env


def _run(tmp_path: Path, env: dict[str, str], output: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(SCRIPT), output],
        cwd=tmp_path,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def test_fresh_ampere_pod_passes_and_records_cuinit(tmp_path: Path) -> None:
    env = _fake_gpu_tools(tmp_path)
    result = _run(tmp_path, env, "hardware.json")

    assert result.returncode == 0, result.stdout
    assert "cuInit(0) -> 0" in result.stdout
    hardware = json.loads((tmp_path / "hardware.json").read_text())
    assert hardware["gpu"] == "NVIDIA RTX A6000"
    assert hardware["driver"] == "570.211.01"
    assert hardware["cuinit_ok"] is True


def test_driver_older_than_525_is_rejected(tmp_path: Path) -> None:
    env = _fake_gpu_tools(tmp_path)
    env["FAKE_DRIVER"] = "510.47"
    result = _run(tmp_path, env, "hardware.json")

    assert result.returncode != 0
    assert "driver 510.47 is too old" in result.stdout
    assert not (tmp_path / "hardware.json").exists()


def test_failed_cuinit_is_rejected_without_success_json(tmp_path: Path) -> None:
    env = _fake_gpu_tools(tmp_path)
    env["FAKE_CUINIT_RC"] = "4"
    result = _run(tmp_path, env, "hardware.json")

    assert result.returncode != 0
    assert "cuInit(0) did NOT return CUDA_SUCCESS" in result.stdout
    assert not (tmp_path / "hardware.json").exists()
