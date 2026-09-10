"""Harness-aware mode is DECLARED at pass creation, never inferred.

    ./.venv/bin/python -m pytest tests/test_pass_manifest_harness.py -q

If a pass were treated as harness-aware just because its traces happen to carry
attribution, then a pass that LOST its attribution would read as a perfectly
valid legacy pass. The declaration is what makes that failure loud, so it is a
flag, and the four combinations of flag and content are checked explicitly.

No corpus and no network: every pass here is written into tmp_path.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PY_BIN = str(ROOT / ".venv" / "bin" / "python")
TOOL = str(ROOT / "src" / "pass_manifest.py")
sys.path.insert(0, str(ROOT / "src"))
import harness_distill as hd                                  # noqa: E402


def block(**over) -> dict:
    value = {
        "schema_version": 1, "name": "h", "status": "candidate",
        "digest": "a" * 64, "compatible_registry_models": ["m1"],
        "execution_model_registry": "m1",
        "prompt_sha256": "b" * 64, "prompt_verified": True,
        "tool_policy_digest": "c" * 64, "verification_policy_digest": "d" * 64,
    }
    value.update(over)
    return value


def make_pass(tmp_path: Path, *records: dict) -> Path:
    directory = tmp_path / "pass"
    directory.mkdir(exist_ok=True)
    (directory / "traces.r0.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    return directory


def run(directory: Path, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run([PY_BIN, TOOL, str(directory), *extra],
                          capture_output=True, text=True, cwd=ROOT)


LEGACY = {"family": "f1", "provenance": {"teacher": "t", "prompt_sha256": "p"}}


def attributed(**over) -> dict:
    return {"family": "f2", "provenance": {"teacher": "t", "prompt_sha256": "q"},
            "harness_provenance": block(**over)}


# --- the four rows of the table ---------------------------------------------

def test_no_flag_and_unattributed_traces_is_a_valid_legacy_pass(tmp_path):
    proc = run(make_pass(tmp_path, LEGACY, LEGACY))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    manifest = json.loads((tmp_path / "pass" / "MANIFEST.json").read_text())
    assert manifest["harness"] == {
        "required": False, "attributed": False, "name": None, "digest": None,
        "prompt_verified": False, "prompt_sha256": None,
        "execution_model_registry": None}


def test_no_flag_but_attributed_traces_is_refused(tmp_path):
    """The declaration must be deliberate. Accepting this silently is how a
    harness-aware corpus loses the fact that it is one."""
    proc = run(make_pass(tmp_path, attributed()))
    assert proc.returncode != 0
    assert "--harness-aware" in proc.stdout + proc.stderr


def test_the_flag_with_uniform_attribution_pins_the_harness(tmp_path):
    proc = run(make_pass(tmp_path, attributed(), attributed()), "--harness-aware")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    harness = json.loads((tmp_path / "pass" / "MANIFEST.json").read_text())["harness"]
    assert harness["required"] is True and harness["attributed"] is True
    assert harness["digest"] == "a" * 64
    assert harness["execution_model_registry"] == "m1"


@pytest.mark.parametrize("records,marker", [
    ([{"family": "f", "provenance": {}}, {"family": "g", "provenance": {},
                                          "harness_provenance": block()}],
     "not attributed"),
    ([{"family": "f", "harness_provenance": block()},
      {"family": "g", "harness_provenance": block(digest="e" * 64)}],
     "harness digests"),
    ([{"family": "f", "harness_provenance": block()},
      {"family": "g", "harness_provenance": block(execution_model_registry="m2")}],
     "execution models"),
    ([{"family": "f", "harness_provenance": block(prompt_verified=False,
                                                  prompt_sha256=None)}],
     "unverified"),
])
def test_the_flag_with_broken_attribution_is_refused(tmp_path, records, marker):
    proc = run(make_pass(tmp_path, *records), "--harness-aware")
    assert proc.returncode != 0
    assert marker in proc.stdout + proc.stderr


# --- the historical manifests are left alone --------------------------------

def test_checked_in_pass_manifests_are_not_rewritten_to_add_the_field():
    """Manifests written before harness attribution existed are historical
    records. Adding {required: false} to them would be inventing a declaration
    nobody made."""
    existing = sorted(ROOT.glob("data/**/MANIFEST.json"))
    if not existing:
        pytest.skip("no checked-in pass manifests in this checkout")
    for path in existing:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "harness" in payload:
            # If one does carry it, it must be a full canonical summary rather
            # than a half-written marker.
            assert set(payload["harness"]) == set(hd._LEGACY_SUMMARY), path
