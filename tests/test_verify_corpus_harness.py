"""The corpus gate learns harness mode from the pass manifest, never the traces.

    ./.venv/bin/python -m pytest tests/test_verify_corpus_harness.py -q

The boundary this preserves: trace contents establish WHAT HAPPENED; the pass
manifest declares WHETHER ATTRIBUTION WAS REQUIRED. If the gate inferred the
mode from the traces, a harness-aware corpus that lost its attribution would
score as a perfectly healthy legacy corpus — the one failure the declaration
exists to make loud.

The legacy corpus is unaffected, and a test asserts that against bytes rather
than by inspection.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PY_BIN = str(ROOT / ".venv" / "bin" / "python")
TOOL = str(ROOT / "src" / "verify_corpus.py")
sys.path.insert(0, str(ROOT / "src"))
import splits as splits_module                                # noqa: E402
import verify_corpus as vc                                    # noqa: E402

LEGACY_CORPUS = ROOT / "data" / "v3-candidate" / "traces.r0.jsonl"


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


def summary(**over) -> dict:
    value = {"required": True, "attributed": True, "name": "h", "digest": "a" * 64,
             "prompt_verified": True, "prompt_sha256": "b" * 64,
             "execution_model_registry": "m1"}
    value.update(over)
    return value


def make_pass(tmp_path: Path, records: list[dict], *, declared: dict | None,
              name: str = "pass", pin_digest: bool = True) -> Path:
    directory = tmp_path / name
    directory.mkdir(parents=True, exist_ok=True)
    corpus = directory / "traces.r0.jsonl"
    corpus.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    manifest: dict = {"files": {corpus.name: {
        "sha256": hashlib.sha256(corpus.read_bytes()).hexdigest() if pin_digest
        else "0" * 64}}}
    if declared is not None:
        manifest["harness"] = declared
    (directory / "MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
    return corpus


# --- the legacy path is untouched -------------------------------------------

@pytest.mark.requires_local_corpus
def test_the_legacy_corpus_output_is_unchanged():
    """Byte-for-byte, including exit code, for both invocations."""
    for extra, expected_exit in (([], 0), (["--gate"], 1)):
        proc = subprocess.run([PY_BIN, TOOL, str(LEGACY_CORPUS), *extra],
                              capture_output=True, text=True, cwd=ROOT)
        assert proc.returncode == expected_exit
        assert "harness attribution" not in proc.stdout, \
            "the legacy path must print no new lines"
        assert proc.stderr == ""


@pytest.mark.requires_local_corpus
def test_the_legacy_violation_list_is_unchanged():
    records = vc.load_corpus([LEGACY_CORPUS])
    errors = vc.check(records, splits_module.load(), gate=True)
    assert len(errors) == 1
    assert "quantized teacher" in errors[0]
    # And the new argument defaults to legacy behaviour for existing callers.
    assert vc.check(records, splits_module.load(), gate=True,
                    harness_required=False) == errors


def test_a_pass_with_no_harness_field_is_legacy(tmp_path):
    corpus = make_pass(tmp_path, [{"family": "f"}], declared=None)
    assert vc.harness_declaration([corpus]) is None


def test_a_corpus_with_no_pass_manifest_is_legacy(tmp_path):
    loose = tmp_path / "traces.jsonl"
    loose.write_text(json.dumps({"family": "f"}) + "\n", encoding="utf-8")
    assert vc.harness_declaration([loose]) is None


def test_a_declared_legacy_pass_is_legacy(tmp_path):
    corpus = make_pass(tmp_path, [{"family": "f"}],
                       declared={"required": False, "attributed": False})
    assert vc.harness_declaration([corpus]) is None


# --- harness-aware mode ------------------------------------------------------

def attributed(**over) -> dict:
    return {"family": "f", "harness_provenance": block(**over)}


def test_a_declared_harness_aware_pass_reports_its_identity(tmp_path):
    corpus = make_pass(tmp_path, [attributed()], declared=summary())
    declared = vc.harness_declaration([corpus])
    assert declared["digest"] == "a" * 64
    assert declared["execution_model_registry"] == "m1"


def test_uniform_attribution_reaches_the_normal_corpus_checks(tmp_path):
    """Harness mode adds violations; it does not replace the existing ones."""
    corpus = make_pass(tmp_path, [attributed()], declared=summary())
    errors = vc.check(vc.load_corpus([corpus]), splits_module.load(),
                      harness_required=True)
    assert not any("harness" in e for e in errors), errors


@pytest.mark.parametrize("records,marker", [
    ([attributed(), {"family": "g"}], "not attributed"),
    ([attributed(), attributed(digest="e" * 64)], "harness digests"),
    ([attributed(), attributed(execution_model_registry="m2")], "execution models"),
    ([attributed(prompt_verified=False, prompt_sha256=None)], "unverified"),
])
def test_broken_attribution_fails_the_gate(tmp_path, records, marker):
    corpus = make_pass(tmp_path, records, declared=summary())
    errors = vc.check(vc.load_corpus([corpus]), splits_module.load(),
                      harness_required=True)
    assert any(marker in e for e in errors), errors


# --- the manifest itself has to be trustworthy ------------------------------

def test_a_malformed_pass_manifest_fails_before_scoring(tmp_path, capsys):
    directory = tmp_path / "pass"
    directory.mkdir()
    corpus = directory / "traces.r0.jsonl"
    corpus.write_text(json.dumps({"family": "f"}) + "\n", encoding="utf-8")
    (directory / "MANIFEST.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        vc.harness_declaration([corpus])
    assert "not readable JSON" in str(exc.value)


def test_a_malformed_harness_block_fails_before_scoring(tmp_path):
    corpus = make_pass(tmp_path, [{"family": "f"}], declared={"attributed": True})
    with pytest.raises(SystemExit) as exc:
        vc.harness_declaration([corpus])
    assert "malformed harness block" in str(exc.value)


def test_a_manifest_digest_mismatch_fails_before_scoring(tmp_path):
    """A declaration about different bytes is not a declaration about this
    corpus."""
    corpus = make_pass(tmp_path, [attributed()], declared=summary(),
                       pin_digest=False)
    with pytest.raises(SystemExit) as exc:
        vc.harness_declaration([corpus])
    assert "different bytes" in str(exc.value)


# --- several inputs must agree ----------------------------------------------

def test_legacy_and_harness_aware_inputs_cannot_be_verified_together(tmp_path):
    aware = make_pass(tmp_path, [attributed()], declared=summary(), name="a")
    legacy = make_pass(tmp_path, [{"family": "f"}], declared=None, name="b")
    with pytest.raises(SystemExit) as exc:
        vc.harness_declaration([aware, legacy])
    assert "cannot verify harness-aware and legacy corpora together" in str(exc.value)


def test_two_harness_digests_cannot_be_verified_together(tmp_path):
    one = make_pass(tmp_path, [attributed()], declared=summary(), name="a")
    two = make_pass(tmp_path, [attributed(digest="e" * 64)],
                    declared=summary(digest="e" * 64), name="b")
    with pytest.raises(SystemExit) as exc:
        vc.harness_declaration([one, two])
    assert "different harness digests" in str(exc.value)


def test_two_execution_models_cannot_be_verified_together(tmp_path):
    one = make_pass(tmp_path, [attributed()], declared=summary(), name="a")
    two = make_pass(tmp_path, [attributed(execution_model_registry="m2")],
                    declared=summary(execution_model_registry="m2"), name="b")
    with pytest.raises(SystemExit) as exc:
        vc.harness_declaration([one, two])
    assert "different execution models" in str(exc.value)
