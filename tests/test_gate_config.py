"""One definition of "which gates", read by the freezer and the gate runner.

    ./.venv/bin/python -m pytest tests/test_gate_config.py -q

The Tinker experiment trains a different base checkpoint but must be judged
against the SAME held-out gates as the QLoRA run. It references them by path
plus digest rather than copying them, so the failure this module has to prevent
is the two readers — src/run_gates.py when measuring, src/freeze_training_run.py
when recording what was measured — quietly resolving different gate sets.

Everything here is tmp files and dicts. No model, no endpoint, no network.
"""
from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
PY = str(ROOT / ".venv" / "bin" / "python")
sys.path.insert(0, str(ROOT / "src"))
import gate_config                                           # noqa: E402

GATES = [
    {"name": "indonesian_voice", "min_pass_rate": 0.95, "model_dependent": True},
    {"name": "office_json_contract", "min_pass_rate": 0.98, "model_dependent": False},
]


def shared_config(tmp_path: Path, gates=GATES) -> Path:
    path = tmp_path / "shared.yaml"
    path.write_text(yaml.safe_dump({"base_model": "org/Model", "eval_gates": gates}),
                    encoding="utf-8")
    return path


def referring_config(tmp_path: Path, shared: Path, *, digest: str | None = None,
                     stops: list[str] | None = None) -> Path:
    if digest is None:
        digest = hashlib.sha256(shared.read_bytes()).hexdigest()
    evaluation = {"config": str(shared), "config_sha256": digest}
    if stops is not None:
        evaluation["stop_sequences"] = stops
    path = tmp_path / "referring.yaml"
    path.write_text(yaml.safe_dump({"base_model": "org/Model-Base",
                                    "evaluation": evaluation}), encoding="utf-8")
    return path


# --- the two shapes ---------------------------------------------------------

def test_a_config_that_declares_its_own_gates_is_returned_unchanged(tmp_path):
    direct = shared_config(tmp_path)
    effective, shared = gate_config.load_gate_config(direct)
    assert [g["name"] for g in effective["eval_gates"]] == [g["name"] for g in GATES]
    assert shared is None, "nothing was referenced, so nothing is recorded"


def test_a_digest_pinned_reference_resolves_to_the_shared_gates(tmp_path):
    shared = shared_config(tmp_path)
    effective, record = gate_config.load_gate_config(referring_config(tmp_path, shared))
    assert [g["name"] for g in effective["eval_gates"]] == [g["name"] for g in GATES]
    assert record == {"path": str(shared),
                      "sha256": hashlib.sha256(shared.read_bytes()).hexdigest()}
    # The referring config's own keys survive; only the gates come from elsewhere.
    assert effective["base_model"] == "org/Model-Base"


def test_the_referring_config_supplies_the_stop_sequences(tmp_path):
    """Stops belong to the ENDPOINT being measured — a Base checkpoint needs
    them, an instruct-tuned one does not — while the gates belong to the
    experiment. So they are inherited from the referrer, not the shared file."""
    shared = shared_config(tmp_path)
    effective, _ = gate_config.load_gate_config(
        referring_config(tmp_path, shared, stops=["\n\nUser:"]))
    assert all(g["stop_sequences"] == ["\n\nUser:"] for g in effective["eval_gates"])
    # and the shared file was not mutated on disk or in memory
    assert all("stop_sequences" not in g for g in
               yaml.safe_load(shared.read_text())["eval_gates"])


def test_no_stop_sequences_means_the_key_is_absent_not_empty(tmp_path):
    effective, _ = gate_config.load_gate_config(
        referring_config(tmp_path, shared_config(tmp_path)))
    assert all("stop_sequences" not in g for g in effective["eval_gates"])


# --- the refusals -----------------------------------------------------------

def test_a_stale_digest_refuses(tmp_path):
    """The digest is the whole point of referencing rather than copying: a
    change to the shared gates must be explicit, never silently inherited."""
    shared = shared_config(tmp_path)
    referring = referring_config(tmp_path, shared)
    shared.write_text(yaml.safe_dump({"eval_gates": [
        {"name": "indonesian_voice", "min_pass_rate": 0.50}]}), encoding="utf-8")

    with pytest.raises(gate_config.GateConfigError) as exc:
        gate_config.load_gate_config(referring)
    assert "STALE or changed" in str(exc.value)


def test_a_missing_shared_file_refuses(tmp_path):
    shared = shared_config(tmp_path)
    referring = referring_config(tmp_path, shared)
    shared.unlink()
    with pytest.raises(gate_config.GateConfigError) as exc:
        gate_config.load_gate_config(referring)
    assert "missing" in str(exc.value)


def test_a_shared_config_without_eval_gates_refuses(tmp_path):
    empty = tmp_path / "shared.yaml"
    empty.write_text(yaml.safe_dump({"base_model": "org/Model"}), encoding="utf-8")
    with pytest.raises(gate_config.GateConfigError) as exc:
        gate_config.load_gate_config(referring_config(tmp_path, empty))
    assert "declares no eval_gates" in str(exc.value)


def test_an_absent_digest_refuses_as_loudly_as_a_wrong_one(tmp_path):
    shared = shared_config(tmp_path)
    with pytest.raises(gate_config.GateConfigError):
        gate_config.load_gate_config(referring_config(tmp_path, shared, digest=""))


def test_a_missing_config_refuses(tmp_path):
    with pytest.raises(gate_config.GateConfigError):
        gate_config.load_gate_config(tmp_path / "nope.yaml")


def test_declared_gate_specs_drops_unnamed_entries(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump({"eval_gates": [
        {"name": "indonesian_voice"}, {"min_pass_rate": 0.9}, "not-a-mapping"]}),
        encoding="utf-8")
    assert [s["name"] for s in gate_config.declared_gate_specs(path)] == \
        ["indonesian_voice"]


# --- both consumers, one answer ---------------------------------------------

def test_the_runner_and_the_freezer_resolve_the_same_gates():
    """The integration this module exists for, against the real shipped
    configs: train/tinker_sft_9b.yaml references train/qlora_9b.yaml's gates."""
    import freeze_training_run
    import run_gates

    tinker = ROOT / "train" / "tinker_sft_9b.yaml"
    runner_effective, runner_shared = run_gates.load_gate_config(tinker)
    runner_names = [g["name"] for g in runner_effective["eval_gates"]]
    freezer_names = [s["name"] for s in freeze_training_run.declared_gate_specs(tinker)]

    assert runner_names == freezer_names
    assert runner_names == [g["name"] for g in yaml.safe_load(
        (ROOT / "train" / "qlora_9b.yaml").read_text())["eval_gates"]]
    assert runner_shared["sha256"] == hashlib.sha256(
        (ROOT / "train" / "qlora_9b.yaml").read_bytes()).hexdigest()


def test_the_freezer_does_not_import_the_gate_runner():
    """A freeze must not depend on a CLI module to answer a configuration
    question; that coupling is what this split removed."""
    source = (ROOT / "src" / "freeze_training_run.py").read_text(encoding="utf-8")
    assert "import run_gates" not in source
    assert "import gate_config" in source


@pytest.mark.parametrize("module,expected_exit,marker", [
    ("run_gates", 2, "GATE RUN ABORTED"),
    ("freeze_training_run", 1, "Refusing to freeze"),
])
def test_each_consumer_refuses_in_its_own_voice(tmp_path, module, expected_exit, marker):
    """Same underlying error, translated by each caller rather than leaking the
    other's exit convention."""
    shared = shared_config(tmp_path)
    referring = referring_config(tmp_path, shared, digest="0" * 64)
    call = {
        "run_gates": f"run_gates.load_gate_config(Path({str(referring)!r}))",
        "freeze_training_run":
            f"freeze_training_run.declared_gate_specs(Path({str(referring)!r}))",
    }[module]
    proc = subprocess.run(
        [PY, "-c", "import sys; from pathlib import Path; "
         f"sys.path.insert(0, {str(ROOT / 'src')!r}); import {module}; {call}"],
        capture_output=True, text=True, cwd=ROOT)
    assert proc.returncode == expected_exit, proc.stdout + proc.stderr
    assert marker in proc.stdout + proc.stderr
