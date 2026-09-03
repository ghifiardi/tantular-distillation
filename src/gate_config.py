"""Resolve which eval gates a training config declares — one definition, two readers.

The gate runner (src/run_gates.py) and the run freezer (src/freeze_training_run.py)
must agree about which gates exist, what they threshold on, and which stop
sequences they generate with. They read it at different moments — one when
measuring, one when recording what was measured — so a second, drifting copy of
that answer is the failure this module prevents.

TWO SHAPES, ONE ANSWER.

  direct        the config declares `eval_gates:` itself (train/qlora_9b.yaml).

  digest-pinned the config names another config's gates by reference
                (train/tinker_sft_9b.yaml: evaluation.config +
                evaluation.config_sha256). The Tinker experiment trains a
                different base checkpoint but must be judged against the SAME
                held-out gates; copying the definitions would let them drift
                silently, so they are referenced and the digest makes any change
                to them explicit and deliberate.

FAILS CLOSED, AND WITHOUT A CLI. Every refusal raises GateConfigError rather
than exiting, so each caller can translate it into its own abort — the runner
through fail(), the freezer through its own refusal — and neither has to import
the other. Importing a CLI module to answer a configuration question is how the
freezer came to depend on the gate runner's argument parsing.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

try:
    import yaml
except ImportError:  # keep the failure legible
    raise SystemExit("pyyaml is required: pip install -r requirements.txt")

ROOT = Path(__file__).resolve().parent.parent


class GateConfigError(Exception):
    """A gate definition could not be resolved. Never a warning: a config whose
    gates cannot be read must not be measured or frozen."""


def digest_file(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def load_gate_config(config: Path) -> tuple[dict, dict | None]:
    """Return (effective config, shared-config record or None).

    The second element is None when the gates were declared directly, and
    otherwise records the resolved path and the digest that was verified — which
    is what the freeze writes down, so a later reader can tell which definitions
    were in force.
    """
    config = Path(config)
    if not config.is_file():
        raise GateConfigError(f"config missing: {config}")
    cfg = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
    if not isinstance(cfg, dict):
        raise GateConfigError(f"config must be a mapping: {config}")

    if cfg.get("eval_gates"):
        return cfg, None

    evaluation = cfg.get("evaluation") or {}
    shared = evaluation.get("config")
    expected = evaluation.get("config_sha256")
    if not shared:
        # Neither direct gates nor a reference. Returned as-is so the caller can
        # give its own "declares no eval_gates" refusal in its own words.
        return cfg, None

    shared_path = Path(shared)
    if not shared_path.is_absolute():
        shared_path = ROOT / shared_path
    if not shared_path.is_file():
        raise GateConfigError(f"shared evaluation config missing: {shared_path}")

    actual = digest_file(shared_path)
    if not expected or actual != expected:
        raise GateConfigError(
            "shared evaluation config is STALE or changed.\n"
            f"  pinned  {expected}\n"
            f"  on disk {actual}\n"
            "Review the gate changes, update the digest, and regenerate the "
            "run freeze."
        )

    shared_cfg = yaml.safe_load(shared_path.read_text(encoding="utf-8")) or {}
    if not isinstance(shared_cfg, dict) or not shared_cfg.get("eval_gates"):
        raise GateConfigError(
            f"shared evaluation config declares no eval_gates: {shared_path}")

    # Stop sequences are inherited from the REFERRING config, not the shared one:
    # they belong to the endpoint being measured (a Base checkpoint needs them;
    # an instruct-tuned one does not), while the gates belong to the experiment.
    inherited_stops = evaluation.get("stop_sequences") or []
    effective = dict(cfg)
    effective["eval_gates"] = [
        {
            **gate,
            **({"stop_sequences": list(inherited_stops)} if inherited_stops else {}),
        }
        for gate in shared_cfg["eval_gates"]
    ]
    return effective, {"path": str(shared_path), "sha256": actual}


def declared_gate_specs(config: Path) -> list[dict]:
    """Just the named gate specs. Shared so the freezer and the runner cannot
    disagree about which gates a config declares."""
    effective, _shared = load_gate_config(config)
    return [spec for spec in (effective.get("eval_gates") or [])
            if isinstance(spec, dict) and spec.get("name")]
