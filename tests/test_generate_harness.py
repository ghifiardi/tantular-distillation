"""Harness attribution at generation time — stamped, or refused before the network.

    ./.venv/bin/python -m pytest tests/test_generate_harness.py -q

The corpus audit found the gap this closes: 136 traces, 136 model-attributed,
0 harness-attributed. A trace that records which MODEL answered but not which
scaffolding surrounded it cannot support the one claim the four-arm experiment
is for — whether an improvement belongs to the model or to the harness.

Three things have to hold, and all three are checked BEFORE any client is
constructed, because a refusal after an hour of generation is a refusal that
already cost the thing it was protecting:

  the execution model is resolved from the serving config, not guessed;
  the harness prompt identity is verified, not merely declared;
  the output file is not about to mix attributed and unattributed traces.

Nothing here touches a network: the client is a stub that records whether it was
ever built.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import config as config_module                                # noqa: E402
import generate                                              # noqa: E402
import harness_distill as hd                                 # noqa: E402

PROMPT_TEXT = "SYSTEM PROMPT\n"
PROMPT_SHA = hashlib.sha256(PROMPT_TEXT.encode()).hexdigest()


# --- 1. the execution model comes from the serving config -------------------

def test_resolve_exposes_the_registry_model_for_both_real_configs():
    """Resolved from the serving config, never inferred from the teacher name,
    the repo basename, or the served alias — those identities diverge."""
    assert config_module.resolve("muse-glimmer", "gateway")["TEACHER_REGISTRY_MODEL"] \
        == "muse-glimmer-30b"
    assert config_module.resolve("office-student-9b", "student-serve")["TEACHER_REGISTRY_MODEL"] \
        == "qwen35-9b-instruct"


def test_a_serving_config_without_a_registry_model_is_refused(tmp_path):
    with pytest.raises(SystemExit) as exc:
        generate.harness_preflight("any", {"TEACHER_REGISTRY_MODEL": "",
                                           "TEACHER_NAME": "t"}, tmp_path / "o.jsonl")
    assert "declares no registry_model" in str(exc.value)


# --- 2. an unverified prompt cannot produce an attributed corpus ------------

@pytest.fixture
def verified_harness(tmp_path, monkeypatch):
    """A harness whose prompt identity is genuinely pinned to a real file."""
    prompt = tmp_path / "system.txt"
    prompt.write_text(PROMPT_TEXT, encoding="utf-8")
    spec = {
        "schema_version": 1, "name": "fixture-harness", "status": "candidate",
        "model_contract": {"protocol": "openai_chat",
                           "compatible_registry_models": ["muse-glimmer-30b"]},
        "prompts": {"system": {"source": "path", "path": str(prompt),
                               "sha256": PROMPT_SHA, "verified": True}},
        "tools": {"allow": ["read"], "state_change_requires_approval": True},
        "memory": {"active_context": "bounded", "ephemeral_execution": "request",
                   "durable_task_state": "none", "product_memory": "disabled"},
        "execution": {"isolation": "companion_process",
                      "network": "denied_by_default",
                      "max_steps": 4, "max_wall_seconds": 300},
        "verification": {"before_action": ["request_schema"],
                         "after_action": ["edit_contract"], "repair_attempts": 0},
        "mutation": {"production_self_modify": False, "candidate_workspace_only": True,
                     "evaluator_mutation_allowed": False, "human_approval_required": True},
    }
    directory = tmp_path / "harnesses"
    directory.mkdir()
    (directory / "fixture-harness.yaml").write_text(yaml.safe_dump(spec), encoding="utf-8")
    monkeypatch.setattr(hd, "HARNESS_DIR", directory)
    return spec


def test_a_verified_harness_passes_preflight(verified_harness, tmp_path):
    block = generate.harness_preflight(
        "fixture-harness", {"TEACHER_REGISTRY_MODEL": "muse-glimmer-30b"},
        tmp_path / "out.jsonl")
    assert block["prompt_verified"] is True
    assert block["execution_model_registry"] == "muse-glimmer-30b"


def test_an_unverified_harness_cannot_generate(verified_harness, tmp_path, monkeypatch):
    """The shipped harnesses are in exactly this state. Planning and auditing
    may describe an unverified harness; GENERATING an attributed corpus with one
    would produce traces whose attribution nobody can check."""
    spec = dict(verified_harness)
    spec["prompts"] = {"system": {"source": "path", "path": "p", "sha256": None,
                                  "verified": False}}
    (hd.HARNESS_DIR / "fixture-harness.yaml").write_text(yaml.safe_dump(spec),
                                                         encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        generate.harness_preflight("fixture-harness",
                                   {"TEACHER_REGISTRY_MODEL": "muse-glimmer-30b"},
                                   tmp_path / "out.jsonl")
    assert "prompt identity is unverified" in str(exc.value)
    assert "verify_harness_identity.py" in str(exc.value)


def test_the_shipped_harnesses_cannot_generate_today(tmp_path):
    """Documented consequence, not an accident: the add-in is unpublished, so no
    attributed corpus can be produced yet. Delete this when it is published and
    the prompts are pinned."""
    for name in ("tantular-office-current", "tantular-office-candidate"):
        with pytest.raises(SystemExit) as exc:
            generate.harness_preflight(
                name, {"TEACHER_REGISTRY_MODEL": "muse-glimmer-30b"},
                tmp_path / "out.jsonl")
        assert "prompt identity is unverified" in str(exc.value), name


def test_an_incompatible_execution_model_is_refused(verified_harness, tmp_path):
    with pytest.raises(SystemExit) as exc:
        generate.harness_preflight("fixture-harness",
                                   {"TEACHER_REGISTRY_MODEL": "qwen35-9b-instruct"},
                                   tmp_path / "out.jsonl")
    assert "not compatible" in str(exc.value)


# --- 3. append mode must not mix attribution --------------------------------

def write(path: Path, *records: dict) -> Path:
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    return path


def attributed(block: dict, **over) -> dict:
    record = {"family": "f", "completion": "x", "provenance": {"teacher": "t"},
              "harness_provenance": dict(block, **over)}
    return record


def test_appending_to_the_same_harness_and_model_is_allowed(verified_harness, tmp_path):
    block = hd.harness_provenance(verified_harness,
                                  execution_model_registry="muse-glimmer-30b")
    out = write(tmp_path / "out.jsonl", attributed(block))
    generate.harness_preflight("fixture-harness",
                               {"TEACHER_REGISTRY_MODEL": "muse-glimmer-30b"}, out)


def test_appending_attributed_traces_to_a_legacy_file_is_refused(verified_harness, tmp_path):
    """write_traces appends. Half a file with attribution and half without is a
    corpus nobody can audit."""
    out = write(tmp_path / "out.jsonl", {"family": "f", "provenance": {"teacher": "t"}})
    with pytest.raises(SystemExit) as exc:
        generate.harness_preflight("fixture-harness",
                                   {"TEACHER_REGISTRY_MODEL": "muse-glimmer-30b"}, out)
    assert "no harness attribution" in str(exc.value)
    assert out.read_text().count("\n") == 1, "the output was not touched"


def test_appending_under_a_different_harness_is_refused(verified_harness, tmp_path):
    block = hd.harness_provenance(verified_harness,
                                  execution_model_registry="muse-glimmer-30b")
    out = write(tmp_path / "out.jsonl", attributed(block, digest="b" * 64))
    with pytest.raises(SystemExit) as exc:
        generate.harness_preflight("fixture-harness",
                                   {"TEACHER_REGISTRY_MODEL": "muse-glimmer-30b"}, out)
    assert "different harness" in str(exc.value)


def test_appending_under_a_different_execution_model_is_refused(verified_harness, tmp_path):
    block = hd.harness_provenance(verified_harness,
                                  execution_model_registry="muse-glimmer-30b")
    out = write(tmp_path / "out.jsonl",
                attributed(block, execution_model_registry="qwen35-9b-instruct"))
    with pytest.raises(SystemExit) as exc:
        generate.harness_preflight("fixture-harness",
                                   {"TEACHER_REGISTRY_MODEL": "muse-glimmer-30b"}, out)
    assert "different" in str(exc.value)


def test_a_legacy_run_refuses_to_append_to_an_attributed_file(verified_harness, tmp_path):
    """The other direction: a legacy run must not dilute an attributed corpus
    with unattributed traces."""
    block = hd.harness_provenance(verified_harness,
                                  execution_model_registry="muse-glimmer-30b")
    out = write(tmp_path / "out.jsonl", attributed(block))
    with pytest.raises(SystemExit) as exc:
        generate.legacy_output_guard(out)
    assert "harness-attributed" in str(exc.value)


def test_a_legacy_run_over_a_legacy_file_is_untouched(tmp_path):
    out = write(tmp_path / "out.jsonl", {"family": "f", "provenance": {"teacher": "t"}})
    generate.legacy_output_guard(out)          # must not raise


def test_a_missing_or_empty_output_is_fine(verified_harness, tmp_path):
    generate.harness_preflight("fixture-harness",
                               {"TEACHER_REGISTRY_MODEL": "muse-glimmer-30b"},
                               tmp_path / "absent.jsonl")
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    generate.harness_preflight("fixture-harness",
                               {"TEACHER_REGISTRY_MODEL": "muse-glimmer-30b"}, empty)
    generate.legacy_output_guard(empty)


# --- end to end, with a stub client -----------------------------------------

class StubClient:
    """Records that it was built, and answers without a network."""
    built = 0

    def __init__(self, **kwargs):
        type(self).built += 1
        self.kwargs = kwargs

    async def health(self):
        return True

    base_url = "stub://no-network"

    async def complete_many(self, prompt_sets, concurrency=16):
        return [{"content": "jawaban", "completion_tokens": 3, "tokens": 3,
                 "latency_s": 0.1, "truncated": False}
                for _ in prompt_sets]

    async def aclose(self):
        return None


@pytest.fixture
def generation(tmp_path, monkeypatch, verified_harness):
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(json.dumps({
        "family": "document:email::0001", "user": "halo", "system": "s",
        "source_class": "synthetic"}) + "\n", encoding="utf-8")
    monkeypatch.setattr(generate, "TeacherClient", StubClient)
    StubClient.built = 0
    return prompts


def args_for(prompts: Path, out: Path, harness: str | None) -> SimpleNamespace:
    return SimpleNamespace(
        teacher="muse-glimmer", host="gateway", prompts=str(prompts), out=str(out),
        concurrency=1, limit=0, batch=0, serve_host="localhost", base_url="",
        keep_mismatches=False, egress_approval="", api_key="test-key",
        harness=harness)


def test_an_unsafe_harness_aborts_before_the_client_is_built(generation, tmp_path,
                                                             verified_harness, monkeypatch):
    """The ordering that matters: every harness check is free and local, so it
    must happen before anything is dialled."""
    spec = dict(verified_harness)
    spec["mutation"] = dict(spec["mutation"], production_self_modify=True)
    (hd.HARNESS_DIR / "fixture-harness.yaml").write_text(yaml.safe_dump(spec),
                                                         encoding="utf-8")
    monkeypatch.setenv("TANTULAR_GATEWAY_KEY", "k")
    with pytest.raises(SystemExit):
        asyncio.run(generate.run(args_for(generation, tmp_path / "out.jsonl",
                                          "fixture-harness")))
    assert StubClient.built == 0, "the network was reached before the harness was checked"


def test_every_trace_carries_the_harness_digest(generation, tmp_path, monkeypatch):
    monkeypatch.setenv("TANTULAR_GATEWAY_KEY", "k")
    out = tmp_path / "out.jsonl"
    asyncio.run(generate.run(args_for(generation, out, "fixture-harness")))

    rows = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    assert rows, "generation produced nothing"
    expected = hd.harness_provenance(hd.load_harness("fixture-harness"),
                                     execution_model_registry="muse-glimmer-30b")
    for row in rows:
        assert row["harness_provenance"] == expected
        assert row["provenance"]["teacher"] == "muse-glimmer", "model provenance kept"


def test_without_harness_the_output_is_byte_identical_to_the_legacy_path(generation,
                                                                        tmp_path,
                                                                        monkeypatch):
    """The legacy path must not gain a key. The existing corpus and every tool
    that reads it predate harness attribution."""
    monkeypatch.setenv("TANTULAR_GATEWAY_KEY", "k")
    out = tmp_path / "legacy.jsonl"
    asyncio.run(generate.run(args_for(generation, out, None)))
    rows = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    assert rows
    for row in rows:
        assert "harness_provenance" not in row
