"""Safe Ollama /api/chat adapter and the pinned 4B registry entry.

Everything here runs against an injected FAKE transport. No socket is opened,
no model answers, no gold prompt exists in this file. The adapter's contract:

* it posts to /api/chat, never to /v1/chat/completions;
* every generation body carries think:false and stream:false;
* decoding is translated into Ollama `options`;
* served identity comes from /api/tags and feeds the runner's existing
  served-identity check, which still refuses a mismatch;
* there is no path that falls back to /v1 when /api/chat is unavailable;
* a reasoning channel in the reply is refused, not stripped;
* the 4B registry entry loads through the runner's loader as a student.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import ollama_chat_client as oc                            # noqa: E402
import run_gold_evaluation as rge                          # noqa: E402

REGISTRY = ROOT / "configs" / "models" / "qwen35-4b-instruct.yaml"
GOLD_FIXTURE = ROOT / "tests" / "fixtures" / "gold" / "valid.synthetic.jsonl"
EXPERIMENT = ROOT / "configs" / "experiments" / "harness-before-weights.yaml"

TAG = "ghifidanukusumo/tantular:lite"
ENDPOINT = "http://127.0.0.1:11434"


class FakeTransport:
    """Records every request; answers from a small script. Never a socket."""

    def __init__(self, *, tags=None, chat=None, chat_status=200, tags_status=200):
        self.requests: list[tuple[str, str, dict | None]] = []
        self.tags = tags if tags is not None else [{"name": TAG, "digest": "b2b2"}]
        self.chat = chat if chat is not None else {
            "model": TAG, "done": True,
            "message": {"role": "assistant", "content": "Bandung"}}
        self.chat_status = chat_status
        self.tags_status = tags_status

    def get(self, path: str) -> tuple[int, dict]:
        self.requests.append(("GET", path, None))
        if path == "/api/tags":
            return self.tags_status, {"models": self.tags}
        return 404, {"error": f"no route {path}"}

    def post(self, path: str, body: dict) -> tuple[int, dict]:
        self.requests.append(("POST", path, json.loads(json.dumps(body))))
        if path == "/api/chat":
            return self.chat_status, self.chat
        if path == "/v1/chat/completions":
            # An OpenAI-compatible route that WOULD answer. The adapter must
            # never reach it.
            return 200, {"choices": [{"message": {"content": "leaked"}}]}
        return 404, {"error": f"no route {path}"}


def client(transport: FakeTransport | None = None, **kwargs) -> oc.OllamaChatClient:
    return oc.OllamaChatClient(ENDPOINT, TAG, transport=transport or FakeTransport(),
                               **kwargs)


# --- the wire ---------------------------------------------------------------


def test_generation_posts_to_api_chat_not_v1():
    transport = FakeTransport()
    client(transport).chat([{"role": "user", "content": "halo"}])
    posts = [(m, p) for m, p, _ in transport.requests if m == "POST"]
    assert posts == [("POST", "/api/chat")]
    assert not any("/v1" in p for _, p, _ in transport.requests)


def test_every_generation_body_disables_thinking_and_streaming():
    transport = FakeTransport()
    c = client(transport)
    c.chat([{"role": "user", "content": "satu"}])
    c.answer({"id": "x", "prompt": "dua"}, dict(rge.DECODING))
    bodies = [b for m, p, b in transport.requests if m == "POST" and p == "/api/chat"]
    assert len(bodies) == 2
    for body in bodies:
        assert body["think"] is False
        assert body["stream"] is False
        assert body["model"] == TAG


def test_decoding_is_translated_into_options():
    transport = FakeTransport()
    client(transport).answer({"id": "x", "prompt": "p"}, dict(rge.DECODING))
    body = [b for m, p, b in transport.requests if p == "/api/chat"][0]
    assert body["options"] == {
        "temperature": rge.DECODING["temperature"],
        "top_p": rge.DECODING["top_p"],
        "num_predict": rge.DECODING["max_tokens"],
        "seed": rge.DECODING["seed"],
    }
    assert "max_tokens" not in body and "max_tokens" not in body["options"]


def test_translate_decoding_refuses_an_unknown_key():
    with pytest.raises(oc.OllamaAdapterError, match="unknown"):
        oc.translate_decoding({"temperature": 0.0, "mirostat": 2})


def test_the_answer_is_the_assistant_content_string_only():
    transport = FakeTransport(chat={"done": True, "model": TAG,
                                    "message": {"role": "assistant",
                                                "content": "  jawaban  "}})
    assert client(transport).answer({"id": "x", "prompt": "p"}, {}) == "jawaban"


def test_a_reasoning_channel_in_the_reply_is_refused_not_stripped():
    transport = FakeTransport(chat={"done": True, "model": TAG,
                                    "message": {"role": "assistant",
                                                "content": "42",
                                                "thinking": "6 kali 7"}})
    with pytest.raises(oc.OllamaAdapterError, match="think"):
        client(transport).chat([{"role": "user", "content": "p"}])


def test_an_unfinished_reply_is_refused():
    transport = FakeTransport(chat={"done": False, "model": TAG,
                                    "message": {"role": "assistant", "content": "partial"}})
    with pytest.raises(oc.OllamaAdapterError, match="done"):
        client(transport).chat([{"role": "user", "content": "p"}])


def test_a_reply_from_another_model_is_refused():
    transport = FakeTransport(chat={"done": True, "model": "qwen3.5:4b",
                                    "message": {"role": "assistant", "content": "x"}})
    with pytest.raises(oc.OllamaAdapterError, match="model"):
        client(transport).chat([{"role": "user", "content": "p"}])


# --- no fallback ------------------------------------------------------------


def test_a_missing_api_chat_route_never_falls_back_to_v1():
    transport = FakeTransport(chat_status=404)
    with pytest.raises(oc.OllamaAdapterError) as error:
        client(transport).chat([{"role": "user", "content": "p"}])
    assert "/v1/chat/completions" in str(error.value)
    assert "refus" in str(error.value).lower()
    assert not any("/v1" in p for _, p, _ in transport.requests)


def test_an_endpoint_that_names_the_v1_path_is_refused_at_construction():
    with pytest.raises(oc.OllamaAdapterError, match="/v1"):
        oc.OllamaChatClient("http://127.0.0.1:11434/v1", TAG, transport=FakeTransport())


def test_the_adapter_source_has_no_openai_route_in_code():
    """The /v1 route may be NAMED in prose and refusal messages; it may never
    be a request target. Every request goes through transport.get/post, so a
    call site naming /v1 is the thing to look for."""
    source = (ROOT / "src" / "ollama_chat_client.py").read_text("utf-8")
    assert oc.CHAT_PATH == "/api/chat" and oc.TAGS_PATH == "/api/tags"
    for line in source.splitlines():
        stripped = line.strip()
        if ("transport.get(" in stripped or "transport.post(" in stripped
                or "_get(" in stripped or "_post(" in stripped):
            assert "/v1" not in stripped, line
        if "/v1/chat/completions" in stripped:
            assert not stripped.startswith(("return", "path", "CHAT_PATH", "TAGS_PATH")), line


def test_an_explicit_fallback_request_is_refused():
    with pytest.raises(oc.OllamaAdapterError, match="refus"):
        client().fallback_to_openai_compatible()


def test_a_transport_error_on_generation_is_retried_with_the_same_body_only():
    class Flaky(FakeTransport):
        def __init__(self):
            super().__init__()
            self.failures = 1

        def post(self, path, body):
            if path == "/api/chat" and self.failures:
                self.failures -= 1
                self.requests.append(("POST", path, json.loads(json.dumps(body))))
                raise ConnectionError("simulated transport failure")
            return super().post(path, body)

    transport = Flaky()
    assert client(transport, max_retries=2).chat([{"role": "user", "content": "p"}]) == "Bandung"
    bodies = [b for m, p, b in transport.requests if p == "/api/chat"]
    assert len(bodies) == 2 and bodies[0] == bodies[1]


# --- identity ---------------------------------------------------------------


def test_served_models_come_from_api_tags():
    transport = FakeTransport(tags=[{"name": TAG, "digest": "b2b2"},
                                    {"name": "tantular-office:lite", "digest": "b2b2"}])
    c = client(transport)
    assert c.served_models() == [TAG, "tantular-office:lite"]
    identity = c.identity()
    assert identity["kind"] == "real"
    assert identity["produces_real_measurements"] is True
    assert identity["endpoint"] == ENDPOINT
    assert identity["served"] == [TAG, "tantular-office:lite"]
    assert identity["protocol"] == "ollama_chat"
    assert ("GET", "/api/tags", None) in transport.requests


def test_the_served_identity_check_accepts_the_declared_tag_and_rejects_others():
    model = rge.resolve_model(ROOT / "configs" / "models", "qwen35-4b-instruct")
    rge.check_served(model, client().identity())
    other = FakeTransport(tags=[{"name": "qwen3.5:4b", "digest": "x"}])
    with pytest.raises(rge.GoldEvaluationError, match="served"):
        rge.check_served(model, client(other).identity())


def test_an_endpoint_that_serves_nothing_is_refused():
    with pytest.raises(rge.GoldEvaluationError, match="served"):
        rge.check_served(rge.resolve_model(ROOT / "configs" / "models", "qwen35-4b-instruct"),
                         client(FakeTransport(tags=[])).identity())


# --- the registry entry -----------------------------------------------------


def test_the_4b_registry_entry_loads_as_a_student_with_a_pinned_revision():
    model = rge.resolve_model(ROOT / "configs" / "models", "qwen35-4b-instruct")
    assert model["model_id"] == "Qwen/Qwen3.5-4B"
    assert model["revision"] == "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
    assert model["serving"]["protocol"] == "ollama_chat"
    assert TAG in model["serving"]["tags"]


def test_the_4b_registry_entry_records_the_verified_identities():
    spec = yaml.safe_load(REGISTRY.read_text("utf-8"))
    assert spec["role"] == "student"
    assert spec["family"] == "qwen3.5" and spec["generation"] == "3.5"
    assert spec["params"]["total_b"] == 4.7
    assert spec["chat_template"]["sha256"] == \
        "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715"
    files = spec["tokenizer"]["files_sha256"]
    assert files["tokenizer.json"] == \
        "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"
    assert files["tokenizer_config.json"] == \
        "316230d6a809701f4db5ea8f8fc862bc3a6f3229c937c174e674ff3ca0a64ac8"
    ollama = spec["serving"]["ollama"]
    assert ollama["weights_blob"] == \
        "sha256-81fb60c7daa80fc1123380b98970b320ae233409f0f71a72ed7b9b0d62f40490"
    assert ollama["canonical_profile_sha256"] == \
        "15368614046c19ce4e63e2b5506507bfbd9d8e3bd3600571d9ad27af3c43f11b"
    assert spec["license"]["evidence_sha256"] == \
        "bbedc3fda3305820b977265f01b8619d87570a6739de3a5582c3464840f1e57a"
    # This composite was measured by verify_model_identity from the pinned
    # provenance-bearing Hugging Face cache snapshot; it is not hand-typed.
    assert spec["tokenizer"]["sha256"] == \
        "6f3a76fa0ff84cba487813d4024623233c4664ecedfc3f3857536f95d25504af"
    assert spec["digests_verified"] is True


def test_a_teacher_role_entry_is_still_refused_in_a_student_arm():
    with pytest.raises(rge.GoldEvaluationError, match="teacher"):
        rge.resolve_model(ROOT / "tests" / "fixtures" / "gold_evaluation" / "models",
                          "teacher-fixture")


# --- runner wiring ----------------------------------------------------------


def test_a_registry_entry_declaring_ollama_refuses_the_openai_client():
    model = rge.resolve_model(ROOT / "configs" / "models", "qwen35-4b-instruct")
    with pytest.raises(rge.GoldEvaluationError, match="ollama"):
        rge.check_client_protocol(model, "openai")
    rge.check_client_protocol(model, "ollama")


def test_the_9b_entry_without_a_serving_block_still_accepts_the_openai_client():
    model = rge.resolve_model(ROOT / "configs" / "models", "qwen35-9b-instruct")
    rge.check_client_protocol(model, "openai")


def test_a_fake_run_through_the_ollama_adapter_produces_receipts(tmp_path):
    """The adapter, the runner and the verifier layer, end to end, with the
    transport faked. Proves the wiring without any model."""
    gold = tmp_path / "gold"
    gold.mkdir()
    shutil.copy(GOLD_FIXTURE, gold / "valid.synthetic.jsonl")
    transport = FakeTransport(chat={"done": True, "model": TAG,
                                    "message": {"role": "assistant", "content": "tidak"}})
    fake_real = oc.OllamaChatClient(ENDPOINT, TAG, transport=transport)
    # A real-kind client cannot run in fixture posture: the runner refuses.
    with pytest.raises(rge.GoldEvaluationError, match="fixture"):
        rge.run_evaluation(
            gold_dir=gold, model_registry="qwen35-4b-instruct",
            model_dir=ROOT / "configs" / "models", client=fake_real, repetitions=2,
            output=tmp_path / "out", run_id="run-1", allow_fixture=True,
            experiment_path=EXPERIMENT)
    assert not any(p == "/api/chat" for _, p, _ in transport.requests), \
        "no generation may happen before the posture check"


def test_plan_for_the_4b_on_the_empty_production_set_is_blocked():
    plan = rge.plan_evaluation(
        gold_dir=ROOT / "data" / "gold", model_registry="qwen35-4b-instruct",
        model_dir=ROOT / "configs" / "models", endpoint=ENDPOINT, repetitions=2,
        experiment_path=EXPERIMENT, allow_fixture=False)
    assert plan["executable"] is False
    assert "gold set is BLOCKED: 0 approved items" in plan["blockers"]
    assert plan["model_identity"]["expected"] == "Qwen/Qwen3.5-4B"
    assert plan["serving"]["protocol"] == "ollama_chat"
    assert plan["training_authorized"] is False


def test_every_artifact_from_the_adapter_says_training_authorized_false():
    c = client()
    assert c.identity()["training_authorized"] is False
    assert c.describe()["training_authorized"] is False
