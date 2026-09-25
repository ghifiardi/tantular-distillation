"""Safe Ollama /api/chat adapter for the gold evaluation runner.

Why this exists. The product model card says Qwen3.5 on Ollama must be driven
through /api/chat with `think: false`. Ollama's OpenAI-compatible
/v1/chat/completions ignores that switch: the model reasons first, spends the
budget, and can return an empty answer -- which would turn every 4B
measurement into a measurement of a thinking budget. So the runner must speak
/api/chat here, and it must never quietly take the /v1 route instead.

The contract, enforced on every generation request:

    think:  false     always
    stream: false     always
    options: the runner's fixed decoding, translated by name (max_tokens ->
             num_predict); an unknown key is refused rather than dropped

and on every reply:

    done must be true; the reply must name the requested model; a `thinking`
    channel is a refusal, not something to strip -- an answer produced after
    reasoning is not the behaviour being measured.

The transport is INJECTED. `HttpxTransport` is the only thing that opens a
socket, and it is constructed only by the runner's --real path. Tests use a
fake. Reads (GET /api/tags, POST /api/chat with an identical body) may be
retried on transport errors; nothing here retries onto a different route.

There is deliberately no fallback. `fallback_to_openai_compatible()` exists
only to refuse, so a caller that reaches for one gets a loud error instead of
a silently different protocol.

NO TRAINING. Every identity block carries training_authorized: false.
"""
from __future__ import annotations

import time
from typing import Any, Protocol

PROTOCOL = "ollama_chat"
CHAT_PATH = "/api/chat"
TAGS_PATH = "/api/tags"

# OpenAI-style decoding names the runner uses -> Ollama option names. Closed
# list: an option this table does not know is not passed through under a
# guessed name, because a silently dropped decoding setting changes what two
# measurements have in common.
OPTION_NAMES = {
    "temperature": "temperature",
    "top_p": "top_p",
    "max_tokens": "num_predict",
    "seed": "seed",
    "top_k": "top_k",
    "presence_penalty": "presence_penalty",
    "repeat_penalty": "repeat_penalty",
}


class OllamaAdapterError(Exception):
    """The adapter could not obtain an answer it can vouch for."""


class Transport(Protocol):
    def get(self, path: str) -> tuple[int, dict[str, Any]]: ...
    def post(self, path: str, body: dict[str, Any]) -> tuple[int, dict[str, Any]]: ...


class HttpxTransport:
    """The one real transport. Opens sockets; never used in tests."""

    def __init__(self, endpoint: str, *, timeout_s: float = 600.0):
        import httpx                                       # noqa: PLC0415
        self._httpx = httpx
        self.endpoint = endpoint
        self.timeout_s = timeout_s

    def get(self, path: str) -> tuple[int, dict[str, Any]]:
        response = self._httpx.get(f"{self.endpoint}{path}", timeout=30.0)
        return response.status_code, _json_or_empty(response)

    def post(self, path: str, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        response = self._httpx.post(f"{self.endpoint}{path}", json=body,
                                    timeout=self.timeout_s)
        return response.status_code, _json_or_empty(response)


def _json_or_empty(response: Any) -> dict[str, Any]:
    try:
        value = response.json()
    except ValueError:
        return {}
    return value if isinstance(value, dict) else {}


def translate_decoding(decoding: dict[str, Any]) -> dict[str, Any]:
    unknown = sorted(k for k in decoding if k not in OPTION_NAMES)
    if unknown:
        raise OllamaAdapterError(
            f"unknown decoding key(s) {unknown!r}; the adapter passes only "
            f"{sorted(OPTION_NAMES)} and refuses to guess an Ollama option name")
    return {OPTION_NAMES[k]: v for k, v in decoding.items() if v is not None}


class OllamaChatClient:
    """Drives one Ollama model through /api/chat, thinking off, no streaming."""

    kind = "real"
    protocol = PROTOCOL

    def __init__(self, endpoint: str, model_tag: str, *, transport: Transport | None = None,
                 max_retries: int = 3, retry_delay_s: float = 1.0):
        endpoint = str(endpoint).rstrip("/")
        if endpoint.endswith("/v1") or "/v1/" in endpoint:
            raise OllamaAdapterError(
                f"endpoint {endpoint!r} names the OpenAI-compatible /v1 path. This "
                "adapter speaks Ollama /api/chat and refuses to be pointed at "
                "/v1; use the OpenAI client explicitly if that is what you mean")
        if not str(model_tag).strip():
            raise OllamaAdapterError("a model tag is required")
        self.endpoint = endpoint
        self.model_tag = str(model_tag)
        self.transport: Transport = transport or HttpxTransport(endpoint)
        self.max_retries = max(1, int(max_retries))
        self.retry_delay_s = retry_delay_s

    # --- identity -----------------------------------------------------------

    def served_models(self) -> list[str]:
        status, data = self._get(TAGS_PATH)
        if status != 200:
            raise OllamaAdapterError(
                f"{self.endpoint}{TAGS_PATH} answered {status}; the served "
                "identity cannot be read")
        models = data.get("models")
        if not isinstance(models, list):
            raise OllamaAdapterError(f"{TAGS_PATH} returned no models list")
        names = [str(m.get("name")) for m in models
                 if isinstance(m, dict) and m.get("name")]
        return names

    def identity(self) -> dict[str, Any]:
        status, data = self._get(TAGS_PATH)
        if status != 200:
            raise OllamaAdapterError(
                f"{self.endpoint}{TAGS_PATH} answered {status}; the served "
                "identity cannot be read")
        models = [m for m in (data.get("models") or []) if isinstance(m, dict)]
        served = [str(m.get("name")) for m in models if m.get("name")]
        digests = {str(m.get("name")): m.get("digest") for m in models if m.get("name")}
        return {
            "kind": self.kind,
            "protocol": self.protocol,
            "produces_real_measurements": True,
            "served": served,
            # Manifest digests as the local store reports them. A pointer for a
            # later investigation, not an equivalence test (see
            # src/model_identity.py): they change across a push/pull.
            "served_digests": digests,
            "endpoint": self.endpoint,
            "model_tag": self.model_tag,
            "think": False,
            "stream": False,
            "training_authorized": False,
        }

    def bind_model(self, model_tag: str) -> None:
        """Generate against THIS served tag from now on.

        Called by the runner with the tag that matched the endpoint's served
        list, so a request never names a tag the endpoint would reject. The
        tag must be served: rebinding to an unserved name is refused.
        """
        tag = str(model_tag).strip()
        if not tag:
            raise OllamaAdapterError("a model tag is required")
        served = self.served_models()
        if tag not in served:
            raise OllamaAdapterError(
                f"{self.endpoint} does not serve {tag!r} (served: {served!r}); "
                "refusing to bind a tag generation would fail on")
        self.model_tag = tag

    def describe(self) -> dict[str, Any]:
        """What the adapter will send, without sending anything."""
        return {
            "kind": self.kind,
            "protocol": self.protocol,
            "endpoint": self.endpoint,
            "model_tag": self.model_tag,
            "chat_path": CHAT_PATH,
            "think": False,
            "stream": False,
            "fallback_to_v1": "refused",
            "training_authorized": False,
        }

    # --- generation ---------------------------------------------------------

    def chat(self, messages: list[dict[str, str]],
             decoding: dict[str, Any] | None = None) -> str:
        body = {
            "model": self.model_tag,
            "messages": [dict(m) for m in messages],
            "think": False,
            "stream": False,
            "options": translate_decoding(dict(decoding or {})),
        }
        status, data = self._post(CHAT_PATH, body)
        if status == 404:
            raise OllamaAdapterError(
                f"{self.endpoint}{CHAT_PATH} is not served (404). Falling back to "
                "/v1/chat/completions is refused: that route ignores think:false "
                "and would measure a thinking budget instead of the model")
        if status != 200:
            raise OllamaAdapterError(
                f"{CHAT_PATH} answered {status}: {str(data.get('error', ''))[:200]}")
        if data.get("done") is not True:
            raise OllamaAdapterError(
                "reply is not done; a partial generation is not an answer")
        if str(data.get("model", "")) != self.model_tag:
            raise OllamaAdapterError(
                f"reply names model {data.get('model')!r}, not the requested "
                f"{self.model_tag!r}; refusing an answer from another model")
        message = data.get("message")
        if not isinstance(message, dict):
            raise OllamaAdapterError("reply carries no message object")
        thinking = message.get("thinking")
        if thinking is not None and str(thinking).strip():
            raise OllamaAdapterError(
                "reply carries a thinking channel although think:false was sent; "
                "the server ignored the switch and the answer is refused, not "
                "stripped")
        content = message.get("content")
        if not isinstance(content, str):
            raise OllamaAdapterError("reply message.content is not a string")
        return content.strip()

    def answer(self, record: dict[str, Any], decoding: dict[str, Any],
               instruction: str = "") -> str:
        messages: list[dict[str, str]] = []
        if str(instruction).strip():
            messages.append({"role": "system", "content": str(instruction)})
        messages.append({"role": "user", "content": str(record["prompt"])})
        return self.chat(messages, decoding)

    def fallback_to_openai_compatible(self) -> None:
        raise OllamaAdapterError(
            "refused: this adapter has no /v1/chat/completions fallback. The "
            "OpenAI-compatible route ignores think:false; choose the client "
            "explicitly instead of downgrading")

    # --- transport with read-only retries ------------------------------------

    def _get(self, path: str) -> tuple[int, dict[str, Any]]:
        return self._attempt(lambda: self.transport.get(path), path)

    def _post(self, path: str, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        # The SAME body every attempt. A retry that changed the request would
        # be a different measurement wearing the first one's receipt.
        return self._attempt(lambda: self.transport.post(path, body), path)

    def _attempt(self, call: Any, path: str) -> tuple[int, dict[str, Any]]:
        last: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                return call()
            except (ConnectionError, TimeoutError, OSError) as error:
                last = error
            except Exception as error:                     # noqa: BLE001
                # httpx errors when the real transport is in play; still a
                # transport failure, never a reason to change route.
                if type(error).__module__.startswith("httpx"):
                    last = error
                else:
                    raise
            if attempt + 1 < self.max_retries and self.retry_delay_s > 0:
                time.sleep(self.retry_delay_s * (2 ** attempt))
        raise OllamaAdapterError(
            f"{path} failed after {self.max_retries} attempt(s): {last!r}")
