"""OpenAI-compatible client for a locally-served teacher.

Deliberately thin: vLLM, Ollama, and a LiteLLM gateway all speak the same
/v1/chat/completions, so a run does not care which one is behind the URL.
That is what makes host-agnostic work in practice.

Mirrors the interface of tantular/finetune/bridge_client.py so generated
traces drop straight into the existing judge.py / dedup.py / review_promote.py
pipeline rather than needing a parallel one.
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field

try:
    import httpx
except ImportError:  # pragma: no cover
    raise SystemExit("httpx is required: pip install -r requirements.txt")


@dataclass
class TeacherClient:
    base_url: str
    model: str
    api_key: str = ""          # local vLLM needs none; a gateway does
    timeout_s: float = 600.0   # a 30B on PCIe-split cards is not fast
    max_retries: int = 3
    sampling: dict = field(default_factory=dict)

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def complete(self, client: httpx.AsyncClient, messages: list[dict]) -> dict:
        """Returns {content, completion_tokens, truncated}.

        `truncated` is inferred from token count, not finish_reason: the
        gateway reports "stop" even when generation clearly ran out of budget
        mid-object, so finish_reason cannot be trusted here.
        """
        body = {
            "model": self.model,
            "messages": messages,
            **{k: v for k, v in self.sampling.items() if v is not None},
        }
        last_error = None
        for attempt in range(self.max_retries):
            started = time.perf_counter()
            try:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=self._headers(),
                    json=body,
                    timeout=self.timeout_s,
                )
                if response.status_code == 200:
                    data = response.json()
                    used = (data.get("usage") or {}).get("completion_tokens", 0)
                    budget = body.get("max_tokens", 0)
                    return {
                        "content": (data["choices"][0]["message"]["content"] or "").strip(),
                        "completion_tokens": used,
                        "latency_s": round(time.perf_counter() - started, 3),
                        # Within a few tokens of the ceiling means the model was
                        # still going when the budget ran out.
                        "truncated": bool(budget and used >= budget - 2),
                    }
                # A model-access 403 or a bad-key 401 will never succeed on
                # retry — fail immediately rather than burning the budget.
                if response.status_code in (401, 403):
                    raise RuntimeError(
                        f"teacher rejected the request ({response.status_code}): "
                        f"{response.text[:200]}"
                    )
                last_error = f"HTTP {response.status_code}: {response.text[:200]}"
            except (httpx.TimeoutException, httpx.TransportError) as error:
                last_error = repr(error)
            await asyncio.sleep(2 ** attempt)
        raise RuntimeError(f"teacher failed after {self.max_retries} attempts: {last_error}")

    async def complete_many(
        self,
        prompt_sets: list[list[dict]],
        concurrency: int = 16,
    ) -> list[dict | None]:
        """Run prompts concurrently. A failed prompt yields None rather than
        aborting the batch — a corpus run should not lose four hours of work
        to one bad generation."""
        results: list[dict | None] = [None] * len(prompt_sets)
        semaphore = asyncio.Semaphore(concurrency)

        async with httpx.AsyncClient() as client:
            async def worker(index: int, messages: list[dict]) -> None:
                async with semaphore:
                    try:
                        results[index] = await self.complete(client, messages)
                    except RuntimeError as error:
                        print(f"  [{index}] {error}")

            await asyncio.gather(
                *(worker(i, m) for i, m in enumerate(prompt_sets))
            )
        return results

    async def health(self) -> bool:
        async with httpx.AsyncClient() as client:
            try:
                # Must carry auth: a gateway answers an unauthenticated /models
                # with 401, which would look identical to "nothing is running".
                response = await client.get(
                    f"{self.base_url}/models", headers=self._headers(), timeout=15.0
                )
                return response.status_code == 200
            except httpx.TransportError:
                return False


def write_traces(path, records: list[dict]) -> int:
    """Append JSONL. Provenance fields stay on every record so a promoted
    training example can always be traced back to the teacher, host, and
    quantization that produced it — which is the first thing you want when a
    fine-tune regresses and you need to know what fed it."""
    written = 0
    with open(path, "a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
    return written
