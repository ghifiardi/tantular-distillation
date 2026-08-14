"""Normalized generation for the precision comparison — identical bytes, both arms.

    ./.venv/bin/python src/generate_normalized.py --host ai19-ollama \
        --prompts prompts/calibration.jsonl \
        --out data/calibration/int4-normalized/traces.jsonl

Separate from generate.py on purpose. generate.py talks to a chat endpoint,
which is what the product does and therefore what the DEPLOYMENT baseline must
measure. This path exists so the two calibration arms differ ONLY in
quantization and kernels:

  render (official harmony template) -> hash -> raw completion -> parse channels

Everything a server could otherwise default is set explicitly: reasoning
strength, temperature, seed, token budget, model revision, quantization.

Channel parsing is strict. The model emits reasoning on a `to=self` channel and
its answer after `<|start|>assistant to=user<|message|>`. A response with no
final channel is MALFORMED and fails validation — silently keeping the
reasoning text as if it were an answer would poison a corpus with the model's
private deliberation.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from pathlib import Path

try:
    import httpx
    import jinja2
except ImportError:
    sys.exit("run with ./.venv/bin/python (needs httpx + jinja2)")

import splits as splits_module
from config import resolve

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_PATH = ROOT / "calibration" / "parity" / "chat_template.jinja"

# The model's answer channel. Everything before it is private reasoning.
# Prompt-set fields that must survive into every trace. A field missing here
# does not fail loudly — it disables whatever downstream check reads it, while
# that check keeps reporting success.
CARRY_FROM_PROMPT = ("source_class", "corpus_role", "source_sha256")

FINAL_MARKER = "<|start|>assistant to=user<|message|>"
STOP_TOKENS = ("<|eot|>", "<|end|>", "<|return|>")

# Optional protocol stop sequences — OFF BY DEFAULT, and deliberately so.
#
# The hypothesis that raw mode ignores stops and runs to the token budget was
# REFUTED: the diagnostic run returned 129-2024 tokens (median 388), none at
# the 4096 ceiling, done_reason "stop" throughout. Generation terminates on EOS
# without help.
#
# Worse, forcing these would be actively risky. `<|eot|>` is a channel
# delimiter as well as a terminator, so stopping at the first occurrence could
# cut generation off BEFORE the final answer channel — turning a good answer
# into a malformed trace. The no-stops configuration is the one proven to
# reproduce production output exactly (52/52 identical).
#
# Kept available because a different runtime may not honour EOS the same way.
# Whatever is chosen must be identical on both arms.
DEFAULT_STOPS: list[str] = []


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_template() -> tuple[jinja2.Template, str]:
    if not TEMPLATE_PATH.exists():
        sys.exit(f"no chat template at {TEMPLATE_PATH}")
    source = TEMPLATE_PATH.read_text(encoding="utf-8")
    env = jinja2.Environment()
    env.globals["raise_exception"] = lambda m: (_ for _ in ()).throw(Exception(m))
    return env.from_string(source), sha256(source)


def render(template: jinja2.Template, system: str, user: str, reasoning_strength: str) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    return template.render(messages=messages, add_generation_prompt=True,
                           reasoning_strength=reasoning_strength)


def classify_termination(raw_text: str, done_reason) -> dict:
    """Derive why generation stopped, conservatively.

    Runtimes disagree on vocabulary and precision. Ollama and vLLM both report
    a bare "stop" for several distinct causes — a stop-sequence hit, an emitted
    EOS token, or a template-derived stop — and both usually strip the matched
    text, so the output itself rarely proves which happened.

    So: report the runtime's own word verbatim, record whether the text
    visibly ends at a known marker as separate evidence, and only derive a
    specific cause where the evidence supports it. A bare "stop" becomes
    `ambiguous`, not an unsupported claim that our stop list fired.
    """
    reason = str(done_reason or "").lower()
    text = raw_text.rstrip()
    ended_at = next((m for m in STOP_TOKENS if text.endswith(m)), None)

    if reason in ("length", "max_tokens"):
        derived = "length"
    elif reason in ("error", "failed"):
        derived = "error"
    elif reason in ("eos", "eos_token"):
        derived = "eos"
    elif reason == "stop":
        # The runtime did not say which stop fired. If the marker is still
        # visible it was emitted rather than matched-and-stripped, which is
        # evidence of EOS — but not proof, so stay ambiguous either way and
        # let ended_at_marker carry the evidence.
        derived = "ambiguous"
    elif not reason:
        derived = "ambiguous"
    else:
        derived = reason
    return {"terminated_by": derived, "ended_at_marker": ended_at}


def parse_channels(raw: str) -> dict:
    """Split raw harmony output into reasoning and final answer.

    Returns {"answer", "reasoning", "malformed"}. A missing final channel is
    malformed rather than "the answer is the reasoning".
    """
    if FINAL_MARKER not in raw:
        return {"answer": "", "reasoning": raw.strip(), "malformed": True}
    reasoning, _, answer = raw.rpartition(FINAL_MARKER)
    for token in STOP_TOKENS:
        answer = answer.split(token)[0]
    return {"answer": answer.strip(),
            "reasoning": reasoning.strip(),
            "malformed": not answer.strip()}


async def complete(client: httpx.AsyncClient, base: str, model: str, prompt: str,
                   *, temperature: float, seed: int, max_tokens: int,
                   runtime: str, timeout: float, stops: list[str]) -> dict:
    if runtime == "ollama":
        response = await client.post(
            f"{base}/api/generate",
            json={"model": model, "prompt": prompt, "raw": True, "stream": False,
                  "options": {"temperature": temperature, "seed": seed,
                              "num_predict": max_tokens,
                              **({"stop": stops} if stops else {})}},
            timeout=timeout)
        response.raise_for_status()
        payload = response.json()
        return {"raw": payload.get("response") or "",
                "completion_tokens": payload.get("eval_count"),
                "done_reason": payload.get("done_reason"),
                # Ollama exposes no per-stop attribution. Recorded as absent so
                # a reader can see the evidence is missing rather than assume
                # it was checked.
                "stop_reason": "__absent__"}
    # vLLM and anything else OpenAI-compatible: the completions path, not chat,
    # so no server-side template is applied to our already-rendered prompt.
    response = await client.post(
        f"{base}/v1/completions",
        json={"model": model, "prompt": prompt, "temperature": temperature,
              "seed": seed, "max_tokens": max_tokens,
              **({"stop": stops} if stops else {})},
        timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    choice = payload["choices"][0]
    return {"raw": choice.get("text") or "",
            "completion_tokens": (payload.get("usage") or {}).get("completion_tokens"),
            "done_reason": choice.get("finish_reason"),
            # vLLM-specific and genuinely informative: `stop_reason` names the
            # stop string or token id that ended generation, and is null when
            # the model emitted its own EOS. That is the evidence Ollama does
            # not give us, so it must not be discarded.
            "stop_reason": choice.get("stop_reason", "__absent__")}


async def run(args: argparse.Namespace) -> None:
    resolved = resolve(args.teacher, args.host)
    template, template_hash = load_template()
    base_v1 = resolved["HOST_BASE_URL"]
    if not base_v1:
        sys.exit(f"host '{args.host}' has no base_url")
    base = base_v1.rsplit("/v1", 1)[0]
    model = resolved["TEACHER_REPO"]
    runtime = "ollama" if args.host.endswith("ollama") else "openai"
    timeout = float(resolved["HOST_REQUEST_TIMEOUT_S"])
    concurrency = args.concurrency or int(resolved["HOST_CONCURRENCY"])

    stops = args.stop or DEFAULT_STOPS

    prompts = [json.loads(l) for l in
               Path(args.prompts).read_text(encoding="utf-8").splitlines() if l.strip()]

    manifest = splits_module.load()
    splits_module.verify(manifest)
    for prompt in prompts:
        prompt["split"] = splits_module.split_of(prompt.get("family", ""), manifest)

    # Same data-handling gate as generate.py. Omitting it here would have left
    # a second generation path able to send real Office material off-premises
    # while the policy appeared to be enforced.
    egress = resolved["HOST_DATA_EGRESS"]
    if egress != "internal":
        carried = {p.get("source_class", "internal") for p in prompts}
        needs_approval = sorted(c for c in carried if c != "synthetic")
        if needs_approval and not args.egress_approval:
            raise SystemExit(
                f"host '{args.host}' is not internal (data_egress: {egress}) and "
                f"{len(prompts)} prompt(s) are classified {needs_approval}.\n"
                "Real or unclassified Office material needs explicit approval. Mark "
                "prompts \"source_class\": \"synthetic\", use an internal host, or "
                "pass --egress-approval <reference>."
            )
        if needs_approval:
            print(f"EGRESS APPROVED [{args.egress_approval}]: {needs_approval} -> {args.host}")

    print(f"normalized run — {args.teacher} @ {args.host} ({resolved['HOST_QUANTIZATION']})")
    print(f"  template sha256   {template_hash[:16]}")
    print(f"  reasoning_strength {args.reasoning_strength}")
    print(f"  temperature {args.temperature}  seed {args.seed}  max_tokens {args.max_tokens}")
    print(f"  stop sequences    {stops or '(none — EOS)'}")
    print(f"  {len(prompts)} prompts, concurrency {concurrency}\n")

    rendered = [render(template, p.get("system", ""), p["user"], args.reasoning_strength)
                for p in prompts]

    results: list[dict | None] = [None] * len(prompts)
    infrastructure: list[dict] = []
    semaphore = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient() as client:
        async def worker(index: int) -> None:
            async with semaphore:
                try:
                    results[index] = await complete(
                        client, base, model, rendered[index],
                        temperature=args.temperature, seed=args.seed,
                        max_tokens=args.max_tokens, runtime=runtime, timeout=timeout,
                        stops=stops)
                except (httpx.HTTPError, httpx.TransportError) as error:
                    # Endpoint failure: never mixed into quality denominators.
                    infrastructure.append({"index": index,
                                           "family": prompts[index].get("family"),
                                           "error": repr(error)})
        await asyncio.gather(*(worker(i) for i in range(len(prompts))))

    records, malformed = [], []
    for index, (prompt, result) in enumerate(zip(prompts, results)):
        if result is None:
            continue
        parsed = parse_channels(result["raw"])
        termination = classify_termination(result["raw"], result["done_reason"])
        record = {
            "family": prompt["family"],
            "split": prompt["split"],
            # Everything the prompt set knows that the corpus needs travels
            # with the trace. Carried as a declared list rather than
            # field-by-field: source_class, corpus_role and source_sha256 were
            # each lost in turn, and each loss silently disabled a downstream
            # check that looked like it was running.
            **{field: prompt[field] for field in CARRY_FROM_PROMPT if field in prompt},
            "system": prompt.get("system", ""),
            "user": prompt["user"],
            # The answer channel only. Reasoning is recorded separately and
            # never presented as the model's answer.
            "completion": parsed["answer"],
            "reasoning_chars": len(parsed["reasoning"]),
            "provenance": {
                "teacher": resolved["TEACHER_NAME"],
                "repo": model,
                "license": resolved["TEACHER_LICENSE"],
                "host": resolved["HOST_NAME"],
                "quantization": resolved["HOST_QUANTIZATION"],
                "protocol": "normalized-harmony-raw",
                "template_sha256": template_hash,
                "prompt_sha256": sha256(rendered[index]),
                "prompt_bytes": len(rendered[index].encode("utf-8")),
                "reasoning_strength": args.reasoning_strength,
                "temperature": args.temperature,
                "seed": args.seed,
                "max_tokens": args.max_tokens,
                "runtime": runtime,
                "completion_tokens": result["completion_tokens"],
                # Termination is recorded as evidence plus a derived category,
                # never as a claim the runtime did not actually make.
                "raw_done_reason": result["done_reason"],
                "raw_stop_reason": result.get("stop_reason", "__absent__"),
                # Did the runtime terminate on its own EOS rather than on a
                # sequence we supplied? Only answerable where the runtime says.
                "eos_applied_by_runtime": (
                    None if result.get("stop_reason") == "__absent__"
                    else result.get("stop_reason") is None),
                "stop_sequences": stops,
                "ended_at_marker": termination["ended_at_marker"],
                "terminated_by": termination["terminated_by"],
                "truncated": termination["terminated_by"] == "length",
                "split_seed": manifest["seed"],
                "split_fingerprint": manifest["fingerprint"],
            },
        }
        if prompt.get("checks"):
            record["checks"] = prompt["checks"]
        if prompt.get("expected"):
            record["expected"] = prompt["expected"]

        if parsed["malformed"]:
            record["malformed_reason"] = ("no final channel" if FINAL_MARKER not in result["raw"]
                                          else "empty final channel")
            malformed.append(record)
            continue
        records.append(record)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"wrote {len(records)}/{len(prompts)} traces -> {out_path}")

    if malformed:
        path = out_path.with_suffix(".malformed.jsonl")
        with path.open("w", encoding="utf-8") as handle:
            for record in malformed:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"MALFORMED {len(malformed)} (failed validation, not accepted) -> {path.name}")
    if infrastructure:
        path = out_path.with_suffix(".errors.json")
        path.write_text(json.dumps({"infrastructure_failures": infrastructure}, indent=2) + "\n",
                        encoding="utf-8")
        print(f"INFRASTRUCTURE {len(infrastructure)} endpoint failures -> {path.name} "
              "(excluded from quality denominators)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", default="muse-glimmer")
    parser.add_argument("--host", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--reasoning-strength", default="high",
                        help="chat-template variable; the template's own default is 'high'")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--egress-approval", default="",
                        help="approval reference permitting non-synthetic material "
                             "on a non-internal host")
    parser.add_argument("--stop", action="append", default=None,
                        help="explicit stop sequence (repeatable). Default: none — "
                             "EOS terminates generation, verified against production. "
                             "Must be identical on both arms.")
    parser.add_argument("--concurrency", type=int, default=0)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
