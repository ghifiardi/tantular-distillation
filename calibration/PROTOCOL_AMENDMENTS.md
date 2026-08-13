# Protocol amendments — int4-vs-FP8 calibration

Pre-registered thresholds in `acceptance.yaml` are **unchanged** by anything
here. These are execution-path amendments, recorded because the environment
moved during the study and a reader must be able to tell what was measured.

---

## Amendment 1 — baseline endpoint changed mid-study (infrastructure)

**When:** 2026-08-13, before any baseline traces were scored.

**What happened.** The first 52-prompt baseline attempt ran against the
LiteLLM gateway at `https://openai.ina17.com/v1` and failed **all 52 calls**
with `HTTP 400 — Invalid model name passed in model=ollama/muse-glimmer-30b`.
The id was still listed by `/v1/models`; only the deployment behind it was
gone. `ollama/nemotron-3.5-lightning` failed identically. `qwen3.6-35b` and
`ghifidanukusumo-Tantular-latest` continued to route.

**Classification: infrastructure failure, not model quality.** Those 52 calls
produced zero traces, are excluded from every quality denominator, and were
**not** retried into the result set. The run was re-executed in full against
the new endpoint. No partial results from the failed attempt were merged.

**Amendment.** The baseline arm is served by **ai19's Ollama reached directly
over an SSH tunnel** (`http://localhost:11435/v1` → `ai19:11434`) rather than
through the gateway. Same weights, same GPUs — the gateway was a proxy in
front of this exact server.

**Effect on scope.** The comparison is now explicitly:

> `int4 on ai19 / Ollama` versus `FP8 on rented Ada-or-Hopper / vLLM`

This answers the practical shipping question. It does **not** isolate
precision: quantization and serving runtime vary together. Any result must be
reported with that scope attached. An int4-under-vLLM arm would isolate
precision, and is worth running only if this result is marginal or the
Ollama-vs-vLLM difference is operationally important — it is an attribution
experiment, not a prerequisite.

---

## Amendment 2 — two runtime facts that bear on comparability

Both discovered while capturing `data/calibration/int4/environment.txt`.

**`TEMPLATE {{ .Prompt }}` — the Ollama model card uses a passthrough
template.** It applies no chat template of its own. A vLLM FP8 arm will apply
the chat template from the model's `tokenizer_config.json`. So the two arms
may not present identical token sequences to the model even from identical
`messages`. This is a real confound for the behavioural comparison and must be
recorded alongside any result. Mitigation if it proves material: pin an
explicit template on both arms, or compare rendered prompts directly.

**`PARAMETER temperature 1` is baked into the model card.** The study sends
`temperature: 0.0` per request, which Ollama's OpenAI-compatible endpoint
applies over the card default. The FP8 arm must set temperature explicitly
too, and not rely on any server default.

---

## Amendment 3 — preflight required before either arm

Because a model disappeared underneath a running study once already, both arms
now run `src/preflight.py` first. It records endpoint, model id, runtime,
quantization, template and the full `/v1/models` response, and **fails the run**
if the model is absent or its signature differs from what was recorded for that
arm. A study that silently changes model underneath itself produces numbers
that mean nothing.

---

## Amendment 4 — data-handling scope of the direct endpoint

Reaching ai19 directly removes the **Cloudflare and LiteLLM operator path**.
It does **not** make prompts unobserved: the ai19 host operator and the Ollama
service itself still see every prompt, and Ollama logs requests.

So this is an improvement in the number of parties with visibility, not a
guarantee of confidentiality. It does not by itself authorise sending real
Office corpus material — `inventory/sources.yaml` still governs that, and the
calibration prompts remain `source_class: synthetic`.

---

## Recorded environment — baseline arm

| Field | Value |
|---|---|
| Endpoint | `http://localhost:11435/v1` (SSH tunnel → `ai19:11434`) |
| Model id | `muse-glimmer:30b` |
| Digest | `de878ce33ad8` |
| Runtime | Ollama 0.32.9 |
| Quantization | **Q4_K_M** |
| Parameters | 27.9B (+1.9B CLIP projector) |
| Context length | 131072 |
| Template | `{{ .Prompt }}` (passthrough) |
| Card defaults | temperature 1, top_k 64, top_p 0.95 |
| Request decoding | temperature 0.0, top_p 1.0, max_tokens 4096 |
| GPUs | 2× RTX 3090, compute capability 8.6, shared/contended |

Full capture: `data/calibration/int4/environment.txt`.

Latency and throughput from this arm measure a **contended production server**,
not the hardware. They stay informational and non-gating.

---

## Amendment 5 — rendering parity resolved, and what it revealed

**Parity by naive rendering: FAILED.** Seven candidate serializations
(`sys_blank_user`, `sys_nl_user`, `labelled_turns`, `user_only`, and three
harmony variants) were run through Ollama's raw path at temperature 0 and
compared against its chat endpoint. None reproduced it. Recorded in
`calibration/parity/ai19-ollama.probe.json`.

**Root cause.** The model is **harmony-format**, and Ollama renders it with a
built-in architecture renderer — which is why the Modelfile shows a bare
`TEMPLATE {{ .Prompt }}` and the GGUF carries no `chat_template`. The
authoritative template is `chat_template.jinja`, shipped in the FP8 repo the
treatment arm will load, and vendored here at
`calibration/parity/chat_template.jinja`.

Canonical rendering (probe hash `3c923547bfe3acab`):

```
<|start|>system<|message|>{system}

Reasoning strength: high.

# Valid recipients: "self", "user".<|eot|><|start|>user<|message|>{user}<|eot|><|start|>assistant
```

**The divergence is on the RESPONSE side, not the prompt side.** Raw generation
returns every channel; the chat endpoint parses them and returns only the final
message. Full raw shape:

```
 to=self<|message|>{reasoning}<|start|>assistant to=user<|message|>{final answer}
```

**Normalized protocol for both arms:**

1. Render with `chat_template.jinja`; record the SHA-256 of the rendered string.
2. Send that exact string via a completion path — `/api/generate` with
   `raw: true` (Ollama), `/v1/completions` (vLLM). Prompt parity then holds by
   construction and the hash proves it.
3. Extract the final message identically on both arms: take the content after
   the last `<|start|>assistant to=user<|message|>`, strip trailing `<|eot|>`.
4. Set temperature, seed, max tokens, model revision and `reasoning_strength`
   explicitly on both arms; rely on no server default.

**Consequence.** The existing 52-prompt int4 result was produced through
Ollama's chat endpoint, i.e. a different response path. It stands as the
**deployment baseline** — it measures what the pane would actually get today —
but the **int4-vs-FP8 verdict must use a normalized re-run**. Thresholds are
unchanged.

### Side finding with product impact

`reasoning_strength` is a **chat-template variable**, defaulting to `'high'`:

```jinja
set rs = reasoning_strength if reasoning_strength is defined and reasoning_strength else 'high'
```

It is not the `reasoning_effort` API field. That explains two earlier
observations: `reasoning_effort: "none"` was silently ignored, and ~230 tokens
went to reasoning before any content appeared.

This reaches beyond the study. `tantular_office_addin/src/tantularClient.js`
sends `reasoning_effort: "none"` specifically to stop thinking from consuming
short budgets — the intent router allots 4 tokens. Against any harmony-format
model that field does nothing, and the control must be passed as a chat
template kwarg instead.
