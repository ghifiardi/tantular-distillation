# The shipped profile does not answer — a serving defect, not a model one

**Measured 2026-08-21 on local Ollama 0.24.0, Apple GPU.**

This is the first real, user-facing failure this project has found. It is not a
capability gap, it is not fixed by training, and it is severe.

## What is affected

`ghifidanukusumo/tantular:latest` and `tantular-office:0.4-9b` are the SAME
weights — digest `912936474e07a209`. The add-in's
`tantularClient.js` sets `DEFAULT_DECK_MODEL = "tantular-office:0.4-9b"`, so
every Studio/deck action hits this profile. Plain chat defaults to `qwen3.5:9b`,
which reasons the same way.

**This is a user endpoint, not a calibration endpoint.** That distinction was
the condition for treating the finding as real.

## The measurements

| request | latency | reasoning | answer |
|---|---|---|---|
| short summary, OpenAI endpoint, `max_tokens 600` | 37 s | 2,529 chars | **empty**, `finish_reason: length` |
| short summary, OpenAI endpoint, `max_tokens 2500` | 134 s | 8,819 chars | 57 chars |
| `fce::0001` edit task, OpenAI endpoint, **`max_tokens 8192`** — the Modelfile's own `num_predict` | **512 s** | **21,808 chars** | **empty**, `finish_reason: length` |
| same task, **native `/api/chat` with `think: false`** | **2 s** | 0 chars | correct |

The 10-item faithful-editing pilot against the shipped profile: **3 of 10
produced an answer.** The other seven were empty with `finish_reason: length`
and `truncated: true`. The same 10 items against bf16 base with thinking
disabled: **10 of 10 passed all six properties.**

## Root cause

The Modelfile carries `RENDERER qwen3.5` / `PARSER qwen3.5`, which enables
Qwen3.5's thinking, and `PARAMETER num_predict 8192`.

Thinking cannot be turned off on the endpoint the product uses. Ollama's
OpenAI-compatible `/v1/chat/completions` **ignores** both
`chat_template_kwargs.enable_thinking` and `think: false` — verified, both
return HTTP 200 and reason anyway. Ollama's **native** `/api/chat` honours
`think: false`.

The add-in calls the OpenAI-compatible path (`tantularClient.js`,
`tools/dev-server.mjs`, and the README's documented contract).

## The fix is serving, not training

Route through Ollama's native `/api/chat` with `think: false`, or have the
companion translate. **67x faster on the measured case, 2 s against 134 s, and
answers where there were none.**

Training was considered and is NOT the answer here. The behaviour is controlled
by the template and the endpoint; a fine-tuned adapter served through the same
path would inherit the same problem. Per the decision of 2026-08-21, a serving
fix comes first, and training is only justified if thousands of reasoning
tokens persist on the production endpoint after it.

## What this says about the existing gates

They measured the quality of an answer and never asked whether one arrived.
Voice 0.9500, edit contract 0.9500, faithful editing 10/10 — all true, all
measured on a path with thinking disabled, and none of them can see an
8-minute empty response on the path users actually take.

Proposed as deployment gates, distinct from capability gates:

- **latency budget** per request;
- **reasoning-token budget**;
- **empty-answer rate**, which must be zero;
- **`finish_reason` distribution** — any `length` is a failure, not a quirk.

A measurement gap the finding also exposed: `reasoning_chars` read only vLLM's
`reasoning_content` and recorded 0 against Ollama, which uses `reasoning` —
reporting nothing while 21,808 characters were being generated. Fixed.


---

# After the serving fix — 2026-08-21

Add-in commit `96b3d80` routes local requests to Ollama's native `/api/chat`,
forces `think: false`, and translates `max_tokens` to `num_predict`. The
companion was restarted (the running process was four days old and predated the
fix). Measured through the companion — the path the add-in actually uses — not
against Ollama directly.

| | before (OpenAI path) | after (companion → native) |
|---|---|---|
| budget | `max_tokens 4096` | **`max_tokens 600`** |
| answers | **3 / 10** | **10 / 10** |
| empty | 7 | **0** |
| `finish_reason: length` | 7 | **0** |
| single short summary | 37 s, empty at 600 tokens | **7.4 s**, answered |
| `fce::0001` at production budget | 512 s, 21,808 reasoning chars, empty | answered |

The comparison is stricter than like-for-like: the fixed path was given a **7x
smaller budget** and still answered every item.

## Quality on the fixed path

**9 / 10** on the six-property faithful-editing scorer — lands, preserves,
structure, no_new_facts and voice all clean on every measurable item.

The single failure is `fce::0005` (multi-edit): the model emitted ONE edit with
a 203-character `find` against a 200-character limit, trying to replace a whole
paragraph rather than making three targeted edits. The output is complete,
well-formed JSON, so this is behaviour and not truncation. bf16 base produced
three correct edits on the same item.

That is a real contract violation, by three characters, on one item of ten. It
is worth watching and is not a shipping blocker on this evidence.

## Status

The serving defect is fixed and the fix is verified on the product's own path.
Latency went from minutes to seconds, and from mostly-no-answer to always-an-
answer at a fraction of the budget.

Still true: no capability gap has been demonstrated that would justify training.
This finding was a serving bug, exactly as diagnosed, and training would not
have fixed it.

---

# Full gate run on the shipped path — 2026-08-21

All 60 capability items plus the 10-item pilot, generated through the companion
(`https://localhost:3000/api/chat-completions` → Ollama native `/api/chat`,
`think: false`) against `tantular-office:0.4-9b`, digest `912936474e07a209` —
the same weights as `ghifidanukusumo/tantular:latest`. Protocol `plain-chat`,
`max_tokens 1200`, concurrency 1. Artifacts in `data/gates/companion/`.

## Deployment gate: PASS

| condition | result |
|---|---|
| empty answers | **0** / 60 |
| `finish_reason: length` | **0** / 60 |
| reasoning characters | **0** |
| latency | p50 **5.4 s**, p95 **22.8 s**, max 32.1 s (budget 30 s p95) |

The serving defect is closed and stays closed under a 60-item load.

## Capability gates: the shipped artifact is WEAKER than the base

| gate | shipped (Q4_K_M, companion) | bf16 base (vLLM, thinking off) | threshold |
|---|---|---|---|
| indonesian_voice | **0.9250 (37/40) — FAIL** | 0.9500 (38/40) | 0.95 |
| edit_contract_output | **0.9000 (18/20)** — exactly at the bar | 0.9500 (19/20) | 0.90 |
| faithful editing pilot | 9/10 | 10/10 | none set |

**The shipped model does not pass its own voice gate.**

All three voice failures are terminology, and all are the same class:

    voice::0013  backup -> pencadangan, di-update -> diperbarui
    voice::0014  disable -> nonaktifkan, maintenance -> pemeliharaan
    voice::0037  maintenance -> pemeliharaan

Both edit failures are contract violations the base did not make: one `find` of
203 characters against the 200 limit, and one unparseable JSON.

## Reading this honestly

Every difference is one item, on samples of 40 and 20. That is not a large
effect and the direction could reverse on a rerun. But it is consistent across
all three evaluations, and the shipped artifact is on the wrong side of a
threshold on one of them.

Two differences could explain it, and neither is a capability gap:

1. **Q4_K_M quantization.** int4 degradation is the reason the corpus carries a
   signed waiver; the base numbers are bf16.
2. **The profile's system prompt and sampling** — `temperature 0.2`,
   `presence_penalty 1.5`, `top_k 20` — versus greedy decoding for the base.

The cheap experiments come before any training: serve the same profile at a
higher precision (Q5/Q6/Q8) and re-run, and re-run at temperature 0. If the gap
closes, it was quantization or sampling. Only if it survives both is there a
model-side failure to discuss.
