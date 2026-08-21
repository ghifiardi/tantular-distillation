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
