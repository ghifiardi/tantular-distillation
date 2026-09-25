# Qwen3.5-4B identity qualification for the separation study

**Date:** 2026-09-25
**Registry entry:** `configs/models/qwen35-4b-instruct.yaml`
**Sources:** `docs/QWEN35_4B_IDENTITY_INSPECTION_2026-09-25.md`,
`docs/qwen35-4b-identity-comparison.json`,
`docs/CLAUDE_CODE_RSI_4B_ADAPTER_HANDOFF_2026-09-25.md`

**No inference was run to produce this document or the registry entry.** No
prompt was sent to any model. No gold data was read. No training occurred. The
values below were recorded by an identity-only inspection of the local Ollama
store and of public upstream model metadata, and are copied here literally.

`training_authorized: false`

## 1. What is being qualified

The RSI separation gate (`src/eval_harness.py separation-gate`) compares a
stronger student against a weaker one on the same gold items. The weaker arm
is the locally installed Ollama model `ghifidanukusumo/tantular:lite`, which
is the upstream `Qwen/Qwen3.5-4B` weights (Q4_K_M) under Tantular's runtime
profile. This document records the evidence that the tag is what it claims to
be, and what still stands between it and a live measurement.

## 2. Local Ollama identity

| Item | Value |
|---|---|
| tag | `ghifidanukusumo/tantular:lite` |
| alias (same manifest) | `tantular-office:lite` |
| parent tag recorded by Ollama | `qwen3.5:4b` |
| local registry (manifest) digest | `b2b24b8d3517401d53b56dd144081ca4d76e5e09180376c64b37d2e3065a6a35` |
| architecture | `qwen35` |
| measured GGUF parameter count | 4,659,865,088 (displayed 4.7B) |
| quantization | Q4_K_M |
| GGUF context length | 262144 |
| configured runtime context | 32768 |
| capabilities | completion, vision, tools, thinking |
| licence | Apache-2.0 |

The manifest digest is a pointer, not an equivalence test: Ollama recomputes
it across a push/pull round-trip (`src/model_identity.py`). The weights blob
and the canonical profile below are the identity claims.

## 3. Weight identity and full-file verification

| Item | Value |
|---|---|
| weights blob | `sha256-81fb60c7daa80fc1123380b98970b320ae233409f0f71a72ed7b9b0d62f40490` |
| byte size | 3,389,971,840 |
| full local SHA-256 of the blob file | **PASS**: the file bytes hash to the blob name |
| shared with | local `qwen3.5:4b` uses the same weight blob |

## 4. Tantular runtime profile, and the shared-profile finding

| Item | Value |
|---|---|
| system layer SHA-256 | `c32dc580a89c26e07364ee0dd6bfbf512f738d21bb3cd007aac120de764d3dc5` |
| local Ollama template SHA-256 | `b507b9c2f6ca642bffcd06665ea7c91f235fd32daeefdf875a0f938db05fb315` |
| parameter layer SHA-256 | `00427f1947f5c700d8c4f329ff7ba52594f6757b4ae7eee36764842051ee95b7` |
| canonical profile SHA-256 | `15368614046c19ce4e63e2b5506507bfbd9d8e3bd3600571d9ad27af3c43f11b` |
| profile-only SHA-256 | `dc69dd6e4f2f5669bb212509c1f21c00e470463a251bac47e51f41d5cb6b81fc` |

Configured parameters: `num_ctx 32768`, `num_predict 8192`,
`presence_penalty 1.5`, `repeat_penalty 1.05`, `temperature 0.2`, `top_k 20`,
`top_p 0.9`.

**Finding versus the 9B.** The canonical profile digest is identical to that
of `ghifidanukusumo/tantular:q8-0.5` (the current 9B): same system layer, same
template, same sorted parameters. The comparison record reports
`equivalent: false` solely because the weights blobs differ
(`sha256-73b25b60…` for the 9B, `sha256-81fb60c7…` for the 4B). That is the
property the separation study needs: the two arms differ in model size and in
nothing else about how they are served.

## 5. Official upstream identity

| Item | Value |
|---|---|
| model | `Qwen/Qwen3.5-4B` |
| pinned revision | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` |
| licence metadata | `apache-2.0` |
| pipeline | `image-text-to-text` |
| config.json SHA-256 | `ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670` |
| tokenizer.json SHA-256 | `5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42` |
| tokenizer_config.json SHA-256 | `316230d6a809701f4db5ea8f8fc862bc3a6f3229c937c174e674ff3ca0a64ac8` |
| chat_template.jinja SHA-256 | `a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715` |
| LICENSE SHA-256 | `bbedc3fda3305820b977265f01b8619d87570a6739de3a5582c3464840f1e57a` |

Text architecture at the pinned revision: hidden 2560, intermediate 9216,
32 layers, 16 attention heads, 4 key/value heads, vocabulary 248320, head
dimension 256, full-attention interval 4.

The chat template digest equals the 9B registry entry's `chat_template.sha256`.

## 6. What the registry entry claims, and what it does not

- `revision`, `chat_template.sha256`, the per-file tokenizer digests, the
  Ollama weights blob, the canonical profile digest and the licence evidence
  digest are recorded literally from the tables above.
- On 2026-09-25, `src/verify_model_identity.py --offline --write` measured the
  composite tokenizer compatibility digest from the pinned Hugging Face cache
  snapshot at revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
  `tokenizer.sha256` is
  `6f3a76fa0ff84cba487813d4024623233c4664ecedfc3f3857536f95d25504af`
  and `digests_verified` is **true**.
  The snapshot's commit is known from the Hugging Face cache layout. The
  per-file digests are evidence for that measurement, not a substitute; a
  hand-typed composite would make a Mode C decision on fiction.
- `capabilities.logprobs` is false: Ollama `/api/chat` exposes none, so this
  entry can never be a Mode C party.
- The entry carries a `serving:` block (`protocol: ollama_chat`, the two tags,
  the Ollama identity digests). The gold runner's served-identity check matches
  the endpoint's `/api/tags` against those tags, and refuses the OpenAI client
  for this entry.

## 7. Endpoint contract

The product model card requires Qwen3.5 on Ollama to be driven through
`/api/chat` with `think: false`. The OpenAI-compatible `/v1/chat/completions`
ignores that switch and can return an empty answer after spending the budget
on reasoning. `src/ollama_chat_client.py` therefore:

- posts to `/api/chat` only, with `think: false` and `stream: false` on every
  request and fixed decoding under `options`;
- reads served identity from `/api/tags`;
- refuses a reply that is not done, names another model, or carries a
  `thinking` channel;
- has no fallback to `/v1`; a 404 on `/api/chat` is a refusal.

## 8. Verdict

`ghifidanukusumo/tantular:lite` is qualified as the 4B **identity** for the
separation study. Live evaluation of it remains **BLOCKED** until `data/gold`
holds approved, human-authored items (`gold set is BLOCKED: 0 approved items`)
and the owner accepts the fixed decoding settings for a Stage 6 run. Nothing
in this document authorizes training.

`training_authorized: false`
