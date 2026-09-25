# Tantular Lite 4B identity-only inspection

No inference was run. No gold data was read or transmitted. No training occurred.

## Local Ollama identity

- tag: `ghifidanukusumo/tantular:lite`
- alias with the same manifest: `tantular-office:lite`
- local registry digest: `b2b24b8d3517401d53b56dd144081ca4d76e5e09180376c64b37d2e3065a6a35`
- parent tag recorded by Ollama: `qwen3.5:4b`
- architecture: `qwen35`
- measured GGUF parameter count: `4,659,865,088` (displayed as 4.7B)
- quantization: `Q4_K_M`
- context length in GGUF: `262144`
- configured runtime context: `32768`
- capabilities: completion, vision, tools, thinking
- license: Apache-2.0

## Weight identity

- weights blob: `sha256-81fb60c7daa80fc1123380b98970b320ae233409f0f71a72ed7b9b0d62f40490`
- byte size: `3,389,971,840`
- full local SHA-256 verification: PASS; file bytes hash to the blob name above
- the Tantular Lite tag and local `qwen3.5:4b` use the same weight blob

## Tantular runtime profile

- system layer SHA-256: `c32dc580a89c26e07364ee0dd6bfbf512f738d21bb3cd007aac120de764d3dc5`
- local Ollama template SHA-256: `b507b9c2f6ca642bffcd06665ea7c91f235fd32daeefdf875a0f938db05fb315`
- parameter layer SHA-256: `00427f1947f5c700d8c4f329ff7ba52594f6757b4ae7eee36764842051ee95b7`
- canonical profile SHA-256: `15368614046c19ce4e63e2b5506507bfbd9d8e3bd3600571d9ad27af3c43f11b`
- profile-only SHA-256: `dc69dd6e4f2f5669bb212509c1f21c00e470463a251bac47e51f41d5cb6b81fc`
- the canonical Tantular runtime profile is identical to `ghifidanukusumo/tantular:q8-0.5`; only the model weights differ

Configured parameters:

- `num_ctx 32768`
- `num_predict 8192`
- `presence_penalty 1.5`
- `repeat_penalty 1.05`
- `temperature 0.2`
- `top_k 20`
- `top_p 0.9`

## Official upstream identity

- upstream model: `Qwen/Qwen3.5-4B`
- pinned Hugging Face revision: `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- license metadata: `apache-2.0`
- official pipeline: `image-text-to-text`
- official config SHA-256: `ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670`
- official tokenizer.json SHA-256: `5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42`
- official tokenizer_config.json SHA-256: `316230d6a809701f4db5ea8f8fc862bc3a6f3229c937c174e674ff3ca0a64ac8`
- official chat_template.jinja SHA-256: `a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715`
- official LICENSE SHA-256: `bbedc3fda3305820b977265f01b8619d87570a6739de3a5582c3464840f1e57a`

Text architecture from the pinned config:

- hidden size: 2560
- intermediate size: 9216
- layers: 32
- attention heads: 16
- key/value heads: 4
- vocabulary: 248320
- head dimension: 256
- full-attention interval: 4

## Endpoint compatibility

- Ollama `/v1/models` exposes `ghifidanukusumo/tantular:lite`, so identity discovery works.
- The existing product model card states that Qwen3.5 must be evaluated through Ollama `/api/chat` with `think: false`.
- The current gold runner's real client uses OpenAI-compatible `/v1/chat/completions` through `bridge_client.TeacherClient`.
- Therefore live evaluation is still BLOCKED: an Ollama `/api/chat` adapter with an explicit `think: false` contract must be implemented and tested before any gold prompt is sent.

## Verdict

`ghifidanukusumo/tantular:lite` is a suitable 4B *identity candidate* for the separation study because it is the Qwen3.5-4B weights under the same Tantular runtime profile as the current 9B. It is not yet authorized for live evaluation because the registry entry, serving declaration, and safe Ollama adapter have not been landed and reviewed.

`training_authorized: false`
