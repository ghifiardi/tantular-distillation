# Claude Code handoff — pin the 4B comparison model and add a safe Ollama adapter

Date: 2026-09-25
Worktree: /private/tmp/tantular-rsi-next
Base branch: feat/rsi-executable-loop-20260924 (PR #24, already green)
NEW branch for this work: feat/rsi-4b-ollama-adapter-20260925

Read this whole file before writing code. This task is identity + adapter
plumbing only. Do not run model inference. Do not send any gold prompt to any
model. Do not train. No external spending. No data egress beyond reading public
model metadata already recorded here.

Every artifact you add or emit must carry: training_authorized: false

## Why this task exists

The separation study compares Tantular-9B against Tantular-4B. The 4B candidate
is the already-installed Ollama model ghifidanukusumo/tantular:lite. Its identity
was inspected on 2026-09-25 (no inference). Two gaps block live evaluation:

1. There is no pinned 4B entry under configs/models/.
2. The gold runner's real client speaks OpenAI /v1/chat/completions, but the
   product model card requires Qwen3.5 on Ollama to be driven through /api/chat
   with think:false; /v1 ignores the thinking switch and can return an empty
   answer, which would corrupt every 4B measurement.

## Verified identity values (use literally; do not re-derive by guessing)

Local Ollama 4B candidate:
- tag: ghifidanukusumo/tantular:lite (alias tantular-office:lite, same manifest)
- local registry digest: b2b24b8d3517401d53b56dd144081ca4d76e5e09180376c64b37d2e3065a6a35
- weights blob: sha256-81fb60c7daa80fc1123380b98970b320ae233409f0f71a72ed7b9b0d62f40490
- weights byte size: 3389971840
- measured parameter count: 4659865088
- quantization: Q4_K_M
- canonical Tantular profile SHA-256: 15368614046c19ce4e63e2b5506507bfbd9d8e3bd3600571d9ad27af3c43f11b
  (identical to ghifidanukusumo/tantular:q8-0.5; only weights differ)
- system layer SHA-256: c32dc580a89c26e07364ee0dd6bfbf512f738d21bb3cd007aac120de764d3dc5
- params layer SHA-256: 00427f1947f5c700d8c4f329ff7ba52594f6757b4ae7eee36764842051ee95b7

Upstream, official:
- model: Qwen/Qwen3.5-4B
- pinned revision: 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
- license: apache-2.0
- config.json SHA-256: ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670
- tokenizer.json SHA-256: 5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42
- tokenizer_config.json SHA-256: 316230d6a809701f4db5ea8f8fc862bc3a6f3229c937c174e674ff3ca0a64ac8
- chat_template.jinja SHA-256: a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715
- LICENSE SHA-256: bbedc3fda3305820b977265f01b8619d87570a6739de3a5582c3464840f1e57a
- text arch: hidden 2560, intermediate 9216, layers 32, heads 16, kv-heads 4,
  vocab 248320, head_dim 256, full_attention_interval 4

## Stage 1 - pinned 4B registry entry

Add configs/models/qwen35-4b-instruct.yaml, mirroring the field shape of the
existing configs/models/qwen35-9b-instruct.yaml. Requirements:
- role: student (a separation arm must be a student, never a teacher).
- model_id: Qwen/Qwen3.5-4B
- revision: 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a (full 40 hex).
- tokenizer + chat_template digests from the verified values above.
- record the local Ollama serving identity (tag, weights blob, canonical profile
  digest) so the served model can be matched at run time.
- family qwen3.5, generation "3.5", params.total_b 4.7.
- keep the same digests_verified / license shape the 9B entry uses; only claim
  digests you can point at in this document.

Confirm src/run_gold_evaluation.py:load_model_spec accepts it: role student,
model_id present, 40-char revision. If that loader needs a serving handle, add it
rather than loosening the loader.

## Stage 2 - safe Ollama /api/chat adapter

Add a new client (e.g. OllamaChatClient) next to BridgeEvaluationClient in
src/run_gold_evaluation.py (or a small importable module it uses). It must:
- POST to <endpoint>/api/chat
- always send "think": false and "stream": false
- send fixed decoding via options (temperature, top_p, num_predict), translating
  the OpenAI-style names the runner already uses.
- implement served_models() by reading Ollama /api/tags (or /v1/models) and
  returning the tag list, so the existing served-identity check works.
- return only the assistant message content string to the verifier layer.
- reads may retry; never retry in a way that changes semantics.
- refuse, loudly, if asked to fall back to /v1/chat/completions. No silent
  downgrade path.

Wire selection so --real against an Ollama endpoint uses this adapter. Keep the
OpenAI path unchanged and keep --real mandatory for any network call.

## Stage 3 - identity qualification doc

Add docs/QWEN35_4B_IDENTITY_QUALIFICATION.md recording every hash here, the
full-file weight-blob verification result, the shared-profile finding versus the
9B, and the explicit statement that no inference was run to produce it.

## Stage 4 - tests (offline, deterministic, no network)

Add tests/test_ollama_adapter.py (extend tests/test_run_gold_evaluation.py only
if needed). Prove, with a fake transport - never a real socket:
1. the adapter posts to /api/chat, not /v1/chat/completions
2. every request body contains think:false and stream:false
3. decoding options are passed under options
4. served_models() parses the tag list and the served-identity check rejects a mismatch
5. a teacher-role registry entry is refused in a student arm
6. there is no code path that silently falls back to /v1
7. the new registry entry loads through load_model_spec
8. every artifact carries training_authorized: false

Do not instantiate the real client against a live endpoint in any test.

## Stage 5 - verification

Run, using the base worktree interpreter (document which one):
  <py> -m pytest tests/test_ollama_adapter.py tests/test_run_gold_evaluation.py tests/test_run_gepa.py tests/test_gold_set.py tests/test_verifiers.py tests/test_eval_harness.py -q
  <py> -m pytest tests/ -q -m "not requires_local_corpus"

No new failing test name versus the recorded baseline
(docs/authorizations/origin-main-baseline-2026-09-19.md plus the known
test_the_legacy_corpus_is_absent_not_malformed corpus-coupled failure). Do not
edit thresholds or unrelated assertions to go green.

Smoke (must exit 0 and contact nothing):
  <py> src/run_gold_evaluation.py plan --gold-dir data/gold --model-registry qwen35-4b-instruct --endpoint http://127.0.0.1:11434 --repetitions 2

With an empty data/gold this must still refuse with
"gold set is BLOCKED: 0 approved items" - that refusal is correct and proves the
plan path is wired without a model.

## Stage 6 - publish

Commit on feat/rsi-4b-ollama-adapter-20260925, push, and open a PR based on
feat/rsi-executable-loop-20260924 (not main), so it stacks on PR #24. The PR body
must state: no inference run, no data egress beyond public model metadata, no
spending, no training, and training_authorized: false.

## Out of scope
- Running 9B or 4B on any prompt.
- Producing receipts or measurements.
- Running GEPA.
- Authoring or approving gold items.
- Any training or weight change.

## Final report

Return: new branch, files added, test results vs baseline, smoke results,
confirmation of no inference / no egress / no spend / no training, the PR URL,
and the literal line training_authorized: false.
