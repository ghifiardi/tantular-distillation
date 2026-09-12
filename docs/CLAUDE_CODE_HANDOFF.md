# Implementation handoff — teacher-agnostic distillation (for Claude Code)

**Read this whole file before writing any code. Execute tasks in order. Do not
skip the guardrails.**

This turns the DRAFT artifacts already in the repo into wired, tested code. It
does **not** authorize any training run.

## Context (what already exists, as drafts)

Created in a prior session, none wired into the trainer or gates:

- `docs/TEACHER_AGNOSTIC_ARCHITECTURE.md` — the design.
- `configs/models/model.schema.md` — registry field + gate reference.
- `configs/models/{qwen35-9b-instruct,muse-glimmer-30b,qwen35-122b-a10b}.yaml`
- `configs/architectures/{qwen35-hybrid-dense-9b,qwen35-hybrid-moe}.yaml`
- `configs/distillation/office-v2-sequence.yaml`
- `src/distill_plan.py` — fail-closed planner: subcommands `plan`,
  `arch-signature`, `provenance-audit`.
- `configs/hosts/rented-48gb.yaml` — gained `gpu_memory_gb`/`gpu_count`.

The `sha256`/`revision` fields in `configs/models/*.yaml` are PLACEHOLDERS
(e.g. `TOKENIZER_DIGEST_QWEN35_9B`) and every spec has `digests_verified: false`.

## GUARDRAILS — non-negotiable

1. `train/TRAINING_BLOCKED.md` is controlling. **Do not start, enable, or
   authorize any training or cloud run.** No task here needs a GPU.
2. **Do not fabricate** digests, revisions, corpora, sources, or eval items. If a
   real value is unavailable, leave the placeholder and keep `digests_verified:
   false`. Failing closed is the correct outcome.
3. **Do not loosen any gate** (corpus, licence, compatibility-key, held-out eval)
   or retarget the product to the Base checkpoint (`train/BASE_VS_INSTRUCT.md`).
4. New code is **fail-closed**: a check that cannot run must refuse, not pass.
   Match the style of `src/verify_corpus.py` / `src/train_qlora.py` (`die()` /
   `sys.exit` with a specific message).
5. Keep everything **offline and deterministic**. No network calls in tests.
6. Single-source shared rules (e.g. import `verify_corpus.UNTRAINABLE_QUANTIZATION`
   rather than re-listing quantizations).
7. After each task, run the full suite and the verification commands. Do not
   proceed to the next task with a red suite.

Interpreter: use `./.venv/bin/python` when present, else `python3`.
Test command: `./.venv/bin/python -m pytest tests/ -q`.

---

## Task 1 — Tests for the planner (do this FIRST, no product code yet)

**Why first:** locks the intended behaviour before anything else touches it.

Create `tests/test_distill_plan.py` covering, using tmp files / fixtures only:

- `license_status` / `license_gate`: permitted+fresh passes; `output_training_
  permitted: false` refuses; `reviewed_at` older than `recheck_max_age_days`
  refuses (STALE); missing `reviewed_at`/`recheck_max_age_days` refuses (UNKNOWN);
  empty `evidence_sha256` yields `FRESH_NO_EVIDENCE` and a gate failure.
- `tokenizers_compatible` / mode selection: equal tokenizer sha256 -> Mode C for
  `auto`; differing -> `auto` falls back to preference (pairs present) then
  sequence; **explicit `on_policy_kd` on a mismatch REFUSES** (SystemExit).
- `_mode_c_preflight`: missing logprobs, unpinned revision, `targets_lm_head:
  true`, or missing architecture profile each block Mode C.
- Hardware: a host with declared capacity >= need is servable; no declared
  capacity -> not servable (fail-closed); `supports_fp8: false` excluded at fp8.
- `architecture_signature`: stable for the same config, changes when
  `num_hidden_layers`/`layer_types`/`model_type` change.
- `provenance-audit`: int4 corpus -> `fp8_gate.status == UNMET`; a
  `(remote, gateway)` trace -> `uncovered_by_waiver` non-empty; a synthetic trace
  -> `real_office_claim.supported == false`; an unregistered teacher ->
  `unresolved_corpus_teachers` non-empty.

**Acceptance:** `pytest tests/test_distill_plan.py -q` passes; no network; runs
under a few seconds. Do NOT modify `src/distill_plan.py` in this task except to
add narrow, behaviour-preserving hooks if a function is not importable.

## Task 2 — `src/verify_model_identity.py` (fill the placeholder digests)

Build an offline verifier that, given a registry model name, loads the tokenizer
and chat template from a local HF snapshot and:

- computes `tokenizer.sha256` (a stable digest over the tokenizer files) and
  `chat_template.sha256` (over the effective template);
- compares to the values in `configs/models/<name>.yaml`;
- with `--write`, fills the fields and sets `digests_verified: true` **only** on a
  clean match; otherwise leaves the file and exits non-zero;
- supports `--offline` (`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`) like
  `src/train_tinker_sft.py --offline`, and never reads a credential.

Do not invent digests. If the snapshot is absent, print how to obtain it and
exit non-zero. Keep heavy imports (transformers) lazy so `--help` works without
them, mirroring the repo's lazy-import pattern.

**Acceptance:** running against a real local snapshot flips one model's
`digests_verified` to `true` and makes `distill_plan.py plan` drop that model's
`digests_verified` warning. With no snapshot, it fails closed. Add
`tests/test_verify_model_identity.py` with a tiny fake tokenizer dir (no network).

## Task 3 — Architecture-signature enforcement at train time

- Fill the real `signature:` values in `configs/architectures/*.yaml` using
  `./.venv/bin/python src/distill_plan.py arch-signature <path/to/config.json>`
  (only from a real, local `config.json`; leave the placeholder otherwise).
- Add a preflight check in the trainer path (`src/train_qlora.py` and/or
  `src/smoke_train.py`) that, before LoRA attaches, recomputes the signature from
  the loaded model config and **aborts on mismatch** against the profile named by
  the student spec. Reuse `distill_plan.architecture_signature`.

**Acceptance:** a matching profile proceeds; a deliberately wrong pinned
signature aborts before any GPU work. Add a unit test that drives the
signature comparison with two dict configs (no model load).

## Task 4 — Wire `provenance-audit` into the freeze

In `src/freeze_training_run.py`, run the corpus through
`distill_plan.cmd_provenance_audit` logic (refactor the report into an importable
`audit_corpus(path, today, teacher_overrides) -> dict` in `distill_plan.py`) and
record the returned verdict as a `provenance_audit` block in the written run
manifest. Bump the manifest `schema_version` and update
`src/train_qlora.py`'s expected version + `tests/test_training_manifest.py`.

Fail closed: if the audit cannot run, the freeze refuses. Do **not** change what
the corpus gate decides; the audit is additive evidence.

**Acceptance:** a freeze writes a manifest containing `provenance_audit` with the
`fp8_gate`, `source_classes`, and `license_freshness` fields; existing
freeze/trainer tests pass after the version bump.

## Task 5 — Reconcile the registry with the existing configs

`configs/teachers/office-student-9b.yaml` (existing) and
`configs/models/qwen35-9b-instruct.yaml` (new) describe the same checkpoint.
Pick ONE source of truth and make the other reference it, or document the mapping
explicitly so the gates and the planner cannot disagree (the repo already treats
this class of drift as a hazard — see `src/model_ids.py`). Add truthful
`gpu_memory_gb`/`gpu_count` to other real hosts **only where known**; leave
unknown (fail-closed) otherwise. Do not touch `ai19`'s `supports_fp8: false`.

**Acceptance:** `distill_plan.py plan office-v2-sequence` resolves the product
student/teacher consistent with the existing gate configs; no gate config is
weakened.

## Task 6 — Design-only: plan -> generate -> promote -> gates (NO execution)

Extend `docs/TEACHER_AGNOSTIC_ARCHITECTURE.md` with a short, concrete section
mapping a `configs/distillation/*.yaml` plan onto the existing tools
(`src/generate.py` for Mode A, the judge/dedup/promote pipeline, and
`src/run_gates.py` before/after). Write it as documentation and, at most, a
`--dry-run` planner that prints the command sequence. **Emit no traces, train
nothing.** Execution stays gated behind a future `train/TRAINING_JUSTIFIED.md`.

**Acceptance:** the doc section exists; any new dry-run path writes nothing and
performs no network call.

---

## Global acceptance (run at the end)

```
./.venv/bin/python -m pytest tests/ -q
./.venv/bin/python src/distill_plan.py plan office-v2-sequence --today <today>
./.venv/bin/python src/distill_plan.py provenance-audit data/promoted/train.jsonl --today <today>
```

Expected, unchanged from the current drafts unless a task deliberately changed
it: the plan selects `preference` for the non-Qwen `muse-glimmer -> qwen`
pair (compatibility key mismatch) and is servable only via `rented-48gb` at fp8;
the audit reports `fp8_gate: UNMET`, waiver in-scope, 100% synthetic,
`trainable_as_is: false`.

## Definition of done

- All tasks' acceptance criteria met; full suite green.
- No placeholder digest was invented; unverified specs still say
  `digests_verified: false`.
- No training was started, enabled, or authorized; `train/TRAINING_BLOCKED.md`
  is untouched.
- A short PR description lists every file changed and states explicitly that no
  training/cloud run was performed.
