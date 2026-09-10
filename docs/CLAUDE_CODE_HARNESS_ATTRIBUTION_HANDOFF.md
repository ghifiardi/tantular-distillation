# Implementation handoff: harness attribution for distillation (for Claude Code)

**Read this whole file before writing code. Execute tasks in order. Obey the
guardrails.** This wires harness identity into trace generation, provenance, and
the freeze so a future distillation corpus can distinguish model quality from
harness quality. It authorizes no training.

## Context (already in the repo, uncommitted new files)

- `docs/HARNESS_AWARE_DISTILLATION_ARCHITECTURE.md` - the design.
- `configs/harnesses/harness.schema.md`
- `configs/harnesses/tantular-office-current.yaml`
- `configs/harnesses/tantular-office-candidate.yaml`
- `configs/experiments/harness-before-weights.yaml`
- `src/harness_distill.py` - `plan`, `evaluate`, `audit-traces`; canonical
  harness digest; safety-invariant validation. 15 tests pass.
- `tests/test_harness_distill.py`

The audit already proved the gap this handoff closes: the existing
`data/promoted/train.jsonl` is 136 traces, 136 model-attributed, 0
harness-attributed. New traces must carry `harness_provenance.digest`.

## GUARDRAILS - non-negotiable

1. `train/TRAINING_BLOCKED.md` is controlling. Do not start, enable, or
   authorize any training, cloud run, model call, or trace generation against a
   live endpoint. No task here needs a GPU or network.
2. Do not fabricate harness digests, prompt hashes, model revisions, or corpora.
   If a real value is unavailable, leave it null and fail closed.
3. Do not break the existing corpus or the 333-passing suite. The legacy int4
   `data/v3-candidate` / `data/promoted` corpus predates harness attribution and
   has no `harness_provenance`. Every new harness requirement must apply ONLY to
   harness-aware corpora (opt-in by an explicit marker), never retroactively to
   the legacy corpus.
4. New code is fail-closed and matches the style of `src/harness_distill.py`,
   `src/distill_plan.py`, and `src/verify_corpus.py` (raise/die with a specific
   message; a check that cannot run refuses).
5. Tests are offline and deterministic: no network, no real model, no real
   add-in required. Mock the client; use tmp fixtures.
6. Single-source shared logic. Reuse `harness_distill.canonical_digest` and a
   new `harness_provenance()` helper everywhere; do not recompute digests
   inline.
7. After each task run the targeted tests, then the corpus-free partition. Do
   not proceed with a red suite.

### Branch and dirty-tree caution

`src/generate.py`, `src/bridge_client.py`, `src/pass_manifest.py`,
`src/freeze_training_run.py` may already carry unrelated in-flight edits in the
working tree. Before editing any of them:

- Work on a dedicated branch off the current committed HEAD of
  `distill-plan-wiring` (do not silently branch from the dirty tree state).
- When staging edits to a pre-modified tracked file, use `git add -p` and stage
  only your harness hunks. Never `git add` a whole pre-modified file blindly.
- Never run `git checkout`/`git restore` on a file that carries uncommitted work.

Interpreter: `./.venv/bin/python` if present, else `python3`.
Targeted tests: `./.venv/bin/python -m pytest tests/test_harness_distill.py -q`.

---

## Task 1 - Shared harness-provenance helper (tests first)

Add to `src/harness_distill.py` a single source of truth:

```python
def harness_provenance(spec: dict) -> dict:
    """Return the provenance block stamped onto every harness-aware trace."""
    # {name, status, digest, model_registry, prompt_sha256|None,
    #  prompt_verified, tool_policy_digest, verification_policy_digest,
    #  schema_version}
```

It must reuse `canonical_digest` and must not invent a prompt hash: if the
harness prompt is unverified, `prompt_sha256` is null and the block records
`prompt_verified: false`.

**Acceptance:** new unit tests in `tests/test_harness_distill.py` prove the
block is deterministic, contains the same digest as `plan`, and carries
`prompt_verified: false` for the current draft harnesses. Existing 15 tests
still pass.

## Task 2 - src/verify_harness_identity.py (resolve the prompt warning)

Offline verifier that resolves the "system prompt identity is unverified"
warning:

- given a harness name, read the declared `prompts.system.path`;
- if the file exists, compute its sha256, and with `--write` fill
  `prompts.system.sha256` and set `verified: true` only on a clean match;
- if the path is declared but missing (e.g. the add-in sibling is absent), fail
  closed with a specific message; do not guess a hash;
- never read a credential or the network.

**Acceptance:** against a tmp prompt file it flips `verified: true`; against a
missing path it exits non-zero; `tests/test_verify_harness_identity.py` proves
both with fixtures (no add-in required).

## Task 3 - --harness in generation, fail-closed, attribution stamped

Add an optional `--harness <registry-name>` to `src/generate.py`:

- load and `validate_harness` the named harness BEFORE any network/client
  construction; an unsafe or missing harness aborts before generation;
- when `--harness` is given, stamp `harness_provenance` (Task 1) onto every
  written trace, alongside the existing `provenance` block;
- when `--harness` is omitted, behaviour is unchanged and no `harness_provenance`
  key is added (legacy path stays byte-compatible);
- the harness's declared `model_contract.registry_model` must be consistent with
  the teacher/model actually being generated from, or abort.

Keep the edit minimal and stage only these hunks.

**Acceptance:** `tests/test_generate_harness.py` drives generation with a mocked
client (no network) and asserts: unsafe harness aborts pre-network; with
`--harness` every trace has a matching `harness_provenance.digest`; without
`--harness` output is unchanged. Mark any test that needs the real corpus with
`requires_local_corpus`; these should need only fixtures.

## Task 4 - Pin the harness digest in manifest and freeze (additive, opt-in)

- In `src/pass_manifest.py`, when a pass's traces carry `harness_provenance`,
  record the single harness digest (and refuse a pass that mixes two digests).
  A pass with no `harness_provenance` records `harness: null` and stays valid.
- In `src/freeze_training_run.py`, add a `harness` block to the freeze:
  - for a legacy (unattributed) corpus, `harness: {attributed: false}` and the
    freeze proceeds exactly as today;
  - for a harness-aware corpus, pin the digest and refuse on a missing or mixed
    digest.
  Bump the freeze `schema_version`, update the trainer's expected version and
  `tests/test_training_manifest.py`, and regenerate the checked-in
  `train/RUN_MANIFEST.v1.json` through the freezer (do not hand-edit); archive
  the prior schema byte-for-byte under `train/archive/` as before.

**Acceptance:** the legacy corpus still freezes green with
`harness.attributed: false`; a synthesized harness-aware fixture pins its digest
and refuses when a trace is missing the digest or two digests are mixed. The
existing manifest tests pass after the version bump. This work reads the real
corpus, so its new tests are `requires_local_corpus`.

## Task 5 - Corpus gate for harness attribution (must not touch legacy)

Add a gate that requires a uniform, present harness digest ONLY when the corpus
declares harness-aware mode (an explicit field on the pass/candidate manifest,
not inferred). The legacy int4 corpus, which does not declare it, is unaffected
and its gate result is unchanged.

Reuse the `verify_corpus` fail-closed style. Do not add a silent skip.

**Acceptance:** a harness-aware fixture with a missing/mixed digest fails the
gate; the legacy corpus gate output is byte-identical to today. Prove the legacy
result is unchanged.

## Task 6 - Wire audit into reporting; document the bakeoff precondition

- Have `distill_plan.py provenance-audit` (or a thin call into
  `harness_distill.audit_traces`) include harness coverage in its report so one
  command shows model quant, source_class, FP8 status, licence freshness, and
  harness attribution together.
- Extend `docs/HARNESS_AWARE_DISTILLATION_ARCHITECTURE.md` with a short "how to
  run the four-arm bakeoff" section that is documentation only: it requires a
  real, approved capability gap and produces a measurements JSON for
  `harness_distill.py evaluate`. Emit no traces and train nothing.

**Acceptance:** the combined audit prints harness coverage; the doc section
exists; no new code performs a network call or generation.

---

## Global acceptance

```
./.venv/bin/python -m pytest tests/test_harness_distill.py \
  tests/test_verify_harness_identity.py tests/test_generate_harness.py -q
./.venv/bin/python -m pytest tests/ -q -m "not requires_local_corpus"
# on a machine with the corpus present:
./.venv/bin/python -m pytest tests/ -q
./.venv/bin/python src/harness_distill.py audit-traces data/promoted/train.jsonl
```

Expected: harness tests green; the corpus-free partition green; the full suite
still 333-passed-class (plus the new corpus-free tests); the legacy corpus audit
still reports `harness_attributed: 0` and `harness_coverage: 0.0` until it is
regenerated with `--harness`.

## Commit plan (separate, ordered)

```
feat: shared harness provenance helper
feat: verify harness system-prompt identity
feat: stamp harness provenance in generation
feat: pin harness digest in pass manifest and freeze
feat: gate harness-aware corpora on attribution
docs: harness bakeoff procedure and combined audit
```

Inspect each with `git show --name-status` and `git diff --cached` before
committing; confirm no unrelated in-flight strand entered the commit.

## Definition of done

- All tasks' acceptance criteria met; targeted and corpus-free suites green;
  full suite green on a corpus-equipped machine.
- No training/cloud run/model call/generation was performed;
  `train/TRAINING_BLOCKED.md` untouched.
- No harness digest, prompt hash, or model revision was fabricated; unverified
  harnesses still report `prompt_verified: false`.
- The legacy corpus, its gate result, and the 333-passing baseline are
  unchanged; harness attribution is enforced only for harness-aware corpora.
- Every new command still emits `training_authorized: false` where applicable.
