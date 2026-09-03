# Tinker SFT runbook

**Status on August 29, 2026: integrated for local preflight; cloud execution
remains blocked.**

Tinker is the managed training backend.  Ollama remains the local runtime.
Nothing in this integration changes the decision in
`train/TRAINING_BLOCKED.md`: no Tantular training is presently justified by an
observed product failure.

The checked-in schema-v3 manifest is consequently a **preview**:

`train/RUN_MANIFEST.tinker-sft-v1.preview.json`

It pins the backend, corpus, promoted train/eval bytes, exact rendered upload
payload, package versions, pricing observation, failed int4 corpus gate, and
signed waiver.  It has `execution_authorized: false` and cannot start a Tinker
run.

## Why the base model differs

The Tinker experiment uses `Qwen/Qwen3.5-9B-Base`, following Tinker's SFT and
distillation recipes.  The existing local config uses `Qwen/Qwen3.5-9B`.
Those are different checkpoints.

Therefore:

- the existing Qwen3.5-9B baseline cannot authorize this experiment;
- before and after gates must both evaluate Qwen3.5-9B-Base;
- the base is represented separately by
  `configs/teachers/office-student-9b-base.yaml`;
- a cross-model before/after comparison must never be reported.

The renderer is `role_colon`, which Tinker's current supported-model table
recommends for Qwen3.5-9B-Base.  The answer-only corpus does not contain
thinking blocks, but that is not a reason to substitute an instruct-model
renderer for a Base checkpoint.  Both Base-model endpoints are started with
the explicit `templates/role_colon.jinja` vLLM chat template and the
`"\n\nUser:"` stop sequence.  The Tinker renderer and the product-path
`/v1/chat/completions` endpoint must therefore produce the same token prefix.

## Client environment

The client can run on a Mac or another control machine; Tinker supplies the
training GPUs.  Keep its dependency set isolated from both the lightweight
repository environment and the local CUDA training environment.

```bash
python3.12 -m venv .venv-tinker
./.venv-tinker/bin/pip install -r requirements-tinker.txt
```

The runner asserts:

- `tinker==0.26.1`
- `tinker-cookbook==0.5.5`

before constructing a Tinker client.

## Safe operations available now

### Local dry run

```bash
./.venv/bin/python src/train_tinker_sft.py
```

This verifies:

- schema-v3 preview freeze and the existing schema-v2 corpus controls;
- promoted file digests and split fingerprint;
- held-out voice and edit-contract prompts;
- exact `source_class == "synthetic"` for all 136 train and 47 eval rows;
- deterministic upload and audit-sidecar digests;
- disjoint train/eval families;
- a conservative no-truncation sequence bound;
- the recorded pricing age and worst-case planning estimate.

It writes nothing, imports no Tinker package, reads no credential, and performs
no network call.

### Verify the real renderer and serving template

Create the client environment once. It is deliberately separate from `.venv`:
the pinned client pulls torch and transformers, which must not enter the CPU
test environment.

```bash
python3.12 -m venv .venv-tinker
./.venv-tinker/bin/pip install -r requirements-tinker.lock
./.venv-tinker/bin/python src/train_tinker_sft.py --verify-renderer
```

Install the **lock**, not `requirements-tinker.txt`. The direct pins do not
determine behaviour on their own: transformers and tokenizers decide the exact
token ids the renderer produces, and torch and safetensors decide the adapter
layout the exporter checks. `train/tinker_sft_9b.yaml: runtime_packages`
asserts that load-bearing subset at runtime and freezes it into the manifest,
so a run cannot be attributed to an environment nobody recorded.

The first run downloads the tokenizer. Afterwards, verification is reproducible
with no network at all — which is also how to run it under restricted
networking, where the Hub would otherwise raise a connection error:

```bash
env -u TINKER_API_KEY ./.venv-tinker/bin/python \
  src/train_tinker_sft.py --verify-renderer --offline
```

`--offline` sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` before any
loader runs. Without it, a cold cache plus no network is a controlled
`TINKER RUNNER ABORTED` naming the flag, never a traceback.

**Executed 2026-08-29 against tinker 0.26.1 / tinker-cookbook 0.5.5 and the
real `Qwen/Qwen3.5-9B-Base` tokenizer: PASS.** All 183 rows render 90–536
tokens, every loss span carries the assistant completion and no thinking
tokens, and the Tinker RoleColon generation prompt matches the Jinja template
token-for-token on every row.

Two findings from that run are why the template looks the way it does:

- `Qwen/Qwen3.5-9B-Base` publishes **no `chat_template.jinja`** (the instruct
  `Qwen/Qwen3.5-9B` does), so vLLM's chat endpoint has nothing to apply unless
  `--chat-template` is passed. That is not a preference; without it the
  product-path gates cannot run against this checkpoint at all.
- Its `tokenizer.bos_token` is `None`. `RoleColonRenderer._bos_tokens` returns
  `[]` in that case, but a bare `{{ bos_token }}` in Jinja renders the literal
  string `None`, prefixing every served prompt with it. The template therefore
  guards the token with `{% if bos_token %}`.

This still performs no Tinker API call and reads no credential.  It loads the
exact Qwen tokenizer and Tinker renderer, verifies every loss-bearing span,
rejects thinking tokens or truncation, and compares the renderer's generation
prompt token-for-token with the Jinja template passed to vLLM.  Cloud execution
also repeats this check before confirmation.

It additionally introspects the pinned cookbook surface — `train.Config`,
`FromConversationFileBuilder`, `ChatDatasetBuilderCommonConfig`, and both
`weights` helpers — and aborts locally if a release renamed or dropped an
argument this integration passes.  The local trainer learned that lesson when
`warmup_ratio` became `warmup_steps` and would have raised only after the model
had finished loading; here it would raise only after the billable client
existed.

### Render the would-be upload

```bash
./.venv/bin/python src/train_tinker_sft.py \
  --render-only ~/tantular-runs/tinker-v1-preview/payload
```

Each upload line contains only:

```json
{"messages":[
  {"role":"system","content":"..."},
  {"role":"user","content":"..."},
  {"role":"assistant","content":"..."}
]}
```

A local sidecar pins family, split, source class, source-document digest, and
the SHA-256 of the corresponding upload line.  The egress check therefore
applies to the exact bytes intended for upload without sending provenance
metadata to Tinker.

## What must exist before cloud execution

### 1. Evidence that training is justified

An actual observed failure must satisfy the reopening criteria in
`train/TRAINING_BLOCKED.md`.  Reviewers may then create:

`train/TRAINING_JUSTIFIED.md`

The document should identify the observed failure, evidence source, affected
product capability, why serving/prompting is insufficient, approved budget,
owner, and decision date.  The runner does not create this document and an
arbitrary substitute path is refused.

### 2. A live Qwen3.5-9B-Base before baseline

Serve the exact Base checkpoint on a non-production endpoint:

```bash
export HOST_BASE_URL=http://<student-host>:8020/v1
./scripts/serve_student.sh office-student-9b-base student-serve
```

Run the existing gates using the Tinker config:

```bash
./.venv/bin/python src/run_gates.py run \
  --config train/tinker_sft_9b.yaml \
  --stage before \
  --host student-serve \
  --teacher office-student-9b-base \
  --expect-model Qwen/Qwen3.5-9B-Base \
  --out ~/tantular-runs/tinker-v1/gates.before.json
```

The Tinker config digest-pins `train/qlora_9b.yaml` as the shared source of the
three gate definitions.  `run_gates.py` refuses if that shared config changes
without review.  Model-dependent results must be live, not fixtures.

### 3. A non-preview schema-v3 freeze

```bash
./.venv/bin/python src/freeze_training_run.py \
  --corpus data/v3-candidate/traces.r0.jsonl \
  --config train/tinker_sft_9b.yaml \
  --promotion-manifest train/RUN_MANIFEST.v1-mechanical.json \
  --waiver calibration/INT4_WAIVER.md \
  --backend tinker \
  --baseline-report ~/tantular-runs/tinker-v1/gates.before.json \
  --authorization train/TRAINING_JUSTIFIED.md \
  --out train/RUN_MANIFEST.tinker-sft-v1.json \
  --frozen-at <reviewed-ISO-8601-timestamp> \
  --write
```

Without both pinned artifacts, the freezer requires the explicit `--preview`
flag and writes `preview: true`.  The filename has no authorization meaning.
The checked-in preview is regenerated with:

```bash
./.venv/bin/python src/freeze_training_run.py \
  --corpus data/v3-candidate/traces.r0.jsonl \
  --config train/tinker_sft_9b.yaml \
  --promotion-manifest train/RUN_MANIFEST.v1-mechanical.json \
  --waiver calibration/INT4_WAIVER.md \
  --backend tinker --preview \
  --out train/RUN_MANIFEST.tinker-sft-v1.preview.json \
  --frozen-at 2026-08-29T12:00:00+07:00 \
  --write
```

## Starting the managed run

Pricing in the config is an observation, not a billing guarantee.  The runner
refuses execution when it is older than 30 days.  `--max-cost-usd` records the
approved ceiling and must cover the conservative estimate, but Tinker does not
currently expose a client-side hard billing cap through this runner.

The bound includes both training and scheduled evaluation.  For the current
configuration it covers 1,114,112 worst-case training tokens plus 962,560
worst-case evaluation tokens across five evaluations: 2,076,672 total, or
$3.0382 at the price recorded on August 29, 2026.

Five, not the four that `max_steps / eval_every` implies: cookbook training
loops commonly also evaluate at step zero or once after the final step, and a
"worst case" that assumes neither is not a ceiling the operator can approve
against.

```bash
export TINKER_API_KEY=...

./.venv-tinker/bin/python src/train_tinker_sft.py \
  --run-manifest train/RUN_MANIFEST.tinker-sft-v1.json \
  --run-dir ~/tantular-runs/tinker-v1 \
  --train-host tinker \
  --egress-approval <decision-or-ticket-reference> \
  --max-cost-usd <approved-ceiling> \
  --execute
```

The command performs exact renderer/tokenizer length validation before the
billable call.  It then requires an exact phrase from an interactive TTY.
Pipes, EOF, missing credentials, stale pricing, a preview manifest, or a
different host all abort.

Tinker's cookbook writes `checkpoints.jsonl` incrementally.  The Tantular
runner also writes `RUN.json` before the training call, on exceptions, as soon
as the call returns, and after final-checkpoint validation.  Raw checkpoint
lines and every parseable row are retained even if the final checkpoint schema
is unexpected.  Only a validated final row containing both `state_path` and
`sampler_path` produces status:

`trained_unvalidated`

That status is not promotable.

## Export to PEFT and validate

The official cookbook provides a direct Tinker-adapter to PEFT conversion.
Whether its concrete output is usable by the pinned vLLM path is measured, not
assumed:

```bash
./.venv-tinker/bin/python src/export_tinker_weights.py \
  --run-record ~/tantular-runs/tinker-v1/RUN.json \
  --output-dir ~/tantular-runs/tinker-v1-export \
  --execute
```

The exporter uses:

1. `tinker_cookbook.weights.download(sampler_path, ...)`
2. `tinker_cookbook.weights.build_lora_adapter(...)`

The exporter checks the resulting base id, target modules, non-zero LoRA B
tensors, and the measured Qwen3.5 `model.language_model` serving prefix.  It
refuses known-unservable `in_proj_a` / `in_proj_b` targets.  A failed static
check is recorded as `peft_incompatible`; a passing one is only
`peft_unvalidated`.

Do **not** automatically run `convert_adapter_for_vllm.py` in either direction.
That converter was qualified for the local PEFT output layout, not an arbitrary
Tinker export.  If the static check fails, retain both artifacts and investigate
the exact key mapping.  If it passes, the existing live distribution probe is
still required because a load log or plausible key prefix is not proof that
vLLM applied the adapter.

Serve it alongside the exact Base checkpoint:

```bash
./scripts/serve_student.sh office-student-9b-base student-serve \
  ~/tantular-runs/tinker-v1-export/peft-adapter \
  tantular-office-9b-tinker-v1
```

Then run the after gates:

```bash
./.venv/bin/python src/run_gates.py run \
  --config train/tinker_sft_9b.yaml \
  --stage after \
  --host student-serve \
  --teacher office-student-9b-base \
  --expect-model Qwen/Qwen3.5-9B-Base \
  --adapter ~/tantular-runs/tinker-v1-export/peft-adapter \
  --adapter-model-id tantular-office-9b-tinker-v1 \
  --out ~/tantular-runs/tinker-v1/gates.after.json

./.venv/bin/python src/run_gates.py compare \
  --before ~/tantular-runs/tinker-v1/gates.before.json \
  --after ~/tantular-runs/tinker-v1/gates.after.json \
  --json-out ~/tantular-runs/tinker-v1/gates.compare.json
```

`run_gates.py` still verifies that the adapter directory is PEFT-shaped, the
base identity matches, the adapter id is distinct and served, requests name
that adapter id, and the adapter changes the output distribution.  Only a
passing comparison can support a later promotion decision.

## Distillation comes after SFT

The first integration deliberately stops at SFT.  Tinker's official on-policy
recipe supports:

```text
Qwen/Qwen3.5-9B teacher
    → Qwen/Qwen3.5-9B-Base student
    → LoRA checkpoint
```

For Tantular, do not begin that phase until the SFT checkpoint is validated and
the Indonesian voice/edit-contract gates establish a trustworthy starting
point.  A future distillation manifest must separately pin teacher identity,
teacher terms, prompt dataset, KL settings, sampling settings, and its own
before/after evidence.
