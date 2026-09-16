# Harness evaluation protocol

`src/harness_distill.py evaluate` accepts measurements as a plain JSON object.
That is deliberate — it keeps the gate independent of how a score was obtained —
but it means a typed-in pass rate and a measured one are indistinguishable to
it. This protocol is what makes the difference real: a number reaches the gate
only by way of per-case execution receipts that name the experiment, the case
set, the arm, the model revision and the harness digest that produced them.

Nothing in this protocol runs a model, opens a socket, drives Office, or trains.
Every artifact carries `training_authorized: false`.

## The four arms

`configs/experiments/harness-before-weights.yaml` declares a 2×2: two models
(student `qwen35-9b-instruct`, teacher `muse-glimmer-30b`) × two harnesses
(`tantular-office-current`, `tantular-office-candidate`).

`teacher_candidate` used to be listed by `build_plan` and read by nothing.
`HARNESS_DISTILL.REQUIRED_ARMS` now includes it, and `evaluate()` requires every
configured metric on every arm — the capability metric *and* both guardrails.
Without the fourth cell the question the design exists to ask is unanswerable:

```
student harness gain = student_candidate  - student_current
teacher harness gain = teacher_candidate  - teacher_current
interaction          = teacher gain - student gain
residual model gap   = teacher_current - student_candidate
```

A negative interaction says the candidate harness helps the student more than
the teacher — the harness is recovering capability the student lacked. A
positive one says the harness needs capability the student does not have, which
is an argument *about* weight distillation rather than a substitute for it.

## 1. Case set (`schema_version: 1`)

What is asked. Canonically digested (SHA-256 over sorted-key compact JSON,
ignoring any self-declared `digest`), exactly like a harness definition.

| field | meaning |
|---|---|
| `schema_version` | protocol version |
| `name` | case-set identity |
| `approved` | **explicit** `true`/`false`. Absent is refused: an unstated approval is a set nobody reviewed |
| `split` | `held_out` or `fixture`. An approved fixture is refused |
| `source_class` | `real_office`, `synthetic` or `fixture` |
| `provenance` | `origin`, `created_at`, `note`, all required |
| `cases[].case_id` | stable, unique; the join key across arms |
| `cases[].request` **or** `request_ref` | inline payload, or `{uri, sha256}`. Exactly one. An undigested reference is a mutable case |
| `cases[].expected.scorers` | which metrics this case answers |
| `cases[].expects_state_change` / `requires_approval` | booleans; state change without approval contradicts every declared harness and is refused |
| `cases[].fixture` | deterministic replay data for the fake executor |

**There is no approved held-out product set.** `tests/fixtures/harness_cases/fixture-office-v1.yaml`
is a fixture: `approved: false`, `split: fixture`, `source_class: fixture`. It
may drive tests and may never back a capability claim.

## 2. Receipt (`schema_version: 1`)

What happened for one `(arm, case_id)` pair. Written atomically (temp file +
`os.replace`) under `<arm>__<case_id>.json`, so a duplicate collides on disk
rather than becoming two countable receipts, and a crash mid-write cannot leave
a half-file that later reads as malformed.

Binds: experiment name + digest · case-set name + digest · `case_id` · arm ·
model registry name, `model_id`, **pinned `model_revision`** · harness name +
digest · the complete block from `harness_distill.harness_provenance()` ·
`prompt_sha256` + `prompt_verified` · `tools_offered`, `tool_calls`,
`approvals` · `before_action`, `after_action`, `repair_attempts` ·
`termination_reason` · `budgets` and `budget_consumed` · `result_digest` ·
executor identity · `started_at`/`ended_at` · `status` · `scores` ·
`training_authorized: false` · `receipt_sha256` over all of it.

Rules that hold it together:

- the receipt's harness/model/prompt fields must agree with the provenance block
  it carries — otherwise it has two answers to "which harness ran";
- `scores` are **per-case booleans**. A rate is an aggregate, not a case result;
- `status: ok` must carry a result for every declared scorer; a non-`ok` status
  must carry none — a failed execution scored nothing;
- **a failed execution leaves a failure receipt.** Disappearing would make a
  broken arm indistinguishable from an arm nobody ran;
- the output itself is never recorded, only its digest: a receipt is provenance,
  not a copy of the customer's document.

## 3. Measurement (`schema_version: 1`)

What the receipts add up to. `harness_eval.aggregate()` takes the experiment,
the case set and the receipts — **there is no argument for supplying a score.**

Carries the experiment digest, case-set digest/approval/split, `measurement_class`
(`measured` or `fixture`), `produces_real_measurements`, `receipt_set_digest`,
per-arm identity, and per metric a `numerator`, `denominator`, `score`, and
`error`/`refused`/`missing` counts, plus the interaction block.

The denominator is **declared coverage, not observed success**: a case that
errored still asked its question, and dropping it would let a flaky arm score
1.0 on the cases that happened to run.

Aggregation refuses when: a receipt names another experiment or case set · an
arm/case pair has no receipt · the same pair appears twice · a receipt is
malformed or fails its own digest · an arm mixes model or harness identities ·
an `ok` receipt is missing a declared scorer · fixture and real receipts are
mixed · and, for a real measurement, when the prompt identity is unverified or
the case set is not approved.

## 4. Executor boundary

The controller does not import a model SDK — if it did, the thing deciding
whether a measurement is admissible would also be producing it. It talks to an
`Executor`: `identity()` and `execute(ExecutionRequest) -> ExecutionResult`.

**The executor observes; the controller judges.** Tool allowlist, approval for
state-changing tools, step/wall budgets and the repair budget are all applied by
`run_harness_evaluation._judge`, not self-reported. Unknown tools count as
state-changing.

- `FakeOfficeExecutor` replays the case set's `fixture` block. Deterministic
  (its timestamps are `fixture:<run_id>:<arm>:<case>:<rep>:<edge>`, so the
  receipt-set digest reproduces). `produces_real_measurements: false`.
- `RealOfficeExecutor` **refuses**.

### Unresolved dependency

There is no Office harness adapter. Running these harnesses for real means
supplying `office_read`/`office_edit`, enforcing the approval policy, running
`request_schema`/`target_location` before the action and
`edit_contract`/`faithful_edit` after it, honouring the repair budget, all under
companion-process or sandbox isolation. `src/generate.py` is not that runner —
the same gap the harness configs already record as
`trace_generation.supported_by_generate_py: false`. Until the adapter exists,
real measurement is unavailable and the honest output is a refusal.

## CLI

```
python src/run_harness_evaluation.py plan      <experiment> --cases <set>
python src/run_harness_evaluation.py run       <experiment> --cases <set> \
    --executor fake --output <receipts-dir> [--run-id ID]
python src/run_harness_evaluation.py aggregate <experiment> --cases <set> \
    --receipts <dir> --output <measurements.json> [--allow-fixture]
python src/harness_distill.py evaluate <experiment> <measurements.json>
```

`plan` writes nothing. `run` defaults to the fixture executor; an executor
claiming real measurements additionally requires `--real`, and passing `--real`
with a fixture executor is refused rather than silently relabelled. `aggregate`
refuses fixture receipts unless `--allow-fixture` is given, and then labels the
artifact `measurement_class: fixture`. Refusals exit 2.

## 5. Execution surface, and the live Office slice

A receipt now records **where** an execution happened, because "measured" does
not say measured against *what*:

| `execution_surface` | meaning |
|---|---|
| `document_text` | the model answered and the contract was checked against a document **string**. A real measurement of the text contract; **not** evidence about live Office behaviour. |
| `office_live` | the edit was applied to the user's open Word document, under an approval. |

A fixture executor may never claim `office_live`, a single measurement may not
mix surfaces, and both facts are enforced in `harness_eval.validate_receipt` /
`aggregate` rather than left to the reader.

### The approval protocol (add-in side)

A successful `office_live` receipt that called a state-changing tool must carry
approval evidence, or aggregation refuses:

```
token_id · approver · document_version · target_digest · edit_digest
nonce · idempotency_key · single_use: true
```

Three bindings, because there are three ways an approval stops describing what
happens: the **document** changed, the **target** moved (the same `find` occurs
many times — approving the third occurrence must not authorise the first), or
the **change itself** changed. Each is reported separately; they send you to
different places.

The companion mints the token at preview and consumes it once at execute,
deleting it on every path. **The client never asserts a digest** — it sends the
material and the companion hashes what it was actually given, so a pane cannot
replay an approval against different text by repeating an old hash. Digests and
sizes are audited; document text and edit bodies never are.

The executor holds no Office handle by construction: the companion decides, the
task pane performs, and `OfficeLiveExecutor` asks and records. It cannot apply
an edit even by mistake, which is what makes the approval more than ceremony.

### Scope: ONE Word edit

Office.js has no transaction across `context.sync()`, so a batch that fails
halfway leaves earlier edits applied with no rollback. Rather than pretend
otherwise, this slice refuses more than one edit per approval — at the pane, at
`prepareEdit`, and at the executor. A future batch must report explicit per-edit
status; it must never claim atomicity the platform does not provide.

## 6. Verifiers: declared is not implemented

`harness_verifiers.run_declared` resolves every check a harness declares. A name
with no implementation returns **`refused`**, never an empty list and never a
pass — a skipped check reads as a pass to anyone counting.

| check | state |
|---|---|
| `request_schema` | **implemented** — shape of the edit contract, against the add-in's own limits (20 edits, 2000-char `find`) |
| `target_location` | **consumes** the add-in's live resolution; refuses when none is supplied, because it cannot be evaluated from a text copy |
| `edit_contract` | **real**, via `scripts/check_edit_contract.mjs` and the add-in's own parser |
| `faithful_edit` | **refuses** — it needs a case's declared `must_preserve` spans, and no approved held-out set exists |

## 7. What still blocks a measurement

Two things, both out of scope here and neither fixed by this protocol:

1. **`capability_pass_rate` has no scorer.** It is the experiment's decision
   metric and nothing computes it.
2. **There is no approved held-out Office case set.** The three held-out sets in
   `prompts/` are `source_class: synthetic` and unapproved.

Until both are resolved, this protocol can produce receipts and refusals but
**no measurement**. Every artifact stays fixture/preflight class and carries
`training_authorized: false`.
