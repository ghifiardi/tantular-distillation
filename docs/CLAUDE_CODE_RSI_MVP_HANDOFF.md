# Implementation handoff — RSI MVP, Phases 1–2 (for Claude Code)

**Read this whole file before writing any code. Execute tasks in order. Do not
skip the guardrails.** This turns the "Recursive Self-Improvement Report v2"
(17pp, dated 2026-09-19) into wired, tested, executable-now scaffolding **in
this repo**. It does **not** authorize any training run.

Source of the plan: `Recursive Self Improvement/Recursive Self-Improvement
Report v2.pdf` (in the parent workspace). This handoff implements only the parts
the report itself says to do first and that need no gradient step: the
evaluation harness (report §J wk 1–2, §G) and GEPA (report §B.1, §J wk 3–4).

## What the report actually concludes (the parts that bind this work)

- **"Your bottleneck is not the algorithm. It is an Indonesian verifier you
  trust."** Build the eval first; everything downstream is gated on it.
- **Do the no-gradient loop first.** GEPA beat GRPO by up to 20% with ~35× fewer
  rollouts on Qwen3 8B — almost exactly Tantular's base family. It optimizes
  prompts/tool-schemas and touches no weights.
- **The 4B may not self-improve at all** (AdaSTaR: every STaR method failed on a
  3B base). Plan the 4B as a distillation target of the improved 9B, never as
  its own loop. → No 4B self-improvement code in this handoff.
- **Correct answer ≠ correct reasoning.** Skip rationalisation; track a
  false-positive rate. → the verifier gate is the load-bearing component.
- **No published result in the report was measured on an Indonesian-first
  model.** Every number is out-of-distribution for Tantular; the harness exists
  to produce Tantular's *own* first measurement.

This aligns with the repo's existing `docs/HARNESS_AWARE_DISTILLATION_ARCHITECTURE.md`
(harness-before-weights) and the already-present
`configs/experiments/harness-before-weights.yaml`. GEPA is the concrete
mechanism for the `student_candidate` arm defined there.

## GUARDRAILS — non-negotiable

1. `train/TRAINING_BLOCKED.md` is controlling. **Do not start, enable, or
   authorize any training, DPO, RL, or cloud run.** No task here needs a GPU.
   Phases 3–5 of the report (ReSTEM, language-imbalance DPO, 4B distillation)
   are **out of scope** — do not implement, scaffold, or wire them.
2. **Do not fabricate** eval items, gold-set questions, sources, digests, or
   scores. The report's 300-item native gold set is **human-authored approved
   input**, exactly like `data/raw/` (currently empty, awaiting approved
   sources). Build the schema, loader, splitter, scorer, and gate; if the real
   items are absent, the harness **fails closed** — it reports BLOCKED, it never
   invents Indonesian questions or synthetic "gold". Synthetic material may seed
   GEPA's *train* rollouts, never the gate set.
3. **Do not loosen any gate.** Reuse `src/run_gates.py` semantics (before/after,
   "measured" ≠ "passed", fail-closed on missing endpoint/scorer/output).
4. New code is **fail-closed** in the repo's `die()`/`sys.exit(...)` style
   (`src/verify_corpus.py`, `src/harness_distill.py`). A check that cannot run
   must refuse, not pass.
5. **Offline and deterministic tests.** No network in `tests/`. GEPA's live
   rollouts go through the existing `src/bridge_client.py` URL abstraction and
   are exercised in tests only via a fake/stub client.
6. Single-source shared rules; import, don't re-list (e.g. reuse
   `harness_distill.canonical_digest`, the split/leakage logic in `src/splits.py`
   and `src/source_split_audit.py`, and `run_gates` scoring where applicable).
7. After each task run the full suite; do not proceed with a red suite.

### Recorded baseline exception — 2026-09-19

Task 1.1/1.2 exposed a pre-existing red partition in
`tests/test_run_gates.py`. It is not imported by, and does not import, the new
`gold_set` or `verifiers` modules. The root cause was confirmed on 2026-09-19:
`train/qlora_9b.yaml` points `office_json_contract` at
`../tantular_office_addin/tests`, so the model-independent gate executes the
sibling add-in's JavaScript build-health suite. That sibling had 74 modified
files. The observed result was 843/886 passing
(`rate == 0.9514672686230248`) against a `0.98` threshold, which makes existing
positive-path assertions fail. Earlier dirty-tree snapshots produced a
different sub-threshold rate, so the exact rate is not a stable baseline. The
partition also starts roughly 23 subprocesses and can take 20–30 minutes
locally.

This is a **narrow baseline exception**, not permission to ignore regressions:

- Continue Tasks 1.3 onward without modifying `src/run_gates.py`,
  `tests/test_run_gates.py`, its scorer, or its threshold.
- After every task, run the task's focused tests and:
  `./.venv/bin/python -m pytest tests/ -q --ignore=tests/test_run_gates.py`.
  The verified baseline on 2026-09-19 is **354 passed, 4 skipped**.
- Keep each RSI task additive. If new code becomes reachable from
  `run_gates.py`, or if any non-`run_gates` test fails, stop immediately.
- Re-run `tests/test_run_gates.py` only at the Phase 1 and Phase 2 milestones,
  with a bounded timeout. Before comparing results, record the sibling add-in
  HEAD and `git status --short`, and ensure it is not being edited during the
  run. Its known failures may remain; no new failing Python test name is
  allowed for the same sibling snapshot. If the sibling snapshot changed, the
  result is non-comparable evidence, not a regression or a pass.
- Diagnose/fix the existing `run_gates` red in a separate change. Never lower
  `0.98`, mark the gate skipped, or edit an assertion merely to turn the suite
  green. The durable fix is to make the gate run the declared pinned add-in
  baseline rather than an arbitrary dirty sibling, but that is outside this
  RSI change.
- **Never remove, rename, or move RSI source files out of the active working
  tree to prove isolation.** A shell trap or Python `finally` is not recovery
  against `SIGKILL`, host restart, or runtime termination. Use static import
  analysis plus a disposable copy/worktree under `/private/tmp` when an
  absence experiment is genuinely necessary. The active tree must remain
  complete throughout every background test.

Interpreter: `./.venv/bin/python` (Python 3.14 here). Test command:
`./.venv/bin/python -m pytest tests/ -q`.

---

## Phase 1 — Evaluation harness (report §J wk 1–2, §G, §H)

**Goal (report's gate):** the harness reliably separates Tantular-9B from
Tantular-4B, and re-runs are stable within a declared noise floor. Until that is
true, no later phase is worth its compute.

### Task 1.1 — Gold-set contract and loader (tests first)

Add `src/gold_set.py` + `tests/test_gold_set.py`. Define a JSONL record schema
for the native Indonesian gold set with, per item: `id`, `split` ∈
{`reasoning`,`instruction_following`,`knowledge`,`tool_use`}, `prompt`,
`verifier` (`exact_match` | `executor` | `schema` | `judge`),
`expected` (answer/schema/test as applicable), `source` (human author/provenance),
`source_class` (defaults to `internal`, mirroring the repo rule), and
`language` (must be `id`).

- The loader validates every record and **refuses** on: unknown split, missing
  verifier, a `judge`-only item counted toward a correctness score, empty
  provenance, or `language != id`.
- Provide a `data/gold/` directory with a `README.md` and a
  `SCHEMA.md` only — **no items**. Absence of approved items is the correct
  BLOCKED state; the loader says so and exits non-zero when asked to gate.
- Tests use tiny in-repo fixtures under `tests/fixtures/gold/` (a handful of
  clearly-synthetic rows, never presented as production gold).

**Acceptance:** `pytest tests/test_gold_set.py -q` passes; loading the empty
`data/gold/` fails closed with a specific "gold set is BLOCKED: 0 approved
items" message.

### Task 1.2 — Verifier tier (report §A, §H.1)

Add `src/verifiers.py` + tests. Implement, in descending trust order:
`exact_match`, `schema` (JSON-schema/contract validation, reuse the edit-contract
checker in `scripts/check_edit_contract.mjs` semantics where relevant),
`executor` (run code against tests in a subprocess with a timeout; no network),
and a `judge` stub that is **quarantined** to a style corpus and may never enter
a correctness score. Track and return a `false_positive`-eligible flag per item
so §H.3 sampling can consume it later.

**Acceptance:** each verifier has unit tests including a deliberate
false-positive case (right answer / wrong reasoning) that the harness records
rather than silently accepts.

### Task 1.3 — Separation + stability gate (report §J wk 1–2 gate)

Add `src/eval_harness.py` with a `separation-gate` subcommand that consumes two
measurement files (9B and 4B, produced by `run_gates`-style runs against served
endpoints via `bridge_client`) and decides:

- **separation:** 9B minus 4B exceeds the declared effect on each split, using
  the paired-statistics approach already in the repo
  (`src/cross_pass_report.py` / the capability-metric work on `origin/main`);
- **stability:** re-run variance is within `noise_floor` from
  `harness-before-weights.yaml` (0.025).

Fail closed if either measurement file is missing a split, an endpoint, or a
model identity. Reuse `harness_distill`/`run_gates` verification of served model
identity — never accept a teacher's numbers in a student slot.

**Acceptance:** with two fixture measurement files the gate returns a structured
verdict; a missing split or a within-noise "separation" both fail the gate with
specific messages. No network.

### Task 1.4 — Public-suite adapters, reporting-only (report §G)

Add thin, **offline** adapters that *format* SEA-HELM and IndoMMLU results into
the harness's report schema **if** result files are supplied. Do not download,
run, or vendor those suites. Document in the module docstring that these are for
reporting/calibration only and never steer the loop (report §G).

**Acceptance:** given a fixture SEA-HELM/IndoMMLU result file the adapter
normalizes it; given none, it is simply absent from the report, not fabricated.

---

## Phase 2 — GEPA on the tool-use path (report §B.1, §J wk 3–4)

**Only start Phase 2 after Phase 1's suite is green.** GEPA is inference-only:
it optimizes prompts / tool descriptions, keeps a Pareto frontier over
individual instances, and touches no weights. It maps onto the
`student_candidate` arm of `harness-before-weights.yaml`.

### Task 2.1 — GEPA controller (tests first, stub client)

Add `src/gepa.py` + `tests/test_gepa.py` implementing the report's loop:
candidate prompt pool → select from Pareto frontier (not just best-scoring) →
run system on a train minibatch via `bridge_client` → `metric + textual
feedback` → reflect to propose a mutated prompt → accept on a held-out slice →
return the Pareto-frontier merge. Constraints:

- **Reflection prompts and mutated instructions are written in Indonesian**
  (report is explicit: an English-optimized-then-translated prompt loses the
  register control that is the whole point).
- Operate on the **tool-use** slice first (most prompt-structure leverage, most
  automatable verifier).
- Every candidate carries a `harness_identity` digest via
  `harness_distill.canonical_digest`. A trajectory without model+harness
  identity is rejected (matches the architecture doc's attribution rule).
- The rollout budget is an explicit argument; default small. No hidden loops.
- Tests inject a deterministic fake client (no network); assert Pareto
  acceptance never regresses a per-instance score, and that a mutation which
  improves two instances but breaks one is **not** promoted on aggregate
  (the single-lineage regression trap the DGM section warns about).

**Acceptance:** `pytest tests/test_gepa.py -q` passes offline; a `--dry-run`
prints the command/rollout sequence and writes nothing.

### Task 2.2 — Wire GEPA output into the harness-before-weights evidence

A completed GEPA run emits a candidate harness + its held-out score. Feed that
into `harness_distill.evaluate` as the `student_candidate` measurement so the
existing decision gate can compute **harness gain** and **residual model gap**.
Do not change what that gate decides; GEPA only supplies one arm's numbers.
Every result still carries `training_authorized: false`.

**Acceptance:** given a fixture GEPA result, `harness_distill evaluate` produces
the four-arm attribution with `student_candidate` populated from GEPA, and the
verdict remains `training_authorized: false`.

---

## Out of scope (report Phases 3–5) — leave BLOCKED

ReSTEM (§C.3, wk 5–7), Self-Instruct (§C.4), SEAL (§C.5), language-imbalance DPO
(§D.6, wk 8–10), self/meta-rewarding (§D.7), Absolute Zero (§D.8), and 4B
distillation (wk 11–12) all require training and/or GPUs. **Do not implement or
scaffold them.** They stay gated behind a future `train/TRAINING_JUSTIFIED.md`,
and the report itself says none are worth starting before the gold set and gates
exist — which is exactly what Phases 1–2 build.

## Global acceptance (run at the end)

```
./.venv/bin/python -m pytest tests/ -q
./.venv/bin/python src/eval_harness.py separation-gate --help
./.venv/bin/python src/gepa.py --dry-run
./.venv/bin/python src/harness_distill.py plan configs/experiments/harness-before-weights.yaml
```

## Definition of done

- Phase 1 and Phase 2 acceptance criteria met; full suite green; no network in
  tests.
- The gold set fails closed with 0 approved items; no eval item, score, or
  digest was fabricated.
- No training/DPO/RL/cloud run was started, enabled, or authorized;
  `train/TRAINING_BLOCKED.md` untouched; every emitted verdict says
  `training_authorized: false`.
- A short PR description lists every file changed and states explicitly that no
  training/cloud run was performed and that Phases 3–5 remain out of scope.
