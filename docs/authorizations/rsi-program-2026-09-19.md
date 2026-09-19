---
authorization_id: rsi-program-2026-09-19
subject: Recursive Self-Improvement programme, Phases 1-2 landed; later phases gated
granted_by: project owner
granted_at: 2026-09-19
transcribed_by: agent (Claude Opus 5), from the owner's instruction of 2026-09-19
scope: >
  Execution of the RSI programme may proceed, but only after each existing
  evidence, provenance, regression, identity, cost, security and data-egress
  gate passes on its own terms.
conditions:
  - evidence
  - provenance
  - regression
  - identity
  - cost
  - security
  - data_egress
training_authorized: false
external_spending_limit: 0
data_egress: denied
independent_reviewer: null
reviewed_at: null
---

# RSI programme authorization — 2026-09-19

## What was granted

The project owner granted conditional authorization, dated 19 September 2026,
to proceed through the RSI programme.

The authorization permits **execution only after** each existing evidence,
provenance, regression, identity, cost, security and data-egress gate passes.
In the owner's words, it "does not authorize bypassing or weakening a gate."

## What it does not grant

**Training remains unauthorized.** `training_authorized` stays `false` until
the harness-before-weights decision in
`configs/experiments/harness-before-weights.yaml` qualifies weight
distillation, and that decision produces a `weight_distillation_candidate`,
never an authorization. Weight training additionally requires a separate
`train/TRAINING_JUSTIFIED.md`, reviewed and signed. `train/TRAINING_BLOCKED.md`
remains controlling and is untouched by this record.

**External spending defaults to zero.** No cloud run, no rented GPU, no paid
API call. A limit exists only when a number and a currency are recorded here.

**Data egress is denied.** No document text, corpus, trace or measurement
leaves the machine. An allowlist exists only when it is written here, host by
host.

Both defaults are restrictive on purpose: an omitted limit reads as zero, never
as unbounded.

## Status of the gates, at the date of this record

| Gate | State on 2026-09-19 | Evidence |
|---|---|---|
| Evidence (gold set) | **BLOCKED** — 0 approved items | `src/gold_set.py --gate` exits 2 |
| Provenance | enforced | `source.kind == human_authored`, distinct approver, ISO dates |
| Regression | green on the CI-gated partition | `tests-no-corpus` passed on PR #22 head in 16m02s |
| Identity | enforced | teacher-in-student-slot refused; `harness_distill.canonical_digest` per candidate |
| Cost | zero by default | no spending limit recorded |
| Security | executor quarantined | `synthetic_fixture` only; production items fail closed |
| Data egress | denied by default | no allowlist recorded |

Phases 1-2 landed as PR #22 (merge commit `a52cbc2`, 44 files). The separation
gate, live GEPA and any weight work remain blocked behind the sequence below.

## The sequence this authorization permits

1. **Authoring.** The owner and an independent reviewer supply the first 40-60
   native Indonesian gold items, per `docs/gold_set_authoring_packet.md`. An
   agent may not author, approve, or countersign an item.
2. **Separation gate.** Once approved items exist, run
   `src/eval_harness.py separation-gate` for 9B against 4B. It must separate on
   every split and be stable within the 0.025 noise floor. Unproven stability
   is not proven stability.
3. **Live GEPA.** Wired and run only after that gate is valid. Rollouts go
   through `src/bridge_client.py`; the run is inference-only and touches no
   weights.
4. **Weight training.** Not reachable from here. It requires the factorial
   evidence to show a residual model gap that survives both the declared
   minimum effect and the noise floor, plus a reviewed and signed
   `train/TRAINING_JUSTIFIED.md`.

Each step is a gate. A failure stops the sequence; it does not lower the bar.

## Baseline recorded at authorization

The pre-existing `origin/main` test failures are recorded in
`docs/authorizations/origin-main-baseline-2026-09-19.md`. No threshold was
modified to accommodate them, and none is proposed here.

## What a reviewer still owes

`independent_reviewer` and `reviewed_at` above are null. This record is a
transcription of an authorization the owner gave; it has not been independently
reviewed. A reviewer filling those fields is attesting that the scope,
conditions and defaults written here match what was actually granted.
