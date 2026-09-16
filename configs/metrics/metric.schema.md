# Metric registry schema

One file per metric under `configs/metrics/`. A metric is a DEFINITION, not an
implementation: the file says what counts as a pass and what may not, and the
scorer is checked against it rather than the other way round.

The canonical digest is computed over the whole YAML object except an optional
`digest` field — the same construction `harness_distill.canonical_digest` uses
for a harness, and for the same reason: an artifact that carries its own digest
must not be trusted to describe itself.

## Fields

| field | meaning |
|---|---|
| `schema_version` | schema version, currently 1 |
| `name` / `version` | metric identity; a receipt and a measurement bind to both |
| `unit` | what one observation is (`case`) |
| `value_type` | `boolean` — a per-case outcome. A rate is an aggregate, never a case result |
| `denominator_policy` | which cases count. `all_declared_eligible_cases` keeps refusals, errors and timeouts in |
| `minimum_independent_cases` | the floor for calling a run statistically qualified |
| `pass_condition.required` | every one must hold |
| `pass_condition.conditional` | scored only when the case declares it |
| `guardrails_excluded` | metrics deliberately NOT folded into the Boolean |
| `expected_actions` | what a case may correctly ask for |
| `noise_floor` | `declared`, `status` (`provisional` \| `measured`), `evidence` |
| `live_support` | which expected actions the live protocol can actually execute |
| `failure_reasons` | the closed vocabulary; a reason outside it is an error, not a new category |
| `predeclared_exclusions` | case ids fixed as ineligible BEFORE arm assignment |

## Why the denominator keeps failures

A case that errored still asked its question. Dropping it lets a flaky arm
score 1.0 on the cases that happened to run, which is the most flattering
possible reading of the least reliable behaviour. The only exclusion is
`predeclared_exclusions`, and it is fixed before arm assignment precisely so it
cannot be chosen after seeing results.

## Why guardrails are excluded

`edit_contract_output` and `indonesian_voice` are configured in
`configs/experiments/harness-before-weights.yaml` with their own minimums and a
zero-regression rule. If they were also inside capability, a voice regression
would move both numbers and `evaluate()` could no longer distinguish
`harness_optimization_sufficient` from `guardrail_failures`.

## `noise_floor.status`

`provisional` means the value is declared but unmeasured, and a report must say
so. It becomes `measured` only when `evidence` names a real record of observed
spread across repetitions at fixed decoding. Nothing may set it by hand from a
number that looked reasonable.

## `live_support`

Separate from `expected_actions` because what a case may ask for and what the
runner can execute are different questions. `no_edit: false` records that
`OfficeLiveExecutor` requires exactly one edit per case, so a case whose
correct answer emits none is scorable from fixtures and not executable live.
Reporting those together would hide a protocol gap behind a metric definition.
