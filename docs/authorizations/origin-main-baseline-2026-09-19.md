---
record_id: origin-main-baseline-2026-09-19
subject: Pre-existing test failures on origin/main at the RSI authorization date
recorded_by: agent (Claude Opus 5)
recorded_at: 2026-09-19
commit: a52cbc2
thresholds_modified: none
assertions_weakened: none
---

# origin/main test baseline — 2026-09-19

Recorded at the date of `docs/authorizations/rsi-program-2026-09-19.md`, so a
later run can tell a new failure from an old one. **No threshold was modified
and no assertion was weakened to produce these numbers.**

## The authority is CI, not a laptop

`tests-no-corpus` passed on the PR #22 head in 16m02s. It runs
`pytest tests/ -q -m "not requires_local_corpus"` in a clean environment, and
it is the check the repository actually gates on.

Everything below is a **local** measurement on one developer machine. It is
recorded because a local run is what a person sees first, and knowing which
failures are expected there prevents a future change being blamed for them.

## Local measurement, merged main (`a52cbc2`)

Partition CI gates, `-m "not requires_local_corpus"`:

| Run | Passed | Failed | Skipped | Deselected |
|---|---|---|---|---|
| without the RSI test files | 892 | 28 | 12 | 48 |
| with RSI | 1070 | 28 | 12 | 48 |
| **Delta** | **+178** | **0** | **0** | **0** |

Whole suite excluding `tests/test_run_gates.py`: **35 failed, 1030 passed,
12 skipped, 12 errors**.

The two failure counts differ (28 vs 35) because the second includes
corpus-marked tests that CI deselects.

## Where the local failures live

| File | Failed | Errors |
|---|---|---|
| `tests/test_tinker_sft.py` | 21 | 0 |
| `tests/test_training_manifest.py` | 6 | 12 |
| `tests/test_distill_plan.py` | 5 | 0 |
| `tests/test_verify_corpus_harness.py` | 2 | 0 |
| `tests/test_harness_distill.py` | 1 | 0 |

**No RSI test file appears in any failure or error list, in any run.**

## A measurement error worth recording

An earlier measurement reported **48** failures rather than 35. That run was
made in a fresh git worktree with **no `.venv`**, and several suites spawn
subprocesses via `<repo>/.venv/bin/python`. Thirteen of those "failures" were
`FileNotFoundError` on the interpreter path — an artefact of the measurement
environment, not a property of `origin/main`.

That number reached the PR #22 description before it was caught, and is
corrected in a comment on that PR. It is recorded here because the failure mode
generalises: **a count taken in a worktree is a measurement of the worktree**,
and a governance record that quotes one without saying so is misleading even
when every digit is accurate.

The delta was unaffected: it was always an apples-to-apples comparison inside
one environment.

## Known environment-coupled partition

`tests/test_run_gates.py` is excluded from the counts above. It runs the
model-independent `office_json_contract` build-health gate against the sibling
`tantular_office_addin` working tree (`train/qlora_9b.yaml:86`) rather than a
pinned baseline, measuring **0.9515** against the **0.98** threshold, and it
spawns roughly 23 subprocesses taking over 20 minutes locally.

The threshold stays at 0.98. Diagnosing that partition is separate work; this
record exists so it is not mistaken for RSI fallout.
