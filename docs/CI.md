# Continuous integration — what it covers, and what it does not

CI runs one job, **`tests-no-corpus-no-addin`**. The name is the summary: it is
not the full suite, and it is not named as though it were.

    pytest tests/ -q -m "not requires_local_corpus and not requires_addin"

Two partitions are excluded, each for a stated reason, each counted and
asserted in the workflow. Neither is a runtime skip.

## The rule both markers follow

There is no `skipif` and no existence check anywhere. Run the suite locally
without these inputs and the marked tests **run and fail** — a check that cannot
run must refuse, not pass, which is the same rule the gates themselves follow.

Deselection happens only in `.github/workflows/tests.yml`, as an explicit `-m`
expression a reader can see. It is a workflow policy, not a property of the
tests.

## Gap 1 — the corpus (47 tests)

`.gitignore` excludes `*.jsonl` (except `prompts/*.jsonl`), `data/raw/` and
`data/promoted/`: *"Corpora and weights never belong in git — HF hosts those."*
A fresh clone therefore lacks:

    data/v3-candidate/traces.r0.jsonl      the source corpus a freeze pins
    data/promoted/train.jsonl              the promoted training split
    data/promoted/eval.jsonl               the promoted held-out split

Forty-seven tests read them, across `tests/test_training_manifest.py`,
`tests/test_tinker_sft.py` (the freeze, payload-rendering and checkpoint-label
paths all start from a real freeze) and
`tests/test_run_gates.py::test_trainer_refuses_ai19_end_to_end`.

That last one is marked rather than refactored because its dependency is
STRUCTURAL: `check_run_freeze` re-verifies the corpus digest before the host
guard is reached, so the real trainer path cannot be exercised without the real
corpus. A temporary fixture would mean forging a freeze whose digests match a
fake corpus and a matching promotion manifest — defeating exactly the pinning
the freeze exists to enforce.

**Residual risk.** Four corpus-free tests still cover the guard's behaviour
(`test_ai19_refuses_to_train`, `test_a_silent_host_refuses_to_train`,
`test_declaring_a_rental_does_not_help_when_running_ON_ai19`,
`test_training_allowed_on_a_declared_rental`). What CI stops proving is that the
guard is WIRED INTO `train_qlora.main()` — only that it refuses when called. A
regression removing the call site would pass CI.

*Optional follow-up, not taken here:* that wiring test is impossible only
because of the ORDER in `train_qlora.main()` — `check_run_freeze` runs before
the host guard. If the guard ran first, a wrong `--train-host` would fail fast
without touching the corpus and the wiring assertion could return to CI. That is
a trainer-ordering question with its own tradeoffs (failing early on a bad
freeze is also worth something), so it is recorded rather than decided.

## Gap 2 — the Office add-in (37 tests)

`train/qlora_9b.yaml` points the `office_json_contract` and
`edit_contract_output` gates at `../tantular_office_addin` — a SIBLING of this
repository, living in `ghifiardi/LLM-Indonesia`. CI cannot obtain it:

- `tantular_office_addin` is **not on that repository's default branch**. `main`
  there contains only `docs` and `eval_sets`. The directory exists on several
  feature branches, but not on the branch a checkout defaults to.
- `tantular_office_addin/package-lock.json` is **not tracked on any ref**. It
  exists only as an untracked file locally, so `npm ci` — a lockfile-reproducible
  install — cannot run in CI against anything.

Pinning a feature branch and using `npm install` instead would produce a
non-reproducible dependency tree pinned to a moving ref. That trades away the
determinism the gates depend on, so it is not done.

**Thirty-seven tests** carry `requires_addin`: twenty-eight in
`tests/test_run_gates.py`, which fail without the add-in source, its parser or
its Node suite, eight in `tests/test_faithful_edit_scorer.py`, and one in
`tests/test_verify_harness_identity.py`.

That last one is the ONLY harness test needing the real add-in. The synthetic
prompt-registry tests beside it need `node` but not the add-in, and they run in
CI — they are what prove that changing a prompt moves the harness prompt digest
while changing an unrelated source file does not, which is the distinction the
whole prompt-registry approach rests on. Marking them would have removed that
proof from CI to save nothing.

Those eight were nearly missed. They already carried a pre-existing
`pytest.mark.skipif` (`needs_addin`) that makes them SKIP when the add-in is
absent, so a probe that looks for failures does not see them — they would have
disappeared from CI silently and uncounted, which is exactly what counting the
exclusions is meant to prevent. They now carry `requires_addin` as well, so CI
deselects them explicitly and the assertion counts them.

*Known inconsistency, not fixed here:* for those eight, an absent add-in
produces a skip locally rather than the failure the corpus markers produce. The
`skipif` predates this change and belongs to the faithful-edit strand; making
them fail-closed like everything else is a follow-up.

The `node_suite_lock` tests are NOT marked: they exercise POSIX `flock` and
never invoke the add-in. They do not exist on `main` — they arrive with the
teacher-agnostic distillation branch — so that branch must confirm they remain
SELECTED when it updates these counts.

**What loses CI cover:** `office_json_contract`, `edit_contract_output`, and the
faithful-edit scorer path. These remain local pre-merge checks.

**The durable fix** is upstream: publish the add-in and its `package-lock.json`
to a stable ref in `ghifiardi/LLM-Indonesia`, then restore the cross-repository
checkout. That is a decision for that repository's owner, tracked as a
follow-up, not worked around here.

Evidence that the cross-repo approach itself works: the first Actions run on
this branch checked out `ghifiardi/LLM-Indonesia` successfully without a token
and applied the sparse-checkout; it failed only on the layout guard, because the
directory is absent from `main`.

## Reproducibility risk, recorded

**The 333-passing baseline is not reproducible from currently published
repository refs.** Two independent causes:

1. The add-in state it was tested against is local `ab73a0b`; the published
   feature-branch state is `bb7918b`.
2. The tested `package-lock.json` is absent from every published ref, so the
   exact dependency graph cannot be reconstructed from Git alone.

Anyone reconstructing that result today would need files that exist outside
version control. This is the standing argument for the upstream fix above.

## Counting the exclusions

The workflow collects each partition into the job log every run and asserts its
size independently:

    EXPECTED_EXCLUDED_CORPUS: 47
    EXPECTED_EXCLUDED_ADDIN:  37

Both assertions fail in **both** directions. If a number moves, investigate; do
not update it to match. A changed count means a new dependency was introduced or
an existing test stopped exercising one, and both are worth knowing.

The **final deselected total is read from the run**, never computed as 47 + 37.
A test could in principle carry both markers, in which case the union is smaller
than the sum. Today the union collects 84, which happens to equal the sum — that is a
measurement, not an assumption, and it is re-measured whenever either number
moves. It has already moved twice: 13/28 on `main` before the distillation
branch landed, 37/36 after it, 44/37 with harness attribution, 46/37 once the
readiness block was pinned by tests that read the real corpus, 47/37 once the
legacy corpus was pinned as absent-not-malformed.

### A test that only LOOKED add-in dependent

`test_the_contract_checker_writes_its_cases_outside_the_repository` mocks its
subprocess and never runs the real parser, but `run_contract_checker` refuses
before calling anything unless `<addin_src>/chat/editContract.js` exists. It
therefore failed without the add-in while testing nothing about it. It now
builds a temporary fake add-in tree containing that sentinel file, and stays in
CI. Marking it `requires_addin` would have been the easy fix and the wrong one:
the hermeticity property it guards — that scoring leaves no artifact in the
working tree — is exactly the kind of thing CI should be checking.

*Measurement caveat:* the add-in partition was discovered by running the
corpus-free partition on a checkout with the add-in absent. It therefore
identifies add-in dependencies **only among the corpus-free tests**, and does not
establish whether any corpus-required test also depends on the add-in. That does
not affect CI, which excludes the union. Establishing independent marker
semantics would need a second controlled probe — corpus present, add-in absent —
which this bootstrap does not require.

## Interpreter: 3.14 only, for now

CI pins Python 3.14 because that is the interpreter the recorded full-suite run
was verified on, so a CI result and a local result mean the same thing. That is a
REPRODUCIBILITY choice, not a declared support floor: nothing here states 3.14 is
the minimum this project supports, and a matrix (3.12, 3.13, 3.14) is
deliberately deferred rather than decided by omission.

## One environment fact CI has to get right

**A real `.venv` at the repository root is required.** 55 files hardcode
`PY = str(ROOT / ".venv" / "bin" / "python")` and shell out to it. Installing
dependencies into the ambient interpreter passes collection and then fails at
subprocess time, so the workflow builds the venv and invokes pytest through it.

## Full pre-merge verification

A green `tests-no-corpus-no-addin` is **not** equivalent to a green full suite,
and should not be described as one. Full verification still requires, on a
machine with both the corpus and the add-in:

    pytest tests/ -q

Latest recorded result: **441 passed, 4 skipped**, exit 0, with
`git status --short` empty afterwards.
