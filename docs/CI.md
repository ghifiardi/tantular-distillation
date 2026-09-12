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

## Gap 2 — the Office add-in (36 tests)

`train/qlora_9b.yaml` points the `office_json_contract` and
`edit_contract_output` gates at `../tantular_office_addin` — a SIBLING of this
repository, living in `ghifiardi/LLM-Indonesia`. CI cannot obtain it:

- `tantular_office_addin` is **not on that repository's default branch**. `main`
  there contains only `docs` and `eval_sets`. The directory exists on several
  feature branches, but not on the branch a checkout defaults to.
- `tantular_office_addin/package-lock.json` **is now tracked**, at the baseline
  tag `tantular-office-addin-harness-baseline-2026-09-11` (peeled commit
  `3e14d25468ab0cd793ba8dc48cf5f755796c94e2`). It is still absent from that
  repository's `main`. `npm ci` is reproducible against the tag, and a clean
  clone of it was verified to install and pass 71 of 72 add-in test files.

Pinning a feature branch and using `npm install` instead would produce a
non-reproducible dependency tree pinned to a moving ref. That trades away the
determinism the gates depend on, so it is not done.

**Thirty-six tests** carry `requires_addin`: twenty-eight in
`tests/test_run_gates.py`, which fail without the add-in source, its parser or
its Node suite, and eight in `tests/test_faithful_edit_scorer.py`.
`tests/test_verify_harness_identity.py` no longer contributes one: prompt
identity is pinned to a published snapshot and asserted offline.

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

**Half of the durable fix has landed.** The add-in and its `package-lock.json`
are published at the baseline tag above, which is what unblocked prompt
verification: both shipped harnesses now pin that snapshot and carry
`prompts.system.verified: true`, asserted offline in CI with no add-in checkout.

**This does not make the add-in partition green.** Restoring it needs two more
things that the tag does not supply:

- `tests/devServerCancellation.test.mjs` remains open. It ran past 704 seconds
  standalone — no pytest, no `run_gates`, no repository code — and is an
  upstream defect. It was excluded from the baseline's own acceptance run and is
  recorded there as a gap, not a pass.
- the workflow has not been redesigned around a tag checkout. It still points at
  a sibling path, and rewiring it is separate work.

So `office_json_contract`, `edit_contract_output` and the faithful-edit scorer
path remain local pre-merge checks.

Evidence that the cross-repo approach itself works: the first Actions run on
this branch checked out `ghifiardi/LLM-Indonesia` successfully without a token
and applied the sparse-checkout; it failed only on the layout guard, because the
directory is absent from `main`.

## Reproducibility risk, recorded

**The 333-passing baseline is not reproducible from currently published
repository refs.** Two independent causes:

1. The add-in state it was tested against is local `ab73a0b`; the published
   feature-branch state is `bb7918b`.
2. The tested `package-lock.json` was absent from every published ref, so the
   exact dependency graph could not be reconstructed from Git alone.

**Cause 2 is now resolved, and cause 1 is superseded rather than fixed.** The
lockfile is published at the baseline tag. `ab73a0b` itself is still not
reproducible from a pristine workspace: its `/api/workspace` handler answered
304 to an unconditional read whenever `data/workspace.json` was absent, so
`tests/devServerStatic.test.mjs` fails in any clean checkout. Every developer
tree masked it by already holding that file. The baseline tag is therefore cut
from a commit that fixes it, not from `ab73a0b`.

The 333-passing figure remains a local historical result. What is reproducible
today is the baseline tag: a clean clone installs from the lockfile and passes
71 of 72 add-in test files, with `devServerCancellation.test.mjs` excluded.

## Counting the exclusions

The workflow collects each partition into the job log every run and asserts its
size independently:

    EXPECTED_EXCLUDED_CORPUS: 47
    EXPECTED_EXCLUDED_ADDIN:  36

Both assertions fail in **both** directions. If a number moves, investigate; do
not update it to match. A changed count means a new dependency was introduced or
an existing test stopped exercising one, and both are worth knowing.

The **final deselected total is read from the run**, never computed as 47 + 36.
A test could in principle carry both markers, in which case the union is smaller
than the sum. Today the union collects 83, which happens to equal the sum — that is a
measurement, not an assumption, and it is re-measured whenever either number
moves. It has already moved: 13/28 on `main` before the distillation branch
landed, 37/36 after it, 44/37 with harness attribution, 46/37 once the
readiness block was pinned by tests that read the real corpus, 47/37 once the
legacy corpus was pinned as absent-not-malformed, and 47/36 when prompt
verification removed the last add-in-dependent test in
`tests/test_verify_harness_identity.py`.

That last move is the only one in this milestone, and it is a removal, not a
reclassification. `test_the_real_harnesses_measure_but_stay_unpinned` asserted
that the shipped harnesses stay unpinned *because* the add-in was unpublished,
and it needed a real add-in checkout to say so. The add-in is published now, so
that assertion is false and its replacement — the exact pin, in
`test_the_shipped_harnesses_pin_the_published_baseline` — is a config assertion
that runs offline in CI and needs no checkout. The partition shrank because a
dependency genuinely went away.

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
