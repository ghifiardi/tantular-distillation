# Claude Code handoff: Milestone 5 — qualify the add-in suite,
# restore CI cover

**Read this whole file before changing anything. Execute the phases in order and
stop on a failed invariant.**

Applicable date: **13 September 2026**.

## Objective

> Qualify the complete published add-in suite and restore the 36
> `requires_addin` tests to CI.

Two repositories, two separate changes, in this order:

1. **Upstream, `ghifiardi/LLM-Indonesia`** — fix the test-harness defects
   found by measurement, then publish a new immutable **CI baseline tag**.
2. **Downstream, `ghifiardi/tantular-distillation`** — check out that tag at
   its exact commit and *select* the 36 `requires_addin` tests instead of
   deselecting them.

Do not merge these into one change. The upstream fix must be published and
reproduced before CI is rewired to depend on it.

## What changed since the last handoff

The previous exclusion rested on a recorded observation that
`tests/devServerCancellation.test.mjs` "remained active for more than 704
seconds when run directly". **That conclusion is withdrawn.** A bounded,
read-only diagnosis at the published baseline measured the opposite.

Measured at add-in baseline `3e14d25468ab0cd793ba8dc48cf5f755796c94e2`, Node
v22.17.0, valid local certificate, no concurrent test processes:

```text
each test alone, --test-concurrency=1, 45s external deadline, own process group

  client cancellation closes upstream      5.22s wall   118ms assertion   pass
  json_schema reaches native /api/chat      0.18s wall   118ms assertion   pass
  normal completed request unaffected       0.18s wall   117ms assertion   pass
  whole file, all three                     5.21s wall   304ms total       pass

ten sequential whole-file runs: 5.21-5.22s, 10/10 pass, 0 hangs, 0 survivors
```

Instrumented timeline of the cancellation test:

```text
t+   91.6ms  dev server ready
t+  109.6ms  requestReceived: 5s guard timer ARMED (never cleared)
t+  118.5ms  client request DESTROYED (simulated cancel)
t+  118.7ms  waitForClose: 5s guard timer ARMED (never cleared)
t+  119.9ms  upstream socket close        <- cancellation propagated, 1.4ms
t+  120.0ms  assertion settled
t+  120.1ms  server.close() requested
t+  120.1ms  close CALLBACK fired         <- 0.0ms, not unbounded
t+  121.9ms  child exit code=null signal=SIGTERM, stdio drained
t+ 5120.1ms  beforeExit
t+ 5125.3ms  process exit
```

`118.7 + 5000 = 5118.7`. The process is idle for five seconds waiting on an
orphaned guard timer. Nothing is hanging.

### The two defects, both in test code

**1. Uncleared guard timers.** Two `Promise.race` guards create a `setTimeout`
that is never cleared when the real promise wins. A ref'd timer holds the event
loop open, so the process cannot exit until it fires.

- `startFakeOllama().waitForClose(timeoutMs)` — the rejecting timer in the
  race.
- the `requestReceived` race inside the cancellation test.

Tests 2 and 3 have no guard timers, which is exactly why they finish in 180ms.

**2. Unbounded fixture teardown, currently unreachable.** `fakeOllama.stop()` is
`new Promise(resolve => server.close(resolve))`. `server.close()` invokes its
callback only once **every existing connection** has closed; it does not destroy
sockets. Today `child.kill()` happens first and closes the upstream socket, so
the callback fires in ~0-6ms. That ordering is the only thing preventing an
unbounded wait. A failure-mode probe (cancel suppressed, so the socket stays
open and the assertion fails) still completed in 5.22s with rc=1 for this
reason.

### The 704-second observation

Record it as **historical and unexplained**. It is not a current known defect
and is not a justification for exclusion.

Source cannot account for it: `tools/dev-server.mjs`,
`tests/devServerCancellation.test.mjs` and `package.json` are byte-identical
between `ab73a0b` and the baseline `3e14d25`. The only difference in that range
is the lockfile plus the `/api/workspace` fix, which this test never exercises.

Two causes have already been named for this test and withdrawn — pytest
contention, then the week-old dev server on port 3000. Do not name a third
without measurement. The one condition deliberately **not** tested was
interaction under parallel test execution, because the diagnosis was scoped to
exclude concurrent test processes. If a stall ever reappears, that is the first
place to look.

## Non-negotiable guardrails

1. `train/TRAINING_BLOCKED.md` remains controlling. No training, generation,
   model endpoint, Tinker, GPU/cloud workload or bakeoff.
2. **Do not modify `tools/dev-server.mjs`** unless new measurements demonstrate
   a production defect. Cancellation propagation, teardown and child exit were
   all measured healthy. A change there is a production-server change and needs
   its own evidence, not a test fix wearing a disguise.
3. **Do not repin the harnesses.** Both are pinned to prompt aggregate
   `1e9e96aac3a2012493a7a149c7acb7cf02dd02246b3626739c2fac2f73df638e` at tag
   `tantular-office-addin-harness-baseline-2026-09-11`, peeled commit
   `3e14d25468ab0cd793ba8dc48cf5f755796c94e2`. A test-harness change leaves the
   nine effective prompt rows untouched, so prompt identity does not move.
   `verify_harness_identity.py --write` will refuse a changed digest anyway.
4. The **CI baseline tag is a different role** from the harness prompt-source
   tag. One names the source CI runs its add-in tests from; the other names the
   snapshot prompt identity was measured at. Do not collapse them, and do not
   move the existing tag.
5. Never stop or kill PID 92762, the companion on port 3000, or any other
   pre-existing process. This test uses `findFreePort()` and `listen(0)`; port
   3000 is irrelevant and that is settled.
6. Work in clean disposable worktrees. Do not modify either dirty original
   checkout (`tantular-distillation` on `distill-plan-wiring`, `LLM-Indonesia`
   on `feat/document-studio-reliability`).
7. Tests must not use the network or require a live model, add-in server or
   credential. Local fixtures only.
8. **Measure marker counts. Never update one by arithmetic**, and never
   preserve a marker solely to keep an old number.
9. Do not add a test exclusion. If a test fails, stop and report it.

## Phase 1 — baseline first, no source changes

Disposable clean checkout at the published baseline
`3e14d25468ab0cd793ba8dc48cf5f755796c94e2`. Run `npm ci` and `npm run cert`
inside it; `certs/` is gitignored and absent from a clone, so HTTPS tests need a
locally generated certificate.

Run **all 72** add-in test files sequentially, one file at a time, **including
`tests/devServerCancellation.test.mjs`**, with a per-file deadline (300s is
generous given the measured maximum is 5.22s). Change no source.

Record per-file wall time. Expect 72/72 to pass, with the cancellation file at
roughly 5.2s and everything else well under a second.

If any file fails or exceeds its deadline, **stop and report it**. That is new
information and the milestone's premise changes again.

## Phase 2 — require repeated clean passes

Repeat the full 72-file sequential run **three times consecutively**, all clean,
before changing a single line. Record each run's totals and the per-file time
for the cancellation test.

The point is to establish that the suite is stable *before* it is modified, so
any later instability is attributable. A single green run is not a baseline.

## Phase 3 — fix the two leaked guard timers, test code only

In `tests/devServerCancellation.test.mjs` only. Clear the losing timer when the
race settles:

```js
function withDeadline(promise, ms, message) {
  let timer;
  const guard = new Promise((_, reject) => {
    timer = setTimeout(() => reject(new Error(message)), ms);
  });
  return Promise.race([promise, guard]).finally(() => clearTimeout(timer));
}
```

Apply it to both races.

**`timer.unref()` is not an acceptable substitute. This is a decision, not a
preference.** Clearing makes the guard's lifecycle explicit and removes the
work; unref merely permits the process to ignore a timer that is still armed
and can still fire. Do not swap one for the other on grounds of brevity.

Expected effect: the cancellation file drops from ~5.2s to well under 300ms, and
`getActiveResourcesInfo()` at `beforeExit` no longer waits on a timer. Measure
it; do not assume it.

Do not change any assertion, any timeout value, or any production module.

## Phase 4 — bounded fixture teardown, with a regression test

Make the test-owned fixtures terminate on a deadline rather than on hope.

**Fake upstream server.** Track connections and destroy them before closing:

```js
const sockets = new Set();
server.on("connection", (s) => {
  sockets.add(s);
  s.on("close", () => sockets.delete(s));
});
stop: () => new Promise((resolve, reject) => {
  const timer = setTimeout(
    () => reject(new Error("fake upstream did not close in time")), 5000);
  server.close(() => { clearTimeout(timer); resolve(); });
  for (const s of sockets) s.destroy();   // close() alone waits on these
})
```

**Child process.** Send SIGTERM, then wait for `exit` with a deadline, and
report rather than hang if it does not arrive. Escalation beyond SIGTERM, if
any, must be deliberate and logged — never a silent SIGKILL.

**The regression test that makes this real.** A held-connection test: open a
connection to the fake upstream, deliberately keep it open, and assert `stop()`
resolves within its deadline. Verify by mutation that removing the
`s.destroy()` loop makes exactly that test fail and no other. Do not commit the
mutant.

This closes a latent unbounded wait that is currently masked only by the order
of `child.kill()` and `server.close()`.

## Phase 5 — do not touch the production server

State explicitly in the PR that `tools/dev-server.mjs` is unchanged, and why:
cancellation propagated in 1.4ms, the upstream socket closed, the child exited
on SIGTERM, and the close callback fired in 0.0ms. There is no measured
production defect here.

If some measurement in Phase 1 or 2 contradicts that, stop and report before
writing any production change.

## Phase 6 — publish a new immutable CI baseline tag

After the test-harness fixes pass repeated clean runs, publish from a clean
worktree on a dedicated branch. Commit test changes only.

Suggested names, matching existing practice:

```text
branch: publish/tantular-office-addin-ci-baseline
tag:    tantular-office-addin-ci-baseline-YYYY-MM-DD
```

Refuse if either name already exists. Never force-push, never move a tag; a
correction is a new dated tag.

Prove reproducibility exactly as the previous baseline milestone did: clone the
new tag `--depth 1` into a fresh directory, `npm ci`, `npm run cert`, and run
all 72 files sequentially. The tag annotation must record the peeled commit, the
lockfile SHA-256, Node/npm versions, the certificate procedure, and the full
72/72 sequential result with per-file timings for the cancellation test.

State in the annotation that this tag's role is **CI test source**, and that
harness prompt identity remains pinned at
`tantular-office-addin-harness-baseline-2026-09-11`.

## Phase 7 — downstream: select the 36, in a separate Tantular PR

Separate branch, separate PR, off current `origin/main`
(`1c0fea7dc4c3a8a9cc4d1121f0844914e750c0e6` at the time of writing; verify).

Rewire `.github/workflows/tests.yml` to obtain the add-in from the **CI baseline
tag at its exact commit** — not a branch, not a moving ref — then `npm ci`,
generate local certificates, and **select** the 36 `requires_addin` tests rather
than deselecting them.

Continue excluding only the 47 corpus-required tests. The selection expression
becomes `-m "not requires_local_corpus"` rather than
`-m "not requires_local_corpus and not requires_addin"`.

### Required CI layout

Prescribed, not left to the executor's judgement:

1. **Deterministic external checkout path inside `$GITHUB_WORKSPACE`.** Check
   the add-in out to a fixed, named path under the workspace — never a
   `runner.temp` path, never a path derived from a ref name that could change.
   The location must be identical on every run and stated in `docs/CI.md`.

2. **Verify the exact full commit before use.** After checkout, resolve `HEAD`
   and require it to equal the CI baseline tag's full 40-character peeled
   commit. Also require the tag itself to peel to that commit. A tag name is
   not proof; this is the same rule the harness prompt source already follows.
   Refuse the run on any mismatch, before installing anything.

3. **A verified deterministic mapping for the `../tantular_office_addin`
   contract.** 55 files and several gates resolve the add-in as a *sibling* of
   the repository root. Create that relationship explicitly — a symlink, or an
   equivalent deterministic mapping — and then **verify it resolves** to the
   checked-out commit before any test runs. Do not assume the link exists
   because the step that creates it exited zero: assert that
   `../tantular_office_addin/src/promptRegistry.js` is readable and that the
   add-in checkout's `HEAD` is still the pinned commit.

4. **Pinned Node and npm, plus lockfile digest verification.** Pin the Node
   version explicitly (the qualification runs used v22.17.0) rather than
   floating on the runner default. Record the npm version. Compute the
   SHA-256 of `tantular_office_addin/package-lock.json` after `npm ci` and
   require it to equal the digest recorded in the CI baseline tag annotation —
   `npm ci` must not rewrite it.

5. **Rename the job to `tests-no-corpus`.** The current name,
   `tests-no-corpus-no-addin`, becomes false the moment the 36 tests are
   selected. Rename the job, its `name:` field, and every reference in
   `docs/CI.md`.

6. **Rename the add-in count variable.** `EXPECTED_EXCLUDED_ADDIN` describes an
   exclusion that will no longer exist. Rename it to something that states what
   is actually asserted — `EXPECTED_SELECTED_ADDIN` — and keep asserting the
   count in both directions. `EXPECTED_EXCLUDED_CORPUS` keeps its name, because
   the corpus partition really is still excluded.

The workflow header comment currently explains `requires_addin` as excluded
because the add-in "is not on the default branch ... and its package-lock.json
is not tracked on any ref". Both halves are now false. Rewrite it to describe
what is actually true: the add-in is obtained from an immutable CI baseline tag,
and only the corpus partition remains excluded.

Update `EXPECTED_EXCLUDED_*` and `docs/CI.md` **from measurement**. Collect each
partition with `--collect-only` and diff the node IDs against `main` before
changing any number. That node-ID diff caught a mis-scoped edit in Milestone 4
that had silently transferred a marker onto an unrelated test while leaving the
count unchanged; it is load-bearing, not procedure.

Recorded counts entering this milestone:

```text
requires_local_corpus  47
requires_addin         36
union                  83   (collected directly, never summed)
```

Correct `docs/CI.md` where it still says the add-in partition cannot be restored
because of the cancellation test, and record the 704-second observation as
historical and unexplained.

## Stop conditions

Stop without improvising if:

- any of the 72 files fails or exceeds its deadline in Phase 1 or 2;
- the three repeated runs are not all clean;
- the cancellation test's measured time does not fall after the timer fix;
- the held-connection regression test does not fail under mutation;
- a production change to `tools/dev-server.mjs` appears necessary;
- the new tag or branch name already exists;
- a clean clone of the new tag does not reproduce 72/72;
- marker counts move without a node-ID-level explanation;
- CI is green only on a commit older than the review head;
- the harness prompt digest changes for any reason.

Do not resolve a stop condition by excluding a test, loosening a deadline,
modifying the production server, moving a tag, or repinning a harness.

## Definition of done

- 72/72 add-in files pass sequentially, three consecutive clean runs, before any
  change; and again after the fixes.
- The cancellation file's wall time is measurably reduced and the reduction is
  reported as a number.
- Fixture teardown is bounded, with a held-connection regression test proven by
  mutation.
- `tools/dev-server.mjs` is unchanged.
- A new immutable CI baseline tag is published and reproduced from a clean
  remote clone.
- The Tantular PR selects the 36 `requires_addin` tests, excludes only the 47
  corpus tests, and is green at its exact review head.
- Marker counts are measured, node-ID diffed, and documented.
- Both harnesses remain pinned to `1e9e96aa...` at the 2026-09-11 tag.
- `train/TRAINING_BLOCKED.md` and the legacy corpus are unchanged.
- No pre-existing or unowned process was terminated; only processes created by
  the acceptance run were stopped through their documented lifecycle.
- No model, training, Tinker or cloud workload ran.

## Still out of scope after this milestone

- No product harness executor and no execution receipt;
  `trace_generation.supported_by_generate_py` stays `false`.
- Qwen3.5-9B instruct model identity remains unverified (independent; may
  proceed in parallel).
- No approved real capability gap.
- No bakeoff and no training authorization.

---

## Paste-ready instruction for Claude Code

```text
Execute Milestone 5 exactly as specified in
docs/CLAUDE_CODE_MILESTONE5_QUALIFY_ADDIN_SUITE_HANDOFF.md.

Read the entire handoff first. Work only in clean disposable worktrees; do not
modify either dirty original checkout. Baseline first: run all 72 add-in test
files sequentially, including devServerCancellation.test.mjs, with per-file
deadlines and no source changes, and require three consecutive clean runs
before changing anything.

Then fix the two uncleared guard timers in test code only, add bounded teardown
for the test-owned fake server and child process with a held-connection
regression test proven by mutation, and leave tools/dev-server.mjs unchanged
unless new measurements demonstrate a production defect.

Publish the test-hardened state as a new immutable CI baseline tag, reproduce it
from a clean remote clone, and only then, in a SEPARATE Tantular PR, rewire CI
to check out that tag at its exact commit, run npm ci, generate local
certificates, and select the 36 requires_addin tests. Follow the required CI
layout in Phase 7 exactly: deterministic checkout path inside GITHUB_WORKSPACE,
full-commit verification before use, a verified sibling mapping for the
../tantular_office_addin contract, pinned Node/npm with lockfile digest
verification, job renamed to tests-no-corpus, and the add-in count variable
renamed since those tests are now selected. Continue excluding only the 47
corpus tests. Measure every marker count and diff node IDs against main.

Do not repin the harnesses, move the existing tag, modify the production server,
add an exclusion, kill any pre-existing process, or run models, training,
Tinker or a cloud workload. Record the 704-second observation as historical and
unexplained. Stop and report on any invariant mismatch.
```
