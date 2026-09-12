# Claude Code handoff: Milestone 4 — verify the Office harness prompts

**Read this whole file before changing anything. Execute the phases in order
and stop on a failed invariant.**

Applicable date: **11 September 2026**.

## Objective

Verify the system-prompt identity of both shipped Office harnesses against the
published, reproducible `LLM-Indonesia` baseline:

```text
repository:      https://github.com/ghifiardi/LLM-Indonesia.git
tag:             tantular-office-addin-harness-baseline-2026-09-11
annotated object: 75254143f3d09af2af2369d7657c88d710616e40
peeled commit:   3e14d25468ab0cd793ba8dc48cf5f755796c94e2
registry path:   tantular_office_addin/src/promptRegistry.js
authoritative repository prompt aggregate:
  1e9e96aac3a2012493a7a149c7acb7cf02dd02246b3626739c2fac2f73df638e
publication-probe aggregate (different framing; not the repository identity):
  5086061098575be10267c1b4a0676f75ad3e65723b497ca3d06eee6b4aa26794
```

The two harnesses are:

```text
configs/harnesses/tantular-office-current.yaml
configs/harnesses/tantular-office-candidate.yaml
```

Both use the same production prompt registry, so both should receive the same
`prompts.system.sha256`. Their complete harness digests will still differ
because their execution, memory and verification policies differ.

This milestone must:

1. replace the shipped harnesses' machine-local sibling paths with repository-
   relative paths tied to the stable tag and its exact peeled commit;
2. make `src/verify_harness_identity.py` resolve a repository-backed prompt
   only from a caller-supplied checkout of that exact immutable source;
3. measure the effective prompts through the add-in's real
   `allPromptIds()`/`getPrompt()` exports;
4. set `verified: true` only after the measured content-based aggregate exactly
   matches the authoritative repository value above;
5. make the checked-in source pin and prompt digest testable offline in CI;
6. update documentation that still says the add-in or prompt identity is
   unpublished.

This is Milestone 4 from the process document. It is safe to execute before the
model-identity milestone because it does not use model weights. It does **not**
waive the unresolved add-in cancellation test, model qualification, capability-
gap, bakeoff or training gates.

## Why this milestone is now unblocked

The add-in baseline has been published and reproduced from a clean remote
clone. The publication contains:

```text
63497565cf97aeaf6ef786b6d810fa114bdf1221
  fix(office): treat missing workspace revision as an unconditional read

3e14d25468ab0cd793ba8dc48cf5f755796c94e2
  chore(office): publish reproducible harness dependency lock
```

Its clean-clone prompt probe found nine prompt IDs:

```text
edit
prose:cekAman
prose:draftTeks
prose:ringkas
prose:tanyaDokumen
prose:terjemah
prose:ubahNada
prose:umum
router
```

The publication probe reproduced this aggregate twice:

```text
5086061098575be10267c1b4a0676f75ad3e65723b497ca3d06eee6b4aa26794
```

That probe used newline-joined JSON objects in declaration-key order. The
reviewed implementation already checked into this repository,
`prompt_registry_digest()`, serializes the same sorted rows as one canonical
JSON array with alphabetized keys. Its authoritative aggregate is:

```text
1e9e96aac3a2012493a7a149c7acb7cf02dd02246b3626739c2fac2f73df638e
```

The nine IDs, every per-prompt `content_sha256`, and every
`registry_content_hash` are identical under both probes. Only the final framing
differs. The checked-in repository algorithm is authoritative. Do not change
it to reproduce the ad-hoc publication probe.

The immutable tag remains valid: its load-bearing identity is the tag plus
peeled source commit, not the informational aggregate written in its
annotation. Keep the tag fixed and record the framing discrepancy durably in
the schema/architecture documentation and the Milestone 4 PR.

## Non-negotiable guardrails

1. `train/TRAINING_BLOCKED.md` remains controlling. Do not run training,
   generation, a model endpoint, Tinker, a GPU/cloud workload or a four-arm
   bakeoff.
2. Work in a new clean branch and worktree from current `origin/main`. The
   existing Tantular worktree is dirty with unrelated work. Do not stage,
   restore, reset, clean, stash or switch branches in it.
3. Do not modify the dirty original `LLM-Indonesia` checkout. Obtain the
   published tag in a separate clean clone under `/private/tmp` or another
   disposable location.
4. Never stop or kill PID 92762 or another pre-existing process. No process
   termination is needed for this milestone.
5. Do not fetch from the network inside `verify_harness_identity.py`. Source
   acquisition and source verification are separate operations:
   - Git obtains a clean checkout;
   - the verifier reads that caller-supplied checkout offline.
6. Never trust a tag name alone. The checkout's `HEAD` and the tag's peeled
   commit must both equal
   `3e14d25468ab0cd793ba8dc48cf5f755796c94e2`.
7. Do not use a moving branch as the harness source. In particular, do not pin
   `main`, `feat/document-studio-reliability`, or the publication branch name.
8. Do not vendor the add-in, its prompts, `node_modules`, certificate material
   or generated files into this repository.
9. Do not use the add-in's short `contentHash` as the prompt identity. It is a
   32-bit djb2 value used for cache-busting, not a cryptographic commitment.
10. Do not hand-set `verified: true`. Only
    `src/verify_harness_identity.py --write` may make that transition after a
    clean measurement.
11. Do not silently re-pin a mismatch. A valid existing SHA-256 that differs
    from the measurement is a refusal, including under `--write`.
12. Keep `trace_generation.supported_by_generate_py: false` for both shipped
    harnesses. Prompt verification does not make `generate.py` a product
    harness executor.
13. Do not restore the complete `requires_addin` partition to CI in this
    milestone. `tests/devServerCancellation.test.mjs` is still an independently
    confirmed upstream blocker.
14. Tests must not use the network. Git repositories needed by unit tests must
    be temporary local repositories created by the tests.
15. Measure CI marker counts. Never update an expected count by arithmetic or
    because this handoff predicts that it should move.

## Required config shape

Extend `prompts.system` for repository-backed prompt registries. Use one
documented shape consistently in both harnesses:

```yaml
prompts:
  system:
    source: prompt_registry
    repository:
      url: https://github.com/ghifiardi/LLM-Indonesia.git
      ref: tantular-office-addin-harness-baseline-2026-09-11
      peeled_commit: 3e14d25468ab0cd793ba8dc48cf5f755796c94e2
    path: tantular_office_addin/src/promptRegistry.js
    sha256: null
    verified: false
```

After successful measurement, the verifier — not a manual edit — should
produce:

```yaml
    sha256: 1e9e96aac3a2012493a7a149c7acb7cf02dd02246b3626739c2fac2f73df638e
    verified: true
```

The `repository`, `ref`, `peeled_commit`, path and prompt SHA are part of the
harness YAML and therefore part of its canonical harness digest. A trace must
bind not only to prompt bytes but also to the source snapshot from which those
bytes were measured.

Do not store a temporary absolute checkout path in either YAML file.

The annotated tag-object SHA may be recorded in documentation and completion
evidence. The load-bearing source pin in the harness is the tag name plus its
exact peeled commit: changing tag annotation without changing source is not a
new prompt identity, while changing the peeled commit is.

## Phase 1 — establish isolated worktrees

In the dirty Tantular checkout, perform read-only inspection:

```bash
git status --short
git rev-parse HEAD
git rev-parse origin/main
git log -5 --oneline --decorate origin/main
```

Expected recorded `origin/main` before this milestone:

```text
7ba07ca16289e3382992cb6a3bb81f61f5e1b63c
```

If current `origin/main` differs, do not automatically reset anything. Inspect
the new commits and confirm that the harness-attribution work is still present.
Branch from the actual reviewed current `origin/main`.

Create a clean Tantular worktree outside the dirty checkout, for example:

```text
/private/tmp/tantular-milestone4-verify-prompts
```

Use a dedicated branch, for example:

```text
verify/office-harness-prompts
```

Before editing, prove in the new worktree:

```text
HEAD == current origin/main
git status --short == empty
branch == verify/office-harness-prompts
```

Separately clone the published add-in tag:

```bash
git clone --depth 1 \
  --branch tantular-office-addin-harness-baseline-2026-09-11 \
  https://github.com/ghifiardi/LLM-Indonesia.git \
  /private/tmp/tantular-office-addin-harness-baseline-2026-09-11
```

This clone is source evidence, not a development worktree. Do not edit it.

Record and require:

```bash
git -C /private/tmp/tantular-office-addin-harness-baseline-2026-09-11 \
  rev-parse HEAD

git -C /private/tmp/tantular-office-addin-harness-baseline-2026-09-11 \
  rev-parse 'tantular-office-addin-harness-baseline-2026-09-11^{commit}'

git -C /private/tmp/tantular-office-addin-harness-baseline-2026-09-11 \
  status --short
```

The first two outputs must both be:

```text
3e14d25468ab0cd793ba8dc48cf5f755796c94e2
```

Status must be empty. Also verify the registry path exists. If any invariant
fails, stop; do not fall back to the dirty sibling checkout or a branch head.

## Phase 2 — tests first: define repository-backed resolution

Extend `tests/test_verify_harness_identity.py` before changing the verifier.
Keep all existing content-identity tests.

Add offline tests using temporary local Git repositories. At minimum, prove:

1. a repository-backed registry at the exact pinned `HEAD` and exact locally
   resolvable tag is accepted;
2. `--write` records the measured aggregate and sets `verified: true`;
3. a checkout at a different commit is refused before Node reads the registry;
4. a tag that peels to a different commit is refused;
5. a missing tag is refused;
6. a repository-backed source without the explicit checkout argument is
   refused rather than resolved from a sibling path;
7. tracked changes in the supplied checkout are refused;
8. a missing registry file is refused;
9. a repository-relative path that escapes the checkout is refused;
10. a measured digest that differs from an already pinned valid SHA-256 is a
    mismatch and is not rewritten under `--write`;
11. prompt content changes move the aggregate even if the registry reports a
    constant short `contentHash`;
12. unrelated non-prompt source changes do not move the prompt aggregate;
13. swapping content between two prompt IDs moves the aggregate;
14. both checked-in harnesses carry the exact stable tag, peeled commit,
    registry path and expected aggregate;
15. both checked-in harnesses report `verified: true`.
16. a partial or contradictory `repository` block is rejected by static harness
    validation rather than treated as an ordinary local path.
17. the canonical prompt-row payload is one compact JSON array with keys sorted
    by `json.dumps(..., sort_keys=True, separators=(",", ":"))`; a regression
    to newline-delimited rows or declaration-key order fails.

The repository-backed tests must not contact GitHub. Build and tag their small
fixtures locally.

Replace every shipped-harness test whose intended behavior is “these remain
unverified.” Do not merely weaken or delete the assertions. Replace them with
positive exact-pin assertions.

Known assertions to find include:

```text
test_the_shipped_harnesses_remain_unverified_here
test_the_shipped_draft_harnesses_are_unverified_today
test_the_real_harnesses_measure_but_stay_unpinned
```

Search the full suite for other statements that the shipped harnesses have a
null digest or `prompt_verified: false`.

The old real-add-in test existed only because the prompt source was local and
unpublished. After the exact source metadata and digest are checked into the
harnesses, the permanent config assertion must run offline in CI. If removing
or replacing that `requires_addin` test changes the add-in partition, investigate
and measure the new count; do not preserve a marker solely to keep the old
number.

First run the new tests against the old implementation and record that they
fail for the intended reason.

## Phase 3 — implement an offline pinned-checkout resolver

Modify `src/verify_harness_identity.py` minimally.

Add an explicit CLI argument:

```text
--source-checkout <path>
```

For `prompts.system.repository`:

1. require `--source-checkout`;
2. require `repository.url`, `repository.ref` and
   `repository.peeled_commit`;
3. require `peeled_commit` to be a complete lowercase hexadecimal Git object
   ID in the repository's actual object format (40 characters for SHA-1 or 64
   for SHA-256), never a display abbreviation;
4. run local, read-only Git commands to establish:
   - the path is a Git worktree;
   - its configured `origin` describes `repository.url` (normalize only
     equivalent trailing `.git`/slash spelling; do not accept an unrelated
     repository);
   - its `HEAD` equals `peeled_commit`;
   - `repository.ref^{commit}` resolves locally and equals
     `peeled_commit`;
   - tracked and staged files are clean;
5. resolve `prompts.system.path` relative to the checkout root;
6. refuse an absolute path or any path that escapes that root;
7. only then invoke the existing prompt-registry probe.

The published repository currently uses a 40-character SHA-1:

```text
3e14d25468ab0cd793ba8dc48cf5f755796c94e2
```

Do not incorrectly validate this Git commit as a 64-character SHA-256. Prompt
digests are SHA-256; Git object IDs are a separate type.

For generic fixture harnesses that have no `repository` block, preserve the
existing local `path` behavior. This keeps the verifier useful for a genuine
standalone prompt file and keeps synthetic tests simple. Do not let that legacy
mode become an implicit fallback for a malformed repository-backed source.

Do not:

- call `git clone`, `git fetch`, GitHub, `curl` or another network client from
  the verifier;
- search parent directories for a convenient add-in checkout;
- accept a checkout merely because the registry bytes happen to hash correctly;
- mutate the supplied checkout;
- run `npm install` or `npm ci`;
- trust `repository.url` as proof of source identity without verifying the
  commit and tag;
- silently replace an existing mismatched prompt digest.

Keep the existing `prompt_registry_digest()` algorithm as the single source of
truth. Refactor only as needed to keep path resolution and Git verification
separate and testable.

Also add static validation for the optional repository-backed source shape to
`harness_distill.validate_harness`. It must validate the mapping, URL, tag/ref,
complete Git object ID and repository-relative prompt path without running Git
or accessing the network. A present-but-partial repository block is malformed;
it must not fall back to local-path semantics. Keep actual checkout and tag
verification in `verify_harness_identity.py`.

## Phase 4 — preserve the content-based aggregate exactly

The aggregate identity must remain SHA-256 over one compact canonical JSON
array of sorted rows shaped as:

```json
{
  "id": "<prompt id>",
  "content_sha256": "<sha256 computed here over UTF-8 prompt text>",
  "registry_content_hash": "<the add-in's reported short hash>"
}
```

The ID and its content hash must travel in the same row so swapping prompt text
between IDs changes the aggregate.

The final payload framing is also part of the identity:

```python
json.dumps(
    normalized_rows,
    ensure_ascii=False,
    sort_keys=True,
    separators=(",", ":"),
).encode("utf-8")
```

Do not serialize one object per line. Add a regression fixture that asserts
both the exact canonical bytes and their final digest so the framing cannot
again remain implicit.

`registry_content_hash` remains a cross-checking field. It must not replace
`content_sha256`, and the aggregate must not become a digest of:

- the whole `src/` tree;
- `promptRegistry.js` file bytes alone;
- prompt IDs and contents in separate unbound lists;
- the add-in's short `contentHash` values alone.

Against the published checkout, the verifier must print nine prompt rows and
measure exactly:

```text
1e9e96aac3a2012493a7a149c7acb7cf02dd02246b3626739c2fac2f73df638e
```

If it does not, stop. Do not update this handoff, the expected test value or
the YAML to whatever appeared.

The published tag annotation's `508606109...` value is not the expected
repository result. It is retained only as evidence of the earlier probe's
newline-delimited framing and must not be pinned into either harness.

## Phase 5 — retarget and verify both shipped harnesses

Edit only the source metadata first. For both harnesses:

- retain `source: prompt_registry`;
- add the exact `repository` block shown above;
- make `path` repository-relative:
  `tantular_office_addin/src/promptRegistry.js`;
- leave `sha256: null` and `verified: false` until the verifier runs;
- leave `trace_generation.supported_by_generate_py: false`;
- do not alter tools, memory, execution, verification, mutation or compatible
  model policy.

Run report-only mode first for both:

```bash
./.venv/bin/python src/verify_harness_identity.py \
  tantular-office-current \
  --source-checkout \
  /private/tmp/tantular-office-addin-harness-baseline-2026-09-11

./.venv/bin/python src/verify_harness_identity.py \
  tantular-office-candidate \
  --source-checkout \
  /private/tmp/tantular-office-addin-harness-baseline-2026-09-11
```

Because the configs are not pinned yet, report-only mode should measure the
expected digest and then exit non-zero with the existing “not pinned” refusal.
That non-zero result is expected. A successful exit at this point would mean
the verifier accepted an unpinned identity.

Then run the only commands allowed to write the identity:

```bash
./.venv/bin/python src/verify_harness_identity.py \
  tantular-office-current \
  --source-checkout \
  /private/tmp/tantular-office-addin-harness-baseline-2026-09-11 \
  --write

./.venv/bin/python src/verify_harness_identity.py \
  tantular-office-candidate \
  --source-checkout \
  /private/tmp/tantular-office-addin-harness-baseline-2026-09-11 \
  --write
```

Inspect the diff. The verifier may change only `sha256` and `verified` in this
phase.

Run both again without `--write`. Both must exit zero, print the same measured
and pinned aggregate, and leave the files byte-identical.

Finally, independently reload both harnesses through `harness_distill` and
prove:

```text
validate_harness: no "system prompt identity is unverified" warning
harness_provenance.prompt_verified: true
harness_provenance.prompt_sha256:
  1e9e96aac3a2012493a7a149c7acb7cf02dd02246b3626739c2fac2f73df638e
```

Also prove their complete harness digests are different. Equal complete
digests would mean policy differences were lost.

## Phase 6 — documentation updates

Update:

```text
configs/harnesses/harness.schema.md
docs/HARNESS_AWARE_DISTILLATION_ARCHITECTURE.md
docs/CI.md
```

Required corrections:

1. document the repository-backed `prompts.system` fields and the
   `--source-checkout` trust model;
2. state that the stable add-in baseline is now published at the exact tag and
   peeled commit;
3. state that both shipped prompt identities are verified at the expected
   aggregate after this milestone;
4. remove claims that prompt verification is blocked because the add-in or
   lockfile is unpublished;
5. record that the tag annotation contains `508606109...` from an ad-hoc
   newline-delimited probe, while the repository's authoritative canonical JSON
   array computes `1e9e96aa...` over the identical nine rows;
6. preserve the real remaining blockers:
   - there is no purpose-built product harness executor or execution receipt;
   - both shipped harnesses remain unsupported by `generate.py`;
   - `devServerCancellation.test.mjs` exceeded 704 seconds standalone and the
     `requires_addin` partition is not yet restored to CI;
   - Qwen3.5-9B instruct model identity remains unverified;
   - no approved real capability gap exists;
   - no bakeoff or training is authorized.

In architecture section 8, the “what blocks it today” text must distinguish
prompt identity, which is resolved by this milestone, from execution
attribution, which is not.

In `docs/CI.md`, do not claim that the package lock is absent from every
published ref. It is present at the baseline tag. Do not claim that this alone
makes the entire add-in partition green; the cancellation defect is still open
and the workflow has not yet been redesigned around the stable checkout.

Do not modify `train/TRAINING_BLOCKED.md`. Its no-training decision is
unchanged.

Do not regenerate or edit the binary DOCX/PPTX deliverables in this code
milestone unless separately requested.

## Phase 7 — targeted and partition verification

Run targeted tests first:

```bash
./.venv/bin/python -m pytest \
  tests/test_verify_harness_identity.py \
  tests/test_harness_distill.py -q
```

Run the exact CI-selected partition:

```bash
./.venv/bin/python -m pytest tests/ -q \
  -m "not requires_local_corpus and not requires_addin"
```

If the local corpus is available, also run the changed surface while excluding
the unresolved add-in partition:

```bash
./.venv/bin/python -m pytest tests/ -q -m "not requires_addin"
```

Do not run the complete `requires_addin` partition merely to claim a full
suite. The known Node cancellation test may exceed its budget. Do not add a
second exclusion and do not describe `-m "not requires_addin"` as the full
suite.

Measure all marker partitions with `--collect-only`:

```bash
./.venv/bin/python -m pytest tests/ --collect-only -q \
  -m "requires_local_corpus"

./.venv/bin/python -m pytest tests/ --collect-only -q \
  -m "requires_addin"

./.venv/bin/python -m pytest tests/ --collect-only -q \
  -m "requires_local_corpus or requires_addin"
```

The recorded pre-milestone counts are:

```text
requires_local_corpus: 47
requires_addin:        37
union:                 84
```

The add-in count may decrease if the old real-add-in/unpinned assertion is
replaced by an offline exact-pin test. That is a hypothesis, not an expected
answer. Read the collected node IDs, classify the change, and update workflow
counts and `docs/CI.md` only after proving why the partition changed.

The final union must be collected directly, never calculated as the sum.

## Phase 8 — negative acceptance checks

Before committing, demonstrate the verifier refuses each of these without
modifying the harness files:

1. no `--source-checkout`;
2. a checkout at the parent commit rather than `3e14d254...`;
3. a local tag with the right name that peels to the wrong commit;
4. a tracked modification in the checkout;
5. a pinned SHA changed to another valid 64-character value.

Use disposable copies or test fixtures. Do not dirty the published evidence
clone and do not edit the real harness configs just to stage a refusal.

Also run a mutation check against the load-bearing exact-commit validation:
temporarily bypass that validation in an out-of-tree copy or reversible local
test mutation and prove the relevant regression test fails. Do not commit the
mutant.

## Phase 9 — diff inspection and commits

Before staging:

```bash
git status --short
git diff --check
git diff --stat
git diff -- src/verify_harness_identity.py
git diff -- configs/harnesses/
git diff -- tests/
git diff -- docs/
```

Confirm there is no change to:

```text
train/TRAINING_BLOCKED.md
src/generate.py
training configs
corpus files
run manifests
model registry files
Office add-in source
binary documentation deliverables
```

Use two reviewable commits:

```text
feat: resolve harness prompts from pinned repository snapshots
chore: verify Tantular Office harness prompt identities
```

Suggested split:

1. verifier, schema and offline resolver tests;
2. two exact harness pins, shipped-config assertions, CI count adjustment if
   empirically required, and factual documentation updates.

Before each commit inspect:

```bash
git diff --cached --check
git diff --cached --stat
git diff --cached --name-status
git diff --cached
```

After each commit inspect:

```bash
git show --check --stat --oneline HEAD
git show --name-status --format=fuller HEAD
```

Do not stage unrelated files from the dirty original worktree. Do not squash
the source-resolution behavior and the measured pins into an opaque commit.

## Phase 10 — PR and CI

Push only the new branch; do not force-push.

Open a PR to `main`. The PR body must record:

- published repository, tag, annotated object and peeled commit;
- registry path;
- nine observed prompt IDs;
- expected and measured aggregate SHA-256;
- both harnesses now `verified: true`;
- complete harness digests remain distinct;
- targeted test result;
- CI-selected partition result;
- corpus-present `not requires_addin` result if available;
- empirically measured marker counts and union;
- exact CI run ID and tested head SHA;
- no network access in tests or verifier;
- no training, generation, endpoint, Tinker or cloud workload;
- `trace_generation.supported_by_generate_py` remains false;
- the add-in cancellation partition, model identity, real capability gap and
  harness executor remain unresolved.

Wait for CI on the exact review head. Do not cite an earlier green run after a
later commit.

## Stop conditions

Stop without improvising if:

- the tag is missing or peels anywhere other than `3e14d254...`;
- the clean checkout's `HEAD` differs from the peeled commit;
- the evidence checkout has tracked changes;
- the prompt registry is absent or cannot be evaluated;
- the prompt ID list is empty, duplicated or differs unexpectedly from the
  nine recorded IDs;
- the measured aggregate differs from `1e9e96aac3a201...`;
- either harness cannot be verified through the same source snapshot;
- `--write` modifies anything other than the intended scalar fields;
- a mismatched existing pin is silently replaced;
- the verifier accesses the network;
- a test requires a live model, add-in server or credential;
- the harness becomes executable by `generate.py`;
- marker counts move without a node-ID-level explanation;
- a staged diff contains unrelated dirty-tree work;
- CI is green only on a commit older than the proposed review head.

Do not solve a stop condition by changing the expected digest, using a moving
branch, copying the dirty sibling checkout, weakening a test, adding an
exclusion, or setting `verified: true` manually.

## Definition of done

- Both shipped harnesses pin:
  - tag `tantular-office-addin-harness-baseline-2026-09-11`;
  - peeled commit `3e14d25468ab0cd793ba8dc48cf5f755796c94e2`;
  - registry path
    `tantular_office_addin/src/promptRegistry.js`;
  - prompt aggregate
    `1e9e96aac3a2012493a7a149c7acb7cf02dd02246b3626739c2fac2f73df638e`;
  - `verified: true`.
- The verifier obtains no source from the network and refuses a checkout that
  is not the pinned clean snapshot.
- CI verifies the permanent config pins and resolver behavior offline.
- Content SHA-256, not the add-in's short hash, remains the identity.
- Shipped-harness “must remain unverified” tests are replaced with exact
  positive assertions.
- Documentation describes the published baseline and the remaining blockers
  accurately.
- Targeted tests and the CI-selected partition pass.
- The corpus-present `not requires_addin` surface passes when available.
- Marker counts are measured and documented.
- The PR is green at its exact review head.
- No training, generation, endpoint, Tinker, cloud workload or process
  termination occurred.
- `train/TRAINING_BLOCKED.md` and the legacy corpus remain unchanged.

---

## Paste-ready instruction for Claude Code

```text
Execute Milestone 4 exactly as specified in
docs/CLAUDE_CODE_MILESTONE4_VERIFY_PROMPTS_HANDOFF.md.

Read the entire handoff first. Work only in new clean Tantular and add-in
worktrees; do not modify either dirty original checkout. Verify the published
LLM-Indonesia tag peels to
3e14d25468ab0cd793ba8dc48cf5f755796c94e2, implement an offline
--source-checkout resolver, and use verify_harness_identity.py --write to pin
both shipped harnesses to the measured content-based aggregate
1e9e96aac3a2012493a7a149c7acb7cf02dd02246b3626739c2fac2f73df638e.

Do not hand-set verified, use a moving branch, fetch inside the verifier,
enable generate.py for the product harness, run models/training/Tinker/cloud,
touch TRAINING_BLOCKED.md, add exclusions, or kill any pre-existing process.
Write tests first, measure marker counts empirically, inspect every staged
diff, use the two-commit plan, and stop on any invariant mismatch. Report the
exact head SHA, CI run, test totals, measured prompt IDs/digest, harness
digests, marker counts, and remaining blockers.
```
