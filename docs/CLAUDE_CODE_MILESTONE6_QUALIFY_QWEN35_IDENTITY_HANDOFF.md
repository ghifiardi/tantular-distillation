# Claude Code handoff: Milestone 6 — qualify the Qwen3.5-9B instruct identity

**Read this whole file before changing anything. Execute the phases in order
and stop on a failed invariant.**

**This handoff is deliberately incomplete. It specifies Stage 1 only and ENDS
at a mandatory review checkpoint.** Stage 2 is written afterwards, against what
Stage 1 measures. Do not design the processor contract speculatively, and do
not run `--write` during Stage 1.

Applicable date: **13 September 2026**.

## Objective

Qualify the model identity of the product's student checkpoint:

```text
registry model  qwen35-9b-instruct
model_id        Qwen/Qwen3.5-9B      (NOT Qwen/Qwen3.5-9B-Base)
spec            configs/models/qwen35-9b-instruct.yaml
```

Three properties, measured from one immutable snapshot: the pinned Hub
revision, the tokenizer and chat-template digests, and the architecture
signature. Nothing else.

This is qualification work. It authorizes no generation, no bakeoff, no
training, and no Tinker or cloud workload.

## Why this milestone starts with a stop

A bounded read-only survey at `origin/main`
`f702c823551e20d8b8c699b6b531fda7b7d4754a` established that qualification
cannot begin: **the instruct snapshot does not exist locally.**

```text
effective hub cache   ~/.cache/huggingface/hub
                      HF_HUB_CACHE / HF_HOME / TRANSFORMERS_CACHE unset
models--Qwen--Qwen3.5-9B        ABSENT
models--Qwen--Qwen3.5-9B-Base   present
  refs/main  68c46c4b3498877f3ef123c856ecfde50c39f404
  3 files    config.json, tokenizer.json, tokenizer_config.json
```

The verifier already refuses this correctly, before measuring anything:

```text
$ ./.venv/bin/python src/verify_model_identity.py qwen35-9b-instruct --offline
IDENTITY UNVERIFIED: no local snapshot of 'Qwen/Qwen3.5-9B' under
/Users/.../.cache/huggingface/hub.
...This tool will not download anything itself.
exit=2
```

So Milestone 6 is not "run the verifier". It is: acquire metadata safely,
measure, **stop and report**, and only then decide what the second half needs
to be.

## Facts the survey measured, to carry forward

Do not re-derive these; do verify them still hold.

**The Base snapshot is not a candidate and its values must never be pinned.**

```text
BASE config.json       sha256
  d0883072e01861ed0b2d47be3c16c36a8e81c224c7ffaa310c6558fb3f932b05
BASE tokenizer.json    sha256
  fe000e3ed39ed12b8d2481d527d44f93c65d37e87645d2dcc80d1bf9d50d2927
BASE tokenizer_config  sha256
  3891e840d7dc5fca0af33d3a25083a735e36fe06214e3f707024820cb6b9f89c
BASE arch signature
  52a3c9e2c895c0dc3d81abbb24e6463256cf3869f6e7343974f773a3f002096d
```

`52a3c9e2...` is evidence that `describe_architecture()` handles this model
family. It is **non-transferable evidence, not the student's signature**: it
may not be copied into any config because of where it came from. It is not a
forbidden result. The architecture signature identifies architecture *shape* —
not checkpoint identity, weights, instruction tuning or chat-template
behaviour — so a Base and an instruct checkpoint may legitimately share a
descriptor and therefore a signature. What makes a value pinnable is
provenance, never the digits.

A Base model carries no chat template, so it could never satisfy a template
digest. That, not the signature, is what distinguishes the two here.

**The model family is vision-language.** The Base `config.json` declares
`Qwen3_5ForConditionalGeneration`, outer `model_type: qwen3_5`, a
`text_config` (`qwen3_5_text`) and a `vision_config`
(`depth 27, hidden_size 1152, out_hidden_size 4096`). `describe_architecture()`
already includes both the nested text fields and the vision block, so the
nested-`text_config` fix holds and no regression was found.

**The declared profile is plausible.** The profile at
`configs/architectures/qwen35-hybrid-dense-9b.yaml` expects 32 layers as
`[linear, linear, linear, full] x 8`, which reproduces the observed
`layer_types` exactly. Its `signature:` is still `ARCH_SIGNATURE_QWEN35_9B`.

**`verify_model_identity.py` does not touch the architecture signature.** It
writes exactly five scalars and mentions "signature" zero times:

```text
tokenizer.sha256
chat_template.sha256
revision            (only when the spec's revision is a placeholder)
tokenizer.revision
digests_verified    -> true
```

The architecture pin is a separate action against
`configs/architectures/qwen35-hybrid-dense-9b.yaml`, asserted today by
`tests/test_architecture_signature.py:83-89`, which pins the placeholder and
would need updating.

**Placeholders currently checked in:**

```text
TOKENIZER_DIGEST_QWEN35_9B        configs/models/qwen35-9b-instruct.yaml
CHAT_TEMPLATE_DIGEST_QWEN35_9B    configs/models/qwen35-9b-instruct.yaml
LICENSE_EVIDENCE_DIGEST_QWEN35_9B configs/models/qwen35-9b-instruct.yaml
REPLACE_WITH_PINNED_HUB_COMMIT    qwen35-9b-instruct, muse-glimmer-30b,
                                  qwen35-122b-a10b
ARCH_SIGNATURE_QWEN35_9B          configs/architectures/
                                    qwen35-hybrid-dense-9b.yaml
```

No checked-in manifest, freeze or corpus fixture pins the student's digests.

## Readiness expectations, exact

Qualifying the student removes **one** planning warning from
`distill_plan plan office-v2-sequence`. Six are emitted today; five remain:

```text
REMOVED  student 'qwen35-9b-instruct' digests_verified is not true

REMAINS  auto: on_policy_kd unavailable -> tokenizer compatibility keys differ
REMAINS  auto fell back to preference (judge-gated)
REMAINS  teacher 'muse-glimmer-30b' digests_verified is not true
REMAINS  memory estimate for fp8 is INCOMPLETE (vision tower params unknown)
REMAINS  memory estimate for bf16 is INCOMPLETE (vision tower params unknown)
```

**The legacy corpus's `identity_ready` does NOT become true.** That component
is derived from the corpus's *resolved teachers*, not from the student. The
real corpus resolves to exactly one:

```text
corpus teachers: ['muse-glimmer'] -> registry 'muse-glimmer-30b',
                                     digests_verified = False
unresolved: []
```

Expected legacy readiness after student qualification, unchanged:

```text
fp8_ready       false     136 int4_ollama traces
source_ready    false     136 synthetic traces
identity_ready  false     teacher muse-glimmer-30b still unverified
license_ready   true
harness_ready   true

trainable_as_is      false
authorizes_training  false
```

If any of those moves, stop: something other than this milestone changed.

**Mode C stays unavailable**, and the reason is worth stating precisely.
Today's "compatibility keys differ" verdict compares two placeholders. After
this milestone it compares one measured key against a placeholder. The verdict
is the same — a non-Qwen teacher genuinely differs — but it only becomes a
real comparison when the teacher is qualified too. Do not present Mode C as
"re-evaluated" on the strength of this work.

## Non-negotiable guardrails

1. `train/TRAINING_BLOCKED.md` remains controlling. No training, generation,
   model endpoint, Tinker, GPU/cloud workload or bakeoff. **No model is
   loaded or executed at any point in this milestone.**
2. **No weights.** Metadata only. Downloading any weight file is a stop
   condition, not a recoverable mistake.
3. Acquisition happens **outside** `verify_model_identity.py`. That tool never
   fetches; it reads a snapshot it is handed. Acquisition and verification are
   separate operations, exactly as the prompt-source milestone established.
4. **Do not modify `src/verify_model_identity.py` during Stage 1.** It already
   refuses an absent snapshot, distinguishes instruct from Base, requires a
   cache path carrying a commit, keeps tokenizer compatibility separate from
   chat-template identity, and refuses mismatches and unknown revisions. If the
   instruct snapshot uses a template layout it does not support, **stop and
   report that exact layout** before implementing anything.
5. **No `--write`, no YAML edit, no architecture pin, no commit, no PR in
   Stage 1.** The milestone ends at the checkpoint.
6. Never copy or pin a signature *because it was measured from Base*. Pin only
   a value independently recomputed from the exact instruct snapshot. If that
   independently measured result equals
   `52a3c9e2c895c0dc3d81abbb24e6463256cf3869f6e7343974f773a3f002096d`, **accept
   the equality**, provided the repository id, cache provenance, resolved
   commit and input `config.json` are all proven to be the instruct snapshot.
   Equality is not evidence of a mistake; unproven provenance is.
7. `config.json`, tokenizer files, template files and the processor inventory
   must all come from **the same immutable instruct commit**. No arbitrary
   copied `config.json` is acceptable.
8. Never stop or kill PID 92762 or any other pre-existing process.
9. Work in a clean worktree from `origin/main`. Do not modify the dirty
   original checkout.
10. Do not print tokens, credentials or a full environment dump.

## Stage 1

### Phase 1 — clean worktree and pre-state

Clean worktree from current `origin/main` (verify it; it was
`f702c823551e20d8b8c699b6b531fda7b7d4754a` when this was written). Record the
worktree status and the exact pre-acquisition cache state: repos present, the
Base snapshot's `refs/main`, and that `models--Qwen--Qwen3.5-9B` is absent.

Record the SHA-256 of `configs/models/qwen35-9b-instruct.yaml` and
`configs/architectures/qwen35-hybrid-dense-9b.yaml` before anything runs. Both
must be byte-identical at the checkpoint.

### Phase 2 — resolve the immutable commit, metadata only

Resolve the revision **before** downloading any file, using repository metadata
only and with implicit credentials disabled:

```python
# HF_HUB_DISABLE_IMPLICIT_TOKEN=1
from huggingface_hub import HfApi
api = HfApi()
info = api.model_info(
    "Qwen/Qwen3.5-9B",
    revision="main",
    files_metadata=False,
)
info.sha        # the immutable commit
info.siblings   # the remote file inventory
```

Require a **complete 40-character lowercase hexadecimal commit**. An
abbreviation, a branch name or an empty value is a stop condition.

Record the full remote file inventory from `info.siblings`. That inventory is
what the allowlist is derived from — do not guess filenames.

If the repository is gated, private, renamed or absent, **stop and report**.
Do not authenticate around it.

### Phase 3 — metadata-only download from that exact commit

Download from the resolved commit, never from `main`. Use an exact allowlist
built from the Phase 2 inventory, intersected with the identity-relevant names:

```text
config.json
tokenizer.json
tokenizer.model
tokenizer_config.json
chat_template.jinja
chat_template.json
special_tokens_map.json
added_tokens.json
merges.txt
vocab.json
generation_config.json

processor_config.json
preprocessor_config.json
image_processor_config.json
video_preprocessor_config.json
feature_extractor_config.json
```

**Include only names actually present at the resolved commit.** Requesting a
name that does not exist there is a reporting error, not a download.

Exclude every weight format explicitly, even though the allowlist should
already prevent them:

```text
*.safetensors  *.bin  *.pt  *.pth  *.gguf  *.onnx  *.msgpack  *.h5
```

**Use per-file `hf_hub_download()`, not the `hf download` CLI.** In Stage 1 the
CLI silently defeated the allowlist:

```text
$ hf download Qwen/Qwen3.5-9B --revision <commit> --include config.json ... \
      --exclude "*.safetensors" ...
UserWarning: Ignoring `--include` since filenames have being explicitly set.
UserWarning: Ignoring `--exclude` since filenames have being explicitly set.
path=.../models--Qwen--Qwen3.5-9B/snapshots/<commit>
```

It parsed the `--include` values as positional FILES, discarded both flag
lists, printed a snapshot path as though it had succeeded, and **created
nothing**. The failure was caught only because the printed directory did not
exist. Naming each file explicitly removes the ambiguity entirely:

```python
# HF_HUB_DISABLE_IMPLICIT_TOKEN=1
from huggingface_hub import hf_hub_download
for name in ALLOWLIST:                  # derived from info.siblings
    assert not name.lower().endswith(WEIGHT_SUFFIXES)
    hf_hub_download("Qwen/Qwen3.5-9B", filename=name, revision=COMMIT)
```

Verify the download by inspecting the snapshot, never by trusting the tool's
exit status or printed path.

After the download, **prove no weights arrived**: enumerate every file in the
new snapshot with its resolved size (HF cache entries are symlinks into
`blobs/`, so an unresolved `stat` reports the symlink length — the survey hit
this and it produced misleading 52-byte readings), and assert that no file
matches an excluded pattern. Report the total bytes acquired.

Record the cache provenance: the snapshot directory, the commit it is named
for, and the `refs/` entries written.

### Phase 4 — measure, offline

Everything from here is offline. Run report-only, no `--write`:

```bash
./.venv/bin/python src/verify_model_identity.py qwen35-9b-instruct --offline
# or, pointing at the exact snapshot:
SNAP=~/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/<commit>
./.venv/bin/python src/verify_model_identity.py qwen35-9b-instruct \
    --offline --snapshot "$SNAP"
```

Expect a **non-zero placeholder refusal after printing measured values** — the
spec still carries `REPLACE_WITH_PINNED_HUB_COMMIT` and placeholder digests, so
a successful exit would mean the verifier accepted an unpinned identity.

Record: resolved model id, snapshot commit, the tokenizer files actually
included in the digest, the tokenizer digest, the effective chat-template
source, the template digest, the exit code and the refusal reason.

**If the effective chat template is not where the verifier looks** — for
instance if it lives only in `chat_template.json`, or only inside
`tokenizer_config.json`'s `chat_template` key, or is split across files — stop
and report that exact layout. Do not extend the verifier in Stage 1.

Confirm both config files are still byte-identical to their Phase 1 digests.

### Phase 5 — architecture, from the same snapshot

Using **that snapshot's** `config.json`:

```bash
./.venv/bin/python src/distill_plan.py arch-signature "$SNAP/config.json"
```

Record its SHA-256 and the complete `describe_architecture()` result: outer and
nested model types, hidden-layer count, the full `layer_types` pattern, hidden
size, attention and key/value head counts, expert fields, and vision model
type, depth and hidden sizes.

Compare against `configs/architectures/qwen35-hybrid-dense-9b.yaml`: does
`model_type` match, is it 32 layers, does `[linear, linear, linear, full] x 8`
reproduce the observed `layer_types`? Report agreement or disagreement
field by field.

**Do not fill the placeholder.** A disagreement is a finding for the
checkpoint, not something to reconcile by editing the profile.

If the measured signature happens to equal the Base value `52a3c9e2...`, that
is an acceptable outcome, not an alarm: the signature digests architecture
shape, which the two checkpoints may genuinely share. Report it as measured and
state the provenance that makes it pinnable — repository id, cache path,
resolved commit, and the `config.json` SHA-256 it was computed from. What would
be wrong is reusing the Base number without recomputing it.

### Phase 6 — processor and vision inventory

For every processor-family file acquired, report:

- which exist at the resolved commit;
- their sizes and SHA-256;
- whether they affect the product's **current text path** at all — the add-in
  sends text; a vision preprocessor may be inert for everything shipping today;
- whether the effective chat template lives somewhere the verifier does not
  inspect;
- a **proposed** processor-identity contract, if one is warranted, with the
  argument for and against including it in `digests_verified`.

Note the related gap without acting on it: `params.vision_b` is `null` in the
spec, which is what makes both memory estimates INCOMPLETE. Qualification as
scoped does not fill it.

Do not decide spec fields and do not change `digests_verified` semantics.

## MANDATORY CHECKPOINT — stop here

**Stage 1 ends after Phase 6.** No `--write`, no YAML edit, no architecture
pin, no commit, no PR, no branch push.

Report exactly:

```text
resolved instruct commit
remote file inventory
exact downloaded allowlist
proof no weights were downloaded
snapshot path and cache provenance
tokenizer files and digest
effective template source and digest
config.json SHA-256
architecture descriptor and signature
processor/vision files and proposed treatment
files the eventual --write would modify
legacy readiness before and projected after
cache and worktree status
```

The checkpoint determines Stage 2: whether the existing verifier needs only
measured pins, or a narrowly scoped architecture/processor extension. That
decision is made from measurement, not in advance.

## Stop conditions

Stop without improvising if:

- `model_info()` returns no complete 40-character commit;
- the repository is gated, private, renamed or absent;
- any weight-format file is downloaded, at all;
- the snapshot is missing `config.json`, `tokenizer.json` or
  `tokenizer_config.json`;
- the effective chat template is in a layout the verifier does not inspect;
- the verifier exits zero before anything is pinned;
- the measured architecture disagrees with the declared profile;
- the architecture signature cannot be traced to the instruct snapshot's own
  `config.json` at the resolved commit;
- either config file is no longer byte-identical;
- the legacy readiness block differs from the expected table above.

Do not resolve a stop condition by authenticating around a gate, downloading
weights, copying a `config.json` from another snapshot, pinning the Base
signature, editing the profile to match, or modifying the verifier.

## Definition of done for Stage 1

- A metadata-only snapshot of `Qwen/Qwen3.5-9B` exists at a recorded immutable
  40-character commit, containing no weight file of any format.
- Tokenizer digest, template source and digest, `config.json` SHA-256 and the
  architecture descriptor and signature are all measured from that one
  snapshot, offline.
- The processor/vision inventory is reported with a proposed treatment and no
  decision taken.
- `configs/models/qwen35-9b-instruct.yaml` and
  `configs/architectures/qwen35-hybrid-dense-9b.yaml` are byte-identical to
  their Phase 1 digests.
- The worktree is clean; nothing is committed or pushed.
- No pre-existing or unowned process was terminated; no model was loaded,
  executed, trained or served.
- `train/TRAINING_BLOCKED.md` unchanged; `authorizes_training` still false.

## Out of scope, before and after this milestone

- Teacher qualification (`muse-glimmer-30b` remains unverified, which is what
  keeps the legacy corpus's `identity_ready` false).
- `params.vision_b`, and therefore the INCOMPLETE memory estimates.
- Mode C eligibility, which needs both sides measured.
- No product harness executor and no execution receipt;
  `trace_generation.supported_by_generate_py` stays `false`.
- No approved capability gap, no bakeoff, no training authorization.

## Stage 1 results — measured, and the inputs Stage 2 pins

Stage 1 executed and stopped at the checkpoint. Its measurements:

```text
resolved commit   c202236235762e1c871ad0ccb60c8ee5ba337b9a   (40 chars, public)
snapshot          ~/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/
                    snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a
                  refs/ is EMPTY: downloading by explicit commit writes no
                  branch ref, so provenance rests on the directory name
downloaded        8 metadata files, 22,912,367 bytes, no weight of any format
tokenizer digest
  6f3a76fa0ff84cba487813d4024623233c4664ecedfc3f3857536f95d25504af
  over merges.txt, tokenizer.json, tokenizer_config.json, vocab.json
template source   chat_template.jinja  (supported; no verifier change)
template digest
  a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715
config.json
  d0883072e01861ed0b2d47be3c16c36a8e81c224c7ffaa310c6558fb3f932b05
arch signature
  52a3c9e2c895c0dc3d81abbb24e6463256cf3869f6e7343974f773a3f002096d
verifier exit     1, placeholder refusal, after printing measured values
```

Two findings that shape Stage 2.

**The instruct `config.json` is byte-identical to the Base one**, so the
signature recomputed from the instruct snapshot equals the Base value. That is
shared architecture, not shared checkpoint identity, and it is accepted because
provenance is proven. A stop condition treating that equality as evidence of a
wrong snapshot would have produced a false refusal on the first run; it was
removed for exactly this reason.

**The template exists twice in the snapshot** — as `chat_template.jinja` and
in `tokenizer_config.json`'s `chat_template` key — and the two are
byte-identical
(7756 chars each). The effective template is therefore unambiguous. The
verifier already strips that key before digesting the tokenizer, so editing
the template cannot silently move the tokenizer digest.

## Stage 2

Authorized only after this handoff is merged, executed from the resulting
`origin/main`.

### Processor identity: record, do not gate

**Do not add `processor.sha256`. Do not modify `digests_verified` semantics.**

`digests_verified` is defined narrowly as tokenizer plus effective chat-template
verification. The shipping add-in uses the text path, so the image and video
processors are not part of the compatibility decision this milestone qualifies.

Record as **non-gating evidence**:

```text
preprocessor_config.json        Qwen2VLImageProcessorFast
  sha256 27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516
video_preprocessor_config.json  Qwen3VLVideoProcessor
  sha256 7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13
```

Document explicitly, in the qualification record:

- vision processor identity remains **unqualified**;
- `modality.vision: true` remains accurate as a checkpoint capability;
- `params.vision_b` remains `null`;
- the FP8 and BF16 memory estimates remain INCOMPLETE;
- any future product use of image or video input requires a **separate
  processor-identity gate** before qualification.

This avoids both errors: pretending the vision metadata does not exist, and
letting an unused modality block a text-path compatibility decision.

### Architecture pin: same PR, separate commits

Both pins come from the same immutable snapshot, so they belong in one PR. They
are different claims, so they are separate commits:

```text
chore: verify Qwen3.5-9B instruct text identity
chore: pin Qwen3.5-9B instruct architecture signature
```

The second commit must state that `52a3c9e2...` was **independently
recomputed** from the instruct snapshot, and that its equality with the Base
value means shared architecture, not shared checkpoint identity.

Retain a synthetic regression test proving a placeholder signature is refused.
Replace **only** the shipped-profile assertion requiring the real profile to
stay a placeholder — `tests/test_architecture_signature.py:83-89`. Do not
weaken or delete the placeholder-refusal property itself.

### Exact pins

Model registry, `configs/models/qwen35-9b-instruct.yaml`:

```yaml
revision:             c202236235762e1c871ad0ccb60c8ee5ba337b9a
tokenizer.revision:   c202236235762e1c871ad0ccb60c8ee5ba337b9a
tokenizer.sha256:
  6f3a76fa0ff84cba487813d4024623233c4664ecedfc3f3857536f95d25504af
chat_template.sha256:
  a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715
digests_verified:     true
```

Architecture profile, `configs/architectures/qwen35-hybrid-dense-9b.yaml`:

```yaml
signature: 52a3c9e2c895c0dc3d81abbb24e6463256cf3869f6e7343974f773a3f002096d
```

**Only** this command may write the five model-registry scalars:

```bash
./.venv/bin/python src/verify_model_identity.py qwen35-9b-instruct \
    --offline --snapshot "$SNAP" --write
```

The architecture signature is then copied from a **separately rerun**
`arch-signature` command, recording the source snapshot and the `config.json`
SHA-256 it was computed from. Do not hand-type either value from this document.

### Boundaries

- **No network.** Use the existing `c202236...` cache snapshot.
- No `verify_model_identity.py` implementation change.
- No processor schema change.
- No weights, no model loading.
- No licence-field change.
- No manifest or freeze regeneration.
- No training, generation, endpoint, Tinker or cloud workload.
- `train/TRAINING_BLOCKED.md` untouched.
- The legacy readiness block must remain unchanged:
  `fp8_ready false, source_ready false, identity_ready false, license_ready
  true, harness_ready true, trainable_as_is false, authorizes_training false`.
- **Exactly one** student warning disappears.
- Mode C must **not** be described as genuinely re-evaluated while the teacher
  compatibility key remains a placeholder.

### Deliverables

A qualification record, `docs/QWEN35_9B_IDENTITY_QUALIFICATION.md`, recording:
the resolved commit; the complete metadata allowlist; the measured digests; the
architecture descriptor; the shared Base/instruct architecture finding; the
processor limitation; the no-weight proof; and the readiness impact.

A clarification in `configs/models/model.schema.md` that `digests_verified`:

- covers the tokenizer and the effective chat template;
- does **not** verify weights or vision processors;
- is independent of the architecture pin, which lives in the architecture
  profile.

---

## Paste-ready instruction for Claude Code

```text
Execute STAGE 1 ONLY of Milestone 6, exactly as specified in
docs/CLAUDE_CODE_MILESTONE6_QUALIFY_QWEN35_IDENTITY_HANDOFF.md.

Read the entire handoff first. Work in a clean worktree from origin/main; do
not modify the dirty original checkout.

Resolve Qwen/Qwen3.5-9B's immutable commit first via HfApi.model_info with
HF_HUB_DISABLE_IMPLICIT_TOKEN=1, require a complete 40-character commit, then
download METADATA ONLY from that exact commit using an allowlist derived from
the remote inventory, excluding every weight format. Prove no weights arrived,
following symlinks when you measure sizes.

Then measure offline: run verify_model_identity.py in report-only mode (expect
a non-zero placeholder refusal after it prints measured values), and compute
the architecture signature from that same snapshot's config.json. Never reuse
the Base value 52a3c9e2... -- recompute independently; if the instruct result
happens to equal it, that is acceptable, because the signature digests
architecture shape and provenance is what makes a value pinnable. Inventory the
processor/vision files and propose, but do not decide, a contract.

STOP at the mandatory checkpoint. No --write, no YAML edit, no architecture
pin, no commit, no PR. Do not modify verify_model_identity.py; if the chat
template is in an unsupported layout, stop and report that layout. Do not load,
execute, train or serve a model, and do not kill any pre-existing process.

Report the full checkpoint list from the handoff.
```
