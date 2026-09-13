# Claude Code handoff: Milestone 7 — resolve the Muse Glimmer identity

**Read this whole file before changing anything. Execute the phases in order
and stop on a failed invariant.**

**This is a Stage-1-only investigation.** It ends with an evidence and
schema-fit report and a state classification. It pins nothing, edits no
registry file, and does not set `digests_verified`.

Applicable date: **13 September 2026**.

## Objective

> Establish the documented lineage and effective serving-interface identity of
> Muse Glimmer without downloading weights or executing the model.

## The decision this milestone starts from

**Do not weaken the teacher contract by dropping tokenizer or chat-template
identity.**

A teacher consumed through a URL still tokenizes its input and still applies an
effective chat template. URL-based serving changes *where those artifacts are
owned*; it does not make them irrelevant. Setting `digests_verified: true`
without them would make "teacher identity verified" a materially weaker claim
than the student equivalent while using the same field name — which is worse
than leaving it false, because the field is read by the planner and recorded
into freeze manifests.

So: **keep the teacher unverified until the actual tokenizer and the effective
serving template are identified.** `identity_ready: false` is the correct
current state, not an engineering blocker to work around.

## What a prior survey established

Measured read-only at `origin/main`
`92c401980ee8c5a4b73b8eb13cf6be8cbf4b4467`. Verify these still hold; do not
re-derive them.

The declared BF16 source resolves publicly, unauthenticated:

```text
id            meta-models/Muse-Glimmer-30B-assistant
commit        e8192f3a8f617f74be2ce220360c89ef4789f39f
private       False        gated  False
pipeline_tag  image-text-to-text
lastModified  2026-08-11 00:04:51+00:00
```

Its complete inventory is six files:

```text
.gitattributes   LICENSE   README.md
USAGE_POLICY.md  config.json   model.safetensors
```

**No tokenizer files. No chat template. No processor metadata.** The registry
nonetheless declares `tokenizer.model_id` as that same repository, so the
nominated tokenizer source publishes no tokenizer.

Both verifier paths refuse, by design and correctly:

- `tokenizer_digest()` requires `tokenizer.json` or `tokenizer.model` —
  "this is not a tokenizer snapshot";
- `chat_template_digest()` requires `chat_template.jinja` or a `chat_template`
  key in `tokenizer_config.json` — "A base checkpoint without a template
  cannot serve the product's chat path".

A metadata-only acquisition would therefore request **nothing useful**: the
intersection of the identity allowlist with that inventory is empty, and the
only remaining file is a weight file.

### Two schema affordances, measured — they are not equivalent

**`tokenizer.model_id` may already differ from `model_id`, and the verifier
honours it.** `src/verify_model_identity.py` resolves the snapshot as:

```python
model_id = tokenizer.get("model_id") or spec.get("model_id")
declared_revision = tokenizer.get("revision") or spec.get("revision") or ""
```

So pointing the tokenizer at a different repository and revision works **today,
with no code change**. It is also guarded: `model_ids.matches()` refuses when
the cached snapshot's own repo id disagrees with the declared one, so this
cannot silently verify one checkpoint's digests into another's entry.

**`chat_template.source: file` is documented but NOT implemented.**
`configs/models/model.schema.md` states `source: model | file` with `path`
required when `file`. The verifier reads **neither**: it has zero references to
`spec["chat_template"]["source"]` or `["path"]`, and always digests the
template found in the snapshot. Proposing `source: file` therefore means
proposing a verifier change, not using an existing path. Say so explicitly if
you propose it.

### Downstream, if the teacher were verified

Simulated on a real copy of the worktree, never on the tree itself:

```text
identity_ready       false -> TRUE
trainable_as_is      false -> false   (fp8_ready, source_ready still false)
authorizes_training  false -> false
planner warnings     5 -> 4
```

Tests that would need replacing: `tests/test_distill_plan.py:363-368`
(unmarked, runs in CI) and `tests/test_training_manifest.py:248`
(`requires_local_corpus`).

**Checked-in run manifests are historical evidence and must not be
regenerated.** `train/RUN_MANIFEST.v1.json` and
`train/RUN_MANIFEST.tinker-sft-v1.preview.json` embed
`provenance_audit.identity_verification = {all_verified: false,
unverified_models: [muse-glimmer-30b]}`, but `freeze_training_run.py`
recomputes the audit fresh at freeze time and refuses to freeze without it.
Those blocks
record what was true when each run was frozen. A future freeze carries the new
verdict on its own.

`LICENSE_EVIDENCE_DIGEST_MUSE_GLIMMER_30B` is a **separate** unresolved
placeholder. `verify_model_identity.py` mentions `evidence_sha256` zero times.
Licence files may be collected as evidence here, but resolving that digest is a
different decision and is out of scope.

## Non-negotiable guardrails

1. **No weights.** `model.safetensors` is never downloaded, at any size, for
   any reason.
2. **No model execution.** No inference, generation, completion, or
   model-loading endpoint request. Local metadata inspection through
   `ollama list` and `ollama show` **is allowed** — those read the local
   daemon's metadata, they do not load or run a model. Do not start, restart,
   pull, create, or execute a model.
3. **No authentication.** The repository is public and ungated; a credential
   would change nothing and must not be introduced.
4. **No derivative substitution.** `RedHatAI/Muse-Glimmer-30B-FP8-block`, the
   MLX 4-bit derivative, `muse-glimmer:30b` and the LiteLLM gateway alias are
   serving artifacts. They may **corroborate** lineage. None may become the
   canonical tokenizer source merely because its files are reachable.
5. No registry, schema, test or documentation edits.
6. No `digests_verified` simulation presented as qualification. If you simulate
   to measure downstream effects, do it on a copy and label it a simulation.
7. No regeneration of historical manifests.
8. Never stop or kill PID 92762 or any other pre-existing process.
9. Work in a clean worktree from `origin/main`; do not modify either dirty
   original checkout.
10. Do not print tokens or credentials.

## Phase 1 — exact source documentation

Download **only** these four files from commit
`e8192f3a8f617f74be2ce220360c89ef4789f39f`, using per-file
`hf_hub_download()` with `HF_HUB_DISABLE_IMPLICIT_TOKEN=1`:

```text
config.json   README.md   LICENSE   USAGE_POLICY.md
```

Not `model.safetensors`. Not `.gitattributes`. Prove afterwards, by inspecting
the snapshot with sizes resolved through `blobs/`, that no weight file arrived.

Search that documentation for anything stating, in the repository's own words:

- a `base_model` declaration (config, model card metadata, or card text);
- the tokenizer it expects, and where that tokenizer comes from;
- a processor or preprocessing requirement;
- a chat template, prompt format, or conversation format;
- conversion provenance (what it was converted from, by whom, with what);
- serving requirements or a recommended runtime.

**Record exact quotations with their file and location.** Do not infer lineage
from architecture similarity: two models sharing `model_type` or layer counts
is not evidence of a shared tokenizer, and a `config.json` field that merely
resembles another family proves nothing about provenance.

If the documentation states a base model or tokenizer source, that is the
strongest available evidence and the likely route to state **A**.

## Phase 2 — derivative metadata, for corroboration only

For each of:

```text
RedHatAI/Muse-Glimmer-30B-FP8-block
mlx-community/Muse-Glimmer-30B-4bit
```

record whether it resolves publicly, its immutable commit, its complete sibling
inventory, whether it publishes tokenizer files, whether it publishes a chat
template, and **what upstream source it claims**.

An upstream claim may live only inside `config.json` or the card text, so a
strict no-download rule would ask for evidence it forbids reading. Use this
bounded escalation:

1. Inspect `model_info().cardData` and the remote inventory **first**. If the
   upstream claim is there, stop — nothing is downloaded.
2. Only if the claim is absent from `cardData`, download **only**
   `config.json` and `README.md`, from that derivative's **exact resolved
   commit**, and only if those files are present in its inventory.
3. Everything else stays prohibited at this stage: tokenizer files, templates,
   processor metadata, weights, and every other derivative artifact.
4. Record every derivative metadata file downloaded, and prove no weight file
   arrived — enumerate the snapshot with sizes resolved through `blobs/`.

These are conversions. If they carry a tokenizer the BF16 source lacks, that is
a strong hint about where the real tokenizer lives and a reason to go looking
for it — it is **not** permission to pin a derivative's tokenizer as the BF16
checkpoint's identity. A quantized conversion may also have altered the
template. Treat everything here as a pointer to be confirmed, never as the
answer.

## Phase 3 — the actual local serving artifact

The legacy corpus was produced by an Ollama artifact, not by the HF checkpoint.

**Begin by checking availability**: is the `ollama` CLI present, and is the
`muse-glimmer:30b` tag listed locally? Then inspect, in this canonical order:

```bash
ollama list
ollama show muse-glimmer:30b
ollama show --modelfile muse-glimmer:30b
```

These read local metadata. Do not start, restart, pull, create or execute a
model, and issue no inference, generation, completion or model-loading request.

Record: the manifest identity and digest, the source blob digests, the
`TEMPLATE`, the `SYSTEM` prompt, the parameters, and the declared
quantization.

Then answer the question that matters: **does the effective template belong to
the Ollama artifact rather than the HF checkpoint?** If the Ollama modelfile
carries a `TEMPLATE` and the HF repository carries none, then the serving
artifact owns the chat interface, and that is state **B**.

**If the CLI or the local tag is unavailable:** record the local
serving-artifact evidence as *unavailable*; do not install, pull or start
anything; and **continue** through historical attribution (Phase 4) and
schema-fit analysis (Phase 5).

Local absence is **not** evidence that the serving artifact has no template.
It is absence of evidence, and the report must say so in those terms rather
than concluding anything about the artifact.

## Phase 4 — historical corpus attribution

The legacy corpus is 136 traces recorded as
`int4_ollama @ muse-glimmer:30b @ ai19-ollama`.

Establish, from the corpus rows and the checked-in manifests only:

- exactly which artifact produced them;
- whether an immutable identity was recorded **at generation time** — an
  Ollama manifest digest, a blob digest, a model profile digest, anything
  that pins the bytes that ran;
- if such an identity was recorded, whether the artifact it names is still
  resolvable locally.

**If no immutable identity was recorded at generation time, say plainly that
the historical corpus cannot be retroactively upgraded to fully verified
identity from the BF16 repository alone.** Qualifying today's teacher would not
retroactively establish what produced traces months ago. That is state **D**,
and it can coexist with any other state.

## Phase 5 — schema-fit analysis

Evaluate, in this order, and report which the real evidence supports:

1. **`tokenizer.model_id` pointing at a different repository.** Already
   supported by the verifier, already guarded by `model_ids.matches()`. If
   Phase 1 or 2 identifies an authoritative tokenizer repository and revision,
   this is the honest representation and needs no code change. Evaluate it
   first.

2. **`chat_template.source: file` with a `path`.** Documented in the schema,
   **not implemented** by the verifier. If the effective template lives in the
   Ollama modelfile, representing it this way means committing the template
   text into this repository and teaching the verifier to read `source`/`path`.
   State the code change explicitly; do not present it as an existing
   affordance.

3. **A schema extension** — a distinct served-artifact or serving-profile
   identity, separate from the HF checkpoint identity. Propose this **only** if
   options 1 and 2 cannot represent the real evidence honestly. If you do
   propose it, say what it would assert, what would verify it, and what it
   would refuse.

The test is honesty of representation, not convenience. A representation that
makes the field fillable but the claim vaguer is worse than leaving the field
unfilled.

## Mandatory outcome

Two **independent** dimensions. Report both. Neither answer is expected, and
each must emerge from the evidence gathered above rather than from a
preference stated here.

### Dimension 1 — current teacher interface: exactly one of A, B, C

```text
A  authoritative tokenizer and template sources identified and verifiable
   an exact repository and revision for the tokenizer, and an exact source
   for the effective template, both acquirable and verifiable

B  some or all interface identity belongs to the serving artifact
   and needs an explicit serving-profile binding

C  insufficient evidence to verify the current interface
```

### Dimension 2 — historical corpus: an independent yes/no flag

```text
D  historical_unpinned = true | false
   true  -> generation-time artifact identity was NOT recorded, so the legacy
            corpus cannot inherit any current qualification
   false -> an immutable generation-time identity was recorded, and is named
```

Report as, for example,
`current = <A|B|C>, historical_unpinned = <true|false>`.
Any pairing is possible; the dimensions do not constrain each other. State the
evidence for each independently, and say what the next milestone would do
under the pair you actually found.

## Stop conditions

Stop without improvising if:

- any weight file is downloaded;
- a model is executed, served or queried;
- the BF16 repository stops resolving, or resolves to a different commit than
  `e8192f3a...` without an explanation;
- a derivative's tokenizer is about to be adopted as the canonical source;
- Phase 3 would require pulling or creating an Ollama model;
- a registry or schema file is about to be edited.

Do not resolve a stop condition by authenticating, downloading weights,
substituting a derivative, weakening the teacher contract, or setting
`digests_verified` on partial evidence.

## Definition of done

- The four documentation files are acquired from the exact commit, with proof
  no weight file arrived.
- Lineage evidence is recorded as exact quotations with locations, or its
  absence is recorded explicitly.
- Both derivatives' inventories and claimed upstreams are recorded, together
  with which of them required a `config.json` / `README.md` download and which
  were answered from `cardData` alone, and a clear statement that neither is
  being adopted as canonical.
- The local Ollama artifact's manifest, template, system prompt and parameters
  are recorded — or its unavailability is stated as absence of evidence, with
  Phases 4 and 5 completed regardless.
- The historical corpus's generation-time identity is established, or its
  absence is stated plainly.
- The schema-fit analysis names which option the evidence supports and what
  code change, if any, each would require.
- The result is reported on both dimensions:
  `current = <A|B|C>` and `historical_unpinned = <true|false>`, each with its
  own evidence.
- No registry, schema, test, manifest or documentation file is edited.
- The worktree is unchanged. The Hugging Face cache has changed **only** by
  the four primary documentation files from
  `meta-models/Muse-Glimmer-30B-assistant@e8192f3a...`, plus any explicitly
  permitted derivative `config.json` / `README.md` acquired under the Phase 2
  escalation rule. Every such file is listed in the report, and no weight file
  of any format is present in any snapshot this milestone created.
- `identity_ready` remains false and `train/TRAINING_BLOCKED.md` is untouched.

---

## Paste-ready instruction for Claude Code

```text
Execute Milestone 7, Stage 1 only, exactly as specified in
docs/CLAUDE_CODE_MILESTONE7_RESOLVE_MUSE_GLIMMER_IDENTITY_HANDOFF.md.

Read the entire handoff first. Work in a clean worktree from origin/main; do
not modify either dirty original checkout.

Download ONLY config.json, README.md, LICENSE and USAGE_POLICY.md from
meta-models/Muse-Glimmer-30B-assistant at commit
e8192f3a8f617f74be2ce220360c89ef4789f39f, using per-file hf_hub_download with
HF_HUB_DISABLE_IMPLICIT_TOKEN=1, and prove no weight file arrived. Search that
documentation for base_model, tokenizer, processor, template, conversion and
serving statements, recording exact quotations -- never inferring lineage from
architecture similarity.

Then examine RedHatAI/Muse-Glimmer-30B-FP8-block and
mlx-community/Muse-Glimmer-30B-4bit: do they publish tokenizer/template files,
and what upstream do they claim? Read cardData and the inventory first; only if
the upstream claim is absent there, download just config.json and README.md
from that derivative's exact resolved commit. They corroborate; they never
become the canonical source.

Check whether the ollama CLI and the muse-glimmer:30b tag exist, then run
ollama list, ollama show muse-glimmer:30b, and ollama show --modelfile
muse-glimmer:30b -- local metadata reads are allowed; do not start, restart,
pull, create or execute a model, and issue no inference request. If the CLI or
tag is unavailable, record that as absence of evidence and continue with
Phases 4 and 5. Establish whether the legacy corpus recorded an immutable
generation-time identity.

Evaluate schema fit in order: tokenizer.model_id pointing elsewhere (already
supported), chat_template.source: file (documented but NOT implemented by the
verifier), then a schema extension only if neither represents the evidence
honestly.

Report BOTH dimensions -- current = A, B or C, and historical_unpinned =
true or false -- from the evidence, not from any expected answer, and STOP.
Do not edit any
registry, schema, test or documentation file, do not set digests_verified, do
not download weights, do not execute a model, do not authenticate, and do not
kill any pre-existing process.
```

---

# Stage 2 — the three-PR correction sequence

Stage 1 executed and reported:

```text
current = A         historical_unpinned = true
```

Stage 2 is written against that result. It is **three separate pull requests
in a fixed order**, with a mandatory stop after the first.

## What Stage 1 proved

**The registry names the wrong repository.**
`meta-models/Muse-Glimmer-30B-assistant` is the **DFlash speculative-decoding
drafter**, not the teacher. Its own card opens with:

> "This model card is for the lightweight "drafter" model for Muse Glimmer
> 30B, based on DFlash... The main model then verifies these proposals in
> parallel."

Its `config.json` agrees — `architectures: ["MuseGlimmerAssistantModel"]`,
`model_type: muse_glimmer_assistant`, `num_hidden_layers: 5`,
`block_size: 16`, `target_layer_ids: [1, 13, 25, 37, 49]`. Five layers, not a
30B model. Its missing tokenizer is not an oversight: a drafter shares its
parent's tokenizer and legitimately ships none.

**The real teacher exists and is fully verifiable.**

```text
meta-models/Muse-Glimmer-30B   a4e59da52a7bc87ae7251dd5545c0dd437c44b68
public, ungated
tokenizer.json  tokenizer_config.json  chat_template.jinja
config.json  generation_config.json  processor_config.json
```

Corroborated from `cardData` alone by both derivatives, which each declare
`base_model: meta-models/Muse-Glimmer-30B`. No derivative was downloaded.

**The historical corpus cannot inherit any of this.** All 136 traces record
`repo: "muse-glimmer:30b"` — a mutable Ollama tag. No manifest digest, no blob
digest, no weights digest, anywhere in the corpus or the pass manifests. The
recorded `template_sha256`
(`114f55ebdc1804c1af371197b9fdf2d6bb925966c9dfe46b73782a71bc07965e`, identical
across all 136) pins the **rendered interface**, not the model bytes.

## The defect that dictates the order

Correcting the repository name and qualifying it in one step would introduce a
false claim, and this was **measured**, not predicted. Simulated on a copy of
the worktree with the teacher's `digests_verified` flipped to true:

```text
identity_ready       false -> TRUE
trainable_as_is      false -> false
authorizes_training  false -> false
```

`identity_ready` becoming true would be **wrong**. The current audit derives it
from the resolved teachers' registry verification alone, so verifying today's
parent would retroactively assert something about traces generated months ago
by an artifact nobody pinned. The HF parent cannot prove which Ollama artifact
ran.

So registry qualification and historical execution identity must become
**separate readiness claims before the teacher is marked verified**. That is
why PR B precedes PR C, and why PR A must not qualify anything.

## PR A — correct the registry source only

**Correct a false statement. Claim no qualification.**

Three source fields change:

```text
configs/models/muse-glimmer-30b.yaml
  model_id:            meta-models/Muse-Glimmer-30B
  tokenizer.model_id:  meta-models/Muse-Glimmer-30B

configs/teachers/muse-glimmer.yaml
  repos.bf16:          meta-models/Muse-Glimmer-30B
```

**`repos.bf16` lives in the serving config, and the two files are coupled.**
`src/distill_plan.py:240-242` reconciles `repos.bf16` against the registry's
`model_id` and refuses to plan when they disagree. Changing one file without
the other breaks the planner. Change both in this PR.

**Retain, unchanged:**

```text
revision:            REPLACE_WITH_PINNED_HUB_COMMIT
tokenizer.revision:  REPLACE_WITH_PINNED_HUB_COMMIT
digests_verified:    false
```

**Fields that are already correct and must not be touched:**

- `params.total_b: 30.0` — describes the parent, always did;
- `params.vision_b: null` — honest; do not invent a value;
- `repos.fp8: RedHatAI/Muse-Glimmer-30B-FP8-block` — a parent derivative;
- `repos.int4_mlx: mlx-community/Muse-Glimmer-30B-4bit` — a parent derivative;
- `repos.int4_ollama: muse-glimmer:30b` and
  `repos.remote: ollama/muse-glimmer-30b` — serving aliases. They may remain,
  but **neither is immutable identity evidence** and neither may be treated as
  one;
- the licence block, including its unresolved evidence placeholder.

`LICENSE_EVIDENCE_DIGEST_MUSE_GLIMMER_30B` remains untouched deliberately.
PR A corrects a false source-identity claim; it does not perform or claim a
licence review. The verifier does not own this field, and the drafter
repository's licence files must not be reused as evidence for the parent
without a separate review against the parent's exact commit.

### Required test

One offline, unmarked test proving:

1. the registry names the **parent**, not the drafter — assert
   `model_id == "meta-models/Muse-Glimmer-30B"` **and** explicitly assert it is
   not `...-assistant`, so the drafter can never silently return;
2. `tokenizer.model_id` equals the parent;
3. `repos.bf16` equals the parent;
4. the FP8 and MLX entries remain their existing parent derivatives;
5. the registry remains **unverified** after this correction —
   `digests_verified is False` and both revisions still placeholders.

Assertion 5 is the point of the test. It is what stops PR A from drifting into
a qualification.

### Invariants

Planning and readiness verdicts must be **unchanged** by PR A. The teacher
warning still fires, `identity_ready` stays false, `trainable_as_is` stays
false, `authorizes_training` stays false. Marker counts unchanged. If any
verdict moves, stop: the PR did more than correct a name.

`model_ids.matches()` already refuses to conflate the two:

```text
matches('meta-models/Muse-Glimmer-30B',
        'meta-models/Muse-Glimmer-30B-assistant') = False
```

### MANDATORY STOP

**Stop after PR A is merged.** Do not begin PR B in the same execution. The
readiness semantics change is a separate review.

## PR B — separate registry and execution-artifact readiness

Split the single `identity_ready` verdict into two claims plus their
conjunction. Report, at minimum:

```text
registry_identity_ready   the current canonical teacher revision, tokenizer
                          and template are verified
execution_artifact_ready  the traces identify the immutable artifact that
                          actually generated them
identity_ready            registry_identity_ready AND execution_artifact_ready
```

`identity_ready` keeps its name and its meaning as the conjunction, so every
existing consumer of it stays correct.

For the legacy corpus, after this PR and before PR C:

```text
registry_identity_ready   false   teacher not yet qualified;
                                  becomes true in PR C
execution_artifact_ready  false   mutable tag only; no manifest,
                                  blob or weights digest
identity_ready            false
```

`execution_artifact_ready` must be derived from **what the traces record**, not
from the registry. A mutable tag is not an artifact identity. Future traces
need an immutable execution receipt — an Ollama manifest digest, blob digest
or model profile identity — and this PR should say what such a receipt must
contain, even though generating one is out of scope.

**Do not regenerate historical manifests.** `train/RUN_MANIFEST.v1.json` and
the others embed `identity_verification: {all_verified: false,
unverified_models: [muse-glimmer-30b]}`, which was true when each was frozen and
remains historically correct. `freeze_training_run.py` recomputes the audit at
freeze time, so a future freeze carries the new shape on its own.

Tests currently asserting the old single verdict —
`tests/test_distill_plan.py:363-368` (unmarked) and
`tests/test_training_manifest.py:248` (`requires_local_corpus`) — are updated
to the new shape, not deleted.

## PR C — qualify the current parent

**Only after PR B is merged**, so the split prevents retroactive upgrading.

1. Acquire metadata only from
   `meta-models/Muse-Glimmer-30B@a4e59da52a7bc87ae7251dd5545c0dd437c44b68`,
   per-file `hf_hub_download()` with `HF_HUB_DISABLE_IMPLICIT_TOKEN=1`, an
   allowlist derived from the remote inventory, every weight format excluded,
   and a proof no weight file arrived (sizes resolved through `blobs/`).
2. Measure the tokenizer digest and the effective template digest **offline**.
   The parent ships `chat_template.jinja`, a layout the verifier already
   supports — confirm that rather than assuming it.
3. Pin the revision and both digests **through the verifier only**:
   `verify_model_identity.py --offline --snapshot ... --write`. Never by hand.
4. Add an exact-pin test for the five scalars, with literal expectations, in
   the manner of the student's
   `test_the_shipped_qwen35_instruct_spec_pins_the_qualified_snapshot`.
5. Confirm `execution_artifact_ready` stays **false** for the legacy corpus, and
   therefore `identity_ready` stays **false**.

The result: the current teacher is reproducibly qualified, and the historical
corpus remains honest about what it cannot prove.

## What remains out of scope throughout

- `LICENSE_EVIDENCE_DIGEST_MUSE_GLIMMER_30B` is unresolved and stays so; the
  verifier does not touch `evidence_sha256`.
- Processor identity remains a separate text-versus-vision decision. The parent
  publishes `processor_config.json`; that does not make it part of
  `digests_verified`.
- `fp8_ready` false (136 int4 traces) and `source_ready` false (136 synthetic)
  are untouched by all three PRs.
- `trainable_as_is` and `authorizes_training` remain false throughout.
  `train/TRAINING_BLOCKED.md` is never edited.
- **No schema change is required to identify the parent's tokenizer or template
  source** — correcting `model_id` is sufficient, and `tokenizer.model_id`
  then needs no divergence. The schema work that PR B implies concerns
  execution-artifact provenance, which is a different question.

## Stop conditions

Stop without improvising if:

- PR A changes any planning or readiness verdict;
- PR A and the serving config disagree, and the planner's reconciliation fires;
- PR B would make `identity_ready` true for the legacy corpus by any route;
- PR B regenerates a historical manifest;
- PR C is reached before PR B is merged;
- the parent stops resolving, or resolves to a commit other than
  `a4e59da5...` without explanation;
- any weight file is downloaded, or any model is loaded or executed;
- a serving alias (`muse-glimmer:30b`, `ollama/muse-glimmer-30b`) is about to
  be used as immutable identity evidence.

Do not resolve a stop condition by qualifying early, by widening
`digests_verified`, by treating a tag as an artifact identity, or by editing a
frozen manifest.
