# Muse Glimmer 30B: teacher identity qualification record

Date: **14 September 2026**. Milestone 7, PR C.

What was qualified, what was corrected first, and what remains out of reach.

## The correction this depended on

The registry named `meta-models/Muse-Glimmer-30B-assistant` for months. That
repository is the **DFlash speculative-decoding drafter** — a five-layer
companion network whose card opens "This model card is for the lightweight
'drafter' model for Muse Glimmer 30B", and whose `config.json` declares
`MuseGlimmerAssistantModel`, `num_hidden_layers: 5`, `block_size: 16` and
`target_layer_ids: [1, 13, 25, 37, 49]`.

It publishes no tokenizer and no chat template, which is correct for a drafter:
it shares its parent's. That absence is what exposed the error, because the
verifier refused with "this is not a tokenizer snapshot" rather than inventing
one.

The teacher is the parent. Both published derivatives agree —
`RedHatAI/Muse-Glimmer-30B-FP8-block` and `mlx-community/Muse-Glimmer-30B-4bit`
each declare `base_model: meta-models/Muse-Glimmer-30B` in card metadata, which
was read from `cardData` without downloading either.

## Source

```text
repository     https://huggingface.co/meta-models/Muse-Glimmer-30B
registry model muse-glimmer-30b
commit         a4e59da52a7bc87ae7251dd5545c0dd437c44b68
snapshot       ~/.cache/huggingface/hub/models--meta-models--Muse-Glimmer-30B/
                 snapshots/a4e59da52a7bc87ae7251dd5545c0dd437c44b68
```

Resolved from repository metadata before any file was fetched, with
`HF_HUB_DISABLE_IMPLICIT_TOKEN=1`. Public, ungated, unauthenticated. Every file
was downloaded from that exact commit, never from `main`. `refs/` is empty:
pinning by commit writes no branch ref, so provenance rests on the snapshot
directory name — which is what the verifier reads.

## Metadata allowlist

Six files, being the intersection of the identity-relevant list with what
exists at that commit:

```text
config.json                5,109 B
tokenizer.json        28,129,897 B
tokenizer_config.json     79,936 B
chat_template.jinja        9,992 B
generation_config.json       202 B
processor_config.json      1,084 B
```

Absent at this commit, so not requested: `tokenizer.model`,
`chat_template.json`, `special_tokens_map.json`, `added_tokens.json`,
`merges.txt`, `vocab.json`, `preprocessor_config.json`,
`image_processor_config.json`, `video_preprocessor_config.json`,
`feature_extractor_config.json`.

Present but deliberately not requested: `.gitattributes`, `LICENSE`,
`README.md`, `USAGE_POLICY.md`, `model.safetensors.index.json`.

### Proof no weights were downloaded

```text
files in snapshot   6
total bytes         28,226,220  (28.2 MB)
matching *.safetensors *.bin *.pt *.pth *.gguf *.onnx *.msgpack *.h5   NONE
largest file        tokenizer.json at 28,129,897 B
blob bytes          28,226,220 — equals snapshot bytes, so nothing else came
```

The repository holds two weight shards, `model-00001-of-00002.safetensors` and
`model-00002-of-00002.safetensors`. Neither was requested. A 30B checkpoint's
BF16 weights are roughly 60 GB.

## Measured identity

```text
tokenizer.sha256
  f945b361c8e542b5d2cb6737d03537d656aaeb94f5fe80b9ec7c00234260cf76
  over tokenizer.json, tokenizer_config.json

chat_template.sha256
  cfc67e5f349f37690dfd31ed1f18bc4442a9dd32fe39a648f993cb4eb3cae678
  source: chat_template.jinja
```

`tokenizer_config.json` is digested with its `chat_template` key removed, so
editing the template cannot silently move the tokenizer digest. In this
snapshot that key is absent entirely — the template exists only as
`chat_template.jinja`, a layout the verifier already supports, so no verifier
change was needed.

The tokenizer covers two files rather than the student's four: this repository
publishes no `merges.txt` or `vocab.json`, and the digest covers what is
actually present rather than asserting a fixed file list.

## What this does NOT establish

**The legacy corpus is not identity-ready.**

```text
registry_identity_ready   true    <- this qualification
execution_artifact_ready  false   <- unchanged by it
identity_ready            false   <- the conjunction
trainable_as_is           false
authorizes_training       false
```

All 136 legacy traces record `repo: "muse-glimmer:30b"` — a mutable Ollama
tag, with no manifest, blob or weights digest anywhere in the corpus or its
pass manifests. Qualifying today's checkpoint says nothing about which artifact
generated traces months ago. The audit reports `136 traces, 0 valid receipts,
136 missing, 0 malformed`.

The legacy corpus cannot become execution-artifact-ready from the evidence
currently recorded in this repository. It remains false unless contemporaneous
immutable artifact evidence is discovered and independently verified — an
archived Ollama manifest, a machine image, a registry log or a recorded blob
digest would each be candidates. Absence of evidence here is not proof that no
such evidence exists anywhere.

That separation is deliberate and was built before this qualification precisely
so this qualification could not launder it. See
`docs/TEACHER_AGNOSTIC_ARCHITECTURE.md` §5 for the corpus-composition policy.

**Weights are not verified.** Metadata-only qualification. A checkpoint whose
weights changed under an unchanged tokenizer and template would still read as
verified.

**The licence evidence is still unresolved, as of this record.**
`LICENSE_EVIDENCE_DIGEST_MUSE_GLIMMER_30B` remains a placeholder. This measured
a tokenizer and a template; it did not perform a licence review, and the
verifier does not own that field. The drafter repository's licence files must
not be reused as evidence for the parent without a separate review against the
parent's exact commit.

This is a statement of current state, not an invariant. Resolving that digest
through a real review is expected future work, and no test pins the placeholder
— a test asserting it would make a legitimate licence review look like a
regression.

**Processor identity is not part of the contract.** The parent publishes
`processor_config.json`; it is not digested, for the same reason the student's
vision processors are not. See `configs/models/model.schema.md`.

## Readiness impact

One planning warning disappears from `distill_plan plan office-v2-sequence`:

```text
REMOVED  teacher 'muse-glimmer-30b' digests_verified is not true

REMAINS  auto: on_policy_kd unavailable -> tokenizer compatibility keys differ
REMAINS  auto fell back to preference (judge-gated)
REMAINS  memory estimate for fp8 is INCOMPLETE (vision tower params unknown)
REMAINS  memory estimate for bf16 is INCOMPLETE (vision tower params unknown)
```

**Mode C is now a real comparison for the first time.** Both sides carry
measured tokenizer digests, so "compatibility keys differ" is a fact about two
real tokenizers rather than about two placeholders. The verdict is unchanged —
a non-Qwen teacher genuinely differs from a Qwen student — but it is now
evidence rather than a default.

`fp8_ready` false (136 int4 traces) and `source_ready` false (136 synthetic)
are untouched, so `trainable_as_is` and `authorizes_training` remain false.
`train/TRAINING_BLOCKED.md` is unchanged, and no historical manifest was
regenerated.
