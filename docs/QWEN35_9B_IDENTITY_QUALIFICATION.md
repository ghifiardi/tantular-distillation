# Qwen3.5-9B instruct: model-identity qualification record

Date: **13 September 2026**. Milestone 6.

What was qualified, and — just as importantly — what was not.

## Source

```text
repository     https://huggingface.co/Qwen/Qwen3.5-9B
registry model qwen35-9b-instruct
commit         c202236235762e1c871ad0ccb60c8ee5ba337b9a
snapshot       ~/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/
                 snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a
```

The commit was resolved from repository metadata **before** any file was
fetched, with `HF_HUB_DISABLE_IMPLICIT_TOKEN=1`:

```python
api.model_info("Qwen/Qwen3.5-9B", revision="main", files_metadata=False)
```

A complete 40-character commit was required, and every file was then downloaded
from **that commit**, never from `main`. The repository is public and not
gated; the request was unauthenticated.

`refs/` in the cache repo is empty. Downloading by explicit commit writes no
branch ref, so the snapshot's provenance rests on its directory name — which
is what the verifier reads.

## Metadata allowlist

The remote inventory at that commit is 16 files: 12 metadata and 4 weight
shards. Eight were downloaded, being the intersection of the identity-relevant
list with what actually exists there:

```text
config.json                       3,126 B
tokenizer.json               12,807,982 B
tokenizer_config.json            16,710 B
chat_template.jinja               7,756 B
merges.txt                    3,353,259 B
vocab.json                    6,722,759 B
preprocessor_config.json            390 B
video_preprocessor_config.json      385 B
```

Absent at this commit, so not requested: `tokenizer.model`,
`chat_template.json`, `special_tokens_map.json`, `added_tokens.json`,
`generation_config.json`, `processor_config.json`,
`image_processor_config.json`, `feature_extractor_config.json`.

Present but deliberately not requested: `.gitattributes`, `LICENSE`,
`README.md`, `model.safetensors.index.json`.

### Proof no weights were downloaded

```text
files in snapshot   8
total bytes         22,912,367  (22.9 MB)
matching *.safetensors *.bin *.pt *.pth *.gguf *.onnx *.msgpack *.h5   NONE
largest file        tokenizer.json at 12,807,982 B
blob bytes          22,912,367  -- equals snapshot bytes, so nothing else
                    was fetched
```

A 9B checkpoint's weights are roughly 18 GB. Sizes were measured through the
cache's symlinks into `blobs/`; an unresolved `stat` reports the symlink length
instead and would have shown misleading 52-byte readings.

Acquisition used per-file `hf_hub_download()`. The `hf download` CLI was tried
first and silently defeated the allowlist — it parsed the filenames as
positionals, discarded both `--include` and `--exclude`, printed a snapshot
path as though it had succeeded, and created nothing.

## Measured identity

```text
tokenizer.sha256
  6f3a76fa0ff84cba487813d4024623233c4664ecedfc3f3857536f95d25504af
  over merges.txt, tokenizer.json, tokenizer_config.json, vocab.json

chat_template.sha256
  a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715
  source: chat_template.jinja

config.json sha256
  d0883072e01861ed0b2d47be3c16c36a8e81c224c7ffaa310c6558fb3f932b05
```

`tokenizer_config.json` is digested with its `chat_template` key removed, so
editing the template cannot silently move the tokenizer digest.

**The template exists twice in this snapshot** — as `chat_template.jinja` and
inside `tokenizer_config.json`'s `chat_template` key. Both are 7756 characters
and byte-identical, so the effective template is unambiguous and the recorded
digest describes both.

## Architecture

```text
signature  52a3c9e2c895c0dc3d81abbb24e6463256cf3869f6e7343974f773a3f002096d
```

Descriptor from the same snapshot's `config.json`:

```text
model_type            qwen3_5
architectures         Qwen3_5ForConditionalGeneration
text_model_type       qwen3_5_text
num_hidden_layers     32               hidden_size    4096
num_attention_heads   16               num_key_value_heads  4
num_experts           None             num_experts_per_tok  None
layer_types           [linear, linear, linear, full] x 8
vision                model_type qwen3_5, depth 27, hidden_size 1152,
                      out_hidden_size 4096
```

Agrees with `configs/architectures/qwen35-hybrid-dense-9b.yaml` field by field:
`model_type` matches, 32 layers matches, and the declared
`[linear, linear, linear, full] x 8` pattern reproduces the observed
`layer_types` exactly.

### Base and instruct share this architecture

`Qwen/Qwen3.5-9B` and `Qwen/Qwen3.5-9B-Base` ship a **byte-identical**
`config.json` (`d0883072...`), so both produce signature `52a3c9e2...`.

This is expected and acceptable. The signature identifies architecture
**shape** — not checkpoint identity, weights, instruction tuning or
chat-template behaviour — and two releases of one family may legitimately
share it. What distinguishes them here is the chat template: the Base
release ships none, so it could never satisfy a template digest.

The pinned value was **recomputed independently** from the instruct snapshot,
not copied because it had been measured from Base. Equality is not evidence of
a mistake; unproven provenance would be.

## What is NOT qualified

**Weights.** This is a metadata-only qualification. No weight file was
downloaded, loaded or hashed. A checkpoint whose weights changed under an
unchanged tokenizer and template would still read as verified.

**Vision processors.** The snapshot carries two, recorded here as non-gating
evidence:

```text
preprocessor_config.json        Qwen2VLImageProcessorFast
  sha256
  27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516
video_preprocessor_config.json  Qwen3VLVideoProcessor
  sha256
  7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13
```

Neither is part of `digests_verified`. The shipping add-in uses the text path,
so image and video preprocessing are not part of the compatibility decision
this milestone qualifies, and an unused modality should not be able to block a
text-path decision. Consequently:

- vision processor identity remains **unqualified**;
- `modality.vision: true` remains accurate as a *checkpoint capability*;
- `params.vision_b` remains `null`;
- the FP8 and BF16 memory estimates remain **INCOMPLETE**;
- **any future product use of image or video input requires a separate
  processor-identity gate before qualification.**

## Readiness impact

Exactly one planning warning disappears from
`distill_plan plan office-v2-sequence`:

```text
REMOVED  student 'qwen35-9b-instruct' digests_verified is not true

REMAINS  auto: on_policy_kd unavailable -> tokenizer compatibility keys differ
REMAINS  auto fell back to preference (judge-gated)
REMAINS  teacher 'muse-glimmer-30b' digests_verified is not true
REMAINS  memory estimate for fp8 is INCOMPLETE (vision tower params unknown)
REMAINS  memory estimate for bf16 is INCOMPLETE (vision tower params unknown)
```

The legacy corpus's readiness block is **unchanged**:

```text
fp8_ready       false     136 int4_ollama traces
source_ready    false     136 synthetic traces
identity_ready  false     resolves teacher muse-glimmer-30b, still unverified
license_ready   true
harness_ready   true

trainable_as_is      false
authorizes_training  false
```

`identity_ready` is derived from the corpus's *resolved teachers*, not from the
student, and that corpus resolves only to `muse-glimmer-30b`.

**Mode C has not been re-evaluated.** It remains unavailable, and the reason is
worth stating precisely: the "compatibility keys differ" verdict previously
compared two placeholders and now compares one measured key against a
placeholder. The verdict is unchanged — a non-Qwen teacher genuinely differs
— but it does not become a real comparison until the teacher is qualified
too.

## No training authority changed

`train/TRAINING_BLOCKED.md` is untouched. No model was downloaded in weight
form, loaded, executed, served or trained. No Tinker or cloud workload ran.
