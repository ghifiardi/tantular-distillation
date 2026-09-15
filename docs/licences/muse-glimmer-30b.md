---
schema_version: 1
registry_model: muse-glimmer-30b
model_id: meta-models/Muse-Glimmer-30B
revision: a4e59da52a7bc87ae7251dd5545c0dd437c44b68
reviewed_at: 2026-09-15
reviewed_by: Raditio Ghifiardi
sources:
  - path: LICENSE
    sha256: cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30
  - path: README.md
    sha256: 5894e358c58f4f8425645a5b0b890468ecb3e6b049c2e941648db077c4b7df58
  - path: USAGE_POLICY.md
    sha256: 98a14dab9fd97de1666dc8589d399efab1e7fc3ba4e6230d037d6637ba9481d3
determination:
  output_training_permitted: true
  rationale: >-
    For the specifically defined use reviewed here, the pinned Muse Glimmer
    documentation provides affirmative support for using Muse Glimmer to
    generate synthetic training data for downstream model development. I find
    no Muse-Glimmer-specific licence prohibition identified in the reviewed
    evidence that would, by itself, prevent the proposed teacher/distillation
    workflow, provided that Muse Glimmer weights are not redistributed; use
    complies with the applicable Usage Policy; the resulting student
    independently complies with the Qwen3.5-9B licence and other applicable
    component and data licences; prohibited-use restrictions and applicable
    laws are respected; and provenance of Muse Glimmer's role as a
    synthetic-data teacher is retained. This determination is limited to the
    exact proposed use defined in this record and is not a general determination
    covering every possible use of Muse Glimmer.
---

# Licence review — Muse Glimmer 30B

## Reviewer and authority

- **Reviewer:** Raditio Ghifiardi
- **Organization:** tantular.ai
- **Role:** Administrator
- **Authority:** I am authorized by tantular.ai to make binding decisions about
  third-party model licensing and the use of model outputs for synthetic
  training or distillation.
- **Review date:** 15 September 2026
- **Jurisdiction:** Indonesia

## Exact proposed use reviewed

The determination covers this specific workflow:

> Muse Glimmer 30B parent at the pinned revision → inference-generated synthetic
> corpus → Qwen3.5-9B Instruct fine-tuning or distillation → Tantular student →
> external distribution and potential commercial deployment, without
> redistribution of Muse Glimmer weights.

The reviewed activity is:

1. Use Muse Glimmer 30B as a teacher/inference model.
2. Generate outputs as part of the teacher inference process.
3. Retain those outputs as an internal synthetic training corpus.
4. Use that corpus to fine-tune or distill Qwen3.5-9B Instruct.
5. Potentially deploy the resulting student as part of Tantular, commercially.
6. Do not redistribute Muse Glimmer model weights.

The Muse-Glimmer-generated outputs and synthetic dataset will remain internal
and will not be distributed as a standalone dataset. The trained student or its
resulting weights may be distributed outside tantular.ai, including for local
or downloadable deployment. It may also be offered through a commercial hosted
service, API, or licensed deployment.

Muse Glimmer's role as a teacher and synthetic-data source will be disclosed in
appropriate Tantular documentation, such as the model card, training and
provenance documentation, release notes, or equivalent release material.

## Sources reviewed

All documents were reviewed from the parent repository
`meta-models/Muse-Glimmer-30B` at the pinned revision
`a4e59da52a7bc87ae7251dd5545c0dd437c44b68`, never from a moving `main` ref:

| Document | SHA-256 |
|---|---|
| `LICENSE` | `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30` |
| `README.md` | `5894e358c58f4f8425645a5b0b890468ecb3e6b049c2e941648db077c4b7df58` |
| `USAGE_POLICY.md` | `98a14dab9fd97de1666dc8589d399efab1e7fc3ba4e6230d037d6637ba9481d3` |

The DFlash drafter repository is not the reviewed checkpoint. Its licence and
usage-policy files happen to be byte-identical to the parent's, so document
hashes alone cannot establish source identity. The `model_id` and `revision` in
this record provide that binding.

## Human determination

### 1. Teacher-model use

**Permitted, subject to the applicable Usage Policy and other licence
conditions.**

The proposed activity uses Muse Glimmer 30B as a teacher/inference model and
does not redistribute the Muse Glimmer weights. The model documentation
identifies Muse Glimmer as available for commercial and research use and
specifically identifies synthetic-data generation for downstream model
development as an intended use.

### 2. Generation of synthetic training data

**Affirmatively supported by the model documentation.**

The proposed use includes generating outputs from Muse Glimmer and retaining
those outputs as a synthetic corpus. This is directly aligned with the stated
intended use of synthetic-data generation for downstream model development.
The synthetic corpus will remain internal to tantular.ai and will not itself be
distributed as a standalone dataset.

### 3. Training or distilling Qwen3.5-9B

**Supported by the stated intended use, subject to the terms of all applicable
licences and policies.**

The purpose of generating the synthetic data is downstream model development:
fine-tuning or distilling a separate Qwen3.5-9B student. The resulting student
will not contain or redistribute Muse Glimmer model weights.

### 4. Distribution of the resulting student

**No prohibition was identified in the reviewed Muse Glimmer evidence solely
because the student was trained using Muse-Glimmer-generated synthetic
outputs.**

The resulting Tantular student and its weights may be distributed externally,
including for downloadable or local deployment. Distribution must independently
comply with:

- the licence applicable to the Qwen3.5-9B base model;
- applicable Muse Glimmer Usage Policy requirements;
- licences or restrictions for other training data incorporated into the
  student; and
- applicable law and regulatory requirements.

Muse Glimmer weights will not be redistributed.

### 5. Commercial use

**Supported, subject to applicable licence and policy conditions.**

Commercial use is within the stated scope of Muse Glimmer's intended use. The
resulting Tantular student may therefore potentially be commercially deployed,
offered through a hosted service or API, distributed for local execution, or
incorporated into licensed Tantular deployments.

This conclusion does not override independent licensing requirements applicable
to Qwen3.5-9B or other Tantular components.

### 6. Provenance and transparency

Tantular will retain internal provenance identifying Muse Glimmer as a teacher
and synthetic-data source. When the student is externally distributed,
appropriate release documentation will disclose that Muse-Glimmer-generated
outputs were used as part of its training corpus.

## Conditions and limits

For the exact use reviewed here, the pinned Muse Glimmer documentation provides
affirmative support for using Muse Glimmer to generate synthetic training data
for downstream model development. I find no Muse-Glimmer-specific licence
prohibition in the reviewed evidence that would by itself prevent the proposed
teacher/distillation workflow, provided that:

1. Muse Glimmer weights are not redistributed as part of the workflow.
2. Use complies with the applicable Muse Glimmer Usage Policy.
3. The resulting student independently complies with the Qwen3.5-9B licence and
   every other applicable component and data licence.
4. Prohibited-use restrictions and applicable laws are respected.
5. Provenance of Muse Glimmer's role as a synthetic-data teacher is retained.

This determination is limited to the exact proposed use recorded above. It is
not a general determination for every possible use of Muse Glimmer.

This is a licence-review determination only. It does not override or satisfy
`train/TRAINING_BLOCKED.md`, which separately governs whether training may
actually proceed.
