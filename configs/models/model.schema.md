# Model registry schema (DRAFT, schema_version 1)

One file per checkpoint under `configs/models/`. A checkpoint is identity only;
hardware lives in `configs/hosts/`, LoRA layout in `configs/architectures/`, and
a run in `configs/distillation/`.

The planner (`src/distill_plan.py`) treats several fields as GATES. A gate that
cannot be evaluated fails CLOSED (the run is refused), mirroring the repo's
corpus/adapter checks.

## Fields

```yaml
schema_version: 1

model_id: Qwen/Qwen3.5-9B            # exact Hub repo id
revision: <hub-commit-sha>           # pinned commit. null is allowed ONLY for a
                                     # model never used as a Mode-C teacher or in
                                     # a real (non-validation) corpus run.
role: student                        # student | teacher
family: qwen3.5                      # free-form family label (grouping only)
generation: "3.5"                    # so a plan can prefer a current upgrade

modality:
  text: true
  vision: false                      # if true and params.vision_b is null the
                                     # planner marks memory estimates INCOMPLETE

params:
  total_b: 9.0                       # billions, all resident weights
  active_b: null                     # MoE active-per-token; null for dense
  vision_b: null                     # vision tower params if known

tokenizer:
  model_id: Qwen/Qwen3.5-9B
  revision: <hub-commit-sha>
  sha256: <digest over tokenizer files>   # THE COMPATIBILITY KEY (section 4)

chat_template:
  source: model                      # model | file
  path: null                         # required when source: file
  sha256: <digest of the effective template>

architecture_profile: qwen35-hybrid-dense-9b   # -> configs/architectures/<name>.yaml

serving_config: office-student-9b     # -> configs/teachers/<name>.yaml, which must
                                      # point back with registry_model: <this file>

capabilities:
  tools: true
  logprobs: true                     # required true for a Mode-C teacher

license:
  identifier: apache-2.0
  output_training_permitted: true    # GATE: false -> refuse (teacher)
  reviewed_at: 2026-09-03            # GATE: age > recheck_max_age_days -> refuse
  recheck_max_age_days: 180
  evidence_sha256: <digest of saved licence text/decision>

digests_verified: false              # written ONLY by src/verify_model_identity.py,
                                     # and only when both digests match a real local
                                     # snapshot whose commit is known. Never by hand.
```

## Gate semantics (evaluated by the planner)

| gate | applies to | passes when |
|---|---|---|
| licence.output_training_permitted | teacher | is exactly `true` |
| licence.reviewed_at freshness | teacher | `today - reviewed_at <= recheck_max_age_days` |
| licence.evidence_sha256 present | teacher | non-empty |
| revision pinned | teacher (real run / Mode C) | non-null |
| tokenizer.sha256 present | both (Mode C) | non-empty on both sides |
| compatibility key match | teacher vs student (Mode C) | `teacher == student` digest |
| capabilities.logprobs | teacher (Mode C) | `true` |
| architecture signature | student | profile signature matches the loaded config (`src/train_qlora.py`, before LoRA attaches) |
| registry/serving reconciliation | both | `serving_config` and `registry_model` point at each other, and `served_model_name`, `repos.bf16`, `tokenizer` and `license` agree with this file |

`digests_verified: false` does not block Mode A/B *planning* — the planner emits
a WARNING and continues — but it does block a corpus from being called trainable:
`provenance-audit` reports `identity_verification.all_verified: false` and
`trainable_as_is: false` while any resolved teacher is unverified. The
compatibility key that chose the distillation mode lives in these digests, so an
unchecked one means the mode was selected against a checkpoint nobody confirmed.
To set it:

    ./.venv/bin/python src/verify_model_identity.py <name> --offline --write

It measures the digests from the local Hugging Face cache and refuses on any
mismatch, on a snapshot whose commit is unknown, or on a snapshot belonging to a
different repo — `Qwen3.5-9B` and `Qwen3.5-9B-Base` are one suffix apart and are
not the same product (`train/BASE_VS_INSTRUCT.md`).

The reconciliation gate exists because two files describe each checkpoint: this
one owns IDENTITY, `configs/teachers/<name>.yaml` owns SERVING, and they are read
by different tools — `src/config.py` resolves the serving config for the gates
while `src/distill_plan.py` reads this registry. A disagreement does not raise an
error anywhere; it produces a plan written for one checkpoint and gates measuring
another. A serving config's `fp8`/`int4`/gateway repos are deliberately different
artefacts and are not identity claims; `repos.bf16` is.
