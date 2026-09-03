# Teacher-agnostic distillation — architecture (rev. 2026-09-03)

**Status: DESIGN DRAFT. This document authorizes nothing.**
`train/TRAINING_BLOCKED.md` remains controlling: no training is justified until a
real observed product-capability gap appears. This revision incorporates the
findings in `docs/Research Report on Proposal Review.pdf` (3 Sep 2026), which
checked the first draft against the repository and published sources.

What changed from the first draft, and why:

| # | First draft said | Correction folded in here |
|---|---|---|
| 1 | A larger teacher improves the 9B student | **Contested** — u-shaped distillation scaling (arXiv:2502.08606); teacher size is a hyperparameter to *validate*, not a direction to assume. |
| 2 | Add Qwen teachers; registry keyed on size | Repo's real teachers are **non-Qwen** (`muse-glimmer`, `nemotron`); the open risk is a **licence**, not a size. Registry is keyed on **identity**, and licence is a **gate**. |
| 3 | Token-level KL after a size swap | Cross-tokenizer token-level KL is a **live cross-family problem today**; gate Mode C on a tokenizer **compatibility key**. |
| 4 | Reuse the 9B LoRA target list | The 9B target list is architecture-specific; require an **architecture signature** match read from config, never assumed. |
| 5 | `bridge_client.py` should capture finish_reason as the truncation signal | **Contradicted** by that file's own docstring; keep the token-count heuristic; finish_reason is metadata only. |
| 6 | Fixed replay mixture `0.55/0.20/0.15/0.10` | **Unsupported**; sweep the ratio against the existing gates; anchor *scoring* (KL penalty vs NTP) matters as much as the ratio. |
| 7 | 2x80GB FP8 for a 122B teacher | The fleet has no 80GB card; account for the **vision tower**; the planner refuses machines the project does not have. |

---

## 0. Scope and non-goals

- **In scope:** a pipeline that can distill from a teacher of *any* size/family
  into the existing 9B student without re-architecting per teacher.
- **Not in scope:** designing a new student architecture, or shipping a
  Base-trained artifact (see `train/BASE_VS_INSTRUCT.md`).
- **Non-negotiable:** every real run still requires (a) a demonstrated product
  capability gap and (b) a schema freeze. This document is the *how*, gated
  behind the *whether*.

## 1. Keep the 9B student; treat teacher size as a hypothesis

A larger teacher does **not** require a larger or matching student, and does not
reliably produce a better student. The distillation scaling literature reports a
**u-shaped** relationship with an optimal teacher size (the "curse of capacity
gap"), and a plain sequence of distillation steps was found *ineffective* as a
capacity-gap remedy.

Consequence for the plan:

- The `large -> 9B -> 4B` chain stays, but its justification is **behavioural
  alignment**, not capacity repair: the 4B learns from a 9B that already shares
  Tantular's Indonesian voice, edit contracts, and tool schemas.
- Before any spend on a larger teacher, run the **cheap experiment the pipeline
  is already qualified for**: generate verified-trace SFT corpora from two
  teacher sizes, train two students, and compare them on the **existing** gates
  (`indonesian_voice`, `edit_contract_output`, `office_json_contract`). Bigger
  teacher wins only if the student wins.

```
Capability gap / production failure   (required to start at all)
          |
          v
  Distillation Plan Registry     configs/distillation/*.yaml
  teacher + student + objective + mode + replay
          |
          v
  Runtime Planner (fail-closed)  src/distill_plan.py
  licence gate -> arch signature -> compatibility key -> mode -> hardware
          |
          v
  Teacher Service                any size/family; served by URL only
          |
          v
  Candidate Trace Generation     multi-candidate; answers/tools/structured
          |
          v
  Verification + Ranking         contracts, faithfulness, voice (deterministic)
          |
          v
  Versioned Dataset Release      pinned revisions, template digest, provenance
          |
          v
  Mode A (SFT) -> Mode B (preference, judge-gated) -> Mode C (on-policy KD)
          |
          v
  9B student -> before/after gates -> promotion -> optional 9B->4B
```

## 2. A registry keyed on identity, with a licence *gate*

`configs/teachers/` conflates identity and role and reasons about the wrong
family. The registry is split so identity is separate from role and hardware:

- `configs/models/*.yaml` — one file per checkpoint (teacher **or** student).
- `configs/architectures/*.yaml` — LoRA targets, layer expectations, adapter
  key prefixes, and Mode-C eligibility, keyed by an **architecture signature**.
- `configs/distillation/*.yaml` — a run *plan*: which teacher, which student,
  objective, mode, replay policy.

Every model spec pins the three things that silently drift:

1. **`revision`** — the exact Hub commit. A teacher used for a real corpus, or
   any Mode-C run, must not carry a null revision.
2. **`tokenizer.sha256`** — a digest over the tokenizer files. This is the
   **compatibility key** (section 4), not merely provenance.
3. **`chat_template.sha256`** — because `office-student-9b.yaml` already warns
   that a changed template silently changes what `prompt_sha256` measured.

And it carries the field the review flagged as the highest-value part of the
registry — a **licence gate**, not a note:

```yaml
license:
  identifier: apache-2.0
  output_training_permitted: true      # GATE: false => planner refuses
  reviewed_at: 2026-09-03              # GATE: staler than recheck_max_age_days => refuse
  recheck_max_age_days: 180
  evidence_sha256: <digest of the saved licence text/decision>
```

This exists because Apache 2.0 held across Qwen 3.5 and 3.6 and **stopped
holding at the 3.8 flagship**, and because the repo's own `nemotron` teacher
ships under NVIDIA OML with an unresolved synthetic-data question. A licence
recorded once and trusted forever is a real hazard; the planner re-checks
freshness on every run.

`generation` (e.g. `"3.5"`) is recorded so a plan can prefer the cheapest
*current* upgrade (a newer 27B) over a hypothetical one.

## 3. Architecture profiles, matched by signature — never assumed

`train/qlora_9b.yaml` correctly encodes 32 layers, the three-linear-then-one-full
attention pattern, the split projection names, and the vLLM `in_proj_qkvz`
fusion trap. Those are **specific to the 9B** and must not be inherited by, say,
a 48-layer 256-expert MoE.

Each `configs/architectures/*.yaml` pins an **architecture signature** computed
from the model's `config.json` (model_type, layer count, layer-type pattern,
hidden sizes). Before training, the trainer introspects the loaded config,
recomputes the signature, and **aborts on mismatch** — the same class of
fail-closed check the repo already applies to corpora and adapters.

MoE students get deliberate expert handling: automatically targeting every
expert projection produces an enormous adapter, and the serving runtime may not
bind it faithfully. Expert LoRA is opt-in per profile, never the default.

## 4. Three modes, with the tokenizer compatibility key deciding Mode C

### Mode A — sequence distillation (default, family-agnostic)
`prompt -> teacher -> verified final answer -> student SFT`. HTTP-only, needs no
teacher logits, tolerates different tokenizers. Generate several candidates per
prompt; keep only those that pass deterministic validation.

### Mode B — preference distillation (narrowed, judge-gated)
The first draft claimed preference training "extracts more signal than SFT." The
measured evidence is equivocal (gains small, and they reverse as pair count
grows; label-only DPO regressed). So Mode B is **kept but constrained**:

- Scope limited to **sequence-level** properties: Indonesian voice and
  hallucination avoidance.
- **Not** used for edit-operation selection or tool-argument correctness, which
  are token-localized — the `contract_ok` checker already localizes those into
  `parse_ok`/`fields_ok`.
- Preference pairs **must be gated through a judge before training**.

### Mode C — on-policy logit distillation (gated on the compatibility key)
TRL's `DistillationTrainer` has graduated to the stable API (chunked generalized
JSD, vLLM-powered student generation). It is **more constrained** than the first
draft assumed:

- The stable v1.10.0 reference **requires the teacher to share the student's
  vocabulary**. The split-host teacher-server topology (`use_teacher_server`,
  `teacher_model_server_url`) is a **main-branch feature, not a released
  guarantee** — pin it by TRL version and read that version's own docs.
- The loss reads `lm_head.weight` directly and runs the backbone **without**
  `PeftModel.forward()`, so **`lm_head` adapters and prompt-learning are
  rejected**. A Mode-C profile must set `mode_c.targets_lm_head: false`.
- Defaults differ sharply from `qlora_9b.yaml` and must be stated explicitly in
  the plan, never inherited: `learning_rate` 1e-6 (vs 1e-4 — a 100x gap),
  `max_completion_length` 512 (vs a 32k training sequence — watch
  `completions/clipped_ratio`), `vllm_mode` colocate, `beta` 1.0 (reverse KL,
  not JSD's 0.5).

**Compatibility-key gate.** The repo's real teachers are not Qwen, so a teacher
and the Qwen student generally have **different tokenizers today**. Token-level
KL across mismatched vocabularies is a research project, not a config flag.
Therefore the planner:

- selects Mode C **only when `teacher.tokenizer.sha256 == student.tokenizer.sha256`**;
- for `mode: auto`, **falls back to Mode B (if pairs are configured) or Mode A**
  on a mismatch, rather than silently losing supervision;
- for an **explicit** `mode: on_policy_kd` request on a mismatch, **refuses**
  loudly instead of downgrading a hard request.

## 5. Trace schema v4 — richer, but not stronger evidence

A versioned conversation format captures multi-part content (document pages),
tool definitions/calls/results, multiple candidates, chosen/rejected pairs, and
optional compressed top-k logprobs. Two constraints:

- **Do not** make hidden chain-of-thought the training target; distill final
  answers, plans, tool trajectories, and citations — verifiable artifacts.
- `finish_reason` is recorded as **metadata only**. Truncation is still inferred
  from token count, because `bridge_client.py` documents that the gateway
  reports `stop` even when generation ran out of budget mid-object.

A richer schema does **not** change what the current corpus is evidence *of*: it
remains `source_class: synthetic` under a signed int4 waiver, supporting no claim
about real Office documents.

## 6. Replay anchors — sweep, do not fix

Replay is the best-evidenced recommendation. But the first draft's fixed
`0.55/0.20/0.15/0.10` mixture has no source; the published evidence clusters
around **~1:1**, and *how* anchors are scored (a KL penalty against the reference
vs plain next-token loss on substitute data) matters as much as the ratio.

Tantular's advantage: the voice and edit anchors are the **same 40-item and
20-item gate sets** run before/after every training run. So the mixture is a
swept hyperparameter measured against a metric that already exists — the plan
declares a `sweep`, an `anchor_scoring` mode, and the gates to sweep against, and
asserts no single ratio.

## 7. Hardware planning — fleet-aware and fail-closed

Weight math (2 bytes/param BF16, 1 at FP8, 0.5 at int4) is necessary but not
sufficient:

- Qwen3.5 checkpoints are **multimodal**; the vision tower is not optional. When
  a spec does not record `params.vision_b`, the planner flags the estimate as
  **incomplete** rather than pretending it is complete.
- For MoE, `A3B`/`A10B` is active compute, **not** resident memory — all experts
  stay resident.
- The planner matches an estimate against the **declared** fleet
  (`configs/hosts/*.yaml`). If no host qualifies, it **refuses and reports what
  would have to be procured**, rather than emitting a plan for a 2x80GB machine
  the project does not have (the fleet tops out at one rented 48GB Ada; `ai19`
  is Ampere and cannot do FP8 at all).

## 8. What is unchanged

`train/TRAINING_BLOCKED.md` still controls. The evidence that reopens training is
a real observed failure — not "the teacher is bigger." Building this architecture
now is cheap and reversible; qualifying a larger teacher because it is larger is
the failure this design exists to prevent.

## 9. Draft artifacts in this change

- `configs/models/model.schema.md` — field reference and gate semantics.
- `configs/models/{qwen35-9b-instruct,muse-glimmer-30b,qwen35-122b-a10b}.yaml`
- `configs/architectures/{qwen35-hybrid-dense-9b,qwen35-hybrid-moe}.yaml`
- `configs/distillation/office-v2-sequence.yaml`
- `src/distill_plan.py` — the fail-closed planner, with three subcommands:
  `plan` (licence gate, architecture signature, compatibility-key gating,
  fleet-aware hardware check), `arch-signature` (digest a config.json), and
  `provenance-audit` (mechanical limits of a promoted corpus: teacher quant,
  source_class, FP8-gate status vs the signed waiver, and licence freshness).

As of the wiring change below, these are no longer drafts sitting beside the
pipeline:

- `src/verify_model_identity.py` fills the placeholder digests from a real local
  snapshot, or refuses. Nothing else may write them.
- `src/train_qlora.py` recomputes the architecture signature from the loaded
  model and aborts before PEFT attaches anything if it differs from the profile.
- `src/freeze_training_run.py` records the corpus audit in the run manifest
  (`schema_version` 4, Tinker 5).
- `src/distill_plan.py plan` reconciles each registry entry against its
  `configs/teachers/*.yaml` serving config and refuses while they disagree.

The digests and the architecture signatures are still placeholders, because no
snapshot of the product's *instruct* checkpoint is available locally to measure
them from. That is the fail-closed state working, not an oversight: the trainer
refuses, and `digests_verified` stays `false`.

## 10. From a plan to a corpus to a gate verdict

A distillation plan is not a new pipeline. It selects the teacher, the mode and
the host for the sequence the v1 run already used, and every step below is an
existing tool. `src/distill_plan.py plan <name> --dry-run` prints exactly this
sequence for a given plan, filled in with the teacher and host it qualified. It
prints and nothing else — no trace is emitted, no file is written.

| Plan field | Decides | Consumed by |
|---|---|---|
| `teacher` | which serving config generates traces | `src/generate.py --teacher` |
| `mode` (after the compatibility-key gate) | sequence: generate completions. preference: generate pairs and judge them before promotion. `on_policy_kd`: refused across a tokenizer split | `src/generate.py`, the promote step |
| `precision_preference` + the fleet | which host may serve the teacher | `src/generate.py --host` |
| `student` | which endpoint the before/after gates measure | `src/run_gates.py --teacher <serving config> --expect-model` |
| `replay.anchor_gates` | which gates a replay-ratio sweep is scored against | `src/run_gates.py compare` |

The ordered mapping:

1. **generate** — `src/generate.py --teacher <serving config> --host <qualified
   host>`. The host comes from the planner's hardware gate, so a plan cannot
   quietly generate on a box that cannot hold the teacher at the stated
   precision. In Mode B the judge runs here, before anything is promoted:
   `configs/distillation/*.yaml` sets `preference.judge_required: true` and
   scopes pairs to voice and faithfulness, never to token-localized edit ops.
2. **verify** — `src/verify_corpus.py <corpus> --gate`. Unchanged, and still the
   thing that decides. A quantized teacher fails it; a signed waiver authorises
   proceeding *despite* the failure and never converts it to a pass.
3. **promote** — `src/promote_corpus.py`, which writes the mechanical promotion
   manifest.
4. **before gate** — `src/run_gates.py run --stage before`, against the base
   student, on a host that is not the training host.
5. **freeze** — `src/freeze_training_run.py`, which now also records the corpus
   audit (`provenance_audit`): FP8-gate status against the signed waiver, source
   classes, and teacher licence freshness measured against the freeze date.
   Additive evidence; it does not change what the gate decided.
6. **train** — still blocked. `train/TRAINING_BLOCKED.md` is controlling, and no
   planner output lifts it. The `--dry-run` sequence deliberately prints no
   training command. What reopens training is a real observed failure recorded
   in a `train/TRAINING_JUSTIFIED.md`, not a plan that validated.
7. **after gate and compare** — `src/run_gates.py run --stage after` then
   `compare`. `train/qlora_9b.yaml`'s instruction stands: do not promote an
   adapter that regresses either gate.

What the dry run will not fill in: the held-out prompt set, the freeze
timestamp, the training config. Those are decisions, and a planner that guessed
them would produce a corpus measuring something nobody chose.
