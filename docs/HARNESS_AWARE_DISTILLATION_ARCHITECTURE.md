# Harness Aware Distillation Architecture

**Status: design and bounded implementation. This document authorizes no
training.** `train/TRAINING_BLOCKED.md` remains controlling.

## 1. Research conclusion

The supplied research pack is directionally useful, but its headline
“the harness is more important than the model” is too broad.

What the primary record supports:

- A harness can unlock substantially more capability from the same model on a
  particular task. Prime Agent reports raising ARC-AGI-3 RHAE Best@1 from 30%
  to 95.5% through a persistent REPL, recursive sub-agents, state management,
  and additional test-time computation.
- Model choice still matters. OpenJarvis reports that replacing its cloud model
  with Qwen3.5-9B inside an existing stack loses 25-39 percentage points. Prompt
  optimization alone recovers only about 5 points; optimizing a typed,
  five-component system closes much more of the gap.
- Durable state, scoped permissions, and isolated execution are real harness
  architecture, not prompt decoration. QM places sessions, memory, and queues
  in Postgres and runs tools in scope-specific sandboxes.
- Safe self-improvement means generating candidates and evaluating them outside
  the active system. The Darwin Gödel Machine evaluates code mutations on
  benchmarks with sandboxing and human oversight. It does not justify allowing
  a production harness to rewrite itself without review.

Therefore Tantular should optimize the **model and harness as a coupled
system**, while measuring their contributions separately.

## 2. Claims from the research pack

| Claim | Assessment | Architectural consequence |
|---|---|---|
| Harness changes can outperform a raw-model baseline | Supported for specific benchmark and harness combinations | Evaluate the deployed system, not only the checkpoint |
| Harness is always more important than model weights | Unsupported as a general rule | Use a factorial model x harness evaluation |
| Persistent REPL and external state help long-horizon tasks | Supported | Separate active context from durable state and execution |
| Recursive sub-agents are universally beneficial | Conditional | Give them budgets, ownership, lifecycle, and communication limits |
| Local systems can be roughly 800x cheaper | Supported as marginal API cost in the OpenJarvis experiment, not as a universal total-cost law | Record accuracy, latency, energy, and cost together |
| A production harness should modify its own code | Unsafe extrapolation | Mutate candidates only in an isolated workspace; promote after gates and human approval |
| 633 agents, 23 million tokens, and a 100% AVO result | Not verified in the primary sources reviewed | Do not use these numbers as acceptance targets |
| An LLM literally becomes a Von Neumann machine | Metaphor | Keep model, state, tools, and control plane as explicit components |

## 3. Design principle

Distillation starts only after answering:

> Is the observed product gap caused by model capacity, or by the harness that
> exposes and controls that capacity?

The existing pipeline evaluates `student before` versus `student after`. The
harness-aware design adds a controlled matrix:

| Arm | Model | Harness | Question |
|---|---|---|---|
| `student_current` | student | current production harness | What users receive now |
| `student_candidate` | same student | candidate harness | Can scaffolding solve the gap without weight changes |
| `teacher_current` | teacher | current production harness | Does a stronger model solve it under the same controls |
| `teacher_candidate` | teacher | candidate harness | Optional upper-bound and interaction measurement |

This separates:

- **Harness gain:** `student_candidate - student_current`
- **Residual model gap:** `teacher_current - student_candidate`
- **Interaction:** whether a candidate harness helps one model but hurts another

Teacher size is a hyperparameter, not an assumed direction. A larger teacher is
valuable only when its *student* later wins the held-out product gates.

## 4. Architecture

```text
Product failures and approved eval set
                    |
                    v
             Evidence registry
      prompt ids, split, source class, digests
                    |
                    v
     +-----------------------------------+
     | Factorial harness evaluation      |
     | student/current                   |
     | student/candidate                 |
     | teacher/current                   |
     | teacher/candidate (optional)      |
     +-----------------------------------+
                    |
                    v
        Attribution and decision gate
      harness sufficient? residual gap?
                    |
       +------------+-------------+
       |                          |
       v                          v
 Promote harness           Distillation candidate
 no weight update          (still not authorization)
                                  |
                                  v
                       Verified trajectory release
                 answers, tool calls, verifier feedback,
                    harness digest, model identity
                                  |
                                  v
                  Sequence / preference / compatible KD
                                  |
                                  v
                       Existing before/after gates
```

### 4.1 Model registry

The existing `configs/models/` continues to own checkpoint identity, revision,
tokenizer compatibility, architecture profile, and output-training licence.

### 4.2 Harness registry

`configs/harnesses/` owns everything surrounding the checkpoint:

- system and task prompt contracts;
- tool schemas and permissions;
- memory policy and retention;
- routing and retry policy;
- budget and termination rules;
- verifier and repair loop;
- execution isolation;
- candidate-mutation policy.

Every harness receives a canonical digest. Every future training trace should
carry both `model_identity` and `harness_identity`. A trace without the harness
digest cannot establish whether its quality came from the model or scaffolding.

### 4.3 State hierarchy

The source documents use inconsistent L1/L2/L3 labels. Tantular should use
names, not overloaded cache numbers:

1. **Active context** - tokens sent to the model.
2. **Ephemeral execution state** - variables and artifacts in the current
   sandbox or REPL.
3. **Durable task state** - append-only trajectory, checkpoints, and resumable
   task metadata.
4. **Approved product memory** - scoped, retained information with explicit
   provenance, permissions, and deletion policy.

Model weights are not memory in this operational taxonomy.

### 4.4 Tool and permission plane

Tool access is a capability grant:

- smallest usable tool surface;
- input/output schemas;
- source and trust labels;
- per-tool approval requirement;
- egress classification;
- idempotency or rollback contract;
- audit event for every state-changing call.

The model and its sandbox are not authorization authorities.

### 4.5 Learning plane

Use the cheapest reversible intervention first:

1. **Harness compilation** - prompts, routes, budgets, tools, examples.
2. **Trajectory distillation** - verified final answers and tool trajectories.
3. **Preference training** - only for sequence-level properties and
   judge-approved pairs.
4. **On-policy token-level KD** - only when the tokenizer compatibility key
   matches.
5. **Model architecture or pretraining changes** - outside this repository.

## 5. Harness-before-weights decision

Weight distillation becomes a candidate only when all conditions hold:

1. A real, approved capability gap exists in `student_current`.
2. `student_candidate` still fails after harness optimization.
3. `teacher_current` passes the same capability under the same harness.
4. The teacher advantage over the optimized student exceeds both the declared
   minimum effect and measurement noise floor.
5. The candidate harness preserves all product guardrails.
6. Teacher licence, corpus provenance, and existing model compatibility gates
   pass.

The result is called `weight_distillation_candidate`, never
`training_authorized`.

## 6. Safe self-improvement

The active product must never rewrite its own harness in place. A
self-improvement loop is an offline promotion system:

```text
failure traces
    -> candidate mutation in isolated workspace
    -> static checks
    -> unit and integration tests
    -> held-out evaluation
    -> regression and permission gates
    -> human review
    -> immutable release
```

Failed candidates remain evidence; they are not deployed. The mutation process
cannot modify its evaluator, held-out set, permission policy, or promotion
thresholds in the same change.

## 7. Implemented slice

This change adds:

- `configs/harnesses/harness.schema.md`
- `configs/harnesses/tantular-office-current.yaml`
- `configs/harnesses/tantular-office-candidate.yaml`
- `configs/experiments/harness-before-weights.yaml`
- `src/harness_distill.py`
- `tests/test_harness_distill.py`

The implementation:

- validates harness safety invariants;
- computes a canonical harness digest;
- emits the required factorial evaluation arms;
- evaluates a measurement file and distinguishes harness sufficiency from a
  residual model gap;
- audits JSONL traces for model and harness attribution;
- always emits `training_authorized: false`.

It performs no network request, model call, trace generation, or training.

## 8. How to run the four-arm bakeoff

**Procedure only. Nothing below is authorized by this document, and nothing in
this repository will execute it today.** Every command listed either reads local
files or refuses; the one step that would call a model is blocked, deliberately,
and section 8.2 says by what.

### 8.1 Precondition: a real, approved capability gap

The bakeoff exists to answer whether an improvement belongs to the model or to
the harness. It is not a warm-up exercise, and running it "to see" is how a
comparison acquires a conclusion before it has evidence.

So the entry condition is the same one `train/TRAINING_BLOCKED.md` states for
training: a REAL OBSERVED FAILURE in the product, written down, that a better
harness or a better student is a plausible fix for. "The teacher is bigger" and
"the harness has more tools" are not capability gaps.

### 8.2 What blocks it today

Both shipped harnesses declare `prompts.system.verified: false`, and
`src/generate.py --harness` refuses to generate against an unverified harness:

    HARNESS IDENTITY UNVERIFIED: harness 'tantular-office-current' prompt
    identity is unverified; run verify_harness_identity.py against a published,
    reproducible prompt registry before generation

That is not a missing feature. The add-in that owns the prompts is not published
to a stable ref — `docs/CI.md` records that its checkout is ahead of its remote
and its lockfile is tracked nowhere — so a prompt digest measured here could not
be reproduced by anyone else. An attributed corpus carrying an unreproducible
digest would look like evidence without being evidence.

**Unblocking is an upstream decision, not a code change**: publish the add-in and
its `package-lock.json`, then pin the prompts.

### 8.3 The sequence, once the precondition holds

    # 0. Pin the prompt identity of each harness under test. Reads the add-in's
    #    own prompt registry; refuses if node, the registry, or a hash is absent.
    ./.venv/bin/python src/verify_harness_identity.py tantular-office-current --write
    ./.venv/bin/python src/verify_harness_identity.py tantular-office-candidate --write

    # 1. Confirm the plan the experiment declares, and read its warnings.
    #    Emits nothing and authorizes nothing.
    ./.venv/bin/python src/harness_distill.py plan \
        configs/experiments/harness-before-weights.yaml

    # 2. FOUR SEPARATE GENERATION PASSES, one per arm. Separate, because two
    #    execution models or two harnesses in one file is the confound the
    #    design exists to remove — the tooling refuses to mix them.
    #      student x current    student x candidate
    #      teacher x current    teacher x candidate
    ./.venv/bin/python src/generate.py \
        --teacher <serving-config> --host <host> \
        --harness <harness-name> \
        --prompts <held-out-prompts>.jsonl \
        --out data/raw/<arm>.jsonl

    # 3. Declare each pass harness-aware. Never inferred: a pass that lost its
    #    attribution must not read as a valid legacy pass.
    ./.venv/bin/python src/pass_manifest.py data/raw/<arm> --harness-aware

    # 4. Gate each arm. The corpus gate reads the pass manifest's declaration,
    #    and refuses a corpus that claims attribution it does not carry.
    ./.venv/bin/python src/verify_corpus.py data/raw/<arm>/traces.r0.jsonl --gate

    # 5. One combined report per arm: teacher quantization, source_class, the
    #    FP8 gate, teacher licence freshness, and harness coverage together.
    ./.venv/bin/python src/distill_plan.py provenance-audit \
        data/raw/<arm>/traces.r0.jsonl --today <YYYY-MM-DD>

    # 6. Score the arms with the product's own gates, into a measurements JSON.
    ./.venv/bin/python src/run_gates.py run --stage before \
        --host <student-host> --teacher <serving-config> \
        --expect-model <model-id> --out data/gates/<arm>.json

    # 7. Evaluate the comparison. Reads the measurements; decides nothing about
    #    training.
    ./.venv/bin/python src/harness_distill.py evaluate \
        configs/experiments/harness-before-weights.yaml \
        <measurements>.json

### 8.4 What the result can and cannot say

`evaluate` reports whether the candidate harness beat the current one, on the
same model, against the declared guardrails. That is a statement about the
scaffolding.

It is NOT authorization to distil anything into weights. If the harness alone
closes the gap, the correct outcome is to ship the harness — which is the whole
point of testing the harness first. Weight distillation remains behind
`train/TRAINING_BLOCKED.md`, and every command above still reports
`training_authorized: false`.

## 8. Sources

- Prime Agent: A Self-Improving RLM Harness, arXiv:2608.23552
- OpenJarvis: Personal AI, On Personal Devices, arXiv:2605.17172
- Recursive Language Models, arXiv:2512.24601
- Darwin Gödel Machine, arXiv:2505.22954
- QM official repository and architecture documentation
- DSPy optimizer documentation

