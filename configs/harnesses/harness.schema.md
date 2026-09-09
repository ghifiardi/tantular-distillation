# Harness registry schema

Harnesses are versioned independently from models. The canonical digest is
computed over the whole YAML object except the optional `digest` field.

```yaml
schema_version: 1
name: tantular-office-current
status: current                 # current | candidate | retired

model_contract:
  protocol: openai_chat
  # A harness is model-COMPATIBLE, not model-bound: the four-arm comparison
  # runs ONE harness against the student and the teacher. Which model actually
  # ran is recorded per trace, in harness_provenance.execution_model_registry.
  compatible_registry_models:
    - qwen35-9b-instruct
    - muse-glimmer-30b

prompts:
  system:
    source: prompt_registry      # prompt_registry | path
    path: ../tantular_office_addin/src/promptRegistry.js
    sha256: null
    verified: false

tools:
  allow: [office_edit]
  state_change_requires_approval: true

memory:
  active_context: bounded
  ephemeral_execution: sandbox
  durable_task_state: append_only
  product_memory: disabled

execution:
  isolation: process
  network: denied_by_default
  max_steps: 8
  max_wall_seconds: 300

verification:
  before_action: [schema]
  after_action: [edit_contract]
  repair_attempts: 0

mutation:
  production_self_modify: false
  candidate_workspace_only: true
  evaluator_mutation_allowed: false
  human_approval_required: true
```

## Hard gates

- `production_self_modify` must be `false`.
- `candidate_workspace_only` and `human_approval_required` must be `true`.
- `evaluator_mutation_allowed` must be `false`.
- Network policy must be `denied_by_default` or an explicit allowlist.
- State-changing tools require approval.
- Execution budgets must be positive and bounded.



## `model_contract`

`compatible_registry_models` lists every `configs/models/` entry this harness
may run against. It is part of the digested definition, so widening it produces
a different harness — which is the honest reading, since a harness qualified
against one model is not automatically qualified against another.

`harness_provenance(spec, execution_model_registry=...)` refuses a model that is
not on the list. Attribution for an undeclared pairing would be a guess, and the
whole point of the block is that it is not.

## `prompts.system.source`

`prompt_registry` (preferred for the Office add-in) resolves the identity through
the add-in's own `promptRegistry.js`, which enumerates every production prompt
and owns each one's content hash. The digest is canonical JSON over the sorted
`{id, contentHash}` pairs.

The alternative, `path`, digests a file or a directory tree. It is the wrong
answer for the add-in: a tree digest changes on any unrelated JavaScript edit
while the prompts are identical, and does not change when a prompt moves between
modules. It answers "did any source change?", not "did the prompts change?".

Either way `src/verify_harness_identity.py` is the only thing that may set
`sha256` and `verified: true`, and it fails closed rather than guessing — on a
missing registry, a missing `node`, a missing export, an empty prompt list, an
entry without a hash, or an empty directory.
