# Harness registry schema

Harnesses are versioned independently from models. The canonical digest is
computed over the whole YAML object except the optional `digest` field.

```yaml
schema_version: 1
name: tantular-office-current
status: current                 # current | candidate | retired

model_contract:
  registry_model: qwen35-9b-instruct
  prompt_format: chat

prompts:
  system:
    path: ../tantular_office_addin/src/...
    sha256: null

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

