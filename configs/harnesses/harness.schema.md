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

trace_generation:
  supported_by_generate_py: false   # see below

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
and returns each one's text. The digest is sha256 over canonical JSON of the
sorted `{id, sha256}` pairs, where each `sha256` is computed HERE over the
prompt text.

The registry also publishes a `contentHash`, and it is recorded alongside for
cross-checking — but it is deliberately not the identity. That value is a djb2
32-bit hash (about eight hex characters), which is right for the add-in's own
cache-busting and wrong as a commitment: it is short enough to collide and is
not collision-resistant by construction. Inheriting it would make the harness
prompt identity only as strong as that.

The alternative, `path`, digests a file or a directory tree. It is the wrong
answer for the add-in: a tree digest changes on any unrelated JavaScript edit
while the prompts are identical, and does not change when a prompt moves between
modules. It answers "did any source change?", not "did the prompts change?".

Either way `src/verify_harness_identity.py` is the only thing that may set
`sha256` and `verified: true`, and it fails closed rather than guessing — on a
missing registry, a missing `node`, a missing export, an empty prompt list, an
entry without a hash, or an empty directory.


## `trace_generation.supported_by_generate_py`

Defaults to false when absent, and `src/generate.py --harness` refuses any
harness that has not opted in.

`generate.py` sends a prompt through an ordinary chat client. It supplies no
tools, runs no `before_action` or `after_action` verifier, executes no repair
loop, and enforces neither the memory nor the approval policy. Stamping a full
product harness onto a trace produced that way would assert that all of that
ran — which is exactly the attribution ambiguity `harness_provenance` exists to
remove, reintroduced one layer up.

A harness may only opt in if it claims nothing `generate.py` cannot honour: no
tools, no verifiers, no repair loop. In practice that means a PROMPT-ONLY
harness. When it does opt in, `generate.py` additionally requires every prompt's
`system` message to hash to the harness's pinned `prompts.system.sha256` — the
harness prompt is what the harness IS, and sending a different one would
attribute a run that did not happen.

Attributing a full product harness needs a purpose-built runner that returns an
execution receipt: which prompt id and content hash were used, which tools were
offered and called, which verifiers ran and what they returned. Until that
exists, the honest answer is that such a corpus cannot be produced.
