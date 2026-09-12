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
    repository:                  # optional: pin a published snapshot
      url: https://github.com/ghifiardi/LLM-Indonesia.git
      ref: tantular-office-addin-harness-baseline-2026-09-11
      peeled_commit: 3e14d25468ab0cd793ba8dc48cf5f755796c94e2
    path: tantular_office_addin/src/promptRegistry.js
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
sorted `{id, content_sha256, registry_content_hash}` rows, where
`content_sha256` is computed HERE over the prompt text.

Each id travels in one object with its own content hash, so the aggregate BINDS
id to content: swapping text between two prompt ids changes the identity. A
digest over the ids and a separate digest over the contents would both be
unchanged by that swap.

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

## `prompts.system.repository` — pinning the snapshot, not a path

Without it, `path` names a place on whoever's machine ran the verifier. A
sibling checkout can be any revision, or absent, so the measured digest was not
an identity anyone else could reproduce. The optional `repository` block pins
the source itself:

- `url` — the repository the prompts come from. Only `.git` and trailing-slash
  spelling is normalised when comparing against the checkout's `origin`; a
  different repository is refused.
- `ref` — the tag naming the snapshot.
- `peeled_commit` — the commit that tag must resolve to. Complete and
  lowercase: 40 hex characters for a SHA-1 repository, 64 for SHA-256. Never an
  abbreviation, because two commits can share a short prefix. This is a Git
  object id, a different type from the SHA-256 prompt digest that sits beside
  it.
- `path` becomes relative to that repository. An absolute or escaping path is
  refused: it would let the pin name one snapshot and measure another.

`ref` and `peeled_commit` are both load-bearing. A tag name alone is not an
identity — re-pointing a tag must not quietly become a new prompt identity — so
the verifier requires the tag to peel to exactly the pinned commit.

The whole block is part of the harness definition and therefore part of its
canonical digest. A trace binds not only to prompt bytes but to the snapshot
those bytes were measured from.

### The trust model: `--source-checkout`

Acquiring the source and verifying it are separate operations, and only the
first touches the network:

    git clone --depth 1 --branch <ref> <url> /tmp/snapshot
    ./.venv/bin/python src/verify_harness_identity.py <harness> \
        --source-checkout /tmp/snapshot

The verifier never fetches. A verifier that could fetch could be talked into
auditing a snapshot other than the one it was asked about. It reads the
checkout you hand it and refuses unless that checkout *is* the pinned snapshot:
a real Git worktree, `origin` describing the pinned url, `HEAD` equal to
`peeled_commit`, `ref` peeling to the same commit, and no tracked modifications.
Untracked files are tolerated — `node_modules` and build output are untracked by
nature, and refusing them would make a real checkout unverifiable for no
integrity gain.

`validate_harness` checks the shape of this block offline, with no Git and no
network, so CI can assert the pin without a checkout. A present-but-partial
`repository` block is malformed and raises; it never falls back to local-path
semantics, because that silent fallback is how the unreproducible machine-local
path got here in the first place.

### One recorded discrepancy in the published tag

The add-in baseline tag `tantular-office-addin-harness-baseline-2026-09-11`
records this prompt aggregate in its annotation:

    5086061098575be10267c1b4a0676f75ad3e65723b497ca3d06eee6b4aa26794

That is **not** this repository's prompt identity. It was produced by a one-off
probe written during the publication milestone, which serialised the rows as
newline-delimited JSON objects in declaration-key order. The reviewed
implementation here frames the same rows as one compact JSON array with sorted
keys, and measures:

    1e9e96aac3a2012493a7a149c7acb7cf02dd02246b3626739c2fac2f73df638e

The nine prompt ids, every `content_sha256` and every `registry_content_hash`
are identical under both. Only the final framing differs, which is precisely why
the framing is part of the identity and is now asserted byte-for-byte in
`tests/test_verify_harness_identity.py`. The tag is correct and was not moved:
its load-bearing content is the source pin, and the annotated aggregate is
informational evidence of how that probe framed its rows.


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
