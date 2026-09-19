# Authorization records

A record here is a **human decision**, written down and dated. One file per
authorization: `docs/authorizations/<subject>-<YYYY-MM-DD>.md`.

The pattern follows `docs/licences/`: the record is the artifact, the prose and
front matter are hashed together, and no tool in this repository decides the
question the record answers.

## Why the directory exists

The repository already refuses to act on undeclared permission. `train/
TRAINING_BLOCKED.md` blocks training, `license.output_training_permitted`
blocks distillation from a teacher, and `configs/experiments/*.yaml` carries
`training_authorized: false`. Those are gates on *specific* actions.

An authorization is the other half: a dated statement of what a person has
permitted, under what conditions, and what remains forbidden. Without it,
"the owner approved this" lives in a chat log, and a chat log is not evidence.

## What an authorization record does NOT do

- It does not open a gate. Every gate named in a record still runs and still
  has to pass on its own terms. An authorization permits *execution after* the
  gates pass; it is not a substitute for passing them.
- It does not set `training_authorized: true`. That value is derived from the
  harness-before-weights decision plus a signed `train/TRAINING_JUSTIFIED.md`,
  and no authorization short-circuits it.
- It does not grant spending or data egress. Both default to zero and denied.
  A limit exists only if it is written in the record as a number.
- It cannot be written by an agent on its own initiative. An agent may
  transcribe an authorization a person gave, attributed and dated; it may not
  author one, approve one, or countersign one.

## Required front matter

```yaml
authorization_id: rsi-program-2026-09-19
subject: <what is authorized>
granted_by: <person or role, as stated>
granted_at: <YYYY-MM-DD>
scope: <what execution this permits>
conditions: [<gate names that must pass first>]
training_authorized: false          # until the decision gate qualifies it
external_spending_limit: 0          # a number, in a named currency, or 0
data_egress: denied                 # denied | <explicit allowlist>
independent_reviewer: null          # a person, supplied at review time
reviewed_at: null
```

`training_authorized`, `external_spending_limit` and `data_egress` are the
three fields most likely to be widened by accident. They default to the
restrictive value and a record that omits them is read as if it carried the
default, never as unrestricted.

`independent_reviewer` and `reviewed_at` are left null by the transcriber and
filled by the reviewer. An agent that filled them would be manufacturing the
review the field exists to record.
