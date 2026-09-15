---
schema_version: 1
registry_model: example-teacher-7b
model_id: example-org/Example-Teacher-7B
revision: 0123456789abcdef0123456789abcdef01234567
reviewed_at: 2026-01-15
reviewed_by: A. Reviewer <reviewer@example.com>
sources:
  - path: LICENSE
    sha256: e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
  - path: USAGE_POLICY.md
    sha256: da39a3ee5e6b4b0d3255bfef95601890afd80709da39a3ee5e6b4b0d3255bfef
determination:
  output_training_permitted: false
  rationale: >-
    Fictional example. The upstream licence is silent on model outputs and the
    usage policy does not address training, so nothing in the reviewed sources
    grants the right this field would assert.
---

# Licence review — Example Teacher 7B (TEMPLATE, not a real determination)

Copy this file to `docs/licences/<registry-name>.md` and replace every value.
It is deliberately fictional: `example-teacher-7b` matches no registry entry, so
the verifier refuses it against any real model. Nothing here has been reviewed
by anyone.

## What was examined

Name each document, where it came from, and the commit it was read at. The
`sources` list above must contain every document the determination rests on and
nothing else — a hash that did not inform the decision implies a review that did
not happen.

## Reasoning

State why the sources support the determination. Say what the licence covers and
what it is silent about. If the conclusion rests on silence, say so explicitly:
absence of a prohibition is not a grant, and a later reader needs to know which
one this was.

Both answers are legitimate outcomes. Recording `false` is a finding, not a
failure, and it verifies exactly as readily as `true`.

## Scope and limits

Note anything the review did *not* settle — downstream redistribution, trademark,
attribution obligations, or clauses that constrain how outputs may be presented.
The registry gate reads one boolean; this section is where everything the boolean
cannot carry is written down.
