# Licence evidence records

A record here is a **human decision**, written down and hashed. One file per
registry entry, named for it: `docs/licences/<registry-name>.md`.

## Why the field exists

`configs/models/*.yaml` declares `license.output_training_permitted`, and the
planner treats it as a gate. Nothing upstream states that value. Apache 2.0
governs the software and weights and is silent on model outputs; a usage policy
typically prohibits particular *uses* without addressing training at all. So the
field is a judgement someone made, and `license.evidence_sha256` is the digest of
the document in which they made it.

Before these records existed, the registry shipped `evidence_sha256:
LICENSE_EVIDENCE_DIGEST_MUSE_GLIMMER_30B` — a placeholder that validated as
evidence because the check only asked whether the string was non-empty. Every
teacher reported `license_ready: true` on a review nobody had done.

## What the verifier does, and what it refuses to do

`src/verify_license_evidence.py` validates the record's shape, binds it to the
registry entry, computes the digest of the **whole committed file**, and — with
`--write` — copies that digest into the registry.

It does not decide the licence question and contains no rule that could. A record
determining `false` verifies exactly as readily as one determining `true`. The
tool's entire job is to make sure that what the registry points at is a complete,
attributable, unmodified record of a decision a person actually made.

It also cannot confirm that the source hashes belong to real documents — it is
offline, and the upstream documents are not committed here. It checks their shape
and that they are bound to a `model_id` and `revision`. Confirming the hashes is
the reviewer's job, once, at review time.

## The digest covers the whole file

Front matter and prose together, byte for byte.

Hashing only the front matter would let the reasoning be rewritten while the
digest still matched. Hashing the upstream `LICENSE` would bind a document that
does not contain the determination — and would be actively misleading for the
Muse Glimmer teacher, whose DFlash drafter publishes a `LICENSE` and
`USAGE_POLICY.md` **byte-identical** to the parent's. Source hashes cannot tell
those two repositories apart; only `model_id` and `revision` can, which is why
the verifier treats a binding mismatch as fatal.

Any edit — a source added, a rationale reworded, a different reviewer — moves the
digest and invalidates the registry entry until it is reviewed again.

## `--write` never replaces a valid digest

Four states, decided before anything is written:

| registry `evidence_sha256` | state | behaviour |
|---|---|---|
| equals this record's digest | `MATCHED` | no-op |
| placeholder or empty | `UNRECORDED` | `--write` fills it |
| a **different** valid digest | `SUPERSEDED` | refuses, in both modes |
| missing, non-string, or not exactly one field | `UNREADABLE` | refuses |

`SUPERSEDED` is the state the tool exists for. A valid digest on file that no
longer matches the record means the reviewed document changed *after* it was
pinned. That mismatch is the only signal that the decision on file is no longer
the decision that was reviewed, so re-pinning it automatically would make a
tamper-evidence tool erase the evidence of tampering.

Re-review is still possible, and deliberately manual: re-review the record,
clear `license.evidence_sha256` in the same commit, then run `--write`. The
clearing is visible in the diff, which is the point.

A refused run writes nothing. The registry file is byte-identical afterwards.

## Writing one

1. Copy `TEMPLATE.md` to `docs/licences/<registry-name>.md`.
2. Read the actual documents at the **pinned revision**, not at `main`. Upstream
   licence links usually point at `main`, and policies say so themselves: "the
   most recent copy of this policy can be found at…". That mutability is why
   `recheck_max_age_days` exists.
3. Record every document the decision rests on, with its SHA-256.
4. Write the determination and the reasoning. If it rests on silence rather than
   on a grant, say so in those words.
5. Sign it as a person. A model, agent, or automation may not be the reviewer,
   and the verifier refuses a machine byline.
6. Commit the record, align the registry's `output_training_permitted` with it,
   then run the verifier with `--write`.
7. If the record ever changes afterwards, the verifier refuses until someone
   clears the recorded digest as part of re-reviewing it.

No record has been written yet. All three shipped registry entries still carry
placeholders and are correctly refused by the licence gate.
