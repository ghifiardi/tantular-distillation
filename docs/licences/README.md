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

| registry `evidence_sha256` | state | behaviour | exit |
|---|---|---|---|
| equals this record's digest | `MATCHED` | nothing to do | 0 |
| `LICENSE_EVIDENCE_DIGEST_<ID>` | `UNRECORDED` | `--write` fills it | 0 with `--write`, 1 without |
| a **different** valid digest | `SUPERSEDED` | refuses, in both modes | 2 |
| anything else, missing, or not exactly one field | `UNREADABLE` | refuses | 2 |

`UNRECORDED` is deliberately narrow: **only** the recognised placeholder form is
writable. Arbitrary text, a misspelt `LICENSE_EVIDNCE_DIGEST_*`, a truncated
digest or an empty value are `UNREADABLE`. A field whose meaning is unknown must
not be overwritten on the assumption that it meant nothing.

Only `MATCHED` is success. A report-only run against an unrecorded registry
exits **1**, not 0 — the licence gate still refuses that teacher, and a 0 would
read as "verified" to anything scripting this. Exit 1 is the same signal
`src/verify_model_identity.py` gives for unfilled placeholders.

`SUPERSEDED` is the state the tool exists for. A valid digest on file that no
longer matches the record means the reviewed document changed *after* it was
pinned. That mismatch is the only signal that the decision on file is no longer
the decision that was reviewed, so re-pinning it automatically would make a
tamper-evidence tool erase the evidence of tampering.

A refused run writes nothing. The registry file is byte-identical afterwards.

### Re-reviewing a record that has already been pinned

Two commits, in this order:

1. **Invalidate.** Commit the changed record *and* reset
   `license.evidence_sha256` to the placeholder, together. The repository is now
   in a state where the licence gate refuses this teacher, and the human review
   happens against that invalidated state.
2. **Re-pin.** Run the verifier with `--write` and commit the newly measured
   digest.

Doing both in one commit is a mistake, and the reason is Git rather than the
tool: an edit of `old digest -> placeholder -> new digest` made before
committing leaves a final diff reading `old digest -> new digest`, and the
invalidation never existed as far as history is concerned. The two-commit
sequence is what makes "this pin was deliberately withdrawn and re-established"
reviewable.

The verifier cannot enforce this — it sees a working tree, not a commit graph.
It enforces only that the transition must pass through the placeholder at all,
which is what stops a changed record from re-pinning itself silently.

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
7. If the record ever changes afterwards, the verifier refuses until the
   recorded digest is cleared — see the two-commit re-review sequence above.

One human-reviewed record now exists for `muse-glimmer-30b`, but its registry
entry still carries the recognised placeholder until the separate verifier
commit pins the whole-file digest. The other two shipped entries have no
records. All three remain refused by the licence gate in this intermediate
state.
