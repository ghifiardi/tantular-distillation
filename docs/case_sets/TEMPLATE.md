# Case-set approval record — TEMPLATE

FIXTURE DATA. This file binds to no shipped case set and must never be mistaken
for a determination. `src/verify_case_set_approval.py` refuses it by name.

Copy it to `docs/case_sets/<case-set-name>.md`, have a HUMAN complete it, and
only then may the registry carry that set's approval digest. An agent may
prepare every field below except the reviewer identity, the review date and the
confirmations; filling those would make the approval a thing the agent granted
itself, which is the one property this record exists to prevent.

```yaml
case_set: example-office-v1
case_set_sha256: 0000000000000000000000000000000000000000000000000000000000000000
cases: 0
split: held_out
source_class: real_office

reviewed_by: REPLACE_WITH_REVIEWER_NAME
reviewer_role: REPLACE_WITH_REVIEWER_ROLE
reviewed_at: REPLACE_WITH_REVIEW_DATE

confirmations:
  privacy_redaction_reviewed: false
  product_representativeness_reviewed: false
  leakage_reviewed: false
  scorer_inputs_complete: false
  expected_outcomes_correct: false
  both_models_and_harnesses_executable: false
  at_least_320_independent_cases: false
  no_training_or_calibration_overlap: false

notes: >-
  What the reviewer actually checked, in their own words.
```

## What each confirmation means

| confirmation | the reviewer is asserting |
|---|---|
| `privacy_redaction_reviewed` | no real customer content; fixtures are synthetic or redacted |
| `product_representativeness_reviewed` | these cases resemble work users actually do, not ones chosen because they pass |
| `leakage_reviewed` | no case overlaps training, calibration, prompt examples or test fixtures |
| `scorer_inputs_complete` | every case declares what its scorers need — `must_preserve`, `must_not_change`, `allowed_new_facts`, a target assertion |
| `expected_outcomes_correct` | the declared right answer IS right, checked case by case |
| `both_models_and_harnesses_executable` | every case runs in all four arms; a model-specific case cannot enter a shared factorial set |
| `at_least_320_independent_cases` | the qualification floor; unique cases, not repetitions |
| `no_training_or_calibration_overlap` | the held-out claim is evidence, not an assertion in a file |

## Why the digest binds

`case_set_sha256` is the canonical digest of the exact bytes reviewed. Change
any case and the digest moves, so the approval stops matching and the set is
unapproved again — automatically, with nobody having to remember. This is the
same construction `docs/licences/` uses for licence determinations, and for the
same reason: a review of something is only evidence about that something.

The record lives OUTSIDE the case set deliberately. Putting the reviewer inside
the digested set would mean approving it changes its digest, so the thing
approved would never be the thing whose digest was reviewed.
