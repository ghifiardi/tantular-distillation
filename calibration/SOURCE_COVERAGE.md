# Source coverage: per-family artifacts do not exist yet

Measured 2026-08-17 against `prompts/expanded.v2.jsonl`.

## The inventory overstates readiness

`src/inventory.py status` reports:

```
260 kinds   source_class: {'synthetic': 260}
  BLOCKED on source/approval : 0
per-family: 260/260 families ready  (backed by: {'family': 260})
```

That reads as one artifact per family. It is not. Every family *resolves* to an
artifact, but not to a **distinct** one:

| | count |
|---|---|
| families | 260 |
| distinct source documents | **78** |
| distinct prompts | **78** |
| families per document | mean 3.33, max 8 |

All 26 strata have exactly 3 source documents — one per split — shared across
their 10 families. The inventory counts resolution, not distinctness, so
`260/260 ready` and `78 documents` are both true and only the second is
informative.

Distribution:

| documents | shared by |
|---:|---|
| 29 | 1 family |
| 14 | 2 families |
| 5 | 3 families |
| 4 | 4 families |
| 6 | 5 families |
| 5 | 6 families |
| 8 | 7 families |
| 7 | 8 families |

## What is and is not wrong with that

**No split leakage.** 0 documents span more than one split — the source pack is
organised per (kind, split), so a train document never backs an eval family.
The invariant `splits.py` exists to protect holds.

**But a training set inherits each document ~3.33x with an identical prompt.**
Seven documents are repeated eight times. This is the "3.3x repetition
inflation" that `cross_pass_report.py`'s FAMILY-WEIGHTED view has been flagging
all along; the UNIQUE-PROMPT view (n=78) is the honest denominator, and it is
the primary view for that reason.

It also caps the corpus: 260 families carry 78 documents' worth of information,
so eval and challenge are narrower than their family counts suggest — 54 eval
families over roughly 26 distinct documents.

## Two candidate next steps

1. **Per-family artifacts.** Author distinct source material per family rather
   than per (kind, split): 260 documents instead of 78. `author_sources.py`
   already composes from structured per-scenario data, so this is a matter of
   more scenario worlds, not a new mechanism. Keeps everything synthetic.

2. **Approved real sources.** Higher value and gated on people, not code. Every
   stratum is currently `source_class: synthetic`, and the inventory's own note
   is explicit: synthetic material supports pipeline generation and behavioural
   training but **no claim about performance on real Office documents**. Real
   material needs approval, redaction, and an egress decision per host before
   any of it can be generated against.

These are not alternatives so much as different claims. (1) buys corpus breadth
and removes the repetition inflation. (2) is what any statement about real
Office documents requires, and no amount of (1) substitutes for it.

## Status

`data/expanded/` and the v2 passes remain `synthetic_candidate`. Not
training-ready. Promotion has not been approved.
