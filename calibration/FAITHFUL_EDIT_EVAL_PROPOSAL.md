# Faithful constrained editing — eval set proposal

**Status: PROPOSAL. No items authored yet. Approve the design first.**
Approved 2026-08-21: objective and 100-item size. Everything below is the
mechanism, and is what needs review.

## Objective

> Measure **faithful constrained editing**: can the model change an Indonesian
> document as an Office instruction asks, while preserving the figures, names,
> terms, structure and facts it must not touch, and without adding information
> that was not there?

This replaces "more voice evaluation" deliberately. The base model already meets
the voice gate at 0.9500, so another voice set would measure a capability that
is not in question.

## Why this can have headroom when the current gates do not

The existing edit gate asks for a valid contract on short, unambiguous
documents. It scores 19/20. The properties below are the ones that fail *while*
the contract stays valid — a model can emit perfectly parseable JSON that edits
the wrong occurrence, silently rounds a figure, or invents a date. Those are
product defects and the current gates cannot see them.

**This is not "the same test, made harder to force a failure."** It is a
different property. If the base model turns out to be good at it too, that is
the answer and the set stands as a no-regression gate — the thresholds are not
being tuned until the base has been measured.

## Six properties, each scored mechanically

No model judges another model. Every check is deterministic, so a failure can be
pointed at.

| # | property | how it is scored |
|---|---|---|
| 1 | the edit lands where asked | the add-in's REAL `resolveEdits` + `applyEditsToText`; the target span must differ from the source and every declared `must_not_change` span must be byte-identical |
| 2 | protected tokens survive | item declares `must_preserve` (figures, units, names, IDs); each must appear in the output with the SAME count as in the source |
| 3 | structure obeys the instruction | reuses the `calibrate.py` check vocabulary already in the repo: `bullets`, `paragraphs`, `max_paragraphs`, `table_rows`, `absent`, `must_not_contain` |
| 4 | no new facts | no numeric token in the output that is absent from the source, and no capitalised token outside a declared allow-list. A proxy, but a strict and checkable one |
| 5 | the edit contract is valid | `contract_ok` from `scripts/check_edit_contract.mjs` — parsed AND located AND applied |
| 6 | professional Indonesian | the existing rubric v2 dimensions applied to the replacement text only |

An item passes only if **all six** hold, matching how the voice rubric already
works. Per-property rates are reported because they say WHERE a regression
happened, which one number cannot.

## Composition — 100 items, 1 point each

Chosen so each stratum targets a distinct way faithfulness fails:

| stratum | items | what it stresses |
|---|---|---|
| repeated-phrase disambiguation | 15 | `occurrence` and `before`/`after` anchors; the classic wrong-edit |
| figure-dense documents | 15 | preserving amounts, dates, percentages, units under rewrite |
| multi-edit (3–8 edits per item) | 15 | all edits applied, none colliding |
| structure-constrained rewrite | 15 | "exactly five bullets", "one paragraph", table shape |
| terminology-constrained | 15 | required source terms kept while register stays professional |
| absent-information cases | 10 | the instruction asks for something not in the document; the correct answer is to say so, NOT to invent it |
| long documents (2–4k chars) | 15 | locating a small edit in a large context |

The absent-information stratum is the one I would most expect to separate a
trained model from the base, and the one that matters most for the product: a
fabricated figure in an Office document is worse than a clumsy sentence.

## Independence

Every item authored new. Checked, not asserted:

- exact digest against `data/promoted/{train,eval}.jsonl`, `prompts/voice_eval.v1.jsonl`,
  `prompts/edit_contract_eval.v1.jsonl`, and the 60 baseline completions;
- shingle-Jaccard near-duplicate scan using the existing `src/dedup.py`, against
  all of the above;
- no document text reused from the 260-trace candidate corpus.

## Pilot before the other 90

**Author 10 items first**, two or three from each of the larger strata, and
measure the base model on them over the same endpoint and the same `plain-chat`
protocol with `enable_thinking: false`. Then review:

1. do the six checks behave — does each one fire on a hand-made bad answer;
2. is there headroom, or is the base already near-perfect here too;
3. are the items fair — a human should agree the "correct" answer is correct.

Authoring 100 items and then discovering the set is saturated would repeat
exactly the mistake this eval exists to correct. The pilot costs one short
endpoint session.

## Thresholds

**None proposed.** Set after the pilot and the full base measurement, and never
by reading the base's score and adding a margin. The comparison that matters is
base versus adapter on identical prompts, endpoint and protocol.

## Open questions for review

1. Is the absent-information stratum weighted right at 10? It is the strongest
   product signal and the hardest to author fairly.
2. Property 4 flags capitalised tokens outside an allow-list. Indonesian
   sentence-initial words and month names will need care; the allow-list is per
   item, which is authoring effort.
3. Should the 15 long-document items reuse document *shapes* from the corpus
   (memo, report, minutes) with entirely new content, or use shapes not present
   in training at all? Reusing shapes measures the target capability; avoiding
   them measures generalisation too. I lean to reusing shapes.
