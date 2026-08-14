# Pipeline smoke run — predictions recorded BEFORE execution

Written before the run so predicted artifacts cannot later be mistaken for
findings about the teacher.

**Corpus role: `pipeline_smoke`. Not a training corpus.** It validates
inventory → seeds → ai19 → gate end to end. Nothing about teacher quality is
being measured.

## Source diversity in the seed set

| Axis | Prompts | Distinct source bodies |
|---|---|---|
| router | 8 | **1** |
| document | 5 | **2** |
| edit | 6 | 6 |
| prose | 7 | 7 |

## Predicted artifacts — expected, not discoveries

**`router_label_accuracy` ≈ 0.125 (1/8), and the metric is meaningless.** All
eight router prompts are byte-identical while carrying eight different expected
labels. No model can exceed chance; the number measures the source pack, not
the teacher. This is not a router regression and must not be reported as one.

**Document-stratum answers will be near-duplicates.** Four document kinds share
one 460-byte generic memo, so `document:spreadsheet-text` is backed by prose
with no table and `document:slide-text` by prose with no bullets. Those strata
cannot demonstrate the behaviour they name.

**`constraint_satisfaction` is weakly informative.** Most smoke prompts carry
no machine-checkable constraint, since the generic sources do not support
task-specific ones.

## What a PASS means here

Only that the path works: inventory gates correctly, seeds embed their sources,
ai19 generates under the signed waiver, provenance is captured, and the corpus
gate evaluates. It says nothing about corpus quality, and nothing about the
student that would result.

## Required before any training run (option 2)

- a real table for `document:spreadsheet-text`
- bullet/slide structure for `document:slide-text`
- distinct email, memo, report and prose artifacts
- task-specific source text per edit family
- **router examples that actually differ per route** — the most broken axis

---

## Result (2026-08-14) — pipeline PASS, gate correctly FAILED

```
[PASS] traces generated   26/26
[PASS] malformed / empty  0 malformed, 0 empty
[PASS] source embedded    26/26
[PASS] provenance         26/26 muse-glimmer:30b / int4_ollama / ai19-ollama
```

Gate exits 1 with three violations, all correct:

1. 26 kinds missing traces in an assigned split — a property of one family per
   kind, not a defect
2. `prose:umum::0000` present with a single trace — the signed waiver's
   volatile-family condition firing as designed
3. 26 quantized traces — the FP8 gate staying UNMET, waiver named as
   authorization rather than converting it to a pass

### Predictions scored

**Correct:** router label accuracy exactly 1/8. Eight byte-identical prompts,
eight different labels. Source-pack artifact, not a teacher result.

**Wrong:** near-duplicate document answers were predicted; 5 distinct of 5 were
produced. The four kinds share a source, but their instructions differ, so the
answers differ. The underlying defect is unchanged —
`document:spreadsheet-text` is backed by prose containing no table and cannot
demonstrate spreadsheet behaviour — it simply does not surface as duplicate
output. Prediction recorded as wrong rather than quietly reinterpreted.

### Standing

Synthetic pipeline smoke corpus; not suitable as a training corpus. Excluded
from training. Source diversification (option 2) remains required first.
