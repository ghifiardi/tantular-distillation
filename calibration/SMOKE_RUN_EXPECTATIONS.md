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
