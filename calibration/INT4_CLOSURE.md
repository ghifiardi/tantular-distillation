# int4 calibration — closed, with limited claims

Closed 2026-08-17 by the corpus owner. Supersedes nothing; it sits alongside
`INT4_WAIVER.md`, which remains in force.

## What may be claimed

- Every metric **that has coverage** shows an across-pass delta of `0.0`.
- Table fidelity measures `1.0` in both passes.
- Prompt-set-v2 answer-text variation is `6/78`, including one prompt whose
  answers are entirely different between passes (`prose:umum::0005`).

## What may not be claimed

- **Not** evidence of single-session reproducibility. Pass B was resumed across
  a session boundary at trace 80. The correct label is
  **across-pass variance with resumed session boundary**.
- The v1 and v2 floors are **not** a trend and must not be compared as one.
  They use different prompt sets and have different interruption histories.
- `data/expanded/` remains `synthetic_candidate`. **Not training-ready.**

A single-session pass B is *not* scheduled. It would only be needed to claim
reproducibility without a session qualifier, and the qualified claim above is
the one being made.

## Where the numbers live

| Artifact | Prompt set | Label |
|---|---|---|
| `noise_floor.v2.ai19-ollama.json` | v1 | across-run floor, 21/26 strata |
| `noise_floor.promptset-v2.ai19-ollama.json` | v2 | across-pass variance with resumed session boundary |

Both filenames carry a `v2` that means **source pack**, not prompt set. Each
file states which prompt set produced it.

## FP8 is still unmet

Unchanged by any of this. `verify_corpus.py --gate` still fails and still exits
1. No FP8 claim may be made; the int4 waiver is what permits the work above.

## FP8 arm: FROZEN 2026-08-18

Two rentals produced no treatment arm (~$3.35). The int4 waiver therefore stands
unchanged, and its central admission remains literally true: the magnitude of
quantization loss is unmeasured and no number in this repository bounds it.

The infrastructure is solved — driver gate, clean-venv stack check, and the two
vLLM protocol fixes are committed and verified. The blocker is model support:
vLLM has no native implementation for `TransformersMultiModalForCausalLM` and
generates only control tokens.

No further GPU rental until a free or local runtime is shown to produce a valid
answer channel from one prompt. See `FP8_ARM_RUNBOOK.md`.

## Next, and it is not calibration

Corpus quality, before training. The measurement side is now sound enough that
further calibration effort would be measuring a corpus whose material is the
weaker half. See `SOURCE_COVERAGE.md` for the specific gap.
