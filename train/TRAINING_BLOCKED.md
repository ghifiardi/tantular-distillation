# v1 training: infrastructure QUALIFIED, run NOT started

**Current state: 2026-08-21. Repository at `6b07cf8`.**

Every technical blocker to starting v1 has been cleared and verified on
hardware. **No training has been performed and no adapter has been produced.**
What remains is a budget/product decision, not an engineering one.

## What is qualified

| | evidence |
|---|---|
| training path executes | Pod A smoke: NF4 load, LoRA attach, real TRL `trainer.train()`, save, reload |
| expanded LoRA targets attach | **80,216,064** trainable, exactly the projected figure (was 58,195,968) |
| adapter is APPLIED when served | Pod B, vLLM 0.27.1: converted adapter's output differs from the base; binding check exit 0 |
| schema-v2 freeze verified | config, corpus, promotion bytes, gate exit 1, signed waiver |
| held-out sets held out | both model-dependent eval sets |
| host guards | training host accepted; ai19 refused |

The freeze in `train/RUN_MANIFEST.v1.json` is CURRENT and accepted by the
trainer. It was regenerated against the expanded-target config.

## The remaining step

**A valid `before` baseline, which does not yet exist.**

v1 attempt 1 (2026-08-21) was approved and run. It never trained: six aborts in
the `before` gates, all from one omission — the gates had never generated
against a live model, because every test drives them from `--traces` fixtures.
The last of those produced a baseline of exactly 0.0000 on both model-dependent
gates by scoring the model's English reasoning as its answer. All six are fixed;
see `calibration/SMOKE_RESULT.md`.

Before `--confirm-run-v1` is used again:

1. run `src/run_gates.py run --stage before` against a served base model;
2. confirm `thinking_disabled: true` on the traces, answers rather than
   reasoning in the completions, and all 20 edit + 40 voice outputs present;
3. get fresh explicit approval.

A baseline of zero is not a floor to measure from — it makes any output an
improvement and satisfies "no regression" trivially.

## Qualification detail

See `calibration/SMOKE_RESULT.md` for the full record, including the four
defects the smoke rentals found — one of which would have produced a complete,
plausible, and entirely false adapter evaluation.

## What the freeze enforces

- exact `train/qlora_9b.yaml` bytes;
- exact 260-trace candidate corpus and provenance;
- exact `RUN_MANIFEST.v1-mechanical.json` bytes;
- exact promoted train/eval bytes and counts;
- one split fingerprint across corpus and promotion;
- the real `verify_corpus.py --gate` result and exact violation list;
- the accepted `calibration/INT4_WAIVER.md` digest.

The expected corpus gate is still **FAILED** because the teacher is int4. The
waiver authorizes proceeding despite the sole quantized-teacher violation; it
does not turn that result into a pass and cannot authorize unrelated failures.

## Ready

- `data/promoted/train.jsonl` — 136 traces
- `data/promoted/eval.jsonl` — 47 traces
- challenge — 27 held out of both
- 40-item Indonesian-voice gate
- 20-item model-dependent edit-contract gate
- add-in build-health suite
- before/after comparison with absolute thresholds and no-regression policy
- vLLM LoRA serving with a distinct adapter id
- fail-closed checks that the adapter exists, is loaded, and was requested

## Non-blocking follow-up

- 28 non-router near-duplicate flags await human review.
- 50 examples in five subjective prose strata remain excluded pending an
  independent judge or human review.
- All sources are synthetic. No claim about performance on real Office
  documents is supported.
- FP8 equivalence remains unmeasured.
