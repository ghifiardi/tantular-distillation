# v1 training: BLOCKED pending smoke and final freeze

**Current state: 2026-08-19.**

The corpus, trainer, model-dependent gates, adapter-serving path, and manifest
enforcement exist. Training has not started and no adapter has been produced.

## Critical path

1. **Pin the GPU training environment** and run the 2–4-example smoke:
   model/tokenizer load, NF4, LoRA attach, forward/backward, optimizer step,
   save, reload, and vLLM LoRA serving under a distinct model id.
2. **Write the final schema-v2 freeze immediately after that smoke**, once no
   further config or environment change is expected:

   ```bash
   ./.venv/bin/python src/freeze_training_run.py \
       --corpus data/v3-candidate/traces.r0.jsonl \
       --config train/qlora_9b.yaml \
       --promotion-manifest train/RUN_MANIFEST.v1-mechanical.json \
       --waiver calibration/INT4_WAIVER.md \
       --out train/RUN_MANIFEST.v1.json \
       --frozen-at <ISO-8601> --write
   ```

3. Run `src/train_qlora.py --dry-run`. It must verify the complete freeze.
4. Make the explicit v1 decision and pass `--confirm-run-v1`.

The checked-in `train/RUN_MANIFEST.v1.json` deliberately remains stale until
step 2. The trainer refuses it: it predates schema-v2 promotion-manifest
enforcement and its config digest no longer matches.

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
