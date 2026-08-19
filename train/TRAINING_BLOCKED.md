# v1 training: BLOCKED

**Decision 2026-08-19.** The corpus is promoted and frozen; training may not
start. One gate the config requires does not exist.

## The blocker

`train/qlora_9b.yaml` declares two eval gates and says of them:

> GATE EVERY RUN ON THIS. Distillation reliably buys reasoning while quietly
> costing Indonesian voice and JSON-contract adherence — and the Studio features
> break outright when the edit contract drifts.

| gate | source | status |
|---|---|---|
| `office_json_contract` | `../tantular_office_addin/tests` | present, 32 entries |
| `indonesian_voice` | `prompts/voice_eval.v1.jsonl` | **RESOLVED 2026-08-19** — 40 items, rubric v2, wired |

`../tantular/data/eval` does not exist, and `tantular/data/` is gitignored, so it
was never in version control and cannot be recovered from history.

## What the cited commit actually added

The config points at commit `55d10e8` ("add eval set"). That commit added
`eval_sets/id_factual_calibration.jsonl` — **5 entries**, category
`factual_calibration`, of the form *"nama ibukota indonesia"* with
`required_terms` / `forbidden_terms`. The file is absent from disk but is
recoverable from git.

It is **not** a substitute for `indonesian_voice`:

- it measures **factual calibration**, not voice or register;
- **5 items cannot support a `0.95` threshold** — a single failure scores 0.80;
- reusing it under the voice gate's name would report one property while
  measuring another.

## Decision

`id_factual_calibration.jsonl` may be recovered and added as a **separate**
factual-calibration gate, with its own name, path and metric. It must not be
repointed at, renamed to, or counted as `indonesian_voice`.

A real Indonesian-voice eval must be authored from product requirements or
reviewed test fixtures. Until it exists and is wired in as its own gate,
**the trainer is not to be run.**

### RESOLVED 2026-08-19 — but training remains blocked for other reasons

`indonesian_voice` now points at `prompts/voice_eval.v1.jsonl` (40 held-out
items, `249acd7c65ce5c58…`), scored by `src/score_voice.py` at rubric v2,
threshold 0.95 (38/40). Approved after product review.

`id_factual_calibration.jsonl` was NOT repointed at this gate, as decided.

**Two blockers remain, and neither is the eval:**

1. **There is no trainer.** Nothing reads `qlora_9b.yaml`. `office_json_contract`
   has a source but no runner either — "all gates executable" is not yet true.
2. **No explicit decision to run v1.** Required before any training starts.

**Also: `train/RUN_MANIFEST.v1.json` is now STALE.** It froze the config at
`79fd7bd376eb724e…`; wiring the gate changed it to `afb6cdd52f56b998…`. That
freeze predates the gate and must NOT be used for a run. Re-run
`src/freeze_training_run.py` immediately before training so the record matches
what actually executes.

## What is ready, and stays ready

- `data/promoted/train.jsonl` — 136 traces, `79c7b6aca182ceeb…`
- `data/promoted/eval.jsonl` — 47 traces, `636f78642152246e…`
- `challenge` — 27, held out of both
- `train/RUN_MANIFEST.v1-mechanical.json` — promoted families, rejections, audits
- `train/RUN_MANIFEST.v1.json` — the full 260-trace corpus as generated

Also outstanding, and not blocking on their own:

- **28 non-router near-duplicates** await human review (the other 56 flags are
  router closed-set labels, expected by construction).
- **5 prose strata are excluded**, awaiting an independent judge or human review.
  Muse Glimmer was not used to judge its own output.
