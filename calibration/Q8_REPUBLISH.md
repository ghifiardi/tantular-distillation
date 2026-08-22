# Republish at Q8_0 — evidence and closure

**Decision 2026-08-22: Q8_0 becomes the production profile. Q4_K_M is retained
as rollback and is no longer the default.**

## The candidate

`tantular-office:0.5-9b-q8`, digest `ad43e5078243140b`, built `FROM
qwen3.5:9b-q8_0` with the shipped profile's SYSTEM, TEMPLATE, RENDERER and
PARSER copied byte for byte and **the original sampling preserved** —
`temperature 0.2`, `top_k 20`, `top_p 0.9`, `presence_penalty 1.5`,
`repeat_penalty 1.05`, `num_ctx 32768`, `num_predict 8192`.

Sampling was deliberately NOT changed to the greedy settings used in the
experiments. Experiment 1 showed sampling does not move these gates, but
publishing a configuration nobody had measured would repeat the error this
whole sequence exists to avoid. So the shipped sampling was kept, and the
candidate was measured as it will actually run.

Verified before measuring: `SYSTEM/TEMPLATE/RENDERER/PARSER` hash
`dc69dd6e4f2f5669` on both the current default and the candidate. Precision is
the only difference.

## Gate results on the exact tag

| gate | Q4_K_M (current default) | **Q8_0 candidate** | threshold |
|---|---|---|---|
| indonesian_voice | 0.9250 (37/40) FAIL | **0.9500 (38/40) PASS** | 0.95 |
| edit_contract_output | 0.9000 (18/20) | **0.9500 (19/20) PASS** | 0.90 |
| deployment | pass on p95, **fail** on a 30 s per-request ceiling (32.1 s) | **PASS on both** | — |

Deployment detail for the candidate: 0 empty, 0 truncated, 0 reasoning
characters, p50 6.6 s, p95 22.9 s, slowest 29.0 s.

**The slowest request is 29.0 s against a 30 s ceiling — one second of margin.**
That is a pass, and it is thin. A slower machine, a longer document, or a
contended CPU will cross it. Worth a per-request budget review rather than
treating the pass as comfortable.

## Identity record

`data/gates/prod-q8/model_identity.json` holds, for the current default, the
published Q4 tag and the candidate: Ollama digest, blob size, full Modelfile
sha256, profile-body sha256, and the complete parameter list. It also confirms
`ghifidanukusumo/tantular:latest` and `tantular-office:0.4-9b` are the same
digest — the published artifact and the add-in default are one model.

## Known limitation, not a training justification

`edit::0007` fails at **every** precision tested — Q4_K_M, Q8_0 and bf16. The
model emits one `find` of 203 characters against the 200-character limit,
attempting a whole-paragraph replacement instead of several targeted edits. One
item of twenty, above the 0.90 threshold in every arm.

Recorded as a known model limitation. Not sufficient to justify training, and
the cheaper avenue — a system-prompt hint to prefer the shortest unique `find` —
has not been tried.

## What this closes

The quality deficit came from **Q4 packaging**, not from a corpus shortfall and
not from a missing fine-tune. Two controlled experiments established it: sampling
made no difference at all (identical verdicts on every item), and precision
recovered bf16-level scores on both gates.

No GPU was rented and no training was run to reach this conclusion.

## Remaining steps, which need account access

1. Publish the Q8 profile to `ghifidanukusumo/Tantular` and move `:latest` to it.
2. Point the add-in's `DEFAULT_DECK_MODEL` at the tag users will have once that
   publish lands — currently `tantular-office:0.4-9b` in `src/tantularClient.js`,
   with four further references in `taskpane.js` and `taskpane.html`.
3. Keep the Q4 tag available for rollback.

Step 2 depends on step 1: changing the client default to a tag nobody can pull
would break every install.
