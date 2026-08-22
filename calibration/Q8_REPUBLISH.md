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

---

# Published and verified — 2026-08-22

Three tags pushed to `ghifidanukusumo/tantular`:

- `q4-0.4` — the previous Q4_K_M artifact, published FIRST as rollback
- `q8-0.5` — the measured Q8_0 profile
- `latest` — moved to Q8

## Identity, by the three-part method

Raw manifest digests are NOT comparable across a push/pull round-trip: the
pushed tag was `ad43e5078243`, the pulled one `0ed8471c2c9e`, because Ollama
recomputed the manifest and reordered the `PARAMETER` lines. Nothing about the
model changed. `src/model_identity.py` records three separate pieces instead:

| evidence | pushed `q8-0.5` | pulled `latest` |
|---|---|---|
| registry digest | `ad43e5078243…` | `0ed8471c2c9e…` — differs, expected |
| weights blob | `sha256-73b25b60…` | **same** |
| canonical profile (PARAMETERs sorted) | `15368614046c19ce` | **same** |

Verdict: equivalent.

## Behavioural verification — the primary check

The pulled `:latest`, generated fresh through the companion:

| gate | result | budget |
|---|---|---|
| indonesian_voice | **38/40 (0.9500)** | 0.95 |
| edit_contract_output | **19/20 (0.9500)** | 0.90 |
| empty / truncated / reasoning | **0 / 0 / 0** | 0 |
| latency p95 | **17.6 s** | ≤ 30 s |
| slowest request | **28.2 s** | ≤ 45 s |

Per item, the pulled tag matched the locally built one exactly — identical
verdicts on all 40 voice items.

## Client and installer

Add-in commit `447606d`:

- installer pulls `ghifidanukusumo/tantular` by default instead of building
  locally from `qwen3.5:9b`, which reproduces a Q4 profile that scores below
  the gates;
- RAM floor raised 12 GB → 16 GB, since the default grew from 6.6 GB to 10 GB;
- `DEFAULT_DECK_MODEL` → `tantular-office:0.5-9b`, with the 0.4 alias retained
  in the preference chain so existing installs keep working.

## Outstanding

**The product obligation attached to the 45 s budget is NOT implemented.**
`calibration/DEPLOYMENT_BUDGET_POLICY.md` requires the add-in to show progress
during a Studio action and allow cancellation. Neither exists yet. Until they
do, a slow request is still indistinguishable from the hang this sequence began
with — the budget is defensible, the user experience of reaching it is not.

---

# Add-in smoke test on the registry Q8 tag — 2026-08-22

End to end through the companion (`https://localhost:3000/api/chat-completions`
→ Ollama native `/api/chat`, `think: false`) against
`ghifidanukusumo/tantular:latest` pulled from the registry.

| request | latency | finish | reasoning | result |
|---|---|---|---|---|
| two-sentence summary | **4.1 s** | stop | 0 | correct Indonesian, figures preserved |
| edit-contract JSON | **2.0 s** | stop | 0 | `{"edits":[{"find":"setting","replace":"pengaturan","occurrence":1}]}` |
| **cold start** (after `ollama stop`) | **11.9 s** | stop | 0 | correct |

All inside the 45 s per-request budget, cold start included.

## One unexplained observation, recorded rather than smoothed over

The first attempt at this smoke test hung for over ten minutes and was killed by
a tool timeout. It has not reproduced: the two warm runs above took 4.1 s and
2.0 s, and a deliberate cold start took 11.9 s, so model loading is not the
explanation.

The most likely cause is that the request reached the companion while it was
still initialising, since the server had just been restarted in the same
command. That is a guess. It is written down because an unexplained ten-minute
stall in exactly the area this work is about should not be quietly dropped, and
because if it recurs this note is the first evidence.

## UX obligation — now met

Add-in commit `5cf7fee` implements what
`calibration/DEPLOYMENT_BUDGET_POLICY.md` required:

- elapsed time in every progress region, never blank;
- a **Batal** button in all four regions, wired to a real `AbortController`
  whose signal reaches `runTantular`;
- cancellation reported as its own outcome, not as an error;
- budget timeout as a third, distinct outcome;
- once past the budget the progress line says so and points at the cancel
  button.

Deck generation keeps its 480 s budget rather than being forced to 45 s: it
produces many slides, not one answer, and was never what the 45 s figure
measured. It gets the clock and the cancel button; it does not get the cap.

8 new tests cover the rules without a DOM; 397 add-in tests pass.
