# v3 volatility — MEASURED at concurrency 4; UNMEASURED across batching

**Status 2026-08-18.** Three independent passes over all 260 v3 prompts show
**0 of 260 families varying**. That closes the question at the configuration
tested, and leaves one narrower question open — see the scope note at the end.

Superseded sections below are kept for the reasoning trail.

Status as of 2026-08-17. **Not `PASS`. Not `NONE FOUND`.** No family has been
tested, so nothing is known either way.

## Why the gate's silence is not evidence

`verify_corpus.py --gate` enforces the waiver's rule — *replicated, or excluded,
never one arbitrary sample* — against families read from
`calibration/noise_floor.<arm>.json`. Today `volatile_families()` returns `{}`:

- the v1 floor is `_STALE` and is skipped by design;
- the v2 floor never carried a `volatile_families` key.

So the rule passes over an empty set. That is a **vacuous pass**: not every
designated family being replicated, but no family being designated. Reporting it
as satisfied would state the opposite of what is known.

## Why the v2 method cannot be reused

The v2 floor measured volatility from **within-run repeat groups** — 49 prompts
appearing 2–8 times inside a single pass, because 260 families shared 78
documents.

Source pack v3 gives every family its own document:

| prompt set | families | distinct prompts | repeated prompts |
|---|---|---|---|
| v2 | 260 | 78 | 49 |
| **v3** | **260** | **260** | **0** |

There are no repeat groups to measure. This is a direct and accepted consequence
of per-family artifacts, not a defect — distinctness and repeat-based
measurement are in tension, and distinctness was chosen deliberately.

It also lowers the resolution of the cross-pass instrument: with one trace per
prompt per pass, mode-frequency drift can only be 0 or 1. The mixture shifts
(`4:1` against `1:4`) that the drift metric was built to catch are invisible
without three or more samples of the same prompt.

## The v1/v2 volatile list must not be inherited

Previously designated: `prose:cekAman::0001`, `prose:tanyaDokumen::0001`,
`prose:umum::0000`. Those came from prompt set v1 over source pack v2 — different
documents, different prompts. Carrying them into v3 would replicate families
chosen by evidence that no longer applies while leaving genuinely volatile v3
families on a single sample.

**v3's volatile set must be identified from v3's own evidence.**

## What will measure it

Explicit replicates, the way `volatile.r1/r2.jsonl` worked before: re-generate
the same v3 prompts at identical settings and compare answers per prompt. Three
samples distinguish the two cases a single disagreement cannot — a family with
two acceptable modes, versus one that is simply unstable.

`volatile_review.py` then decides per family: INCLUDE ALL / INCLUDE ONE /
EXCLUDE. Only once that has run can the replication rule be *enforced* rather
than vacuous.

Until then this file is the answer to "are v3's volatile families replicated?":
**unknown, because none have been identified.**

---

## Probe round 1 — 2026-08-17: 0 candidates, and why that is not an answer

`volatile_probe.py` over the two completed v3 passes:

| | |
|---|---|
| passes | `data/v3-candidate/traces.r0.jsonl`, `data/v3-pass-a/traces.r0.jsonl` |
| plan consistency | all 9 pinned fields identical |
| prompt digests | identical for all 260 shared families |
| **families that varied** | **0 / 260** |

No candidates, so there is nothing to replicate under the selection rule, and
**the status stays `UNMEASURED`.** Two observations cannot separate a stable
family from one whose modes happen to agree twice.

### The result is not a batching artifact

Worth checking, because the v1 evidence tied mode selection to batching: if both
passes had processed every prompt in the same batch window, identical output
would be near-guaranteed and would say nothing about stability.

They did not. The resumable runner chunks at 40 with retries, and the two passes
landed on different boundaries:

```
r0     (0,40) (40,55) (55,95) (95,135) (135,175) (175,215) (215,255) (255,260)
passA  (0,40) (40,80) (80,120) (120,133) (133,173) (173,213) (213,253) (253,260)
```

**220 of 260 families were generated in a different batch window across the two
passes, with different co-resident prompts, and still produced byte-identical
completions.** Only the first 40 shared a window.

That is real evidence of stability under the one variable v1 implicated. It is
still two observations per family, which is why the status does not move.

### The two pass files are byte-identical

`sha256 1d16204ddc384fce…` for both. Checked rather than assumed, because
identical files are exactly what a copied or mis-pathed run would produce:

- written 84 minutes apart (10:42 and 12:06);
- each runner log shows its own distinct chunk progression;
- provenance records **no wall-clock or timing field** — `latency_s` is not
  captured — so a fully deterministic generation has nothing left to differ on.

Byte-identity is not automatic. The v2 passes over the same host and settings
have different digests (`55b0503a…`, `f21386aa…`) because 6 of 78 prompts
genuinely disagreed. v3 reproduced exactly where v2 did not.

**Hypothesis, unverified:** v2's 260 families shared 78 prompts, so a batch
could contain the same prompt several times; v3's prompts are all distinct. If
mode selection is tied to batching, repeated identical prompts within a batch
are the more likely trigger. Recorded as a lead, not a finding — nothing here
tests it.

### What would move it

A third observation per family, ideally at a deliberately different chunk size
and concurrency to vary batching on purpose rather than by accident of retries.
Three samples let `volatile_review.py` report modes instead of a bare
agree/disagree, which is the whole point of that tool.

Absent that, the honest statement is: **no v3 family has been shown to vary, and
no v3 family has been shown to be stable.**


---

## Result 2026-08-18: three passes, 0/260 varied

| | |
|---|---|
| passes | `data/v3-candidate`, `data/v3-pass-a`, `data/v3-pass-c` |
| plan consistency | all 9 pinned fields identical |
| prompt digests | identical for all 260 families in all 3 passes |
| uniqueness | 260 traces / 260 families / 0 duplicates, each pass |
| **families that varied** | **0 / 260** |

Every completion is byte-identical across all three passes. The only difference
between the files is LINE ORDER — pass C was generated across resumed chunks
after tunnel drops, so its families are written in a different sequence. A
field-by-field comparison finds no other difference at all: not one provenance
value, not one token count.

### Steps 3 and 4 were not run, because there was nothing to run them on

The agreed sequence was: probe, then replicate any varying family, then
`volatile_review` to decide INCLUDE ALL / INCLUDE ONE / EXCLUDE. The probe
returned zero candidates, so there are no families to replicate and no modes to
adjudicate. Running the review on an empty set would produce a verdict with no
evidence behind it.

The waiver's rule — *replicated, or excluded, never one arbitrary sample* —
is therefore satisfied on evidence rather than vacuously. Previously no family
was designated volatile because none had been looked for. Now 260 have been
checked three times.

### What this does and does not establish

**Does:** at `concurrency 4` with identical configuration, the teacher produced
identical answers to all 260 v3 prompts on three separate runs, hours apart,
across tunnel drops and process restarts.

**Does not:** establish stability under DIFFERENT batching. All three passes ran
at concurrency 4. The v1 evidence tied mode selection to batching, and that
remains untested for v3 — a concurrency-8 pass was attempted and abandoned after
180 of 260 requests failed with `RemoteProtocolError`, because ai19 is a shared
production box that drops connections at that load. Those failures are
infrastructure, not quality, and are filed under
`data/_failed-runs/v3-pass-c-concurrency8/`.

**Also does not:** prove a rare mode cannot exist. Three samples would very
likely miss a mode that appears, say, 5% of the time. The honest statement is
*no variation was observed in three observations at these settings*, not *this
teacher is deterministic*.

### Practical consequence

No v3 family requires replication or exclusion before training. The corpus can
be used as generated, and `verify_corpus.py --gate`'s volatile-family check now
passes over an empty set for a measured reason.

Corpus remains `synthetic_candidate`; the int4 waiver is unaffected.
