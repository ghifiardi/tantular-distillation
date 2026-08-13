# int4-only distillation — quality waiver

**Status: DRAFTED, NOT ACCEPTED.** The acceptance block at the bottom is blank
on purpose. Recording the terms is engineering work; accepting the tradeoff is
not, and nothing here takes effect until a named person signs it.

---

## What is being waived

The pre-registered training gate requires FP8 teacher traces
(`calibration/acceptance.yaml`, `critical`). Proceeding with an int4 teacher
means that gate is **UNMET**.

Unmet — not loosened, not removed, not passed by another route. The threshold
stands exactly as committed in `3f05281`. `verify_corpus.py --gate` will
continue to fail on int4 traces, and that failure is correct. Overriding it
requires this waiver, by reference, in the training run's record.

## What may be claimed, and what may not

**May be claimed:**

- The student imitates the teacher behaviour users actually receive today.
  That behaviour is validated: the normalized protocol reproduces the deployed
  Ollama chat endpoint 52/52 per-prompt.
- Generation stayed on-premises, with no third-party operator in the path and
  no rental cost.

**May NOT be claimed:**

- FP8 equivalence, in any wording.
- That the FP8 gate passed, or was satisfied, or was met.
- That FP8 would not have produced a better student. **This was never
  measured.** Absence of evidence is the whole content of this waiver.

The correct description is: *"trained from an int4 teacher under a recorded
waiver; FP8 comparison not performed."*

## What is given up

A quantized teacher's error is baked permanently into the student. The
magnitude here is **unknown and unmeasurable without the treatment arm** — the
control-side work characterises run-to-run noise, not quantization loss. No
number in this repository bounds it.

## Conditions

Accepting this waiver requires all five:

### 1. Freeze the corpus with full provenance

Every trace carries teacher, repo, license, host, quantization, protocol,
template sha256, per-prompt prompt sha256, reasoning_strength, temperature,
seed, max_tokens, runtime, raw termination reason and split fingerprint.
Recorded at freeze time, not reconstructed later.

### 2. Account for measured runtime noise

The teacher is **not reproducible across runs** — 34.6% of families diverge at
production concurrency. Per `calibration/noise_floor.ai19-ollama.json`:

- aggregate floors: `refusal` 0.0385, `constraints_ok` 0.0294
- but these are **not diffuse drift**. At R=2, 3 of 52 families flip a metric
  fully between 0 and 1:
  `prose:cekAman::0001`, `prose:umum::0000` (refusal),
  `prose:tanyaDokumen::0001` (constraints_ok)

Volatile families must be flagged in the corpus. A single trace from a family
that flips is a coin toss, not a teacher's judgment, and training on one as if
it were authoritative teaches the coin toss. Either sample such families
across replicates or exclude them, deliberately and on the record.

### 3. Evaluate on held-out examples across all approved strata

The eval and challenge splits exist and are enforced. Coverage must be real:
**14 of 26 strata currently have no approved source material**
(`inventory/sources.yaml`). This waiver does not touch that constraint — it is
independent and still binding. An int4 waiver permits a worse teacher, not an
absent corpus.

### 4. Compare the student against the deployed int4 teacher

Not against an abstract quality bar. The target is the behaviour users
receive, so the student is measured against that teacher on held-out examples,
plus the task-level acceptance criteria in `train/qlora_9b.yaml`
(`office_json_contract` ≥ 0.98, `indonesian_voice` ≥ 0.95).

### 5. FP8 stays an open calibration track

Not a hidden prerequisite, and not cancelled. The treatment arm is fully
specified and guarded (`src/compare_arms.py`); it needs only a 48GB Ada or
80GB Hopper host. If it runs later and shows a material gap, that is a finding
about this student, not a retrospective invalidation of the decision.

## When this is the right call

Reasonable if the product target is current int4 Ollama behaviour, and privacy
and cost outweigh maximising teacher quality. Tantular's positioning —
Indonesian-first, private, on-device — makes that a coherent choice rather than
a concession.

It is **not** equivalent to proving FP8 would not improve the student.

---

## Acceptance

| Field | Value |
|---|---|
| Accepted by | _(name, role)_ |
| Date | _(YYYY-MM-DD)_ |
| Scope | _(specific corpus freeze / training run id)_ |
| Conditions 1-5 verified by | _(name)_ |
| Review trigger | _(when FP8 hardware becomes available)_ |

Unsigned, this document records terms only. It authorises nothing.
