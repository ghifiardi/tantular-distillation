# int4-only distillation — quality waiver

**Status: ACCEPTED 2026-08-13.** See the acceptance block at the bottom.

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
| Accepted by | ghifiardi (project owner) |
| Date | 2026-08-13 |
| Scope | int4-only distillation programme — Q4_K_M teacher `muse-glimmer:30b` on ai19. Each corpus freeze and training run must cite this waiver by id. |
| Review trigger | availability of a 48GB Ada or 80GB Hopper host |

### Accepted text, verbatim

> I accept proceeding with the Q4_K_M int4 teacher on ai19 despite the failed
> FP8 corpus gate. This does not claim FP8 equivalence. The control noise is
> bimodal, so volatile families will not be represented by an arbitrary single
> trace: they must use replicated/distributional sampling or be excluded from
> the authoritative training corpus. The 14 unapproved strata and gateway-key
> rotation remain independent production blockers.

### What this authorises, and what it does not

Authorises: training from the int4 teacher despite `verify_corpus.py --gate`
failing on teacher quantization.

Does **not** authorise: any FP8 claim; treating the gate as passed; generating
a production corpus (14 strata still lack approved sources); or production
generation before the gateway key is rotated.

### Enforced, not merely promised

The volatile-family condition in the accepted text is checked by
`verify_corpus.py --gate`, which fails if a family known to flip a metric
appears in the corpus with a single trace. Families are read from
`calibration/noise_floor.<arm>.json`. The rule is: **replicated, or excluded —
never one arbitrary sample.**

---

## Addendum — synthetic substitution for blocked strata (2026-08-13)

The 14 strata with no approved real source may be backed by **fabricated**
documents instead, generated via Tinker. This unblocks pipeline generation and
behavioural training. It does **not** demonstrate performance on real Office
documents, and does not remove the need for later real-corpus approval.

### Handling rule for the external service

Tinker processes inputs through its service and AI providers. Its terms state
customer content is not used to develop or train models, but it remains an
external boundary. So:

**Only fabricated inputs.** Invented names, companies, numbers, emails and
document contents. No real Office text, screenshots, templates, or
descriptions containing confidential details — including as *prompts* asking
Tinker to produce something similar.

- Terms: https://thinkingmachines.ai/legal/terms/
- Privacy: https://thinkingmachines.ai/legal/privacy/

### Required record per synthetic stratum

```yaml
source_class: synthetic
source: tinker/<project-or-session>/<artifact>
source_sha256: "<exported-artifact-hash>"
approval: "<internal approval for synthetic substitution>"
redacted: true
redaction_record: "Synthetic from scratch; no real personal or company data"
egress_approved: true
egress_reference: "<approval for Tinker and ai19>"
generator:
  service: tinker
  model: "<model id>"
  version: "<tinker version>"
  prompt_template_sha256: "<hash>"
  generated_at: "YYYY-MM-DD"
```

Enforced by `src/inventory.py`: a stratum marked `synthetic` is not ready to
seed until every field above is present, generator sub-fields included.
`verify_corpus.py` reports synthetic-backed strata at verification time and
states plainly what such a corpus does and does not support.

### Prompt construction for editing tasks

The synthetic source document must be **embedded in the prompt**, with the
requested transformation specified separately. A prompt that refers to a
document without including it produces a teacher asking for the missing text —
observed directly: 4 of 5 traces unusable in the first seed run.

## Addendum 2 — local source material (preferred over Tinker)

Local files are preferable to an external generator on privacy grounds,
**provided use is authorized**. The route is: inventory local sources, approve
and redact them, then send only the sanitized artifacts to ai19.

### Rules, enforced by `src/inventory.py`

| Requirement | Field |
|---|---|
| Originals stay outside Git and outside `data/raw/` | `originals_location` — a path inside the repo is rejected |
| Source path or document ID | `source` |
| Content hash | `source_sha256` |
| Owner and approval reference | `owner`, `approval` |
| Names, account numbers, emails, client identifiers removed | `redacted: true` + `redaction_record` |
| Sanitized artifact may go to ai19 | `egress_ai19_approved` + `egress_reference` |

### Three distinctions that are easy to lose

**Sending sanitized local material to ai19 over the SSH tunnel is still an
egress event.** The tunnel reduces the number of observers; it does not make
the transfer internal-by-definition. It needs its own authorization, recorded
in `egress_ai19_approved`, separate from the approval to *use* the document.

**A local MLX/int4 teacher is a different teacher path** and is NOT covered by
this waiver. Verified: `("int4_mlx", "mac-validate")` is absent from
`WAIVER_COVERED`, so such traces fail the gate as unauthorised. Adding that
path requires amending this waiver explicitly.

**Unknown ownership, provenance, or sensitivity remains blocked.** A file found
on disk without those answers is not a source; it is an unknown. The schema
makes this mechanical rather than a judgment call.

### Fully fabricated local documents

May be marked `source_class: synthetic` with no real-document redaction —
there is nothing real in them to redact. They still require provenance
(`generator`, including handwritten authorship) and synthetic-substitution
approval. Where generation stayed local, no egress reference is required for
the generation step itself.

### What this does not change

- The FP8 gate remains UNMET; these addenda concern source material, not
  teacher precision.
- Gateway key rotation remains a separate production gate.
- Real-corpus approval is deferred, not waived.
