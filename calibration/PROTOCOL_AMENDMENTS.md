# Protocol amendments — int4-vs-FP8 calibration

Pre-registered thresholds in `acceptance.yaml` are **unchanged** by anything
here. These are execution-path amendments, recorded because the environment
moved during the study and a reader must be able to tell what was measured.

---

## Amendment 1 — baseline endpoint changed mid-study (infrastructure)

**When:** 2026-08-13, before any baseline traces were scored.

**What happened.** The first 52-prompt baseline attempt ran against the
LiteLLM gateway at `https://openai.ina17.com/v1` and failed **all 52 calls**
with `HTTP 400 — Invalid model name passed in model=ollama/muse-glimmer-30b`.
The id was still listed by `/v1/models`; only the deployment behind it was
gone. `ollama/nemotron-3.5-lightning` failed identically. `qwen3.6-35b` and
`ghifidanukusumo-Tantular-latest` continued to route.

**Classification: infrastructure failure, not model quality.** Those 52 calls
produced zero traces, are excluded from every quality denominator, and were
**not** retried into the result set. The run was re-executed in full against
the new endpoint. No partial results from the failed attempt were merged.

**Amendment.** The baseline arm is served by **ai19's Ollama reached directly
over an SSH tunnel** (`http://localhost:11435/v1` → `ai19:11434`) rather than
through the gateway. Same weights, same GPUs — the gateway was a proxy in
front of this exact server.

**Effect on scope.** The comparison is now explicitly:

> `int4 on ai19 / Ollama` versus `FP8 on rented Ada-or-Hopper / vLLM`

This answers the practical shipping question. It does **not** isolate
precision: quantization and serving runtime vary together. Any result must be
reported with that scope attached. An int4-under-vLLM arm would isolate
precision, and is worth running only if this result is marginal or the
Ollama-vs-vLLM difference is operationally important — it is an attribution
experiment, not a prerequisite.

---

## Amendment 2 — two runtime facts that bear on comparability

Both discovered while capturing `data/calibration/int4/environment.txt`.

**`TEMPLATE {{ .Prompt }}` — the Ollama model card uses a passthrough
template.** It applies no chat template of its own. A vLLM FP8 arm will apply
the chat template from the model's `tokenizer_config.json`. So the two arms
may not present identical token sequences to the model even from identical
`messages`. This is a real confound for the behavioural comparison and must be
recorded alongside any result. Mitigation if it proves material: pin an
explicit template on both arms, or compare rendered prompts directly.

**`PARAMETER temperature 1` is baked into the model card.** The study sends
`temperature: 0.0` per request, which Ollama's OpenAI-compatible endpoint
applies over the card default. The FP8 arm must set temperature explicitly
too, and not rely on any server default.

---

## Amendment 3 — preflight required before either arm

Because a model disappeared underneath a running study once already, both arms
now run `src/preflight.py` first. It records endpoint, model id, runtime,
quantization, template and the full `/v1/models` response, and **fails the run**
if the model is absent or its signature differs from what was recorded for that
arm. A study that silently changes model underneath itself produces numbers
that mean nothing.

---

## Amendment 4 — data-handling scope of the direct endpoint

Reaching ai19 directly removes the **Cloudflare and LiteLLM operator path**.
It does **not** make prompts unobserved: the ai19 host operator and the Ollama
service itself still see every prompt, and Ollama logs requests.

So this is an improvement in the number of parties with visibility, not a
guarantee of confidentiality. It does not by itself authorise sending real
Office corpus material — `inventory/sources.yaml` still governs that, and the
calibration prompts remain `source_class: synthetic`.

---

## Recorded environment — baseline arm

| Field | Value |
|---|---|
| Endpoint | `http://localhost:11435/v1` (SSH tunnel → `ai19:11434`) |
| Model id | `muse-glimmer:30b` |
| Digest | `de878ce33ad8` |
| Runtime | Ollama 0.32.9 |
| Quantization | **Q4_K_M** |
| Parameters | 27.9B (+1.9B CLIP projector) |
| Context length | 131072 |
| Template | `{{ .Prompt }}` (passthrough) |
| Card defaults | temperature 1, top_k 64, top_p 0.95 |
| Request decoding | temperature 0.0, top_p 1.0, max_tokens 4096 |
| GPUs | 2× RTX 3090, compute capability 8.6, shared/contended |

Full capture: `data/calibration/int4/environment.txt`.

Latency and throughput from this arm measure a **contended production server**,
not the hardware. They stay informational and non-gating.

---

## Amendment 5 — rendering parity resolved, and what it revealed

**Parity by naive rendering: FAILED.** Seven candidate serializations
(`sys_blank_user`, `sys_nl_user`, `labelled_turns`, `user_only`, and three
harmony variants) were run through Ollama's raw path at temperature 0 and
compared against its chat endpoint. None reproduced it. Recorded in
`calibration/parity/ai19-ollama.probe.json`.

**Root cause.** The model is **harmony-format**, and Ollama renders it with a
built-in architecture renderer — which is why the Modelfile shows a bare
`TEMPLATE {{ .Prompt }}` and the GGUF carries no `chat_template`. The
authoritative template is `chat_template.jinja`, shipped in the FP8 repo the
treatment arm will load, and vendored here at
`calibration/parity/chat_template.jinja`.

Canonical rendering (probe hash `3c923547bfe3acab`):

```
<|start|>system<|message|>{system}

Reasoning strength: high.

# Valid recipients: "self", "user".<|eot|><|start|>user<|message|>{user}<|eot|><|start|>assistant
```

**The divergence is on the RESPONSE side, not the prompt side.** Raw generation
returns every channel; the chat endpoint parses them and returns only the final
message. Full raw shape:

```
 to=self<|message|>{reasoning}<|start|>assistant to=user<|message|>{final answer}
```

**Normalized protocol for both arms:**

1. Render with `chat_template.jinja`; record the SHA-256 of the rendered string.
2. Send that exact string via a completion path — `/api/generate` with
   `raw: true` (Ollama), `/v1/completions` (vLLM). Prompt parity then holds by
   construction and the hash proves it.
3. Extract the final message identically on both arms: take the content after
   the last `<|start|>assistant to=user<|message|>`, strip trailing `<|eot|>`.
4. Set temperature, seed, max tokens, model revision and `reasoning_strength`
   explicitly on both arms; rely on no server default.

**Consequence.** The existing 52-prompt int4 result was produced through
Ollama's chat endpoint, i.e. a different response path. It stands as the
**deployment baseline** — it measures what the pane would actually get today —
but the **int4-vs-FP8 verdict must use a normalized re-run**. Thresholds are
unchanged.

### Side finding with product impact

`reasoning_strength` is a **chat-template variable**, defaulting to `'high'`:

```jinja
set rs = reasoning_strength if reasoning_strength is defined and reasoning_strength else 'high'
```

It is not the `reasoning_effort` API field. That explains two earlier
observations: `reasoning_effort: "none"` was silently ignored, and ~230 tokens
went to reasoning before any content appeared.

This reaches beyond the study. `tantular_office_addin/src/tantularClient.js`
sends `reasoning_effort: "none"` specifically to stop thinking from consuming
short budgets — the intent router allots 4 tokens. Against any harmony-format
model that field does nothing, and the control must be passed as a chat
template kwarg instead.

---

## Amendment 6 — three distinct protocols, never merged

Fixed **before** the diagnostic run's results were inspected.

| # | Dataset | Protocol | Status | Used for |
|---|---|---|---|---|
| 1 | `data/calibration/int4/` | Ollama **chat endpoint**, server-side template + channel parsing | immutable | **Deployment baseline** — what the pane actually receives today |
| 2 | `data/calibration/int4-normalized/` | raw completion, official template, no explicit stops | **PARITY VALIDATION** (see Amendment 7) | proves the canonical rendering reproduces production 52/52 |
| 3 | `data/calibration/int4-normalized-controlled/` | raw completion, official template, **explicit stops**, full termination metadata | authoritative | **Precision-comparison baseline** vs FP8 |

These are three different measurements of the same weights. They are reported
side by side and **never concatenated** — merging them would average distinct
protocols into a number describing none of them.

### Decision rule for the diagnostic (fixed in advance)

- **Tokens cluster at 4096** → hypothesis confirmed; run 3 is authoritative and
  run 2 keeps its diagnostic label permanently.
- **Varied short completions with stop metadata** → raw-mode stops were already
  working; inspect termination behaviour and retain run 2 only if both arms can
  be shown to terminate equivalently.
- **No meaningful speedup despite shorter parsed answers** → `options.stop` is
  ignored in raw mode. Fall back to client-side truncation at the marker, which
  preserves answer correctness but is **not** equivalent termination behaviour:
  `completion_tokens` is then marked UNUSABLE and excluded from cost, length and
  performance claims **for both arms**.

Client-side truncation is never reported as if the runtime had stopped
generation. The answer is valid; the length metric is not.

---

## Amendment 7 — diagnostic result, and the finalized protocol

**The budget-exhaustion hypothesis was REFUTED.** Run 2's distribution:
129–2024 completion tokens, median 388, **none at the 4096 ceiling**, 44 of 52
under 1000, `done_reason: "stop"` throughout. Raw mode terminates on EOS
unaided. The slowness was contention on a shared box, not runaway generation.

**Run 2 is therefore reclassified from DIAGNOSTIC to PARITY VALIDATION**, on
much stronger evidence than it was launched to gather. Compared per-prompt
against the deployment baseline:

```
families compared      52
identical token counts 52/52
identical answers      52/52
```

Every aggregate metric matches to four decimals. The canonical harmony
rendering **is** what Ollama's chat endpoint produces. Amendment 5 was right
that the divergence was response-side and wrong that it mattered: the earlier
probe "failed" only because raw multi-channel output was compared against
parsed chat output.

### Finalized protocol

- **No explicit stops by default.** The problem they addressed does not exist,
  and forcing them is actively risky: `<|eot|>` is a channel delimiter as well
  as a terminator, so stopping at the first occurrence could cut generation off
  before the final answer channel — turning good answers into malformed traces.
  The no-stops configuration is the one proven to reproduce production exactly.
- **EOS is left to each runtime**, and how it terminated is recorded rather
  than assumed.
- `--stop` remains available for a runtime that does not honour EOS. If used,
  it must be identical on both arms.
- Ollama's bare `"stop"` stays `ambiguous`. It attributes nothing, so nothing
  is claimed.

### FP8 arm requirements (enforced by `src/compare_arms.py`)

The comparison is **blocked**, not merely annotated, unless:

- per-prompt `prompt_sha256` is identical across arms — the same bytes reached
  both models;
- `template_sha256` is identical;
- `reasoning_strength`, `temperature`, `seed`, `max_tokens` and `stop_sequences`
  all match;
- the family sets are the same.

Termination is reported side by side and never equated. vLLM exposes
`stop_reason` — the stop string or token id that fired, `null` when the model
emitted its own EOS — which is precisely the evidence Ollama withholds. Both
are recorded verbatim, along with `eos_applied_by_runtime`, which is `null`
for Ollama to show the evidence is *missing* rather than checked.

`reasoning_strength: high` on both arms, matching the deployment baseline.
Evaluating `low` is a separate generation-policy experiment.

---

## Amendment 8 — the runtime is NOT deterministic, and the noise floor exceeds the thresholds

The controlled rerun was launched only to confirm reproducible generation. It
refuted it.

**Two runs of the same arm — same weights, same rendered bytes (hash-verified),
temperature 0.0, seed 0, no stops, concurrency 4:**

```
compared 52 families
  identical            34
  token_count_only      7
  answer_divergent     11
NON-DETERMINISTIC — 18 of 52 (34.6%) differ
```

Both runs are preserved. Neither replaces the other; the newer is not "more
correct", and discarding either would destroy the evidence that this variance
exists.

### The noise floor, in metric terms

| metric | run A | run B | \|Δ\| | pre-registered bound |
|---|---|---|---|---|
| refusal_rate | 0.0192 | 0.0577 | **0.0385** | 0.00 — **exceeded** |
| constraint_satisfaction | 0.9706 | 1.0 | **0.0294** | 0.00 — **exceeded** |
| router_label_accuracy | 1 | 1 | 0 | 0.00 |
| source_preservation | 1.0 | 1.0 | 0 | 0.00 |
| empty / truncation rate | 0.0 | 0.0 | 0 | 0.00 |
| indonesian_quality | 1.0 | 1.0 | 0 | 0.05 |

**Run-to-run variance alone breaches the critical tolerance on two metrics.** A
single-run int4-vs-FP8 comparison therefore cannot support a verdict: noise
would produce a FAIL, or a spurious PASS, with no precision effect present.

### What this does and does not change

- Thresholds are **NOT** revised. They were pre-registered; loosening them
  after seeing variance is exactly the move pre-registration exists to prevent.
- The **study design** is what is insufficient. A verdict needs either
  replicates per arm compared as distributions, a paired per-prompt comparison
  (statistically far more efficient — the same prompt through both arms,
  differences tested pairwise rather than as aggregate rates), or a serving
  configuration that is actually deterministic.
- The int4 arm remains **parity-validated**: 52/52 identical to production was
  measured within one run pair and is unaffected. Reproducibility is now
  answered — negatively.

### Suspected cause

Concurrency. Batched inference groups requests differently between runs, and
reduction order within a batch is not fixed, so identical inputs can yield
different logits. Being tested at concurrency 1 on a 12-prompt subset before
any expensive redesign. If serial generation reproduces, the fix is a serving
configuration; if it does not, the fix must be statistical.

---

## Amendment 9 — concurrency finding, and the repeated-crossover design

### Concurrency 1: a serving-configuration finding, not a determinism proof

12-prompt subset, two runs, concurrency 1, everything else unchanged:

```
answers identical    12/12
reasoning identical   8/12   <- variance lives here
```

**Answer text is deterministic at concurrency 1; the reasoning channel is
not.** Since metrics score the answer channel, concurrency 1 looks stable —
but that is a property of a serving configuration, not evidence that
production concurrency is deterministic. A teacher run at concurrency > 1 has
already been shown to diverge on 34.6% of families, and a verdict that only
holds at concurrency 1 describes a configuration nobody will deploy.

So: recorded as a diagnostic. **The FP8 verdict uses the concurrency and
replication policy intended for deployment**, not the one that happens to look
reproducible.

### Repeated crossover replaces the single paired run

`src/crossover.py`. One paired observation per arm cannot separate a precision
effect from a runtime that moves refusal_rate by 0.0385 between identical runs.

- every prompt through **both** arms, so prompt difficulty cancels in the
  within-prompt delta rather than inflating variance;
- **arm order randomized per prompt**, seeded for reproducibility, so a warm
  cache or a drifting box cannot favour whichever arm ran second;
- **R replicates per arm per prompt**, so within-arm variance is measured
  rather than assumed;
- **runtime noise and effect are reported separately and never combined**. An
  effect at or below the floor is not evidence of a precision difference.

Validated by a **null experiment** — the same int4 data fed in as both arms:

```
EFFECT (armB - armA)          all metrics: mean delta 0.0
NOISE FLOOR recovered         constraints_ok 0.0294, refusal 0.0385
```

The harness reports zero effect where there is none, and independently
recovers the noise floor measured directly. A design that could not do that
would be untrustworthy for a real comparison.

### Two claims now kept distinct

- **Production equivalence within a run** — VALIDATED. The canonical rendering
  reproduces Ollama's chat endpoint 52/52 per-prompt inside a single run pair.
- **Repeatability across runs** — REFUTED. 34.6% of families differ between
  runs at production concurrency.

These are different properties. The first is what makes the normalized
protocol a faithful stand-in for production; the second is what determines how
much evidence a verdict needs. Thresholds are unchanged.
