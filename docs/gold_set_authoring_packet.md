# Gold-set authoring packet — first 40–60 native Indonesian items

For the project owner and an independent reviewer. **No agent may author,
approve, or countersign an item in this set.** The loader validates structure;
it cannot supply content, and there is no code path that makes
`approved: false` become true.

Authorized by `docs/authorizations/rsi-program-2026-09-19.md`. The evidence
gate is currently **BLOCKED — 0 approved items**, which is the correct state
until this packet is completed.

## Why you are writing these by hand

The RSI report is blunt about where a self-improvement loop actually fails:

> What separates a method that compounds from one that drifts is almost
> entirely the verifier. **Your bottleneck is not the algorithm. It is an
> Indonesian verifier you trust.**

and, on why translation will not do:

> Do not translate an English seed set; you will inherit its task distribution
> and its cultural assumptions wholesale.

`train/TRAINING_BLOCKED.md` names the only evidence that reopens the training
question: *"a real observed failure, from users, add-in logs or an approved
real Office corpus. Not a constructed one."* Items invented to fill a quota do
not qualify, and a model measured against them would be measured against our
own imagination.

## What to produce

**40–60 items total**, across four splits. A suggested distribution, not a
rule:

| Split | Items | What it tests |
|---|---|---|
| `tool_use` | 15–20 | the add-in's edit contract, tool selection, structured output |
| `instruction_following` | 10–15 | constraints obeyed exactly: length, format, register |
| `knowledge` | 10–15 | Indonesian law, regulation, geography, civics, institutions |
| `reasoning` | 5–10 | multi-step inference with a checkable final answer |

`tool_use` is weighted highest deliberately: the RSI report recommends
optimising it first, because "that is where prompt structure carries the most
weight and where your eval is most automatable."

## The rules that will reject your file

`src/gold_set.py` refuses, per `data/gold/SCHEMA.md`:

1. an unknown `split`, a missing `verifier`, or `language != "id"`;
2. an empty or unapproved `source`;
3. a `judge` item with `score_role: correctness` — judge items are style-only
   until judge–human agreement is calibrated;
4. an empty `expected` on a correctness item;
5. a duplicate `id`;
6. an unknown top-level field;
7. an unenumerated normalization (no fuzzy matching added to make an answer
   pass);
8. an executor item with an out-of-bounds timeout or a runner that is not
   allowlisted.

Run `./.venv/bin/python src/gold_set.py --dir <your dir>` to see every problem
at once. It reports; it does not repair.

## Provenance: what makes an item production-eligible

```yaml
source:
  kind: human_authored          # not synthetic_fixture
  author_ref: <registry id>
  authored_at: 2026-09-__       # ISO date
  approved_by: <registry id>    # MUST differ from author_ref
  approved_at: 2026-09-__
  evidence_ref: <local review record id>
source_class: internal          # never synthetic
```

`approved_by` must be a different person from `author_ref`. An author approving
their own item is not an independent review, and the loader rejects it by name.

## Choosing a verifier

In descending order of trustworthiness — prefer the highest one that fits.

**`exact_match`** — a single unambiguous answer.

```json
{"answer": "42", "normalization": ["trim", "unicode_nfc"],
 "reasoning_check": {"type": "required_claims", "claims": ["6 × 7 = 42"]}}
```

Add `reasoning_check` wherever a wrong method can still reach the right answer.
It is how the harness records a **false positive**: the answer matches, a
checkable claim does not, and the pass is kept alongside the flag rather than
erasing it.

**`schema`** — structured output, especially the Office edit contract. Use a
bounded JSON schema with `required` and `additionalProperties: false`. Parsing
is not passing: a model can emit valid JSON that satisfies nothing.

**`executor`** — currently restricted to `synthetic_fixture` records. The local
backend is timeout-bounded, **not sandboxed**: no network policy, no isolation,
no resource limit beyond wall-clock time. A human-authored executor item will
**fail closed** until a reviewed sandbox backend exists. Please do not author
executor items yet.

**`judge`** — style only, `score_role: style`, report-only. It cannot gate
correctness and the judge itself is not implemented.

## Writing good items

- **Native, not translated.** Write in the register you actually serve: formal
  or `baku`, conversational, regional-flavoured, code-switched, domain jargon.
- **Checkable.** If you cannot say mechanically what makes an answer wrong, it
  belongs in the style corpus, not the correctness one.
- **Real.** Draw on genuine documents, real regulations, real add-in usage.
  Prefer a case that actually failed.
- **Discriminating.** The gate the set must pass is that it **separates
  Tantular-9B from Tantular-4B stably**. An item both models always pass, or
  both always fail, costs a rollout and tells you nothing.
- **Independent.** Avoid items that are trivial rephrasings of each other;
  near-duplicates inflate a pass rate without widening coverage.

## Where to put the file

`data/gold/<name>.jsonl`, one JSON object per line. `data/gold/*.jsonl` is
gitignored — production gold is supplied through the approved local-data
workflow, never committed here. `tests/fixtures/gold/*.jsonl` is un-ignored
because those are tiny synthetic fixtures and are code, not corpus.

## Checking your work

```
./.venv/bin/python src/gold_set.py --dir data/gold            # report
./.venv/bin/python src/gold_set.py --dir data/gold --gate     # exit 0 only when usable
```

The gate additionally requires at least one approved item in **every** split
and at least one correctness item; a judge-only set cannot gate correctness.

## What happens once the gate passes

1. Run `src/eval_harness.py separation-gate` for 9B against 4B. Every split
   must separate by more than the declared effect, and re-runs must stay within
   the 0.025 noise floor. Stability that is merely unproven fails.
2. Only then is live GEPA wired and run.
3. Weight training remains unreachable without a residual model gap in the
   factorial evidence **and** a reviewed, signed `train/TRAINING_JUSTIFIED.md`.

If the separation gate fails, the finding is that the harness does not yet
discriminate — not that the threshold is wrong.
