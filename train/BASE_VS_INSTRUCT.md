# Product stays on INSTRUCT — decided 2026-08-30

**The product is built on the instruct checkpoint: `Qwen/Qwen3.5-9B`, served as
`qwen3.5:9b`.** The Tinker Base path is **not** retargeted. It remains a
preflight-only experiment until a deliberate decision changes it.

Neither option below was taken. The divergence is retained on purpose, with the
two paths scoped differently: instruct is the product, Base is an experiment.

What follows from that, and is binding until revisited:

- **No Base-trained artifact is a product candidate.** A Tinker run can qualify
  infrastructure and measure things about `Qwen/Qwen3.5-9B-Base`, but its
  output cannot be promoted into the product or served to users, because the
  product does not run that checkpoint. The `trained_unvalidated` →
  `peft_unvalidated` → after-gates chain terminates in an experimental result,
  not a shippable adapter.
- **The cross-model guards stay.** `src/model_ids.py`, the expected-model check
  in `run_gates.py compare`, and `required_baseline_model` in
  `baseline_snapshot()` are what keep the two paths from being compared to each
  other. With the divergence deliberately retained, they are load-bearing.
- **The recorded renderer parity stands for the experiment.** `role_colon` on
  `Qwen/Qwen3.5-9B-Base` was verified 2026-08-29 and remains the correct format
  for the Base path. It says nothing about the instruct checkpoint, and would
  have to be re-established if Option 1 were ever taken.
- **`train/TRAINING_BLOCKED.md` is untouched and still controlling.** This
  decision settles which checkpoint, not whether to train. Training remains NOT
  JUSTIFIED.

The rest of this file is the analysis the decision was made against, kept as
the record of what was weighed.

---

# (the open question, as recorded 2026-08-29)

**The Tinker backend trains a different checkpoint, in a different prompt
format, from the one the product serves.** Nothing is wrong with either half on
its own. What does not exist is a decision about which one Tantular Office is
built on, and until that decision is explicit no rental or training spend is
justified — independently of `train/TRAINING_BLOCKED.md`, which still concludes
that training is not justified at all.

This file records the divergence. It does not resolve it.

## The divergence

| | checkpoint | prompt format |
|---|---|---|
| product today | `qwen3.5:9b` via Ollama — `Modelfile.office-9b` is `FROM qwen3.5:9b` plus a SYSTEM prompt | the add-in's chat template, over `/v1/chat/completions` |
| local QLoRA path | `Qwen/Qwen3.5-9B` (`train/qlora_9b.yaml:11`) | same |
| gate endpoint | `Qwen/Qwen3.5-9B` (`configs/teachers/office-student-9b.yaml:17`) | same |
| **Tinker path** | **`Qwen/Qwen3.5-9B-Base`** (`train/tinker_sft_9b.yaml:8`) | **`role_colon`** (`:14`), served through an explicit Jinja template (`:20`) with the `"\n\nUser:"` stop sequence (`:86`) |

A LoRA trained on Base may load on the instruct checkpoint — same
architecture — but the learned delta is relative to Base weights. It would be
neither a valid adapter nor a controlled experiment, and
`docs/TINKER_RUNBOOK.md` already forbids the cross-model comparison.

So a successful Tinker v1 does not drop into the product. It would require
changing the served checkpoint **and** the wire format the add-in speaks.

## How this arose

The independent review of the Tinker integration flagged the `-Base` vs
instruct mismatch as a blocking question. It was resolved by representing the
two as **distinct model identities** — which is correct, and is why the
cross-model guards now exist:

- `src/model_ids.py` — exact id matching, so `Qwen3.5-9B` cannot match
  `Qwen3.5-9B-Base`;
- `run_gates.py compare` — before and after must name the same expected model;
- `freeze_training_run.baseline_snapshot()` — the baseline must be for
  `evaluation.required_baseline_model`, which must equal `base_model`.

Those guards are worth keeping whatever is decided. But separating the
identities deferred the product question rather than answering it, and the
deferral is what now blocks the path.

## Option 1 — stay on instruct; retarget Tinker

Retarget `train/tinker_sft_9b.yaml` to `Qwen/Qwen3.5-9B`, re-verify renderer
parity, regenerate the schema-v3 manifest.

Evidence gathered 2026-08-29 that this is the smaller change:

- `qwen3_5_disable_thinking` is present in the tinker-cookbook 0.5.5 renderer
  registry, so a supported renderer for the instruct checkpoint already exists.
- `Qwen/Qwen3.5-9B` publishes its own `chat_template.jinja` on the Hub.
  `Qwen/Qwen3.5-9B-Base` does not — which is the only reason
  `templates/role_colon.jinja` had to be written.
- These artifacts exist **solely** because of the Base choice and would become
  unnecessary rather than needing rework: `templates/role_colon.jinja`,
  `configs/teachers/office-student-9b-base.yaml`, `serving.chat_template` and
  `evaluation.stop_sequences` in the Tinker config, the `--chat-template`
  plumbing in `scripts/serve_student.sh`, and `TEACHER_CHAT_TEMPLATE` in
  `src/config.py`.
- The promoted corpus suits a thinking-disabled renderer as it stands: all 183
  rows carry `reasoning_chars > 0` with the reasoning stripped, and none
  contains a `<think>` block. The training targets are answers only.
- The whole gate stack, the product baseline, and `office-student-9b.yaml`
  already point at the instruct checkpoint.

Cost: re-run `--verify-renderer` against the instruct tokenizer, regenerate the
preview manifest. No GPU rental. Note that renderer parity for
`qwen3_5_disable_thinking` would be **unverified** until that run happens — the
parity currently recorded is for `role_colon` on Base, and does not transfer.

## Option 2 — move the product to Base

Change what `tantular-office:0.4-9b` serves, and change the format the add-in
puts on the wire.

Blast radius:

- the Ollama model behind the product profile, and `Modelfile.office-9b`;
- the add-in's request format — `role_colon` is not what it sends today;
- serving must pass `templates/role_colon.jinja` explicitly, because the Base
  checkpoint ships no chat template and vLLM would otherwise have none to
  apply;
- the accumulated instruct evidence is retired, including the passing gate
  results in `train/TRAINING_BLOCKED.md` — `indonesian_voice` 0.9500,
  `edit_contract_output` 0.9500, `office_json_contract` 1.0000, and the 10/10
  faithful-editing pilot — which are currently the reason training is not
  justified. A new baseline would have to be established for Base before any
  claim about it could be made.

## What is NOT affected

`train/TRAINING_BLOCKED.md` is unchanged and still controlling. Neither option
lifts it. The evidence that would reopen training is a real observed product
failure — from users, add-in logs, or an approved real Office corpus — and that
is a separate question from which checkpoint the student is built on.

Both questions must clear before a run: the model decision recorded here, and
the justification recorded there.

## Status

**Decided 2026-08-30: the product stays on instruct; the Tinker Base path stays
an experiment.** See the top of this file.

No configuration was retargeted — before the decision, because that would have
made it by default; after it, because the decision was to leave the Base path
as it stands rather than to adopt Option 1.
