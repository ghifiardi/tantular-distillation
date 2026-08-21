# Why v1 training is not justified — 2026-08-21

**Conclusion: no model capability has been demonstrated that Tantular must
provide and `Qwen/Qwen3.5-9B` cannot.** Training is therefore not justified on
the present evidence. This is a finding, not a blocker to route around.

## What was measured

All three evaluations, on the base model, over the same endpoint and protocol
(`plain-chat`, `enable_thinking: false`):

| evaluation | base result | threshold |
|---|---|---|
| indonesian_voice | 0.9500 (38/40) | 0.95 |
| edit_contract_output | 0.9500 (19/20) | 0.90 |
| office_json_contract | 1.0000 (382/382) | 0.98 |
| faithful editing pilot | 10/10 | none set |

The base passes every one. **No model failure exists that training would
address.** A training run could only hold or regress these numbers, and
`indonesian_voice` sits exactly on its threshold, where a single item is 0.025.

## What Tantular still differentiates

As a PRODUCT, not as a model:

- direct Word / Excel / PowerPoint integration;
- a real edit-contract parser and application path;
- intent routing;
- provenance, privacy and data-egress control;
- local or controlled deployment.

None of these require a fine-tuned model. They are the product, and they work
against any competent base.

## What would change this conclusion

A real, observed failure — not a constructed one:

1. failures reported by users of the add-in;
2. failures visible in add-in logs;
3. failures on an approved real Office corpus, which has never been used here —
   every corpus and eval in this repository is `synthetic`.

Only then: build an eval around that failure, confirm the base actually fails
it, and train if the corpus plausibly fixes it.

**Do not** author more eval items hoping to find a gap, and do not harden
existing ones until the base fails. Both were explicitly ruled out on
2026-08-21, and both would manufacture a justification rather than find one.

## What the work produced anyway

The pipeline is qualified end to end and is not wasted:

- the training path executes — NF4 load, LoRA attach, real TRL, save, reload;
- expanded LoRA targets attach at 80,216,064, matching projection exactly;
- a converted adapter genuinely BINDS in vLLM and changes output;
- the gates fail closed on a missing endpoint, a wrong model, an inert adapter,
  a hanging suite, and reasoning scored as an answer;
- three evaluations exist with recorded baselines.

If a real failure appears, training can start from a verified position rather
than from scratch.

## The honest summary

Six aborted attempts, two qualification rentals and three evaluations produced
one durable answer: **on everything we have been able to measure, the base model
is already good enough.** That answer cost far less than a fine-tune that
improved nothing and would have been difficult to disprove.
