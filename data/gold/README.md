# Native Indonesian gold set

This directory is the local input boundary for the evaluation harness described
in `docs/CLAUDE_CODE_RSI_MVP_HANDOFF.md`.

## Current status

**BLOCKED: 0 approved production items.**

That is deliberate. The Recursive Self-Improvement Report requires a native,
human-authored Indonesian set that reflects Tantular's real deployment. An
agent must not generate, translate, infer, or silently promote synthetic rows
to satisfy the target count.

The planned set has approximately 300 approved items across:

- `reasoning`
- `instruction_following`
- `knowledge`
- `tool_use`

The exact count is not a pass condition by itself. The set must also have
reviewed provenance, usable verifiers, no train/eval leakage, and enough
coverage to distinguish the deployed 9B and 4B systems stably.

## Data handling

Production records are intentionally not committed:

- `.gitignore` ignores `*.jsonl` outside explicitly approved fixture paths.
- Put the reviewed local file at `data/gold/items.jsonl`, or pass an explicit
  path to the future loader.
- Treat prompts, expected answers, author identity, and review evidence
  according to their source classification.
- Never use this directory as training, prompt-evolution, or synthetic-data
  input. GEPA may score on a held-out slice but its proposer must not see the
  gold answers or verifier internals.

`SCHEMA.md` defines the proposed contract. Claude Code may refine the contract
while implementing `src/gold_set.py`, but changes must preserve the fail-closed
rules in the handoff.

## What may be committed

Only documentation belongs here. Small, clearly synthetic examples for unit
tests live under `tests/fixtures/gold/`; they are never production gold and
must be rejected by a production gate.
