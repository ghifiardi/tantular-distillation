# Synthetic gold-loader fixtures

These records exist only to implement and test `src/gold_set.py` and the
verifier layer from `docs/CLAUDE_CODE_RSI_MVP_HANDOFF.md`.

- Every structurally valid row is marked `source_class: synthetic` and
  `source.kind: synthetic_fixture`.
- They are **never** production gold, do not count toward the proposed
  300-item set, and must be rejected when a production gate is requested.
- Files prefixed `invalid_` each isolate one required fail-closed behavior.
- `false_positive.synthetic.jsonl` supplies a right-answer/wrong-reasoning test
  case for the verifier implementation. Its `fixture_model_output` field is
  test metadata, not part of the production record schema; the test should
  remove that field before invoking strict gold-record validation.
