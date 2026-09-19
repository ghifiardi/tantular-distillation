# Gold-set JSONL contract

**Status:** starter contract for `src/gold_set.py`. The loader is not yet
implemented.

Each non-empty line is one UTF-8 JSON object. Unknown fields should be rejected
by default so schema drift cannot silently change what the evaluation means.

## Record

```json
{
  "schema_version": 1,
  "id": "gold::tool_use::0001",
  "split": "tool_use",
  "language": "id",
  "prompt": "Gunakan alat ...",
  "verifier": {
    "type": "schema",
    "config": {}
  },
  "expected": {},
  "score_role": "correctness",
  "source_class": "internal",
  "source": {
    "kind": "human_authored",
    "author_ref": "reviewer-registry-id",
    "authored_at": "2026-09-19",
    "approved_by": "independent-reviewer-registry-id",
    "approved_at": "2026-09-19",
    "evidence_ref": "local-review-record-id"
  }
}
```

## Required fields

| Field | Contract |
|---|---|
| `schema_version` | Integer `1`. |
| `id` | Non-empty, unique, stable identifier. Recommended form: `gold::<split>::<number>`. |
| `split` | One of `reasoning`, `instruction_following`, `knowledge`, `tool_use`. |
| `language` | Must be exactly `id`. Translated English material does not become native gold by changing this field. |
| `prompt` | Non-empty native Indonesian prompt. |
| `verifier.type` | One of `exact_match`, `schema`, `executor`, `judge`. |
| `verifier.config` | Object containing only verifier-specific bounded settings. |
| `expected` | Verifier-specific expected result. It must not be empty for a correctness item. |
| `score_role` | `correctness` or `style`. A `judge` item must be `style`; it can never contribute to correctness pass rate. |
| `source_class` | Existing repository classification. Missing values default to `internal`, never `synthetic`. |
| `source` | Reviewed provenance object; see below. |

## Verifier-specific `expected`

### `exact_match`

```json
{
  "answer": "42",
  "normalization": ["trim", "unicode_nfc"]
}
```

Permitted normalization must be enumerated by the implementation. Never add
fuzzy matching merely to make an answer pass.

An item may add a mechanical reasoning constraint:

```json
{
  "answer": "42",
  "normalization": ["trim"],
  "reasoning_check": {
    "type": "required_claims",
    "claims": ["6 × 7 = 42"]
  }
}
```

This is how the harness can record a false positive: the final answer matches,
but a checkable reasoning claim does not. A final-answer pass must not erase
that evidence.

### `schema`

```json
{
  "json_schema": {
    "type": "object",
    "required": ["tool", "arguments"],
    "additionalProperties": false
  }
}
```

Use a bounded JSON schema. For Office edit output, reuse the semantics of
`scripts/check_edit_contract.mjs`; do not maintain an incompatible duplicate.

### `executor`

```json
{
  "runner": "python",
  "timeout_seconds": 2,
  "tests": [
    {"stdin": "2 3\n", "stdout": "5\n"}
  ]
}
```

The runner must execute in a subprocess with a timeout, denied network, bounded
input/output, and no shell interpolation. Only explicitly allowed runners may
be used.

### `judge`

```json
{
  "rubric": ["alami", "ringkas", "sesuai ragam bahasa"],
  "calibration_set_ref": "native-rater-calibration-v1"
}
```

`judge` is style-only. Until judge-human agreement is calibrated, a judge item
must remain report-only and cannot gate correctness.

## Provenance

Production gate eligibility requires:

- `source.kind == "human_authored"`
- non-empty `author_ref`
- a parseable `authored_at`
- a distinct, non-empty `approved_by`
- a parseable `approved_at`
- non-empty `evidence_ref`
- `source_class != "synthetic"`

An agent may validate these claims but may not create or approve them. A
synthetic fixture uses `source.kind == "synthetic_fixture"` and is always
ineligible for the production gate, even if structurally valid.

## Fail-closed rules

The loader/gate must refuse:

1. no approved production items;
2. duplicate IDs;
3. missing or unknown split/verifier/language;
4. empty provenance or an unapproved source;
5. `judge` with `score_role: correctness`;
6. empty correctness expectations;
7. fixture or synthetic rows presented as production gold;
8. train/eval leakage or disclosure of expected answers to a proposer;
9. missing model identity or missing split in a measurement;
10. a request to turn validation warnings into passing records.

The expected empty-state message from the handoff is:

```text
gold set is BLOCKED: 0 approved items
```
