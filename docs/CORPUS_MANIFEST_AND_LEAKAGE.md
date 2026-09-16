# Corpus manifests and leakage evidence

`confirmations.leakage_reviewed: true` in an approval record is a person saying
they looked. This is the machinery that makes it possible to look — and to
prove afterwards what was looked at.

See `configs/corpus_manifests/manifest.schema.md` for the artifact shapes.

## The shape of the problem

The training corpora are **gitignored**: `data/raw/` and `data/promoted/` exist
on one machine and in no commit. So a leakage check cannot read them from a
clean checkout, and a manifest is the only way the question can be asked in CI
at all.

They are also **short and templated** — 36 of 183 requests under 120 characters,
drawn from four distinct system prompts. A public unsalted SHA-256 of one is a
membership oracle: anyone can hash a guess and test it. Hence HMAC, and hence
the public/private split.

## What is verifiable, and what is not

| component | status | why |
|---|---|---|
| `system_payload` | available | **overlap expected and non-blocking** — every production case reuses the product's own system prompt |
| `user_payload` | available | the whole prompt, normalized |
| `completion_payload` | available | |
| `canonical_row` | available | |
| `request` | **unverifiable** | needs a versioned extractor to split instruction from document |
| `document` | **unverifiable** | same, and the promoted corpora carry **no document field at all** |
| `expected_target` | **unverifiable** | no structured field in the source |
| `expected_outcome` | **unverifiable** | needs an extractor |
| `full_office_case` | **unverifiable** | requires the components above |

**`unverifiable` is never `clean`.** A component with nothing to compare has
found nothing — and a checker that reported that as clean would be most
confident exactly where it knew least. A required component that cannot be
checked exits 2, not 0.

**A clean `user_payload` does not mean no document was reused.** It means the
whole prompt differs. A reused document inside a rephrased instruction would
pass, and nothing here claims otherwise.

## Exact only

Normalization v1 finds identical text after NFC, line-ending and
whitespace-run normalization, with case and paragraph structure preserved. It
does **not** detect paraphrase, translation, or lightly-edited documents.
Near-duplicate detection is a different technique and is not implemented.

Normalizing harder would be worse, not better: collapse case and paragraphs and
materially different Office documents collide, so the checker starts blocking
legitimate cases while looking rigorous.

## Keys

Supplied externally; this repository generates none. ≥ 256 bits, mode `0600`,
never printed in output, logs or error text. `hmac_key_id` is a label. Rotation
means a new key id and regenerated manifests: historical manifests stay valid
under their historical key but cannot be compared under a new one. **Standard
PR CI never receives the production key**; tests use a key id containing
`fixture`, and the generator refuses to treat such a key as production without
`--fixture` (and refuses `--fixture` with a production-looking id).

## Registration is one-way

A set used to tune or validate the scorer is calibration data forever. It may
never become the held-out set that decides something — not because the bytes
are tainted, but because the decision would be made against material already
seen. `docs/case_sets/REGISTRY.jsonl` is append-only and keyed by **digest, not
name**, so renaming resets nothing.

Editing one character produces a NEW digest that is genuinely unregistered.
That is why a component-overlap check against registered material is still
required: otherwise a one-character edit would launder development material
into a held-out set.

## Production approval is a conjunction

`verify_case_set_approval.production_approval` requires **all** of: a binding
human approval record; a binding leakage-evidence record; required components
clean; ≥ 320 eligible cases; no development or calibration registration; and an
exact metric-contract digest match.

## Still blocked

| blocker | consequence |
|---|---|
| **No versioned document extractor** | `document` and `full_office_case` coverage is unavailable, so an Office held-out set cannot be fully cleared |
| **No production HMAC key** | no production manifest exists; generation waits on human key custody |
| **Corpora are gitignored** | a manifest must be generated on the machine that holds them, and the source digest is the only binding |
| **No near-duplicate detection** | paraphrase and light edits are undetected |

Nothing here was run against a real corpus, and no production manifest exists.
