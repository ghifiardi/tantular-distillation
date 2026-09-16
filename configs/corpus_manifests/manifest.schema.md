# Corpus manifest schema (v1)

Two artifacts per corpus, and the split is the whole point.

| artifact | committed? | holds |
|---|---|---|
| **public** | yes | file digests, counts, classifications, attribution summaries, the private artifact's digest and Merkle root |
| **private** | **no** | per-row HMAC identities |

Neither holds corpus text.

## Why the split

The promoted corpora are short, templated requests — 36 of 183 under 120
characters, drawn from four distinct system prompts. A **public unsigned
SHA-256 of one is a membership oracle**: anyone can hash a guess and test it.
So row identities are HMAC-keyed and stay out of the repository, while the
public manifest carries enough to prove it describes *those exact bytes*
(`raw_sha256`) and that the private half has not been swapped
(`private_manifest_sha256`, `private_merkle_root`).

The cost is real and must be stated rather than discovered: **a case author
without the key cannot self-check for leakage** and must go through CI or a
keyholder.

## Public manifest

```yaml
schema_version: 1
normalization_version: 1
privacy_scheme: hmac-sha256
hmac_key_id: null                    # set at generation; never secret material

source:
  kind: machine_local_gitignored
  repository: ghifiardi/tantular-distillation
  checkout_head: null                # PROVENANCE ONLY
  checkout_head_is_not_a_binding: true
  logical_path: null
  bytes: null
  mtime_utc: null
  rows: null
  raw_sha256: null                   # THE binding

corpus:
  role: training                     # training | evaluation | calibration | development
  source_classes: {}
  model_identities: []
  harness_identities: []
  harness_attribution_status: unattributed
  private_manifest_sha256: null
  private_merkle_root: null

components:
  system_payload:     {available: null,  status: unverified}
  user_payload:       {available: null,  status: unverified}
  completion_payload: {available: null,  status: unverified}
  canonical_row:      {available: null,  status: unverified}
  request:            {available: false, status: unverifiable, reason: no_versioned_extractor}
  document:           {available: false, status: unverifiable, reason: no_versioned_extractor}
  expected_target:    {available: false, status: unverifiable, reason: source_has_no_structured_field}
  expected_outcome:   {available: false, status: unverifiable, reason: no_versioned_extractor}
  full_office_case:   {available: false, status: unverifiable, reason: required_components_unavailable}
```

### `checkout_head_is_not_a_binding`

Stated in the artifact, not just in prose. `data/promoted/` is gitignored, so
**no commit contains or reproduces those bytes**. The checkout head records
where the working tree stood; the binding is `raw_sha256`. A reader who
confuses the two would believe a commit could reconstruct the corpus.

### `status`, and why `unverifiable` is not `clean`

| status | meaning |
|---|---|
| `unverified` | the generator has not run yet |
| `available` | identities exist and can be compared |
| `unverifiable` | **no identity can exist**, with a `reason` |

A component that cannot be compared must never read as one that was compared
and found clean. That distinction is the difference between a leakage gate and
a gate-shaped hole: a checker with nothing to compare would otherwise pass
every time, most confidently when it knew least.

`request` and `document` are `unverifiable` because splitting a prompt into
instruction and document needs a **versioned extractor** — and guessing where
one ends and the other begins is precisely the heuristic this milestone
refuses to ship. `expected_target` has no structured field in the source at
all.

## Private manifest

```yaml
schema_version: 1
normalization_version: 1
privacy_scheme: hmac-sha256
hmac_key_id: corpus-hmac-v1
source_raw_sha256: ...
rows:
  - source_index: 1
    stable_id_hmac: ...
    components:
      system_payload: ...
      user_payload: ...
      completion_payload: ...
      canonical_row: ...
merkle_root: ...
```

No plaintext field appears. `source_index` is a position, not content.

## Keys

Supplied externally, never generated here. At least 256 bits, mode `0600` or
stricter, not group- or world-readable. `hmac_key_id` is an identifier and
carries no secret material. **Rotation creates a new key id and a new manifest
version**; historical manifests stay valid under their historical key but
cannot be compared under a new one until regenerated. Standard PR CI never
receives the production key; tests use an obviously non-production fixture key.

## What exact hashing does not do

It finds **exact** matches after normalization v1. It does **not** detect
paraphrase, translation, or lightly-edited documents. A clean result means
"no identical normalized component was found", never "no related material
exists". Near-duplicate detection is a different technique and is not claimed.
