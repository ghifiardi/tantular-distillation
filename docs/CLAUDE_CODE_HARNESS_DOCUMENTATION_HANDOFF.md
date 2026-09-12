# Claude Code handoff: Harness-aware distillation documentation and deck

Create two polished deliverables that document the completed Tantular harness-aware distillation work:

1. `output/docx/Tantular_Harness_Aware_Distillation_Process.docx`
2. `output/pptx/Tantular_Harness_Aware_Distillation_Process.pptx`

Current date: **10 September 2026**.

## Important starting state

A preliminary DOCX already exists at:

`output/docx/Tantular_Harness_Aware_Distillation_Process.docx`

You may use it as a content and layout starting point, but it is not automatically accepted. Re-render and inspect it. Correct any pagination, numbering, split table rows, stale facts, or awkward wording. No PPTX has been completed yet.

Do not modify or overwrite source research files. Do not run training, generation, a model endpoint, Tinker, or any cloud workload. This is documentation work only.

## Source material

Read and synthesize these sources:

### Research pack

- `docs/Research/Architecting_Intelligence.pdf`
- `docs/Research/The Evolution and Impact of AI Harnesses_ Why the Scaffolding Outperforms the Model.docx`
- `docs/Research/Harness_ Jantung yang Mengubah Prediksi Kata Menjadi Aksi Nyata.docx`
- `docs/Research/Mengapa Harness Lebih Penting daripada Model_ Laporan Briefing YC Paper Club.docx`
- `docs/Research/AI_Harness_Versus_Raw_Models.jpg`

Ignore Word lock files beginning with `~$`.

### Architecture and implementation records

- `docs/HARNESS_AWARE_DISTILLATION_ARCHITECTURE.md`
- `docs/TEACHER_AGNOSTIC_ARCHITECTURE.md`
- `docs/CI.md`
- `train/TRAINING_BLOCKED.md`
- `train/BASE_VS_INSTRUCT.md`
- `configs/harnesses/`
- `configs/models/`
- `configs/experiments/harness-before-weights.yaml`
- `src/harness_distill.py`
- `src/distill_plan.py`
- `src/generate.py`
- `src/pass_manifest.py`
- `src/promote_corpus.py`
- `src/freeze_training_run.py`
- `src/verify_corpus.py`
- `src/verify_harness_identity.py`
- `src/verify_model_identity.py`

### Final Git history

Use the current committed repository history and verify these values before writing:

- PR #2 CI bootstrap merge: `01b88dd`
- PR #1 teacher-agnostic planning merge: `8ded5af`
- PR #4 bounded gate timeout merge: `389ba73`
- PR #3 harness attribution merge: `7ba07ca`
- Final `origin/main` after this work: `7ba07ca`

If current Git history disagrees, use the current committed values and state the discrepancy rather than copying stale values.

## Facts that must be stated accurately

### Research conclusion

Do not repeat “the harness is more important than the model” as a universal law. State the qualified conclusion:

- A harness can unlock substantially more capability from a fixed model on a particular task.
- Model choice remains important.
- Tantular treats the model and harness as a coupled system and measures their contributions separately.
- Teacher size is an experimental variable, not an assumed direction of improvement.

Qualify research numbers:

- The 30% to 95.5% example is specific to the cited ARC-AGI-3 Prime Agent setup, not a universal model multiplier.
- The OpenJarvis 800x figure is a marginal API-cost result in its experiment, not a universal total-cost law.
- Do not use the research pack's 100% AVO, 633-agent/23-million-token, or “local models are always 6-12 months behind” claims as acceptance targets unless primary evidence is present.
- The Turing-to-Von-Neumann framing is a metaphor, not a literal transformation.

### Implemented architecture

Explain:

- model registry;
- harness registry;
- evidence registry;
- four-arm harness-before-weights bakeoff;
- trace-level model and harness identity;
- pass manifest, promotion, freeze and trainer recomputation;
- canonical attribution states;
- provenance audit and readiness block;
- compatibility-key and licence gates;
- fail-closed CI and counted exclusions;
- bounded timeout cleanup.

### Four-arm evaluation

Use this exact conceptual matrix:

| Arm | Model | Harness | Purpose |
|---|---|---|---|
| `student_current` | current student | current harness | product baseline |
| `student_candidate` | same student | candidate harness | measure harness gain |
| `teacher_current` | teacher | current harness | measure residual model advantage |
| `teacher_candidate` | teacher | candidate harness | optional interaction and upper bound |

State:

- `harness_gain = student_candidate - student_current`
- `residual_model_gap = teacher_current - student_candidate`
- if the candidate harness closes the gap, promote the harness and stop;
- weight distillation becomes only a proposal when a significant residual model gap remains.

### Legacy corpus state

Use the verified current facts:

- 136 promoted training traces inspected;
- 136 model-attributed;
- 0 harness-attributed;
- 0 malformed harness blocks;
- harness coverage 0.0;
- `distillation_attribution_ready: false` because no attribution exists, not because it is malformed;
- harness declaration: `required: false`, `attributed: false`, `coverage: 0.0`;
- readiness:
  - `fp8_ready: false`
  - `source_ready: false`
  - `identity_ready: false`
  - `license_ready: true`
  - `harness_ready: true`
- FP8 gate remains UNMET;
- corpus gate remains FAILED under the existing int4 waiver;
- the waiver permits proceeding despite the failure and does not convert it into a pass;
- corpus is synthetic and supports no claim about real Office documents;
- `trainable_as_is: false`;
- `authorizes_training: false`.

### Final verification evidence

Use the final PR #3 evidence:

- CI selected partition: 411 passed, 4 skipped, 84 deselected;
- `requires_local_corpus`: 47;
- `requires_addin`: 37;
- corpus-present suite excluding add-in tests: 458 passed, 4 skipped, 37 deselected; exit 0;
- Git status was empty before and after that run;
- all five reproduced semantic defects refused at the exact reviewed head before merge.

Also explain that the add-in partition is not verified:

- CI explicitly deselects it;
- direct `devServerCancellation.test.mjs` remained active beyond 704 seconds;
- it uses ephemeral ports, so the week-old dev server was cleared as a cause;
- the bounded gate cleanup now returns a fail-closed verdict rather than hanging forever;
- the upstream test itself still requires repair in `LLM-Indonesia`.

### Current blockers

State clearly:

1. `tantular_office_addin` and its `package-lock.json` are not published on a reproducible stable ref. This blocks harness prompt verification and add-in CI coverage.
2. Model identity is not qualified: registry specs remain `digests_verified: false`, architecture signatures are unpinned, and the exact Qwen3.5-9B instruct snapshot has not been verified.
3. No real product capability gap justifies training. `train/TRAINING_BLOCKED.md` remains controlling.

Do not imply that training is ready or authorized.

## DOCX specification

Create an internal engineering process document, approximately 10-14 pages.

### Required structure

1. Title page
2. Executive summary
3. Purpose and scope
4. Research conclusion and qualifications
5. Harness-before-weights design principle
6. System architecture
7. Model and harness identity
8. Safe generation contract
9. Provenance through trace, pass, promotion, freeze and trainer
10. Canonical attribution states
11. Readiness and authorization
12. Verification and CI evidence
13. Defects caught during review
14. Future bakeoff operating procedure
15. Next milestones
16. Repository commands
17. References and evidence

### DOCX design

- Professional internal engineering report.
- A4 or US Letter portrait.
- Black title and headings.
- Aptos, Arial, or another professional sans-serif font.
- Dark navy table headers with white text; alternating pale rows; light gray borders.
- Page numbers in the footer.
- Avoid decorative callout boxes.
- Use tables only for comparisons, controls, evidence matrices and milestones.
- Do not split a table row across pages.
- Avoid an almost-empty page caused by one orphan paragraph or table fragment.
- Number procedures from 1 rather than continuing numbering from earlier lists.
- Keep commands in monospaced text.
- Include document metadata: title, subject and author `Tantular Engineering`.

### DOCX visual QA

Render the DOCX through Microsoft Word or a reliable headless renderer, export to PDF, then inspect every rendered page. Fix:

- clipping or overflow;
- split rows;
- repeated or missing table headers;
- orphan lines;
- incorrect numbering;
- overly dense pages;
- large unintended blank areas;
- broken Indonesian characters or file paths.

The final output directory must contain only the DOCX deliverable, not QA PDFs or page images.

## PPTX specification

Create a concise 12-slide executive engineering deck in 16:9.

### Slide outline

1. **Harness aware distillation** - title and current status
2. **Why the process changed** - model-only attribution could not explain system performance
3. **Model and harness as a coupled system** - qualified research conclusion
4. **Harness before weights** - the decision sequence
5. **Four-arm evaluation** - editable matrix and gain calculations
6. **System architecture** - evidence, registry, generation, promotion, freeze, trainer and audit
7. **Provenance chain** - trace to pass to promotion to freeze to trainer
8. **Fail-closed controls** - licence, compatibility, prompt identity, append guard and canonical declaration
9. **Readiness of the legacy corpus** - five readiness components; harness remains neutral
10. **What review caught** - the most important semantic defects and fixes
11. **Verification and remaining coverage gap** - CI/local evidence plus add-in limitation
12. **Next milestones** - publish add-in, qualify model identity, verify prompts, observe real gap, run bakeoff, consider training

### PPTX design

- Modern technical design, not a dashboard of cards.
- White or very light background, dark navy typography, one restrained orange accent.
- Minimal title slide.
- At least 32 pt slide titles and 18 pt body text.
- Use flat compositions, editable tables and simple editable diagrams.
- Do not turn every point into a separate rounded card.
- Avoid slogans and inflated wording.
- Use direct titles that name the subject.
- Limit each slide to one main purpose.
- Use the supplied infographic only if a slide explicitly critiques or contextualizes it. Do not repeat its unsupported claims as evidence.
- Add concise speaker notes with source paths or arXiv identifiers for claims.

### PPTX visual QA

- Export the deck to PDF or slide PNGs through PowerPoint or a reliable renderer.
- Inspect every slide at full size.
- Confirm no clipping, overlap, tiny text, broken diagrams or table overflow.
- Confirm all tables and diagrams remain editable.
- Confirm slide count is exactly 12.
- Confirm no placeholder text, internal tool output or encrypted payload appears.

## Final acceptance

Before reporting completion:

- Open and render both final files.
- Verify the DOCX page count and all pages visually.
- Verify the PPTX has exactly 12 slides and inspect every slide.
- Confirm both files open successfully in Microsoft Office.
- Confirm factual numbers and commit hashes against the repository.
- Confirm only the final DOCX and PPTX are in their output directories.
- Do not commit the artifacts unless explicitly asked.

Final response must provide the two absolute file paths and summarize any remaining factual limitations. Do not claim the add-in tests or model identity are verified.
