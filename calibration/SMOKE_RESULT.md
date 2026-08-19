# Smoke rental result — 2026-08-19

One L40S, ~1 hour, two pods. Steps 1–6 of the seven-step plan ran. **Step 6
found a defect that would have invalidated the entire v1 `after` gate.**

## Verdict

| step | result |
|---|---|
| 1 CUDA + pinned deps | PASS — all ten pins resolved exactly, including the `tokenizers 0.22.2` ceiling |
| 2 NF4 load | PASS — `BitsAndBytesConfig`, 136.7s including a 19.3GB download |
| 3 LoRA + real TRL path | PASS — 58,195,968 trainable (1.048%), loss 2.1336, grad norm 8.9205, `trainer.train()` completed |
| 4 save + reload | PASS — `adapter_config.json` + weights, base `Qwen/Qwen3.5-9B`, r=32 |
| 5 serve under a distinct id | PASS — `Qwen/Qwen3.5-9B`, `office-student-9b`, `tantular-smoke` |
| 6 the adapter id generates | **text PASS, effect FAIL** |

Total 158.9s for steps 1–4. Library versions are recorded in
`/workspace/runs/smoke/SMOKE.json`.

## The defect

The adapter id returned fluent, correct Indonesian. It was also **byte-identical
to the base model's output**, and every one of the 48 token logprobs was
**bit-identical**. The same weights answered both requests.

The adapter was not at fault: `lora_B` carried real values (max |w| 1.01e-4,
non-zero in all 128 tensors), so the checkpoint had been trained.

Root cause, established from the transformers and vLLM sources rather than
guessed:

- **Training** — `AutoModelForCausalLM` resolves `Qwen3_5ForCausalLM`, whose
  `.model` is a `Qwen3_5TextModel` holding `.layers` directly. PEFT therefore
  wrote keys `base_model.model.model.layers.N.…`
- **Serving** — vLLM resolved `Qwen3_5ForConditionalGeneration`, whose `.model`
  is a `Qwen3_5Model` holding `.visual` and `.language_model`. Its layers are at
  `model.language_model.layers.N.…`

The adapter's paths lack the `language_model.` segment. vLLM loaded it, matched
nothing, bound nothing, and logged `Loaded new LoRA adapter: name
'tantular-smoke'`.

## Why the existing gate missed it

`verify_adapter_served()` checked that the directory was a real PEFT adapter,
that the id differed from every base alias, and that `/v1/models` listed it. All
three were true. **None of them is evidence the LoRA does anything.**

Text comparison would not have caught it either: a small adapter can legitimately
decode to the same string. Only the distribution distinguishes "changed little"
from "never applied".

## Fixes

1. `scripts/serve_student.sh` pins the text-only architecture with
   `--hf-overrides '{"architectures": ["Qwen3_5ForCausalLM"]}'`. vLLM registers
   that class separately and it `SupportsLoRA`, so the served tree matches the
   trained tree. It also skips the vision tower, which a text-only Office
   student never uses. Applied **unconditionally** — before and after must be
   the same architecture, or the baseline is a different model from the thing
   compared against it.
2. `run_gates.py` gained `adapter_changes_the_distribution()`, run at
   `--stage after` before any gate: it probes both ids with `logprobs=1` and
   **fails closed** if every token is bit-identical.

## Still unproven

The fix has not been tested. Step 6 must be re-run on the paused pod:
`--hf-overrides` may not accept a multimodal config's architecture swap, in
which case the fallback is to rewrite the adapter's keys with the
`language_model.` segment inserted.

**Nothing here authorises the v1 run.**

## Incidental findings

- A co-tenanted or wedged GPU passes every static check. The first pod showed
  100% utilisation with 0 MiB used and no visible processes, and failed at the
  first `cudaMalloc` — after a 19.3GB download. `verify_student_host.sh` now
  samples utilisation and attempts a real allocation.
- `torch 2.13.0+cu130` is torch 2.13.0; the build tag is not a version mismatch.
- `serve_student.sh` ran a bare `vllm` and died with exit 127. Worse would have
  been finding a *different* vllm on PATH. Both serve scripts now resolve the
  venv's binaries.
- `transformers` 5.x removed `warmup_ratio`. Found by reading the wheels while
  the pod was busy; `trainer.train()` on the pod then confirmed the fix.
