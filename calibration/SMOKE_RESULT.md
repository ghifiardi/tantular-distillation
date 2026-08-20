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

**Attempt 1, tried and REJECTED.** Pinning the text-only architecture with
`--hf-overrides '{"architectures": ["Qwen3_5ForCausalLM"]}'` swaps the model
class but not the weight-name mapping. The architecture did resolve —
`Resolved architecture: Qwen3_5ForCausalLM` — and then the base itself failed to
load:

```
ValueError: There is no module or parameter named 'language_model' in
Qwen3_5Model
```

The weights on disk are named `model.language_model.…`, so the text-only class
cannot read them. This is the mirror image of the adapter problem, and it rules
the approach out rather than leaving it as an option.

**Attempt 2, the actual fix.** `src/convert_adapter_for_vllm.py` rewrites the
adapter's keys, inserting the `language_model.` segment so they match the served
tree. Names only: every tensor is copied and then verified byte-identical after
the write, because a converter that silently altered weights would be a worse
version of the bug it exists to fix. It is idempotent, and it refuses an adapter
whose layout it does not recognise rather than reporting a vacuous success.

**The gate, regardless of fix.** `run_gates.py` gained
`adapter_changes_the_distribution()`, run at `--stage after` before any gate: it
probes both ids with `logprobs=1` and **fails closed** if every token is
bit-identical.

## Still unproven

The conversion has not been served. Step 6 must be re-run: convert the smoke
adapter, serve the converted directory, and confirm the logprobs differ.

**Nothing here authorises the v1 run.**

## A separate finding: the target modules miss most of the model

The failed load printed the served parameter list, which shows Qwen3.5-9B is a
**hybrid**: only 8 of 32 layers have `self_attn`. The other 24 use `linear_attn`
(`in_proj_qkvz`, `out_proj`, `A_log`, `dt_bias`).

That reconciles exactly with the adapter's 128 `lora_A` tensors:

    8 layers x 4 attention projections (q,k,v,o)  =  32
    32 layers x 3 MLP projections (gate,up,down)  =  96
                                                     128

So `target_modules` in `train/qlora_9b.yaml` — `q_proj, k_proj, v_proj, o_proj,
gate_proj, up_proj, down_proj` — reaches attention in only a quarter of the
layers, and the 24 linear-attention layers get LoRA on their MLP alone. This is
not a bug and does not block anything; it is a config written for a dense
architecture applied to a hybrid one. Worth a decision before v1, separately
from the serving fix.

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
