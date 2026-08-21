# Smoke rental result — round 1, 2026-08-19

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
| 6 the adapter id generates | **FAILED first, PASSES after the fix** |
| 7 stop the pod | done |

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

## Confirmed

The converted adapter was served and step 6 passed. All 256 tensors were
rewritten and verified byte-identical; vLLM bound them, and the adapter's output
now DIFFERS from the base's on the probe prompt:

    base    ...ditunda pekan depan karena data belum lengkap.
    adapter ...ditunda pekan depan karena data realisasi belum lengkap. ...

A one-step adapter over four examples changing the output at all is more than
was required — the gate only demands a distributional difference.

`train_qlora.py` now runs the conversion itself after training and instructs
serving the converted directory, so a v1 run cannot repeat this. `RUN.json`
records both digests: the trained adapter and the key-converted copy served to
the gates.

**Nothing here authorises the v1 run.** Steps 1-7 prove the training path
executes and that an adapter can be measured. They say nothing about whether
training on the promoted corpus produces an adapter worth promoting.

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


---

# Qualification round 2 — 2026-08-20/21

Round 1 proved the training path and found the adapter-binding defect. Round 2
re-qualified everything against the CHANGED config (expanded `target_modules`)
and on DIFFERENT hardware, because neither carries over from an earlier run.

Two hosts, both RTX A6000 48GB:

| | Pod A — training | Pod B — endpoint |
|---|---|---|
| driver | 570.211.01 (max CUDA 12.8) | 580.159.04 (CUDA 13.0) |
| torch | 2.13.0+**cu129** | 2.13.0+cu130 |
| stack | `requirements-train.txt` | `requirements-serve.txt`, vLLM **0.27.1** |

The split CUDA builds are deliberate: the hosts communicate over HTTP only, and
the driver dictates the build tag while the pinned RELEASE is identical.

## Results

| check | result |
|---|---|
| Pod A smoke, steps 1–4 | **PASS** |
| expanded LoRA targets attach | **PASS — 80,216,064 trainable** (was 58,195,968) |
| NF4 load | `BitsAndBytesConfig` |
| real TRL `trainer.train()` | completed; loss 2.1266, grad norm 8.9728 |
| Pod B serves base bf16 | `Qwen/Qwen3.5-9B`, `office-student-9b` |
| converted adapter binds | **PASS** — `Loaded new LoRA adapter`, all three ids served |
| **adapter is APPLIED** | **PASS** — output differs from base; check exit 0 |
| final dry run | exit 0 (verified locally; the Pod A run through the tunnel was reported by the operator and not observed in-session) |
| **v1 training** | **NOT PERFORMED. `--confirm-run-v1` never used.** |

The binding evidence, in full — same prompt, greedy decoding:

    base    ...ditunda pekan depan karena data belum lengkap.
    adapter ...ditunda pekan depan karena data realisasi belum lengkap. ...

The expanded targets were confirmed by arithmetic BEFORE the run and by
measurement after: 8 full-attention layers x 4 projections + 32 layers x 3 MLP
+ 24 linear-attention layers x 3 projections = 80,216,064.

## Defects this round found

Every one of these would have cost paid time or produced a false result:

1. **vLLM bundled into the training requirements.** It pulls the CUDA 13
   runtime, which a driver-570 host cannot run — turning a perfectly usable
   training pod into an unusable one. vLLM is not a training dependency; the
   gates reach the endpoint over HTTP. Split into `requirements-serve.txt`.
2. **A broken partial install.** `accelerate` was present as a version number
   with only one subpackage on disk, on an NFS-backed volume. Imports failed
   with a message pointing at `AutoModel`, three layers from the cause. Repaired
   and now checked per-module after install.
3. **The endpoint probe went through the corpus generator**, which enforces
   split-manifest membership — a rule about corpus attribution, not smoke
   probes. It reported a healthy endpoint as broken and stopped before the
   baseline gates.
4. **`office_json_contract` could hang forever.** `node --test a b c` runs files
   in parallel; the add-in's bridge test waits a fixed 300ms for its worker's
   ready banner and, under contention, waits forever. The gate produced no
   verdict for 15 minutes while every file passed in isolation. Now serialised
   with a per-file timeout: a hang becomes a loud failure.

## Two host failures, neither ours

Both passed every static check and would have been caught by
`verify_student_host.sh` in sixty seconds:

- an L40S at 100% utilisation with 0 MiB used and no visible processes — busy
  for a co-tenant, failing at the first `cudaMalloc` after a 19.3GB download;
- an RTX PRO 6000 Blackwell, idle, matching libcuda and kernel module, where
  `cuInit` returned **999** and `cuDeviceGetCount` reported 0 devices.

`nvidia-smi` speaks NVML and succeeded on both. Only a real allocation, or
`cuInit` directly, distinguishes a working GPU from a broken one.

## What is still unknown

Everything above concerns plumbing. **Nothing here says whether 136 mechanically
promoted traces can lift Indonesian voice to 0.95 without regressing the edit
contract.** The v1 run was made measurable, not likely to succeed.


---

# v1 attempt 1 — 2026-08-21: ABORTED, no training, no valid baseline

`--confirm-run-v1` was used with explicit approval. **No training ever started
and no adapter was produced.** Every attempt stopped in the `before` gates, and
the trainer refuses to train past an unrunnable gate — so the cost was pod time,
not a wasted training run or, worse, a plausible-looking false result.

Hosts: two RTX A6000 48GB, drivers 580.159.04 (train) and 580.159.03 (serve),
both on torch 2.13.0+cu130, vLLM 0.27.1 on the endpoint.

## Six aborts, one omission

Every one of these is the same root cause: **the gates had never generated
against a live model.** All 100+ tests drive them from `--traces` fixtures, so
the entire live generation path — prompt handling, protocol, parsing — was
unexercised code that looked covered.

| # | abort | cause |
|---|---|---|
| 1 | `family '' is not in the split manifest` | split assignment is a CORPUS rule; held-out eval prompts belong to no family by design |
| 2 | egress refused 20 prompts | eval prompts were unclassified, so defaulted to `internal`; a rented endpoint is `external` |
| 3 | `KeyError: 'family'` | the trace record joins on family; eval items have only an id |
| 4 | `bridge.test.mjs did not finish in 300s` | a previous timeout left orphaned node workers that contended with the retry |
| 5 | 20 malformed traces, `no final channel` | normalized-harmony is the TEACHER's protocol; Qwen3.5 emits no channels |
| 6 | baseline scored **0.0000** on both gates | the model returned an English "Thinking Process:" preamble as `content`, and the scorer graded REASONING as the ANSWER |

Abort 6 is the one that matters most, because it did not look like a failure.
The run completed the baseline, wrote its reports, and continued to training. A
0.0000 baseline is not obviously wrong — a base model with no Indonesian office
training is expected to score low. It was only wrong on inspection: 38 of 40
voice items failed on "indonesian ratio 0.35", which is not a voice problem at
all. Had it stood, every after-gate would have compared against a floor of zero,
where any output whatsoever is an improvement and "no regression" is trivially
satisfied.

## Fixes

- `--eval-prompts`: skips split assignment, stamps `split: eval-only` and
  `corpus_role: held_out_eval`, and REFUSES prompts carrying a family, so it
  cannot become a way to bypass split rules for corpus.
- Eval prompts classified `source_class: synthetic` — truthfully, since they
  were authored for the purpose and reuse nothing. Deliberately not
  `--egress-approval`, which suppresses the question rather than answering it.
- Eval items use their own id as the join key, and an item without an id is
  refused: a completion that cannot be attributed is not a measurement.
- Gate timeouts kill the whole process group.
- `--protocol {harmony,chat}`, default harmony so corpus generation is
  unchanged; the gates use chat, which is also how the training data was
  rendered and how the add-in talks to the model.
- Chat requests send `enable_thinking: false` and **fail closed** if the
  endpoint rejects it. The first version of that patch retried with thinking
  enabled, which would have silently restored the invalid baseline.

## What is still true

The training path, expanded LoRA targets (80,216,064) and adapter binding all
remain verified from qualification round 2. What does NOT exist is **a valid v1
baseline**. The gates have still never produced one, and no comparison can be
made until they do.


---

# Valid v1 baseline — 2026-08-21

Measured with `run_gates run --stage before` alone. No trainer, no
`--confirm-run-v1`. Artifacts in `data/gates/v1-baseline/`.

## Trace integrity

| check | result |
|---|---|
| voice traces | 40/40 |
| edit traces | 20/20 |
| `thinking_disabled` | `True` on every trace |
| protocol | `plain-chat` on every trace |
| empty completions | 0 |
| completions containing a reasoning preamble | 0 |

## Rates

| gate | rate | threshold | verdict |
|---|---|---|---|
| office_json_contract | 1.0000 (382/382) | 0.98 | MET_TARGET (model-independent) |
| edit_contract_output | 0.9500 (19/20) | 0.90 | MET_TARGET |
| indonesian_voice | 0.9500 (38/40) | 0.95 | MET_TARGET |

Report verdict: **`BASELINE_MEETS_TARGET`**, `below_target: []`.

## THE FINDING: the gates are saturated

**The untrained base model already meets every threshold.** This is the opposite
of what the whole gate design assumed — `BASELINE_MEASURED_BELOW_TARGET` was
built as the expected outcome, on the reasoning that a base model with no
Indonesian office training would score far below 0.95.

Consequences for v1, stated plainly:

- **These gates cannot show that v1 helps.** There is no room above the bar for
  improvement to register. They can only show that v1 does not hurt.
- **Indonesian voice sits EXACTLY at its threshold.** One item flipping is
  0.025, which takes 0.9500 to 0.9250 and fails the after gate. v1 must not
  lose a single voice item to be promotable.
- The promotion criteria are therefore an extremely tight *no-harm* test, not a
  demonstration of value.

The available headroom is precise and small: the two voice failures are both
TERMINOLOGY — `voice::0013` (backup, user) and `voice::0037` (maintenance) —
which is exactly what fine-tuning on Indonesian office material might fix. The
best possible outcome on this gate is 40/40; the worst promotable one is 38/40.

The edit gate has one failure of 20, at a threshold of 0.90, so it has one item
of slack in each direction.

**This is a question about what v1 is for, not an engineering problem.** The
plumbing works and the baseline is valid. Whether to spend GPU time on a run
whose best case is "two terminology items and no regressions" is a product
decision.
