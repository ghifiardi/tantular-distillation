# First rental — bounded training smoke test

**Scope: seven steps, then stop the pod.** No v1 training run. No promotion. The
decision to train v1 is separate and comes after this passes.

One NVIDIA GPU, **48GB, Ampere or newer. FP8 is not needed** — do not pay for
Ada. Not ai19 (`training_allowed: false`, enforced by `training_guard()`).

**It must be a DEDICATED GPU, not a shared or community instance.** A
co-tenanted card passes every static check — full memory free, no processes,
right capability, right driver — because a container cannot see processes in
other containers. It then fails at the first `cudaMalloc` with
`cudaErrorDevicesUnavailable`. That cost one rental and a 19.3GB download on
2026-08-19; `verify_student_host.sh` now samples utilisation and attempts a
real allocation, so it fails in sixty seconds instead.

## Why this rental exists

`src/train_qlora.py` has never executed a training step. Its GPU imports are
lazy, so everything checkable on a laptop has been checked and the model load,
LoRA attach and checkpoint write have not. The first execution of those should
not be inside a paid v1 run.

Two specific risks this is meant to surface early:

- **TRL API drift.** The smoke and v1 now call the same
  `build_sft_trainer()` helper. The smoke must construct `SFTConfig`,
  construct `SFTTrainer`, and complete `trainer.train()` with `max_steps=1`.
  If TRL 1.10.0 rejects that path, fix the shared helper rather than unpinning.
- **Quantization config.** A 9B that loads without a `quantization_config` is
  bf16, not NF4, and will not fit the memory budget QLoRA assumes.

## Pinned environment

`requirements-train.txt`. Exact pins, not floors — a floor resolves to whatever
was current on the day the pod was made, and two runs a week apart then train on
different libraries.

| package | pin | why this one |
|---|---|---|
| torch | 2.13.0 | `vllm==0.27.1` pins `torch==2.13.0` exactly |
| transformers | 5.15.0 | vLLM needs ≥5.5.3; TRL needs ≥4.56.2 |
| tokenizers | **0.22.2** | transformers 5.15 requires `>=0.22.0,<=0.23.0`, and **there is no 0.23.0 final on PyPI** — only `0.23.0rc0`, then `0.23.1`, which the ceiling excludes. Left unpinned, pip resolves 0.23.1 and fails on the pod. |
| trl | 1.10.0 | needs transformers ≥4.56.2, accelerate ≥1.4, datasets ≥4.7 |
| peft | 0.20.0 | |
| datasets | 5.0.1 | |
| accelerate | 1.14.0 | |
| bitsandbytes | 0.50.1 | NF4 |
| safetensors | 0.8.0 | |
| vllm | 0.27.1 | **endpoint host only** — see below |

**vLLM is not a training dependency.** It lives in `requirements-serve.txt`
because it pulls the entire CUDA 13 runtime (`nvidia-*-cu13`), which a driver
older than 580 cannot run. A training pod reaches the endpoint over HTTP and
needs none of it. Discovered 2026-08-20 on an A6000 pod with driver 570.211.01,
where the training stack itself was perfectly usable.

**Driver, and which wheel to install:**

| driver | training host | endpoint host |
|---|---|---|
| ≥ 580 | default index (`torch 2.13.0+cu130`) | fine |
| 525–579 | install torch from the **cu129** index first, then `requirements-train.txt` | **not usable** — the vllm 0.27.1 wheel is built against CUDA 13 |

There is no cu128 build of torch 2.13.0; cu129 is the lowest available, and
CUDA 12.x builds run on any r525+ driver by minor-version compatibility.

Consistency was checked against each package's own declared constraints on
PyPI, not assumed.

**Build the venv clean, without `--system-site-packages`.** That flag caused
NCCL ABI mixing (`undefined symbol: ncclCommResume`) on an earlier rental.

## The seven steps

```bash
# --- on the pod ------------------------------------------------------------
./scripts/verify_student_host.sh student-hardware.json     # hardware, 60 seconds
python -m venv .venv && ./.venv/bin/pip install -r requirements-train.txt
./.venv/bin/python src/smoke_train.py --out ~/tantular-runs/smoke   # steps 1-4

# --- step 5: serve the adapter under a DISTINCT id -------------------------
./scripts/serve_student.sh office-student-9b student-serve \
    ~/tantular-runs/smoke/adapter tantular-smoke 2>&1 | tee logs/vllm-smoke.log

# --- step 6: prove that id generates ---------------------------------------
curl -s localhost:8020/v1/models | python -c \
  "import json,sys; print([m['id'] for m in json.load(sys.stdin)['data']])"
curl -s localhost:8020/v1/completions -H 'Content-Type: application/json' -d '{
  "model": "tantular-smoke",
  "prompt": "Ringkas kalimat berikut menjadi satu kalimat: ",
  "max_tokens": 64, "temperature": 0}' | python -c \
  "import json,sys; print(repr(json.load(sys.stdin)['choices'][0]['text']))"

# --- step 7 ----------------------------------------------------------------
# stop the pod
```

`src/smoke_train.py` covers steps 1–4 and aborts on the first failure:

1. **CUDA and dependencies** — every pinned package imports, a CUDA device
   exists, compute capability ≥8.0.
2. **NF4 load** — `Qwen3.5-9B` loads 4-bit. A model that comes back with no
   `quantization_config` fails the step; it is bf16 wearing a QLoRA label.
3. **Real TRL path plus one manual optimizer step** — the smoke constructs the
   same `SFTConfig`/`SFTTrainer` path as v1, completes one TRL step, and also
   requires non-zero trainable parameters, finite loss, and a **non-zero
   gradient norm**. The two checks detect different failures.
4. **Save and reload** — the adapter directory must contain exactly what
   `run_gates.verify_adapter_served()` requires (`adapter_config.json` plus
   weights), and the reloaded config must name the configured base model.

Steps 5–6 are the part no local test can prove: that a real PEFT directory,
written by this trainer, loads into vLLM under an id distinct from the base and
answers to that id.

## Stop conditions

Any step failing → stop the pod, do not train. Specifically:

- a dependency will not import, or resolves to a different version than pinned;
- the model loads without a quantization config;
- loss/gradient norm is non-finite or zero, or the real TRL trainer path fails;
- the adapter directory is missing `adapter_config.json` or weights;
- `/v1/models` does not list `tantular-smoke`, or that id returns empty text.

An empty completion is a **failure**, not a quirk. That is exactly how the FP8
arm failed while looking healthy from outside for two rentals.

## What a pass authorises

Regenerating the schema-v2 `RUN_MANIFEST.v1.json`, and asking for the v1
decision. Nothing else. The smoke adapter is garbage — four examples, one step —
and is not a v1 artifact.

`~/tantular-runs/smoke/SMOKE.json` records library versions, GPU, loss, gradient
norm and adapter shape. Copy it back before stopping the pod; it is the evidence
that the training path executes.
