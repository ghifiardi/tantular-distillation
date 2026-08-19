# Student endpoint validation — rental runbook

**Approved 2026-08-19 as a VALIDATION-ONLY rental.** No trainer, no training
dependencies, no `--confirm-run-v1`. The pod exists to answer five questions and
then be stopped.

Running v1 is a **second, separate decision**, made after reading the report this
produces.

## What is being validated

That `Qwen/Qwen3.5-9B` — the model `train/qlora_9b.yaml` names — can be
served at bf16 and gated. Not the teacher. Not NF4, int4 or FP8. The student
config (`configs/teachers/office-student-9b.yaml`) carries a bf16 repo only, so a
host asking for any other precision fails to resolve rather than substituting.

## Pod spec

| | requirement | why |
|---|---|---|
| GPU | any **Ampere or newer**, ≥40GB | bf16 needs cc 8.0+; 9B is ~18GB of weights before KV cache. **FP8 is not needed** — do not pay for Ada on this rental. |
| driver | **≥ 525**, and **≥ 580** if the image ships CUDA 13 | The driver, not the container, decides which CUDA a kernel may use. This line was missing from the first rental spec and that rental was lost to it. |
| image | a vLLM image matching the driver's CUDA | A CUDA 13 build on a 5xx driver fails to launch after the pull. |
| venv | clean, **no `--system-site-packages`** | That flag caused NCCL ABI mixing (`undefined symbol: ncclCommResume`) on an earlier rental. |

Not ai19. `configs/hosts/ai19.yaml` declares `training_allowed: false`, and
`src/config.py: training_guard()` refuses it both by declaration and by hostname.

## Sequence

Stop at the first failure. Each step is cheap; the expensive one is last.

```bash
# --- on the pod, BEFORE serving --------------------------------------------
./scripts/verify_student_host.sh student-hardware.json      # check 1

# --- on the pod ------------------------------------------------------------
./scripts/serve_student.sh office-student-9b student-serve 2>&1 \
    | tee logs/vllm-student.log                             # check 2's evidence

# --- from the laptop, over the tunnel --------------------------------------
ssh -N -L 8020:localhost:8020 <pod>
export HOST_BASE_URL=http://localhost:8020/v1

./.venv/bin/python src/validate_student_endpoint.py \
    --host student-serve --model office-student-9b \
    --hardware student-hardware.json \
    --serve-log logs/vllm-student.log \
    --out data/gates/student-validation                     # checks 3, 4, 5
```

## The five checks

1. **Hardware meets the gate** — read from the JSON `verify_student_host.sh`
   wrote on the pod. Refuses pre-Ampere and a driver too old for the image.
2. **vLLM loaded bf16** — from vLLM's own log. The API does not report dtype, and
   *passing* `--dtype bfloat16` is what we asked for, not evidence of what
   loaded. No log means check 2 fails. A log showing `float16` or any quantizer
   also fails: either would make the baseline a measurement of precision damage
   that the adapter would then appear to have fixed.
3. **Identity** — `/v1/models` must report the model the config names. This is
   the check that stops the endpoint quietly being a teacher.
4. **One prompt returns valid text** — non-empty, ≥40 chars, not a repetition
   loop. An endpoint that answers with empty completions is exactly how the FP8
   arm failed; it looked healthy from the outside for two rentals.
5. **`run_gates --stage before` completes** — all three gates execute and produce
   a rate.

## Reading check 5

**`BASELINE_MEASURED_BELOW_TARGET` is the expected result and is a PASS for this
validation.** The base student was never trained on Indonesian office work; the
0.95 bar is for a promoted adapter. The gate records the number, marks it
`BASELINE_BELOW_TARGET`, and exits 0.

What would be a real failure of check 5 is exit 2 — a gate that could not run.

No threshold moves as a result of this rental.

## Stop conditions

Stop the pod and do not proceed to training if:

- any of checks 1–4 fails;
- check 5 exits 2 (a gate could not be executed);
- `/v1/models` reports anything other than the configured student.

`src/validate_student_endpoint.py` stops at the first failure by design — check 5
generates 60 completions, and running it against a known-bad endpoint spends
rental time to produce a number nobody should read.

## After a pass

`data/gates/student-validation/VALIDATION.json` holds the five verdicts, the
baseline rates, and `training_authorised: false`. Keep
`gates.before.json` — it is the baseline the `after` run will be compared
against, and `compare` refuses reports from a different config.

Then, and only then, ask for the decision to run v1.
