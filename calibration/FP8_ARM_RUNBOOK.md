# Running the FP8 treatment arm

> **FROZEN 2026-08-18 — DO NOT RENT A GPU AGAINST THIS RUNBOOK.**
>
> Two rentals (~$3.35 total) produced no arm. Hardware, driver, stack and
> pipeline are all now correct and gated; the blocker is that vLLM has no
> native implementation for this checkpoint's `TransformersMultiModalForCausalLM`
> architecture and emits only `<|eom|>` tokens with no text.
>
> **Precondition for any future rental:** a free or local runtime must first be
> shown to produce a valid answer channel from ONE prompt. Without that
> evidence, renting repeats the same failure. The steps below are correct for
> the infrastructure and should not be re-derived — but they are on hold.

The pre-registered study in `acceptance.yaml` has two arms. Only the baseline
(int4) was ever generated, so the study has **never been run** — not failed, not
inconclusive. This runbook produces the missing arm and the verdict.

Everything below is prepared and tested. The blind-review harness was exercised
end to end against a simulated treatment arm before this was written.

**Scope:** 52 synthetic calibration prompts. Not the 260-prompt corpus — the
study measures runtime quality loss, and the calibration set is where the
baseline still has headroom to move (`refusal` 0.0192, `constraints_ok` 0.9706;
the v3 corpus sits at ceiling and would compare vacuously).

---

## Before you start

**Use the normalized baseline.** `data/calibration/int4/traces.jsonl` predates
the normalized protocol: it has no `prompt_sha256`, and its prompts were
rendered by the server. Comparing it against a vLLM arm compares two differently
worded questions. The correct baseline is:

    data/calibration/int4-normalized/traces.jsonl

`blind_review.py` refuses the old arm by name rather than letting it through.

**Card choice is load-bearing.** Rent an **RTX 6000 Ada** or **L40S** (48GB,
~$0.74–1.00/hr). Do **not** take an RTX A6000 at $0.33/hr: it is Ampere, has no
FP8 path, and would silently serve int4 — producing a "treatment arm" identical
to the baseline. `blind_review.py prepare` aborts if both arms report the same
quantization, but catching it after an hour of rental is the expensive way.

**DRIVER >= 580 IS A HARD PREREQUISITE — check it before renting.** This is the
constraint that cost a whole rental on 2026-08-17. Compute capability 8.9 says
the *silicon* can do FP8; it says nothing about whether the *driver* can run
current vLLM, whose kernels are built against CUDA 13.

An L40S — correct card, cc 8.9, every hardware check green — on driver
**570.124.06** got all the way through setup, a clean vLLM install and the model
download, then died at engine init with:

```
cudaHostGetDevicePointer failed: CUDA driver version is insufficient for CUDA runtime version
```

Torch was fine throughout (uv resolved cu129; CUDA 12.x minor-version
compatibility covers it), so torch imported, allocated GPU tensors and named the
device correctly. Only vLLM's own extension failed, and only at the last step.

**Filter pods on the template reporting CUDA 13 / driver 580+.** `nvidia-smi`
shows both. `verify_fp8_host.sh` now refuses anything below 580 in seconds,
before a single byte of model is downloaded.

**No egress approval needed.** All 52 calibration prompts are
`source_class: synthetic`, so the external-host gate passes without
`--egress-approval`. Do not point this at real Office material.

---

## Order of operations

Each gate is placed so the cheapest rejection happens first.

| # | step | needs | rejects |
|---|---|---|---|
| 0a | `nvidia-smi` over SSH | nothing | wrong card or driver, in seconds |
| — | rsync the working tree | SSH key | — |
| 0b | `verify_fp8_host.sh` | repo | writes `hardware.json` |
| 0.5 | clean venv + `verify_stack.sh` | repo | broken stack, before any model bytes |
| 1 | serve | working stack | fp8 fallback |
| 2-4 | `run_fp8_arm.sh` | ready server | five further gates |

**Do not download the model until 0.5 passes.** Every failure in the 2026-08-17
rental was detectable before that point.

---

## Step 0a — reject a wrong pod before transferring anything

`verify_fp8_host.sh` is the full gate, but it arrives with the rsync — so it
cannot be the first check. This one needs nothing on the pod and takes seconds:

```bash
ssh -p <port> -i <key> root@<host> \
  "nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv"
```

Require **cc 8.9+ AND driver major ≥ 580**. If either fails, destroy the pod and
pick another — before the SSH key setup, before the rsync, before anything.

Expected good output:

```
NVIDIA L40S, 8.9, 580.65.06, 46068 MiB
```

## Step 0b — prove the card can do FP8 (after the rsync)

```bash
./scripts/verify_fp8_host.sh data/calibration/fp8/hardware.json
```

**`preflight.py` cannot do this.** It reports `quantization` from
`configs/hosts/<host>.yaml` — the config's claim, not a measurement. Point it at
an A6000 while using the `rented-48gb` config and it prints `fp8` for a card with
no FP8 silicon. The run completes, the numbers look plausible, and the
"treatment arm" is int4 compared against itself.

This reads the hardware: compute capability must be **8.9+** (Ada / Hopper), and
A6000 / 3090 / A100 / V100 / T4 are refused by name. Verified against each:

| card | cc | result |
|---|---|---|
| RTX A6000 | 8.6 | refused |
| RTX 3090 | 8.6 | refused |
| A100 | 8.0 | refused |
| RTX 6000 Ada | 8.9 | accepted |
| L40S | 8.9 | accepted |
| H100 | 9.0 | accepted |

## Step 0.5 — build the environment, and prove it works

Measured on 2026-08-17: both of these bit, and both cost more than the checks do.

```bash
cd /workspace/tantular-distillation

# Clean venv. NEVER --system-site-packages: it mixes the container's torch with
# the installed one and yields "undefined symbol: ncclCommResume", an NCCL ABI
# mismatch that no package pinning repairs.
python3 -m venv .venv
./.venv/bin/pip install --upgrade pip uv

# Let uv pick the torch backend from the detected driver. Do not pin torch by
# hand and do not reuse the container's.
VIRTUAL_ENV=$PWD/.venv ./.venv/bin/uv pip install vllm --torch-backend=auto

./scripts/verify_stack.sh
```

`verify_stack.sh` checks venv isolation, a **real GPU allocation** (not just
`is_available()`, which returns True on a driver that then refuses work), the
vLLM import, `pip check`, and — the step that actually failed — whether vLLM's
**compiled CUDA extension** loads. It prints the `LD_LIBRARY_PATH` to export.

Also note the repo transfer: `rsync` the working tree rather than cloning, or
the pod gets whatever is on `origin/main` instead of your local work.

## Step 1 — TERMINAL A: serve the teacher at FP8

`serve_teacher.sh` ends in `exec vllm serve`. **It blocks and never returns.**
Give it its own terminal and leave it open for the whole run — or start it
detached with `setsid nohup`, which survives the browser terminal closing.

```bash
cd /workspace/tantular-distillation
source .venv/bin/activate
export HF_HOME=/workspace/hf-cache
export LD_LIBRARY_PATH=$PWD/.venv/lib/python3.12/site-packages/nvidia/cu13/lib:$PWD/.venv/lib/python3.12/site-packages/nvidia/cuda_runtime/lib
./scripts/serve_teacher.sh muse-glimmer rented-48gb 2>&1 | tee /tmp/vllm.log
```

**Port:** the RunPod PyTorch template runs **nginx on 8001**, the repo's default.
It even answers `HTTP 200` on `/v1/models` with an HTML page. `configs/teachers/
muse-glimmer.yaml` must use a free port — **8010** works.

**Quantization:** do **not** pass `--quantization fp8`. This checkpoint declares
`compressed-tensors` / `FP8_BLOCK`, and forcing the flag makes vLLM refuse:
*"Quantization method specified in the model config (compressed-tensors) does not
match the quantization argument (fp8)"*. Letting the checkpoint decide is also
better evidence — gate 2 then verifies what vLLM independently selected rather
than echoing our own argument.

Keep the log. It is the only evidence of what vLLM actually selected, and gate 2
refuses to proceed without it.

Serves on port 8010, `tensor-parallel-size 1`. Serve one teacher only — two 30B models do not co-resident in 48GB at FP8, and dropping
both to int4 to fit defeats the entire purpose of the run.

Expect a long first load while weights download. Wait until the server reports
it is ready before starting Terminal B.

## Steps 2–4 — TERMINAL B: the gated run

```bash
export RUNPOD_POD_ID="<your-pod-id>"

./scripts/stop_pod.sh --check          # fail now, not from the exit trap

./scripts/run_fp8_arm.sh \
    --vllm-log /tmp/vllm.log \
    --budget-min 90 \
    --rate 0.79 \
    --on-finish  "./scripts/stop_pod.sh --delay 600" \
    --stop-check "./scripts/stop_pod.sh --check"
```

**`runpodctl` on a RunPod pod is UNAUTHENTICATED by default.** Measured
2026-08-17: the binary is at `/usr/bin/runpodctl`, the pod ID is right, and
`stop_pod.sh --check` still fails with *"runpodctl cannot see pod"* — the
container carries no RunPod credentials. Authenticating it means putting an
account-wide API key on a rented third-party box.

If you decline that (recommended for an attended run), drop both flags and stop
the pod from the console yourself:

```bash
./scripts/run_fp8_arm.sh --vllm-log /tmp/vllm.log --budget-min 90 --rate 1.00
```

The budget watchdog still applies; only the unattended auto-stop is given up.
The script says plainly that the instance keeps billing. This also removes the
retrieval race, since nothing stops the pod out from under the `scp`.

**Export `RUNPOD_POD_ID` first** if you do use the auto-stop. The calling shell
expands `--on-finish`, so an unset variable becomes `runpodctl stop pod ` with a
missing argument — and that only fails at the end, from inside the EXIT trap,
with the pod still billing.

Gate 0 catches this before anything runs: it aborts if the stop command's binary
is not on `PATH`, or if the command ends with or contains an empty argument.
Verified against an unset `RUNPOD_POD_ID`, a missing binary, and a valid
command.

Runs preflight, generation and the health checks behind five gates, failing
cheapest-first. Each is one of the ways this can silently produce an
uninterpretable number:

| gate | aborts if |
|---|---|
| 0 stop command | `--on-finish` binary is missing or has an empty argument, or `--stop-check` fails |
| 1 hardware | `hardware.json` does not show cc 8.9+ |
| 2 server | the vLLM log shows a fallback, a non-fp8 method, or never mentions fp8 |
| 3 signature | model differs from the study's, server reports a different model than requested, or **quantization equals the baseline's** |
| 4 arm health | not 52 traces, or any empty or truncated |
| 5 comparability | any prompt digest differs from the baseline, or template / temperature / seed / reasoning_strength differ |

Gate 3's quantization check is the backstop for the A6000 case: an arm that
reports the same quantization as the baseline is int4 compared with itself, and
cannot fail the study no matter what it produces.

**Cost control.** `--budget-min` arms a watchdog that terminates the run and its
children at the cap — it kills the child process first, because signalling only
the shell leaves a stalled preflight or a wedged generation billing until it
returns on its own. `--on-finish` runs from an EXIT trap, so the instance is
stopped on success, on any abort, on the budget timeout, and on Ctrl-C. Without
it the script says plainly that the pod is still billing.

`--rate` is only for the closing cost estimate; it controls nothing.

Generation uses `--resume`, so a dropped pod can be restarted without repeating
completed prompts. Decoding comes from the host and teacher configs — do not
override temperature or max_tokens on the command line, or gate 5 will reject
the arm.

## Step 4.5 — retrieve the arm, THEN confirm the pod stopped

**Order matters, and it is the opposite of what seems natural.** A stopped pod
runs no sshd, so `scp` to it fails. But `--on-finish` stops the pod from inside
the EXIT trap — meaning by the time the run returns, the pod is already stopped
and the arm is stranded on a box you cannot reach.

That tension is deliberate: the auto-stop must survive a crash or an abort, so
it cannot wait for a human to copy files first. Resolve it one of two ways.

**Option A — pull it before the stop lands.** The three artifacts do not appear
together, which decides how to do this safely:

| artifact | written at | race? |
|---|---|---|
| `hardware.json` | step 0, before serving | no — copy any time |
| `signature.json` | gate 3, before generation | no — copy any time |
| `traces.jsonl` | **all at once after all 52 finish** (`generate_normalized.py`) | **yes** |

Gates 4 and 5 take about a second each, so the gap between `traces.jsonl`
appearing and the pod stopping is a few seconds. Too short to copy reliably.

Widen it by delaying the stop, which keeps the crash protection intact. Use the
wrapper rather than a raw compound command:

```bash
export RUNPOD_POD_ID="<your-pod-id>"
./scripts/stop_pod.sh --check          # validate BEFORE anything bills
```

then run the arm with:

```bash
--on-finish  "./scripts/stop_pod.sh --delay 600" \
--stop-check "./scripts/stop_pod.sh --check"
```

**Why the wrapper.** Gate 0 validates only the FIRST binary of `--on-finish`.
Given `sleep 600 && runpodctl stop pod $ID` it checks `sleep` — the delay hides
the binary that matters, and an unset id inside a compound command is invisible
to it. Making the script the first binary means the thing gate 0 checks is the
thing that knows how to check everything else. `--stop-check` then runs that
validation during gate 0, while aborting is still free.

`stop_pod.sh` refuses on: `runpodctl` missing, `RUNPOD_POD_ID` unset or empty,
an id that is not a plausible id, and — the case no string check can catch — a
well-formed id the API cannot see. Each verified. After stopping it polls up to
three times and, if the pod never reports a stopped state, says so loudly rather
than exiting quietly on an unverified stop.

Then from Terminal C: copy `hardware.json` and `signature.json` early, and
`traces.jsonl` as soon as the run prints `FP8 ARM COMPLETE`, with ten minutes of
margin.

```bash
scp -r <pod>:tantular-distillation/data/calibration/fp8 data/calibration/
```

One consequence of the delay, worth accepting knowingly: **it also applies on
abort**, so a failed gate idles ten minutes before stopping. About $0.13 at
$0.79/hr — cheap insurance against a stranded arm.

Verify locally before letting the pod stop:

```bash
wc -l < data/calibration/fp8/traces.jsonl                 # expect 52
./.venv/bin/python -c "import json;d=json.load(open('data/calibration/fp8/hardware.json'));print(d['gpu_name'],d['compute_capability'])"
./.venv/bin/python -c "import json;d=json.load(open('data/calibration/fp8/signature.json'));print(d['reported_model'],d['quantization'])"
```

**Option B — restart briefly to retrieve.** A *stopped* RunPod pod keeps its
volume (a *terminated* one does not). Restart, copy, stop again:

```bash
runpodctl start pod $RUNPOD_POD_ID
scp -r <pod>:tantular-distillation/data/calibration/fp8 data/calibration/
runpodctl stop pod $RUNPOD_POD_ID
```

Costs a few minutes of billing. Reliable, and the safer default if you are not
watching the run.

You need all three files: `traces.jsonl`, `signature.json`, and
`hardware.json` — that last one is the only evidence the silicon could do FP8,
and it exists nowhere else.

Then confirm the pod is stopped. The trap *attempts* the stop and warns if the
command fails, but a warning in scrollback is not confirmation:

```bash
runpodctl get pod $RUNPOD_POD_ID        # expect EXITED / STOPPED, not RUNNING
```

**Do not terminate until the three files are on your machine and verified.**
Terminating destroys the volume; the arm would have to be regenerated, at full
rental cost.

Everything from here runs on any machine, with no GPU.

## Step 5 — study gate 1, the critical metrics

```bash
./.venv/bin/python src/calibrate.py compare \
    data/calibration/int4-normalized/traces.jsonl \
    data/calibration/fp8/traces.jsonl \
    --prompts prompts/calibration.jsonl
```

Scores both arms against `acceptance.yaml` `critical` (all at
`max_regression_abs: 0.00` — int4 must be no worse) and `quality.indonesian`
(0.05 tolerance).

**Expect this half to be weakly informative.** The baseline is already at or
near ceiling on most critical metrics, so they cannot separate the arms. Only
`refusal` (0.0192) and `constraints_ok` (0.9706) have room to move. This is why
step 6 is not optional.

## Step 6 — study gate 2, blind pairwise review

The only metric that can see what the mechanical ones cannot. Quantization
damage shows up as weaker inference, blander drafts, shallower summaries — none
of which move a constraint check or a router label.

```bash
./.venv/bin/python src/blind_review.py prepare \
    --baseline data/calibration/int4-normalized/traces.jsonl \
    --treatment data/calibration/fp8/traces.jsonl \
    --prompts prompts/calibration.jsonl \
    --out data/calibration/_review \
    --salt "<any string you choose>" --write
```

Produces three files:

| file | purpose |
|---|---|
| `REVIEW.md` | 52 items, each with the task and two answers as A / B |
| `VERDICTS.csv` | one row per item — fill in `A`, `B`, or `tie` |
| `KEY.json` | which answer was which arm. **Do not open until verdicts are complete.** |

Blinding is by salted hash *rank*, giving an exact 26/26 split of which arm sits
in slot A, scattered through the item order. Hash parity was tried first and
rejected: it is balanced only in expectation and put the baseline in slot A for
nine consecutive items, which is long enough for a reviewer to start recognising
a house style and partially unblind themselves.

**Who reviews:** someone who did not generate the arms. Judge overall usefulness
— correct reasoning, faithfulness to the source, usable as written. `tie` is a
legitimate and expected verdict; most answers to an easy task genuinely are
equivalent, and forcing a preference manufactures signal.

Then:

```bash
./.venv/bin/python src/blind_review.py score --dir data/calibration/_review
```

Ties count as half a win. Scores `pairwise_win_rate` for the baseline against
`acceptance.yaml` (`min_absolute: 0.40`). The scorer refuses to run if
`REVIEW.md` changed after preparation, so verdicts cannot be matched against an
edited packet.

## Step 7 — record the verdict

`acceptance.yaml`: **all critical pass AND all quality pass**, else FAIL.

- **PASS** → "int4 acceptable for this corpus, as an explicit recorded waiver."
  The waiver still stands; a pass does not delete the FP8 gate, and
  `verify_corpus.py --gate` still fails on int4 traces. What changes is that the
  gap is now *measured* instead of unbounded.
- **FAIL** → `acceptance.yaml` says generate at FP8 before training. The corpus
  would need regenerating on rented hardware.

Either way, update `calibration/INT4_WAIVER.md`, whose current text says the
magnitude of quantization loss is "unknown and unmeasurable without the
treatment arm" and that "no number in this repository bounds it." After this run,
one does.

---

## What this does not settle

- **The 260-prompt corpus is not covered.** The study measures the teacher on 52
  calibration prompts. A pass licenses a waiver; it does not certify v3.
- **Nothing here concerns real Office documents.** All prompts are synthetic;
  that claim still needs approved, redacted real sources.
- **The corpus stays `synthetic_candidate`** until someone decides otherwise on
  the evidence.

---

## BLOCKER as of 2026-08-18: vLLM cannot generate from this checkpoint

A second rental cleared every infrastructure gate and still produced **no arm**.
This is the current stopping point, and it is not a hardware or config problem.

### What worked

| step | result |
|---|---|
| pod: L40S, cc 8.9, **driver 580.159.04** | driver gate passed |
| clean venv, `uv pip install vllm --torch-backend=auto` | torch 2.13.0+**cu132**, vllm 0.27.1, transformers 5.15.0 |
| `verify_stack.sh` | all checks incl. vLLM's compiled CUDA extension |
| model load | 32.32 GiB in 87 s |
| FP8 kernels | `Selected TritonFp8BlockScaledMMKernel for CompressedTensorsW8A8Fp8` |
| gates 1–3 | passed |

FP8 serving on Ada is confirmed possible. The driver gate did its job.

### What failed

All 52 traces malformed. Raw output, once `skip_special_tokens: False` made the
control tokens visible:

```
<|eom|><|eom|><|eom|><|eom|><|eom|>… (repeating, no content)
```

The chat endpoint is equally broken (`content: "\n"`, 4 tokens). The model
generates control tokens and no text, on either request path.

The probable cause is in the startup log:

```
TransformersMultiModalForCausalLM has no vLLM implementation,
falling back to Transformers implementation
```

vLLM has no native implementation for this architecture and runs it through a
generic fallback. Weights load and FP8 kernels engage, but generation is
garbage. No request-level flag fixed it.

### Fixed along the way — real bugs, now committed

The `runtime == "openai"` branch of `generate_normalized.py` had **never been
executed** before this rental; ai19-ollama was the only host ever used to
generate. Two genuine defects were hiding there:

- `add_special_tokens: False` — the harmony template renders `bos_token`
  itself, so vLLM prepending another gave a doubled BOS; every completion came
  back empty.
- `skip_special_tokens: False` — vLLM strips control tokens from returned text
  by default, deleting the very channel markers `parse_channels` needs.

Ollama's single `raw: True` covers both, which is why neither had surfaced.
Also fixed: `rented-48gb` had no `base_url`; the teacher port collided with the
template's nginx; and `serve_teacher.sh` now registers both the short name and
the repo path as served-model aliases.

### Before renting again

**Do not debug this on a rented GPU.** Three incompatibilities were found at
$1/hr, each revealing the next. Resolve the architecture support question
off-GPU first:

1. Determine whether any vLLM version natively implements this architecture, or
   whether a different serving stack is needed.
2. Verify generation produces real text against a free or local endpoint.
3. Only then rent, and expect the run to take about 30 minutes.

Until that is answered, the FP8 arm remains unobtainable and the int4 waiver
stands unchanged.
