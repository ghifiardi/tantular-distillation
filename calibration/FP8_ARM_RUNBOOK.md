# Running the FP8 treatment arm

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
~$0.74–0.79/hr). Do **not** take an RTX A6000 at $0.33/hr: it is Ampere, has no
FP8 path, and would silently serve int4 — producing a "treatment arm" identical
to the baseline. `blind_review.py prepare` aborts if both arms report the same
quantization, but catching it after an hour of rental is the expensive way.

**No egress approval needed.** All 52 calibration prompts are
`source_class: synthetic`, so the external-host gate passes without
`--egress-approval`. Do not point this at real Office material.

---

## Step 0 — prove the card can do FP8 (before serving anything)

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

## Step 1 — serve the teacher at FP8 (on the rented box)

```bash
git clone <this repo> && cd tantular-distillation
pip install -r requirements.txt
./scripts/serve_teacher.sh muse-glimmer rented-48gb 2>&1 | tee /tmp/vllm.log
```

Keep that log — it is the only evidence of what vLLM actually selected, and
step 2 refuses to proceed without it.

Serves on port 8001 via vLLM, `--quantization fp8`, `tensor-parallel-size 1`.
Serve one teacher only — two 30B models do not co-resident in 48GB at FP8, and
dropping both to int4 to fit defeats the entire purpose of the run.

Expect a long first load while weights download.

## Steps 2–4 — one gated command

```bash
./scripts/run_fp8_arm.sh \
    --vllm-log /tmp/vllm.log \
    --budget-min 90 \
    --rate 0.79 \
    --on-finish "runpodctl stop pod $RUNPOD_POD_ID"
```

Runs preflight, generation and the health checks behind five gates, failing
cheapest-first. Each is one of the ways this can silently produce an
uninterpretable number:

| gate | aborts if |
|---|---|
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
