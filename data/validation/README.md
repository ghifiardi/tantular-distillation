# Pipeline-validation artifacts — NOT corpus

Traces here proved the pipeline worked. They are **not** training material and
must never enter the authoritative corpus manifest.

## gateway-canary.jsonl

12 traces, the first end-to-end run through `openai.ina17.com`
(`quantization: "remote"`, `host: "gateway"`).

Excluded because the signed waiver (`calibration/INT4_WAIVER.md`) names the
Q4_K_M `muse-glimmer:30b` teacher on **ai19-ollama** only. A gateway trace is
also int4, but it came through a third-party operator's proxy at a precision we
did not control and cannot attest — from a deployment that vanished mid-study.
Unauthorised, not merely quantized.

`verify_corpus.py --gate` enforces this: these traces are reported as NOT
COVERED BY ANY WAIVER, separately from the quantization failure.

Kept rather than deleted because they are the evidence that the generation
path, split enforcement, provenance capture and quality gates work end to end.
