"""Validate a rented student endpoint. Five checks, no training, no trainer.

    ./.venv/bin/python src/validate_student_endpoint.py \
        --host student-serve --model office-student-9b \
        --serve-log logs/vllm-student.log \
        --out data/gates/student-validation

Approved 2026-08-19 as a VALIDATION-ONLY rental. It exists so the decision to
spend a GPU on the v1 training run is made against evidence rather than hope,
and so a bad pod is discovered in minutes instead of after a 20-minute load.

The five checks, in the order a failure would actually occur:

  1 hardware       scripts/verify_student_host.sh, run ON the pod before serving.
                   Not repeated here — this reads the JSON it wrote.
  2 bf16 load      vLLM does not report dtype over the API, so the evidence is
                   the serve log. No log, no claim: check 2 fails rather than
                   being assumed from the flag we passed.
  3 identity       /v1/models must report the model train/qlora_9b.yaml names.
  4 generation     one prompt must come back as non-empty, non-degenerate text.
  5 baseline gate  run_gates --stage before must COMPLETE. Under 0.95 is the
                   expected outcome and is recorded as BASELINE_BELOW_TARGET;
                   it is not a failure of this validation.

STOP CONDITION: if any of 1-4 fails, or if 5 cannot complete (exit 2), the pod
should be stopped and training does not proceed. This script never passes
--confirm-run-v1 and never installs a training dependency; it cannot start a
training run even if every check passes. That second decision is a human one.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = str(ROOT / ".venv" / "bin" / "python")

# Evidence that vLLM actually loaded bfloat16. Matched against the serve log.
BF16_EVIDENCE = re.compile(r"(?:dtype|torch_dtype)[=' \"]*(?:torch\.)?bfloat16",
                           re.IGNORECASE)
# Precisions that must NOT appear: any of them means the baseline would measure
# something other than the weights QLoRA will train against.
WRONG_PRECISION = re.compile(
    r"(?:dtype|torch_dtype)[=' \"]*(?:torch\.)?(float16|half|fp16)"
    r"|quantization[=' \"]*(?:awq|gptq|fp8|bitsandbytes|compressed-tensors)",
    re.IGNORECASE)


class Check:
    def __init__(self, n: int, name: str):
        self.n, self.name, self.ok, self.detail = n, name, False, ""

    def passed(self, detail: str) -> "Check":
        self.ok, self.detail = True, detail
        return self

    def failed(self, detail: str) -> "Check":
        self.ok, self.detail = False, detail
        return self

    def as_dict(self) -> dict:
        return {"check": self.n, "name": self.name,
                "verdict": "PASS" if self.ok else "FAIL", "detail": self.detail}


def check_1_hardware(path: Path | None) -> Check:
    c = Check(1, "hardware meets the gate")
    if path is None or not path.is_file():
        return c.failed(
            "no hardware report. Run scripts/verify_student_host.sh ON the pod "
            "before serving and pass it with --hardware. Both previous rentals "
            "were lost to hardware facts knowable in the first sixty seconds.")
    hw = json.loads(path.read_text())
    cap = str(hw.get("compute_capability", ""))
    if not cap or int(cap.split(".")[0]) < 8:
        return c.failed(f"compute capability {cap!r} has no bfloat16 path")
    return c.passed(f"{hw.get('gpu')} cc {cap}, driver {hw.get('driver')}, "
                    f"{hw.get('memory_free_mib')} MiB free")


def check_2_bf16(log: Path | None) -> Check:
    c = Check(2, "vLLM loaded the model at bf16")
    if log is None or not log.is_file():
        return c.failed(
            "no serve log. vLLM does not report dtype over the API, so bf16 "
            "cannot be verified from the endpoint. Passing --dtype bfloat16 is "
            "what we ASKED for, not evidence of what loaded; capture the serve "
            "log and pass it with --serve-log.")
    text = log.read_text(errors="replace")
    wrong = WRONG_PRECISION.search(text)
    if wrong:
        return c.failed(
            f"the log shows {wrong.group(0)!r}. The baseline must be bf16: any "
            "other precision makes 'before' a measurement of quantization "
            "damage that the adapter would then appear to have fixed.")
    m = BF16_EVIDENCE.search(text)
    if not m:
        return c.failed("the log never states a dtype. Refusing to infer bf16 "
                        "from the flag we passed.")
    return c.passed(f"log states {m.group(0)!r}")


def check_3_identity(host: str, model: str, expect: str) -> Check:
    c = Check(3, "/v1/models reports the configured student")
    sys.path.insert(0, str(ROOT / "src"))
    from config import base_url, resolve
    try:
        resolved = resolve(model, host)
    except SystemExit as e:
        return c.failed(f"config will not resolve: {e}")
    url = resolved.get("HOST_BASE_URL") or base_url(resolved)
    try:
        import httpx
        served = [m.get("id") for m in
                  httpx.get(f"{url}/models", timeout=30).json().get("data", [])]
    except Exception as e:
        return c.failed(f"cannot reach {url}: {e}")
    if not any(expect == s or expect.split("/")[-1].lower() in str(s).lower()
               for s in served):
        return c.failed(f"endpoint serves {served}, config names {expect}. "
                        "Most likely this is pointed at a teacher.")
    return c.passed(f"{url} serves {served}")


DEGENERATE = re.compile(r"^(?:(.{1,20}?)\1{4,})$", re.DOTALL)


def check_4_generation(host: str, model: str, out_dir: Path) -> Check:
    c = Check(4, "one prompt returns valid text")
    prompt = out_dir / "smoke.jsonl"
    prompt.write_text(json.dumps({
        "id": "student-smoke-1",
        "user": "Sebutkan tiga hal yang perlu diperiksa sebelum menyetujui "
                "laporan anggaran bulanan.",
    }, ensure_ascii=False) + "\n", encoding="utf-8")
    traces = out_dir / "smoke.traces.jsonl"
    proc = subprocess.run(
        [PY, str(ROOT / "src" / "generate_normalized.py"),
         "--teacher", model, "--host", host,
         "--prompts", str(prompt), "--out", str(traces)],
        capture_output=True, text=True, cwd=ROOT)
    if not traces.is_file():
        return c.failed(f"generation produced nothing:\n{proc.stderr[-400:]}")
    rows = [json.loads(l) for l in traces.read_text().splitlines() if l.strip()]
    if not rows:
        return c.failed("generation wrote an empty file")
    text = (rows[0].get("completion") or "").strip()
    if not text:
        # This is exactly how the FP8 arm failed: the endpoint answered, and
        # every completion was empty or a lone control token.
        return c.failed("the completion is EMPTY. The endpoint answers but "
                        "produces no text — the same failure mode that killed "
                        "the FP8 arm. Stop the pod.")
    if len(text) < 40:
        return c.failed(f"the completion is {len(text)} chars: {text!r}")
    if DEGENERATE.match(text):
        return c.failed(f"the completion is a repeating loop: {text[:120]!r}")
    return c.passed(f"{len(text)} chars, first line: {text.splitlines()[0][:80]!r}")


def check_5_baseline(host: str, model: str, expect: str, out_dir: Path) -> tuple[Check, dict | None]:
    c = Check(5, "gate --stage before completes")
    report_path = out_dir / "gates.before.json"
    proc = subprocess.run(
        [PY, str(ROOT / "src" / "run_gates.py"), "run", "--stage", "before",
         "--host", host, "--teacher", model, "--expect-model", expect,
         "--out", str(report_path)], cwd=ROOT)
    if proc.returncode == 2:
        return c.failed("a gate could not be executed (exit 2). This is an "
                        "infrastructure or identity failure, not a low score."), None
    if not report_path.is_file():
        return c.failed("the gate run produced no report"), None
    report = json.loads(report_path.read_text())
    if not report.get("measured"):
        return c.failed("the report does not claim to be measured"), report
    # Under 0.95 is the EXPECTED outcome for an untrained base model and is not
    # a failure of this validation. What matters is that every gate produced a
    # rate, which `measured` asserts and exit != 2 confirms.
    return c.passed(f"{report['verdict']}; "
                    f"below target on: {report['below_target'] or 'nothing'}"), report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="student-serve")
    p.add_argument("--model", default="office-student-9b")
    p.add_argument("--config", default="train/qlora_9b.yaml")
    p.add_argument("--hardware", type=Path, default=None,
                   help="JSON from scripts/verify_student_host.sh, run on the pod")
    p.add_argument("--serve-log", type=Path, default=None,
                   help="vLLM's own log — the only evidence of the loaded dtype")
    p.add_argument("--out", type=Path, default=Path("data/gates/student-validation"))
    args = p.parse_args()

    try:
        import yaml
    except ImportError:
        sys.exit("pyyaml is required")
    expect = yaml.safe_load((ROOT / args.config).read_text())["base_model"]

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== STUDENT ENDPOINT VALIDATION (no training) ===")
    print(f"  config base_model : {expect}")
    print(f"  endpoint          : {args.model} @ {args.host}\n")

    checks, gate_report = [], None
    for fn in (lambda: check_1_hardware(args.hardware),
               lambda: check_2_bf16(args.serve_log),
               lambda: check_3_identity(args.host, args.model, expect),
               lambda: check_4_generation(args.host, args.model, out_dir)):
        c = fn()
        checks.append(c)
        print(f"  [{c.n}] {'PASS' if c.ok else 'FAIL'}  {c.name}\n      {c.detail}")
        if not c.ok:
            # Stop at the first failure: check 5 generates 60 completions, and
            # running it against a broken endpoint burns rental time to produce
            # a number nobody should read.
            print("\nSTOPPING at the first failure — the remaining checks would "
                  "spend rental time\nto produce evidence about an endpoint we "
                  "already know is wrong.")
            break
    else:
        c, gate_report = check_5_baseline(args.host, args.model, expect, out_dir)
        checks.append(c)
        print(f"\n  [{c.n}] {'PASS' if c.ok else 'FAIL'}  {c.name}\n      {c.detail}")

    ok = all(c.ok for c in checks) and len(checks) == 5
    summary = {
        "validation": "student endpoint, no training",
        "config_base_model": expect,
        "endpoint": {"model": args.model, "host": args.host},
        "checks": [c.as_dict() for c in checks],
        "all_passed": ok,
        "baseline": ({"verdict": gate_report.get("verdict"),
                      "below_target": gate_report.get("below_target"),
                      "rates": {g["name"]: g["rate"] for g in gate_report["gates"]}}
                     if gate_report else None),
        "training_authorised": False,
    }
    path = out_dir / "VALIDATION.json"
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    print(f"\nwrote {path}")

    if ok:
        print("\nVALIDATION PASSED — all five checks.")
        print("This authorises NOTHING further. Training v1 needs a second, "
              "separate decision;\nthis script has never passed "
              "--confirm-run-v1 and installs no training dependency.")
    else:
        failed = [f"[{c.n}] {c.name}" for c in checks if not c.ok]
        skipped = 5 - len(checks)
        print(f"\nVALIDATION FAILED — {', '.join(failed)}"
              + (f" ({skipped} check(s) not reached)" if skipped else ""))
        print("STOP THE POD. Do not proceed to training.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
