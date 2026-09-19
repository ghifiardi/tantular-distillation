"""Verifier tier for the RSI evaluation harness (Phase 1, Task 1.2).

The Recursive Self-Improvement Report v2 ranks verifiers by trustworthiness
(§A) and makes the verifier the load-bearing component of any loop that
compounds rather than drifts. This module implements that tier, in the order
the handoff names:

    exact_match   deterministic string comparison under enumerated normalization
    schema        bounded structural validation of the model's output
    executor      run the candidate against tests in a subprocess with a timeout
    judge         QUARANTINED. Style only. Never enters a correctness score.

Two properties carry the weight:

1. **A judge result can never score correctness.** The report's §D.7 warning is
   blunt: "A judge that is 60% aligned with your raters will happily produce
   50,000 confident preference pairs." `counts_toward_correctness` is False for
   every judge result, and `score` refuses a judge item presented as
   correctness rather than quietly dropping it.

2. **A right answer with wrong reasoning is RECORDED, not silently accepted.**
   Report §04: "ReSTEM's authors found STaR-style rationalisation produced a
   substantial increase in false positives — right answer, wrong reasoning —
   and the original STaR paper called filtering those an open question. This is
   the error-amplification channel you asked about, and it is a filtering
   problem, not a training problem." A final-answer match does not erase a
   failed reasoning check; it sets `false_positive=True` and §H.3 sampling can
   consume it later.

NO MODEL. NO TRAINING. `train/TRAINING_BLOCKED.md` is controlling.

The executor is TIMEOUT-BOUNDED, not sandboxed. It sets no network policy, and
a child process is free to open sockets; there is no isolation, no filesystem
confinement beyond a disposable working directory, and no resource limit
beyond wall-clock time and a cap on how much captured output is retained.
Because of that it is QUARANTINED to `synthetic_fixture` records — code this
repository wrote for its own tests. A production or human-authored executor
item fails closed and will keep failing closed until a reviewed sandbox
backend exists. Running third-party code through this is not supported.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unicodedata
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import gold_set as gs  # noqa: E402

# The handoff's order. `judge` is last because it is least trustworthy, and
# nothing in this module may reorder it upward.
TRUST_ORDER = ("exact_match", "schema", "executor", "judge")

QUARANTINED = ("judge",)


class VerifierError(Exception):
    """A verifier could not run. Never the same thing as an item failing."""


@dataclass
class VerificationResult:
    """The outcome of checking one model output against one gold item.

    `passed` is the item's correctness verdict. `false_positive` is a separate
    axis: True means the final answer matched while a mechanical reasoning
    claim did not. A result can be `passed=True, false_positive=True`, and that
    combination is the entire point of recording it.
    """

    item_id: str
    verifier: str
    score_role: str
    passed: bool
    counts_toward_correctness: bool
    false_positive: bool | None = None
    reasoning_checked: bool = False
    detail: str = ""
    problems: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --- normalization ----------------------------------------------------------


def normalize(text: str, steps: list[str] | tuple[str, ...]) -> str:
    """Apply enumerated normalizations only.

    An unknown step raises rather than being skipped: silently ignoring a
    normalization the item asked for changes what the comparison means while
    still reporting a clean pass.
    """
    value = str(text)
    for step in steps or ():
        if step not in gs.ALLOWED_NORMALIZATIONS:
            raise VerifierError(
                f"unknown normalization {step!r}; allowed "
                f"{list(gs.ALLOWED_NORMALIZATIONS)}"
            )
        if step == "trim":
            value = value.strip()
        elif step == "unicode_nfc":
            value = unicodedata.normalize("NFC", value)
        elif step == "casefold":
            value = value.casefold()
        elif step == "collapse_space":
            value = " ".join(value.split())
    return value


# --- reasoning check --------------------------------------------------------


def check_reasoning(expected: dict[str, Any], reasoning: str | None) -> tuple[bool, str]:
    """Mechanical reasoning check. Returns (satisfied, detail).

    Only `required_claims` is implemented, deliberately: a claim either appears
    in the reasoning or it does not, which is checkable. Anything softer would
    be a judge wearing a verifier's name.
    """
    check = expected.get("reasoning_check")
    if not isinstance(check, dict):
        return True, "no reasoning check declared"
    if check.get("type") != "required_claims":
        raise VerifierError(
            f"unsupported reasoning_check.type {check.get('type')!r}"
        )
    claims = [c for c in check.get("claims", []) if str(c).strip()]
    if not claims:
        raise VerifierError("reasoning_check.claims is empty")
    if reasoning is None:
        return False, "no reasoning supplied for a required_claims item"
    text = normalize(reasoning, ("collapse_space",))
    missing = [c for c in claims
               if normalize(str(c), ("collapse_space",)) not in text]
    if missing:
        return False, f"missing claim(s): {missing!r}"
    return True, "all required claims present"


# --- verifiers --------------------------------------------------------------


def _output_parts(output: Any) -> tuple[str, str | None]:
    """(answer, reasoning) from a model output that may be a string or object."""
    if isinstance(output, dict):
        answer = output.get("answer", "")
        reasoning = output.get("reasoning")
        return str(answer), None if reasoning is None else str(reasoning)
    return str(output), None


def verify_exact_match(record: dict[str, Any], output: Any) -> VerificationResult:
    expected = record["expected"]
    steps = expected.get("normalization", [])
    answer, reasoning = _output_parts(output)

    got = normalize(answer, steps)
    want = normalize(str(expected.get("answer", "")), steps)
    passed = got == want

    declared = isinstance(expected.get("reasoning_check"), dict)
    satisfied, detail = check_reasoning(expected, reasoning)

    return VerificationResult(
        item_id=str(record["id"]),
        verifier="exact_match",
        score_role=str(record.get("score_role")),
        passed=passed,
        counts_toward_correctness=record.get("score_role") == "correctness",
        # Only meaningful when the answer matched: a wrong answer with wrong
        # reasoning is simply a failure, not a false positive.
        false_positive=(passed and not satisfied) if declared else None,
        reasoning_checked=declared,
        detail=f"answer {'matched' if passed else 'differed'}; {detail}",
    )


def verify_schema(record: dict[str, Any], output: Any) -> VerificationResult:
    """Bounded structural validation.

    Mirrors the semantics of `scripts/check_edit_contract.mjs`: parsing is not
    passing. A model can emit perfectly valid JSON that satisfies nothing the
    contract requires, so required keys and `additionalProperties: false` are
    both enforced here.
    """
    expected = record["expected"]
    schema = expected.get("json_schema")
    if not isinstance(schema, dict) or not schema:
        raise VerifierError(f"{record['id']}: expected.json_schema is required")

    problems: list[str] = []
    value: Any = output
    if isinstance(output, str):
        try:
            value = json.loads(output)
        except json.JSONDecodeError as error:
            problems.append(f"parse_ok=false: {error}")
            value = None

    if value is not None:
        problems.extend(_schema_problems(value, schema, path="$"))

    passed = not problems
    return VerificationResult(
        item_id=str(record["id"]),
        verifier="schema",
        score_role=str(record.get("score_role")),
        passed=passed,
        counts_toward_correctness=record.get("score_role") == "correctness",
        detail="contract satisfied" if passed else problems[0],
        problems=problems,
    )


def _schema_problems(value: Any, schema: dict[str, Any], path: str) -> list[str]:
    """A deliberately small JSON-schema subset: type, required, properties,
    additionalProperties, enum, items. Enough for a tool-call contract, and
    small enough to read. An unsupported keyword raises rather than passing."""
    supported = {"type", "required", "properties", "additionalProperties",
                 "enum", "items", "minimum", "maximum", "minLength"}
    unknown = set(schema) - supported
    if unknown:
        raise VerifierError(f"unsupported schema keyword(s) {sorted(unknown)!r}")

    problems: list[str] = []
    expected_type = schema.get("type")
    types = {
        "object": dict, "array": list, "string": str,
        "number": (int, float), "integer": int, "boolean": bool,
    }
    if expected_type:
        py = types.get(expected_type)
        if py is None:
            raise VerifierError(f"unsupported schema type {expected_type!r}")
        if expected_type == "integer" and isinstance(value, bool):
            problems.append(f"{path}: expected integer, got boolean")
        elif not isinstance(value, py):
            problems.append(
                f"{path}: expected {expected_type}, got {type(value).__name__}")
            return problems

    if "enum" in schema and value not in schema["enum"]:
        problems.append(f"{path}: {value!r} not in enum {schema['enum']!r}")

    # Numeric and length bounds. Kept as explicit, checkable comparisons: a
    # bound the validator silently ignores is a contract the product does not
    # actually enforce.
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            problems.append(f"{path}: {value} below minimum {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            problems.append(f"{path}: {value} above maximum {schema['maximum']}")
    if isinstance(value, str) and "minLength" in schema \
            and len(value) < schema["minLength"]:
        problems.append(
            f"{path}: length {len(value)} below minLength {schema['minLength']}")

    if isinstance(value, dict):
        for key in schema.get("required", []):
            if key not in value:
                problems.append(f"{path}.{key}: required")
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            extra = sorted(set(value) - set(properties))
            if extra:
                problems.append(f"{path}: unexpected field(s) {extra!r}")
        for key, sub in properties.items():
            if key in value:
                problems.extend(_schema_problems(value[key], sub, f"{path}.{key}"))

    if isinstance(value, list) and isinstance(schema.get("items"), dict):
        for i, item in enumerate(value):
            problems.extend(_schema_problems(item, schema["items"], f"{path}[{i}]"))

    return problems


# Retained output per stream, per test. A cap on what enters a report, applied
# after capture: it bounds what this module keeps and passes on, and does NOT
# bound what the child process may produce or allocate.
MAX_CAPTURED_CHARS = 8192


def executor_backend_allows(record: dict[str, Any]) -> tuple[bool, str]:
    """May the local backend run this item?

    Only for records this repository authored as fixtures. The local backend
    is timeout-bounded and nothing more — no network policy, no isolation, no
    memory or file-descriptor limit — so running a production item through it
    would be asserting a containment guarantee that does not exist.
    """
    source = record.get("source")
    kind = (source or {}).get("kind") if isinstance(source, dict) else None
    if kind == "synthetic_fixture":
        return True, "synthetic fixture: local timeout-bounded backend permitted"
    return False, (
        f"executor item {record.get('id')!r} has source.kind {kind!r}. The "
        "local backend is timeout-bounded only — no network policy, no "
        "isolation, no resource limits — so it is restricted to "
        "synthetic_fixture records. A reviewed sandbox backend is required "
        "before a production executor item can be scored.")


def verify_executor(record: dict[str, Any], output: Any) -> VerificationResult:
    """Run a FIXTURE program against the item's tests.

    Timeout-bounded, not sandboxed. See `executor_backend_allows`: production
    and human-authored items are refused here rather than run.
    """
    allowed, why = executor_backend_allows(record)
    if not allowed:
        raise VerifierError(why)

    expected = record["expected"]
    runner = expected.get("runner")
    if runner not in gs.ALLOWED_RUNNERS:
        raise VerifierError(
            f"{record['id']}: runner {runner!r} not in {list(gs.ALLOWED_RUNNERS)}")
    timeout = float(expected.get("timeout_seconds") or 0)
    if not 0 < timeout <= gs.MAX_EXECUTOR_TIMEOUT_SECONDS:
        raise VerifierError(f"{record['id']}: timeout_seconds out of bounds")

    program, _ = _output_parts(output)
    if not program.strip():
        return VerificationResult(
            item_id=str(record["id"]), verifier="executor",
            score_role=str(record.get("score_role")), passed=False,
            counts_toward_correctness=record.get("score_role") == "correctness",
            detail="no program supplied", problems=["empty candidate"],
        )

    # A minimal environment, so a fixture cannot depend on ambient
    # configuration. This is hygiene, NOT a network control: proxy variables
    # are not a firewall and are deliberately not set here, because setting
    # them would imply a denial this module cannot enforce.
    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": os.environ.get("HOME", ""),
        "PYTHONIOENCODING": "utf-8",
    }

    problems: list[str] = []
    # A disposable working directory, removed afterwards. A fixture that writes
    # a file must not leave it in the repository, and one that reads a relative
    # path must not find repository contents by accident.
    with tempfile.TemporaryDirectory(prefix="tantular-executor-") as workdir:
        for index, case in enumerate(expected.get("tests", []), start=1):
            stdin = str(case.get("stdin", ""))
            want = str(case.get("stdout", ""))
            try:
                proc = subprocess.run(
                    [sys.executable, "-I", "-c", program],
                    input=stdin, capture_output=True, text=True,
                    timeout=timeout, env=env, cwd=workdir, check=False,
                )
            except subprocess.TimeoutExpired:
                problems.append(f"test {index}: timed out after {timeout}s")
                continue
            stdout = proc.stdout[:MAX_CAPTURED_CHARS]
            stderr = proc.stderr[:MAX_CAPTURED_CHARS]
            if len(proc.stdout) > MAX_CAPTURED_CHARS:
                problems.append(
                    f"test {index}: output exceeded {MAX_CAPTURED_CHARS} "
                    "characters and was truncated; treated as a failure "
                    "rather than compared against a prefix")
                continue
            if proc.returncode != 0:
                last = stderr.strip().splitlines()[-1] if stderr.strip() else ""
                problems.append(f"test {index}: exit {proc.returncode}: {last}")
                continue
            if stdout != want:
                problems.append(f"test {index}: stdout {stdout!r} != {want!r}")

    passed = not problems
    return VerificationResult(
        item_id=str(record["id"]), verifier="executor",
        score_role=str(record.get("score_role")), passed=passed,
        counts_toward_correctness=record.get("score_role") == "correctness",
        detail="all tests passed" if passed else problems[0],
        problems=problems,
    )


def verify_judge(record: dict[str, Any], output: Any) -> VerificationResult:
    """Quarantined stub. Report-only, and never a correctness verdict.

    It deliberately does NOT call a model. Until judge-human agreement is
    calibrated on native-rated pairs (report §D.7), a judge score is not
    evidence, and a stub that returned a number would be treated as one.
    """
    rubric = record["expected"].get("rubric", [])
    return VerificationResult(
        item_id=str(record["id"]),
        verifier="judge",
        score_role=str(record.get("score_role")),
        passed=False,
        counts_toward_correctness=False,
        detail=(
            "judge is quarantined: style-only, report-only, and uncalibrated. "
            f"rubric={list(rubric)!r}"
        ),
        problems=["judge_not_calibrated"],
    )


VERIFIERS = {
    "exact_match": verify_exact_match,
    "schema": verify_schema,
    "executor": verify_executor,
    "judge": verify_judge,
}


def verify(record: dict[str, Any], output: Any) -> VerificationResult:
    """Check one model output against one gold record.

    Refuses a structurally invalid record: a verifier that runs on an
    unvalidated item is measuring something nobody declared.
    """
    problems = gs.validate_record(record)
    if problems:
        raise VerifierError(f"invalid gold record: {problems[0]}")
    vtype = record["verifier"]["type"]
    if vtype in QUARANTINED and record.get("score_role") == "correctness":
        raise VerifierError(
            f"{record['id']}: {vtype} may not produce a correctness verdict")
    return VERIFIERS[vtype](record, output)


# --- scoring ----------------------------------------------------------------


def score(results: list[VerificationResult]) -> dict[str, Any]:
    """Aggregate. Correctness counts only items eligible to count.

    `false_positive_rate` is reported next to the pass rate on purpose: §H.3
    asks for it as a time series, and a pass rate that hides it is the
    error-amplification channel the report warns about.
    """
    correctness = [r for r in results if r.counts_toward_correctness]
    passed = [r for r in correctness if r.passed]
    checked = [r for r in correctness if r.reasoning_checked and r.passed]
    false_positives = [r for r in checked if r.false_positive]
    quarantined = [r for r in results if r.verifier in QUARANTINED]

    return {
        "items": len(results),
        "correctness_items": len(correctness),
        "passed": len(passed),
        "pass_rate": (len(passed) / len(correctness)) if correctness else 0.0,
        "reasoning_checked": len(checked),
        "false_positives": len(false_positives),
        "false_positive_rate": (
            len(false_positives) / len(checked)) if checked else 0.0,
        "false_positive_ids": [r.item_id for r in false_positives],
        "quarantined_items": len(quarantined),
        # A measurement is "measured", never "passed", until a gate says so —
        # the run_gates distinction this repo already enforces.
        "measured": True,
        "training_authorized": False,
    }
