"""Evaluation-case protocol, execution receipts and measurement aggregation.

The harness experiment (configs/experiments/harness-before-weights.yaml) asks
whether a capability gap survives a controlled harness change. Answering it
needs MEASUREMENTS, and src/harness_distill.py deliberately accepts those as a
plain JSON object -- which means a typed-in pass rate and a measured one are
indistinguishable to it. This module closes that: a score may only reach
``harness_distill.evaluate`` by way of per-case execution receipts that name the
experiment, the case set, the arm, the model revision and the harness digest
that produced them.

Three artifacts, three schemas, each versioned separately:

  CASE SET      what is asked. Versioned, digested, and marked `approved` or
                not. An unapproved set is a FIXTURE: it may drive tests and it
                may never back a real measurement.

  RECEIPT       what happened for one (arm, case) pair. Immutable, written
                atomically, and carrying harness_distill.harness_provenance()
                verbatim. A failed execution leaves a failure receipt; it does
                not disappear.

  MEASUREMENT   what the receipts add up to. Derived only from receipts, with
                the numerator, denominator and receipt-set digest recorded so
                the arithmetic can be re-checked.

NOTHING HERE RUNS A MODEL. No network call, no weights, no Office automation,
no training. Every artifact carries ``training_authorized: false``.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - mirrors the other entry points
    raise SystemExit("pyyaml is required: pip install -r requirements.txt")

import harness_distill as hd

CASE_SET_SCHEMA = 1
RECEIPT_SCHEMA = 1
MEASUREMENT_SCHEMA = 1

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
CASE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

# Terminal states a receipt may record. `ok` is the only one that can score.
STATUS_OK = "ok"
STATUS_REFUSED = "refused"
STATUS_ERROR = "error"
RECEIPT_STATUSES = (STATUS_OK, STATUS_REFUSED, STATUS_ERROR)


class HarnessEvalError(hd.HarnessPlanError):
    """A fail-closed evaluation-protocol error.

    Subclasses HarnessPlanError so the CLI's single refusal path covers both,
    and so a caller that already fails closed on harness errors does not
    silently let an evaluation error through.
    """


def canonical_json(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def digest_payload(payload: Any) -> str:
    """SHA-256 over canonical JSON. The one construction used everywhere here."""
    return hashlib.sha256(canonical_json(payload)).hexdigest()


_digest = digest_payload      # internal alias, kept for readability below


# --- the case set -----------------------------------------------------------

def case_set_digest(case_set: dict[str, Any]) -> str:
    """Canonical SHA-256 over the case set, ignoring a self-declared digest.

    Same construction as harness_distill.canonical_digest, and for the same
    reason: an artifact that carries its own digest must not be trusted to
    describe itself.
    """
    value = dict(case_set)
    value.pop("digest", None)
    return _digest(value)


def _require(mapping: dict[str, Any], key: str, where: str) -> Any:
    if key not in mapping:
        raise HarnessEvalError(f"{where}: missing required field {key!r}")
    return mapping[key]


def validate_case_set(case_set: Any) -> dict[str, Any]:
    """Validate the whole case set, or refuse. Returns it unchanged."""
    if not isinstance(case_set, dict):
        raise HarnessEvalError("case set must be a mapping")
    if case_set.get("schema_version") != CASE_SET_SCHEMA:
        raise HarnessEvalError(
            f"case set schema_version must be {CASE_SET_SCHEMA}, got "
            f"{case_set.get('schema_version')!r}")
    name = _require(case_set, "name", "case set")
    if not isinstance(name, str) or not name.strip():
        raise HarnessEvalError("case set name must be a non-empty string")

    # `approved` is the gate that keeps a fixture out of a real measurement.
    # Absent is not false-by-default here: an unstated approval is a case set
    # nobody reviewed, and it must read as a refusal rather than a default.
    if "approved" not in case_set or not isinstance(case_set["approved"], bool):
        raise HarnessEvalError(
            "case set must declare approved: true|false explicitly. An absent "
            "approval is not the same claim as a reviewed refusal.")
    split = _require(case_set, "split", "case set")
    if split not in ("held_out", "fixture"):
        raise HarnessEvalError(
            f"case set split must be 'held_out' or 'fixture', got {split!r}")
    if case_set["approved"] and split == "fixture":
        raise HarnessEvalError(
            "a fixture split cannot be approved: fixtures exist to drive tests, "
            "and an approved fixture is how synthetic data becomes a claim")

    source_class = _require(case_set, "source_class", "case set")
    if source_class not in ("real_office", "synthetic", "fixture"):
        raise HarnessEvalError(
            "case set source_class must be real_office, synthetic or fixture, "
            f"got {source_class!r}")
    provenance = _require(case_set, "provenance", "case set")
    if not isinstance(provenance, dict) or not provenance:
        raise HarnessEvalError("case set provenance must be a non-empty mapping")
    for key in ("origin", "created_at", "note"):
        if not isinstance(provenance.get(key), str) or not provenance[key].strip():
            raise HarnessEvalError(
                f"case set provenance.{key} is required and must be a non-empty "
                "string: a case set that cannot say where it came from is not "
                "evidence")

    cases = case_set.get("cases")
    if not isinstance(cases, list) or not cases:
        raise HarnessEvalError(
            "case set must declare a non-empty cases list. An empty set scores "
            "0/0, which is not a measurement.")

    seen: set[str] = set()
    for index, case in enumerate(cases, 1):
        _validate_case(case, index, seen)
    return case_set


def _validate_case(case: Any, index: int, seen: set[str]) -> None:
    where = f"case #{index}"
    if not isinstance(case, dict):
        raise HarnessEvalError(f"{where}: each case must be a mapping")
    case_id = _require(case, "case_id", where)
    if not isinstance(case_id, str) or not CASE_ID_RE.match(case_id):
        raise HarnessEvalError(
            f"{where}: case_id must match {CASE_ID_RE.pattern} (got {case_id!r})")
    if case_id in seen:
        raise HarnessEvalError(
            f"{where}: duplicate case_id {case_id!r}. Case ids are the join key "
            "between arms; a duplicate makes coverage uncountable.")
    seen.add(case_id)
    where = f"case {case_id!r}"

    # A case carries its request inline, or names one by digest. It may not do
    # both and it may not do neither: a mutable reference with no digest is a
    # case whose content can change after the receipts were written.
    request = case.get("request")
    reference = case.get("request_ref")
    if (request is None) == (reference is None):
        raise HarnessEvalError(
            f"{where}: declare exactly one of request (inline payload) or "
            "request_ref (a digest-bound reference)")
    if request is not None and not isinstance(request, dict):
        raise HarnessEvalError(f"{where}: request must be a mapping")
    if reference is not None:
        if not isinstance(reference, dict):
            raise HarnessEvalError(f"{where}: request_ref must be a mapping")
        for key in ("uri", "sha256"):
            if not isinstance(reference.get(key), str) or not reference[key].strip():
                raise HarnessEvalError(
                    f"{where}: request_ref.{key} is required")
        if not SHA256_RE.match(reference["sha256"]):
            raise HarnessEvalError(
                f"{where}: request_ref.sha256 must be 64 lowercase hex. An "
                "undigested reference is a mutable case.")

    expected = _require(case, "expected", where)
    if not isinstance(expected, dict):
        raise HarnessEvalError(f"{where}: expected must be a mapping")
    scorers = expected.get("scorers")
    if not isinstance(scorers, list) or not scorers or \
            not all(isinstance(s, str) and s.strip() for s in scorers):
        raise HarnessEvalError(
            f"{where}: expected.scorers must be a non-empty list of metric names")
    if len(set(scorers)) != len(scorers):
        raise HarnessEvalError(f"{where}: expected.scorers contains duplicates")

    for key in ("expects_state_change", "requires_approval"):
        if not isinstance(case.get(key), bool):
            raise HarnessEvalError(
                f"{where}: {key} must be declared as a boolean")
    # A case that changes Office state without requiring approval contradicts
    # every harness in configs/harnesses/, all of which set
    # state_change_requires_approval: true. Refuse the contradiction here
    # rather than discovering it per-arm at execution time.
    if case["expects_state_change"] and not case["requires_approval"]:
        raise HarnessEvalError(
            f"{where}: expects_state_change is true but requires_approval is "
            "false; every declared harness requires approval for state change")


def case_ids(case_set: dict[str, Any]) -> list[str]:
    return [str(case["case_id"]) for case in case_set["cases"]]


def scorers_for(case_set: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    for case in case_set["cases"]:
        names.update(case["expected"]["scorers"])
    return names


def load_case_set(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise HarnessEvalError(f"missing case set: {path}")
    text = path.read_text(encoding="utf-8")
    try:
        value = yaml.safe_load(text) if path.suffix in (".yaml", ".yml") \
            else json.loads(text)
    except (yaml.YAMLError, json.JSONDecodeError) as exc:
        raise HarnessEvalError(f"unreadable case set at {path}: {exc}") from exc
    return validate_case_set(value)


# --- receipts ---------------------------------------------------------------

def receipt_digest(receipt: dict[str, Any]) -> str:
    """Digest a receipt without its own digest field."""
    value = dict(receipt)
    value.pop("receipt_sha256", None)
    return _digest(value)


def receipt_set_digest(receipts: list[dict[str, Any]]) -> str:
    """Order-independent digest over a receipt set.

    Sorted by (arm, case_id) rather than by file order: the same receipts read
    from a directory in a different order are the same evidence, and a digest
    that disagreed would make the artifact non-reproducible for no gain.
    """
    rows = sorted(
        (r["arm"], r["case_id"], receipt_digest(r)) for r in receipts)
    return _digest(rows)


_RECEIPT_REQUIRED = (
    "schema_version", "experiment", "experiment_digest", "case_set_digest",
    "case_set_name", "case_id", "arm", "model_registry", "model_id",
    "model_revision", "harness_name", "harness_digest", "harness_provenance",
    "prompt_sha256", "prompt_verified", "tools_offered", "tool_calls",
    "approvals", "before_action", "after_action", "repair_attempts",
    "termination_reason", "budgets", "budget_consumed", "result_digest",
    "executor", "started_at", "ended_at", "status", "scores",
    "training_authorized",
)


def validate_receipt(receipt: Any) -> dict[str, Any]:
    """Validate one receipt's shape and internal consistency, or refuse."""
    if not isinstance(receipt, dict):
        raise HarnessEvalError("receipt must be a mapping")
    if receipt.get("schema_version") != RECEIPT_SCHEMA:
        raise HarnessEvalError(
            f"receipt schema_version must be {RECEIPT_SCHEMA}, got "
            f"{receipt.get('schema_version')!r}")
    missing = [key for key in _RECEIPT_REQUIRED if key not in receipt]
    if missing:
        raise HarnessEvalError(f"receipt missing fields: {sorted(missing)}")

    if receipt["training_authorized"] is not False:
        raise HarnessEvalError(
            "receipt.training_authorized must be false; this protocol never "
            "authorizes training")

    status = receipt["status"]
    if status not in RECEIPT_STATUSES:
        raise HarnessEvalError(
            f"receipt status must be one of {list(RECEIPT_STATUSES)}, got "
            f"{status!r}")

    for key in ("case_set_digest", "harness_digest", "experiment_digest",
                "result_digest"):
        value = receipt[key]
        if not isinstance(value, str) or not SHA256_RE.match(value):
            raise HarnessEvalError(
                f"receipt.{key} must be 64 lowercase hex, got {value!r}")

    provenance = receipt["harness_provenance"]
    if not isinstance(provenance, dict) or not provenance:
        raise HarnessEvalError(
            "receipt.harness_provenance must be the non-empty block returned by "
            "harness_distill.harness_provenance()")
    # ONE SOURCE OF TRUTH. The receipt's own harness fields must agree with the
    # provenance block it carries; if they can disagree, the receipt has two
    # answers to "which harness ran" and neither is authoritative.
    if provenance.get("digest") != receipt["harness_digest"]:
        raise HarnessEvalError(
            "receipt.harness_digest disagrees with harness_provenance.digest")
    if provenance.get("name") != receipt["harness_name"]:
        raise HarnessEvalError(
            "receipt.harness_name disagrees with harness_provenance.name")
    if provenance.get("execution_model_registry") != receipt["model_registry"]:
        raise HarnessEvalError(
            "receipt.model_registry disagrees with "
            "harness_provenance.execution_model_registry")
    if provenance.get("prompt_sha256") != receipt["prompt_sha256"]:
        raise HarnessEvalError(
            "receipt.prompt_sha256 disagrees with harness_provenance")
    if provenance.get("prompt_verified") != receipt["prompt_verified"]:
        raise HarnessEvalError(
            "receipt.prompt_verified disagrees with harness_provenance")

    executor = receipt["executor"]
    if not isinstance(executor, dict):
        raise HarnessEvalError("receipt.executor must be a mapping")
    for key in ("name", "version", "kind", "produces_real_measurements"):
        if key not in executor:
            raise HarnessEvalError(f"receipt.executor.{key} is required")
    if not isinstance(executor["produces_real_measurements"], bool):
        raise HarnessEvalError(
            "receipt.executor.produces_real_measurements must be a boolean")

    scores = receipt["scores"]
    if not isinstance(scores, dict):
        raise HarnessEvalError("receipt.scores must be a mapping")
    if status == STATUS_OK and not scores:
        raise HarnessEvalError(
            f"receipt {receipt['arm']}/{receipt['case_id']}: status is 'ok' but "
            "no scorer result was recorded. A missing scorer is not a pass.")
    for metric, value in scores.items():
        if not isinstance(value, bool):
            raise HarnessEvalError(
                f"receipt scores.{metric} must be a boolean per-case outcome; "
                "a rate is an aggregate, not a case result")
    if status != STATUS_OK and scores:
        raise HarnessEvalError(
            f"receipt {receipt['arm']}/{receipt['case_id']}: status {status!r} "
            "must not carry scorer results; a failed execution scored nothing")

    budgets, consumed = receipt["budgets"], receipt["budget_consumed"]
    for label, block in (("budgets", budgets), ("budget_consumed", consumed)):
        if not isinstance(block, dict):
            raise HarnessEvalError(f"receipt.{label} must be a mapping")
        for key in ("max_steps", "max_wall_seconds") if label == "budgets" \
                else ("steps", "wall_seconds"):
            value = block.get(key)
            if not isinstance(value, (int, float)) or value < 0:
                raise HarnessEvalError(
                    f"receipt.{label}.{key} must be a non-negative number")
    return receipt


def write_receipt(directory: Path, receipt: dict[str, Any]) -> Path:
    """Write one receipt atomically. Returns the path.

    Atomic because a receipt is evidence: a half-written file that a later
    aggregation reads as malformed would turn a completed execution into an
    unexplained refusal, and a crash mid-write is exactly when the record
    matters most.
    """
    validate_receipt(receipt)
    stamped = dict(receipt)
    stamped["receipt_sha256"] = receipt_digest(receipt)
    directory.mkdir(parents=True, exist_ok=True)
    # The filename carries the join key, so a duplicate (arm, case) collides on
    # disk instead of silently becoming two countable receipts.
    path = directory / f"{receipt['arm']}__{receipt['case_id']}.json"
    payload = json.dumps(stamped, indent=2, ensure_ascii=False,
                         sort_keys=True) + "\n"
    handle, temporary = tempfile.mkstemp(dir=str(directory), suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    return path


def load_receipts(directory: Path) -> list[dict[str, Any]]:
    if not directory.is_dir():
        raise HarnessEvalError(f"missing receipts directory: {directory}")
    receipts: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise HarnessEvalError(f"malformed receipt at {path}: {exc}") from exc
        if not isinstance(value, dict):
            raise HarnessEvalError(f"receipt root must be an object: {path}")
        declared = value.pop("receipt_sha256", None)
        validate_receipt(value)
        if declared is not None and declared != receipt_digest(value):
            raise HarnessEvalError(
                f"receipt at {path} does not match its own receipt_sha256; it "
                "was modified after it was written")
        receipts.append(value)
    if not receipts:
        raise HarnessEvalError(f"no receipts found under {directory}")
    return receipts


# --- measurement aggregation ------------------------------------------------

def experiment_digest(experiment: dict[str, Any]) -> str:
    return _digest(experiment)


def metric_names(experiment: dict[str, Any]) -> list[str]:
    decision = experiment.get("decision") or {}
    metric = decision.get("capability_metric")
    if not isinstance(metric, str) or not metric.strip():
        raise HarnessEvalError("decision.capability_metric is required")
    guardrails = decision.get("guardrails") or {}
    if not isinstance(guardrails, dict):
        raise HarnessEvalError("decision.guardrails must be a mapping")
    # Sorted for a stable artifact; the capability metric first because it is
    # the one the decision turns on.
    return [metric] + sorted(k for k in guardrails if k != metric)


def required_arms(experiment: dict[str, Any]) -> list[str]:
    arms = experiment.get("arms") or list(hd.REQUIRED_ARMS)
    if not isinstance(arms, list) or not arms:
        raise HarnessEvalError("experiment.arms must be a non-empty list")
    missing = [arm for arm in hd.REQUIRED_ARMS if arm not in arms]
    if missing:
        raise HarnessEvalError(f"experiment is missing required arms: {missing}")
    return list(arms)


def aggregate(experiment: dict[str, Any], case_set: dict[str, Any],
              receipts: list[dict[str, Any]], *,
              allow_fixture: bool = False) -> dict[str, Any]:
    """Turn receipts into the measurement artifact harness_distill consumes.

    Every number here is derived. There is no code path that accepts a score
    from outside: that is the whole point, because harness_distill.evaluate()
    cannot tell a measured rate from a typed one and this is where the
    difference is established.
    """
    validate_case_set(case_set)
    arms = required_arms(experiment)
    metrics = metric_names(experiment)
    ids = case_ids(case_set)
    set_digest = case_set_digest(case_set)
    exp_digest = experiment_digest(experiment)

    declared = scorers_for(case_set)
    unscored = [m for m in metrics if m not in declared]
    if unscored:
        raise HarnessEvalError(
            f"the case set declares no scorer for {unscored}; the experiment "
            "decides on metrics this set cannot measure")

    # --- index receipts, refusing any ambiguity about what a receipt is for
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    for receipt in receipts:
        validate_receipt(receipt)
        if receipt["experiment_digest"] != exp_digest:
            raise HarnessEvalError(
                f"receipt {receipt['arm']}/{receipt['case_id']} was produced for "
                "a different experiment definition (experiment_digest "
                "mismatch); scores from another experiment are not evidence "
                "for this one")
        if receipt["case_set_digest"] != set_digest:
            raise HarnessEvalError(
                f"receipt {receipt['arm']}/{receipt['case_id']} names case-set "
                f"digest {receipt['case_set_digest'][:12]}... but this set is "
                f"{set_digest[:12]}...; the arms did not answer the same "
                "questions")
        if receipt["arm"] not in arms:
            raise HarnessEvalError(
                f"receipt names arm {receipt['arm']!r}, which the experiment "
                f"does not declare (declared: {arms})")
        if receipt["case_id"] not in ids:
            raise HarnessEvalError(
                f"receipt names case {receipt['case_id']!r}, which is not in the "
                "case set")
        key = (receipt["arm"], receipt["case_id"])
        if key in indexed:
            raise HarnessEvalError(
                f"duplicate receipt for arm {key[0]!r} case {key[1]!r}; the same "
                "execution counted twice is not two measurements")
        indexed[key] = receipt

    missing_pairs = [f"{arm}/{cid}" for arm in arms for cid in ids
                     if (arm, cid) not in indexed]
    if missing_pairs:
        raise HarnessEvalError(
            f"{len(missing_pairs)} arm/case pair(s) have no receipt (first: "
            f"{missing_pairs[0]}). An incomplete arm cannot be compared against "
            "a complete one.")

    # --- identity and honesty gates
    fixture_receipts = [r for r in indexed.values()
                        if not r["executor"]["produces_real_measurements"]]
    if fixture_receipts and not allow_fixture:
        raise HarnessEvalError(
            f"{len(fixture_receipts)} receipt(s) came from an executor that "
            "declares produces_real_measurements: false. A fixture executor "
            "replays declared outcomes; presenting that as a product "
            "measurement would be fabrication. Pass allow_fixture=True (or "
            "--allow-fixture) to aggregate it as an explicitly labelled "
            "fixture artifact.")
    if fixture_receipts and len(fixture_receipts) != len(indexed):
        raise HarnessEvalError(
            "receipts mix fixture and real executors; a measurement that is "
            "partly replayed is not interpretable")
    is_fixture = bool(fixture_receipts)

    if not is_fixture:
        unverified = sorted({f"{r['arm']}/{r['case_id']}" for r in indexed.values()
                             if r["prompt_verified"] is not True
                             or not isinstance(r["prompt_sha256"], str)})
        if unverified:
            raise HarnessEvalError(
                f"{len(unverified)} receipt(s) carry an unverified prompt "
                f"identity (first: {unverified[0]}). Attribution naming a "
                "prompt nobody checked is worse than none: it looks like "
                "evidence.")
        if case_set.get("approved") is not True:
            raise HarnessEvalError(
                f"case set {case_set['name']!r} is not approved; a real "
                "measurement needs an approved held-out set")

    # --- per-arm identity must be consistent within the arm
    arm_identity: dict[str, dict[str, Any]] = {}
    for arm in arms:
        rows = [indexed[(arm, cid)] for cid in ids]
        for field in ("model_registry", "model_id", "model_revision",
                      "harness_name", "harness_digest", "prompt_sha256"):
            values = {json.dumps(r[field], sort_keys=True) for r in rows}
            if len(values) != 1:
                raise HarnessEvalError(
                    f"arm {arm!r} mixes {len(values)} values of {field!r}; one "
                    "arm is one model under one harness, by construction")
        first = rows[0]
        arm_identity[arm] = {
            "model_registry": first["model_registry"],
            "model_id": first["model_id"],
            "model_revision": first["model_revision"],
            "harness_name": first["harness_name"],
            "harness_digest": first["harness_digest"],
            "prompt_sha256": first["prompt_sha256"],
            "prompt_verified": first["prompt_verified"],
        }

    # --- score
    scores: dict[str, dict[str, float]] = {}
    evidence: dict[str, Any] = {}
    for arm in arms:
        rows = {cid: indexed[(arm, cid)] for cid in ids}
        per_metric: dict[str, Any] = {}
        arm_scores: dict[str, float] = {}
        for metric in metrics:
            # The denominator is DECLARED coverage, not observed success. A
            # case that errored still asked its question; dropping it would let
            # a flaky arm score 1.0 on the two cases that happened to run.
            applicable = [c for c in case_set["cases"]
                          if metric in c["expected"]["scorers"]]
            denominator = len(applicable)
            if denominator == 0:
                raise HarnessEvalError(
                    f"no case declares scorer {metric!r}; it has no denominator")
            numerator = 0
            errored = refused = 0
            for case in applicable:
                receipt = rows[str(case["case_id"])]
                if receipt["status"] == STATUS_ERROR:
                    errored += 1
                    continue
                if receipt["status"] == STATUS_REFUSED:
                    refused += 1
                    continue
                if metric not in receipt["scores"]:
                    # An 'ok' receipt that never ran a declared scorer is a hole
                    # in the evidence, not a silent zero.
                    raise HarnessEvalError(
                        f"receipt {arm}/{case['case_id']} is 'ok' but records no "
                        f"result for declared scorer {metric!r}")
                if receipt["scores"][metric] is True:
                    numerator += 1
            score = numerator / denominator
            if not 0.0 <= score <= 1.0:      # unreachable by construction; the
                raise HarnessEvalError(      # check is the invariant, not a hope
                    f"{arm}.{metric} scored {score!r}, outside [0, 1]")
            arm_scores[metric] = score
            per_metric[metric] = {
                "numerator": numerator,
                "denominator": denominator,
                "score": score,
                "error": errored,
                "refused": refused,
                # Structurally zero: a case with no receipt refuses aggregation
                # above, and an 'ok' receipt missing a declared scorer raises.
                # The field is kept so the artifact's shape does not depend on
                # which failure happened to be absent.
                "missing": 0,
            }
        scores[arm] = arm_scores
        evidence[arm] = {
            "identity": arm_identity[arm],
            "cases": len(ids),
            "metrics": per_metric,
            "status_counts": {
                status: sum(1 for r in rows.values() if r["status"] == status)
                for status in RECEIPT_STATUSES
            },
        }

    capability = metrics[0]
    student_gain = scores["student_candidate"][capability] - \
        scores["student_current"][capability]
    teacher_gain = scores["teacher_candidate"][capability] - \
        scores["teacher_current"][capability]

    return {
        "schema_version": MEASUREMENT_SCHEMA,
        "experiment": experiment.get("name"),
        "experiment_digest": exp_digest,
        "case_set": {
            "name": case_set["name"],
            "digest": set_digest,
            "cases": len(ids),
            "approved": case_set["approved"],
            "split": case_set["split"],
            "source_class": case_set["source_class"],
        },
        # Stated at the top level, in the artifact itself, so a reader who never
        # opens the receipts still cannot mistake replayed fixture data for a
        # product measurement.
        "measurement_class": "fixture" if is_fixture else "measured",
        "produces_real_measurements": not is_fixture,
        "receipt_set_digest": receipt_set_digest(list(indexed.values())),
        "receipts": len(indexed),
        "metrics": metrics,
        "arms": scores,
        "evidence": evidence,
        "interaction": {
            "student_harness_gain": student_gain,
            "teacher_harness_gain": teacher_gain,
            "interaction": teacher_gain - student_gain,
            "residual_model_gap": scores["teacher_current"][capability]
            - scores["student_candidate"][capability],
        },
        "training_authorized": False,
    }
