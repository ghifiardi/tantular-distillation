"""The live-Office slice: approval evidence, surface honesty, no blind retry.

Nothing here opens a document, calls a model, or touches the network. The
companion is a fake, the transport is a fake, and one test breaks `socket` for
the whole path to prove that structurally rather than by assertion.
"""
from __future__ import annotations

import asyncio
import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import harness_distill as hd
import harness_eval as he
import harness_executors as hx
import harness_verifiers as hv
import run_harness_evaluation as rhe

EXPERIMENT = ROOT / "configs" / "experiments" / "harness-before-weights.yaml"
FIXTURE_CASES = ROOT / "tests" / "fixtures" / "harness_cases" / "fixture-office-v1.yaml"
HEX = "a" * 64


@pytest.fixture
def experiment():
    return hd.load_yaml(EXPERIMENT)


@pytest.fixture
def case_set():
    return he.load_case_set(FIXTURE_CASES)


# --- the bridge: retries are for reads only ---------------------------------

class _AlwaysTimesOut:
    """Counts attempts so the retry policy is observed, not assumed."""

    def __init__(self):
        self.calls = 0

    async def post(self, *_args, **kwargs):
        import httpx
        self.calls += 1
        self.headers = kwargs.get("headers", {})
        raise httpx.TimeoutException("no answer")


def _complete(client, **kwargs):
    import bridge_client
    teacher = bridge_client.TeacherClient(base_url="http://x", model="m",
                                          max_retries=3)
    return asyncio.run(teacher.complete(client, [{"role": "user", "content": "hi"}],
                                        **kwargs))


def test_a_read_still_retries():
    client = _AlwaysTimesOut()
    with pytest.raises(RuntimeError):
        _complete(client)
    assert client.calls == 3, "a generation that times out is safe to retry"


def test_a_mutating_call_is_never_retried():
    """A timeout does not mean the request was not received. Retrying a
    mutating call is how one approved edit becomes two applied ones."""
    client = _AlwaysTimesOut()
    with pytest.raises(RuntimeError, match="NOT retried"):
        _complete(client, mutating=True)
    assert client.calls == 1


def test_an_idempotency_key_travels_as_a_header():
    client = _AlwaysTimesOut()
    with pytest.raises(RuntimeError):
        _complete(client, mutating=True, idempotency_key="key-1")
    assert client.headers.get("Idempotency-Key") == "key-1"


# --- verifiers: declared is not implemented ---------------------------------

def test_request_schema_accepts_the_real_contract_shape():
    result = hv.request_schema(json.dumps(
        {"edits": [{"find": "a", "replace": "b", "occurrence": 1}]}))
    assert result["outcome"] == hv.PASSED
    assert result["edits"] == 1


@pytest.mark.parametrize("payload, reason", [
    ("not json at all", "no JSON"),
    ('{"edits": []}', "no edits"),
    ('{"edits": [{"replace": "b"}]}', "no usable 'find'"),
    ('{"edits": [{"find": "a"}]}', "no 'replace'"),
    ('{"edits": [{"find": "a", "replace": "b", "occurrence": 0}]}', "positive integer"),
])
def test_request_schema_rejects_what_the_product_would_reject(payload, reason):
    result = hv.request_schema(payload)
    assert result["outcome"] == hv.FAILED
    assert reason in result["detail"]


def test_request_schema_enforces_the_add_in_limits():
    too_many = json.dumps({"edits": [{"find": "a", "replace": "b"}] * (hv.MAX_EDITS + 1)})
    assert hv.request_schema(too_many)["outcome"] == hv.FAILED
    too_long = json.dumps({"edits": [{"find": "x" * (hv.MAX_FIND + 1), "replace": "b"}]})
    assert hv.request_schema(too_long)["outcome"] == hv.FAILED


def test_target_location_refuses_without_a_live_resolution():
    """It reads the add-in's answer about the OPEN document. There is no
    text-copy substitute, and an unevaluated check is not a pass."""
    result = hv.target_location(None)
    assert result["outcome"] == hv.REFUSED
    assert result["passed"] is False
    assert "cannot be evaluated from a text copy" in result["detail"]


def test_target_location_consumes_a_real_resolution():
    ok = hv.target_location({"ordinal": 2, "target_digest": HEX})
    assert ok["outcome"] == hv.PASSED and ok["ordinal"] == 2
    moved = hv.target_location({"error": "ambiguous"})
    assert moved["outcome"] == hv.FAILED
    no_digest = hv.target_location({"ordinal": 0})
    assert no_digest["outcome"] == hv.REFUSED


def test_faithful_edit_refuses_until_an_approved_set_exists():
    result = hv.faithful_edit()
    assert result["outcome"] == hv.REFUSED
    assert "No approved held-out" in result["detail"]


def test_a_declared_check_with_no_implementation_refuses_it_does_not_vanish():
    results = hv.run_declared("before_action",
                              ["request_schema", "target_location", "invented"],
                              {"request_schema": '{"edits":[{"find":"a","replace":"b"}]}'})
    by_name = {r["check"]: r for r in results}
    assert len(results) == 3, "a declared check must never silently disappear"
    assert by_name["request_schema"]["outcome"] == hv.PASSED
    assert by_name["target_location"]["outcome"] == hv.REFUSED
    assert by_name["invented"]["outcome"] == hv.REFUSED
    assert all(r["passed"] is (r["outcome"] == hv.PASSED) for r in results)


def test_the_declared_harness_checks_all_resolve_or_refuse():
    """Against the real shipped harnesses, so the module cannot drift from what
    configs/harnesses/*.yaml actually declares."""
    for name in ("tantular-office-current", "tantular-office-candidate"):
        spec = hd.load_harness(name)
        for phase in ("before_action", "after_action"):
            declared = (spec["verification"] or {}).get(phase) or []
            results = hv.run_declared(phase, declared, {})
            assert len(results) == len(declared), (name, phase)
            assert not any(r["outcome"] == hv.PASSED for r in results), (
                "with no inputs supplied, nothing may report a pass")


# --- the live executor ------------------------------------------------------

EDIT = {"find": "lama", "replace": "baru", "occurrence": 1}


class FakeCompanion:
    """The companion's decisions, without the companion."""

    def __init__(self, *, prepare_ok=True, execute_ok=True, status="applied",
                 refuse_reason="document_changed"):
        self.prepare_ok = prepare_ok
        self.execute_ok = execute_ok
        self.status = status
        self.refuse_reason = refuse_reason
        self.executes = 0

    def post(self, path, payload):
        """The companion's two decisions. The executor owns these payloads."""
        if path == hx.PREPARE_PATH:
            if not self.prepare_ok:
                return {"ok": False, "reason": "not_located"}
            return {"ok": True, "token": "tok-1", "nonce": "nonce-1"}
        self.executes += 1
        if not self.execute_ok:
            return {"ok": False, "reason": self.refuse_reason}
        return {"ok": True, "idempotency_key": "idem-1", "nonce": "nonce-1",
                "document_version": HEX, "target_digest": HEX,
                "edit_digest": HEX, "approver": "local-user"}

    def apply(self, idempotency_key, case_id):
        """What the PANE observed. This client never reports an outcome."""
        return {"status": self.status, "steps": 2, "wall_seconds": 3.0,
                "scores": {"capability_pass_rate": True,
                           "indonesian_voice": True,
                           "edit_contract_output": True},
                "before_action": [{"check": "request_schema", "passed": True}],
                "after_action": [{"check": "edit_contract", "passed": True}]}


def live_request(case_overrides=None):
    case = {"case_id": "live-001",
            "request": {"edits": [EDIT]},
            "expected": {"scorers": ["capability_pass_rate", "indonesian_voice",
                                     "edit_contract_output"]},
            "expects_state_change": True, "requires_approval": True}
    case.update(case_overrides or {})
    return hx.ExecutionRequest(
        case=case, arm="student_current", model_registry="qwen35-9b-instruct",
        model_id="Qwen/Qwen3.5-9B", model_revision="c" * 40,
        harness_name="tantular-office-current", harness_digest=HEX,
        prompt_sha256=HEX, prompt_verified=True,
        tool_policy={"allow": ["office_read", "office_edit"],
                     "state_change_requires_approval": True},
        verification_policy={"repair_attempts": 0},
        budgets={"max_steps": 4, "max_wall_seconds": 300}, run_id="r1")


def executor(companion=None, digest="d" * 64):
    return hx.OfficeLiveExecutor(companion or FakeCompanion(),
                                 served_model_digest=digest,
                                 companion_boot_id="boot-1")


def test_a_live_execution_carries_its_approval():
    result = executor().execute(live_request())
    assert result.error is None
    assert result.approval["single_use"] is True
    assert result.approval["token_id"] == "tok-1"
    assert result.approval["idempotency_key"] == "idem-1"
    assert result.tool_calls[0]["approved"] is True


def test_a_refused_approval_produces_no_scores():
    for companion in (FakeCompanion(prepare_ok=False),
                      FakeCompanion(execute_ok=False)):
        result = executor(companion).execute(live_request())
        assert result.error and "approval refused" in result.error
        assert result.scores == {}
        assert result.approval is None


def test_an_expired_or_replayed_approval_is_reported_by_reason():
    for reason in ("expired", "unknown_token", "duplicate", "target_moved"):
        companion = FakeCompanion(execute_ok=False, refuse_reason=reason)
        result = executor(companion).execute(live_request())
        assert reason in result.error


def test_without_a_served_model_digest_the_executor_refuses():
    """A tag is a name, not an identity: it can be repointed under a running
    install, so a receipt naming only a tag cannot say which weights answered."""
    result = executor(digest=None).execute(live_request())
    assert result.error and "no served-model digest" in result.error
    assert result.scores == {}


def test_more_than_one_edit_is_refused_not_half_applied():
    request = live_request({"request": {"edits": [EDIT, dict(EDIT, replace="lain")]}})
    result = executor().execute(request)
    assert "exactly one edit" in result.error
    assert "Batch atomicity is unsolved" in result.error


def test_the_executor_never_applies_anything_itself():
    """It holds no Office handle by construction, so a bug cannot make it edit."""
    live = executor()
    assert not hasattr(live, "apply")
    assert live.identity()["execution_surface"] == "office_live"
    assert live.identity()["produces_real_measurements"] is True


def test_real_office_executor_still_refuses():
    with pytest.raises(he.HarnessEvalError, match="no Office harness adapter"):
        hx.RealOfficeExecutor().execute(live_request())


# --- receipts and aggregation ----------------------------------------------

def live_receipt(**overrides):
    """A minimal valid office_live receipt, to be broken one field at a time."""
    receipt = {
        "schema_version": he.RECEIPT_SCHEMA,
        "experiment": "e", "experiment_digest": HEX,
        "case_set_name": "s", "case_set_digest": HEX,
        "case_id": "live-001", "arm": "student_current",
        "model_registry": "qwen35-9b-instruct", "model_id": "Qwen/Qwen3.5-9B",
        "model_revision": "c" * 40,
        "harness_name": "h", "harness_digest": HEX,
        "harness_provenance": {"digest": HEX, "name": "h",
                               "execution_model_registry": "qwen35-9b-instruct",
                               "prompt_sha256": HEX, "prompt_verified": True},
        "prompt_sha256": HEX, "prompt_verified": True,
        "tools_offered": ["office_edit"],
        "tool_calls": [{"tool": "office_edit", "approved": True}],
        "approvals": [], "before_action": [], "after_action": [],
        "repair_attempts": 0, "termination_reason": "completed",
        "budgets": {"max_steps": 4, "max_wall_seconds": 300},
        "budget_consumed": {"steps": 1, "wall_seconds": 1.0},
        "result_digest": HEX,
        "executor": {"name": "office-live", "version": "1", "kind": "real",
                     "produces_real_measurements": True},
        "started_at": "s", "ended_at": "e", "status": he.STATUS_OK,
        "scores": {"capability_pass_rate": True},
        "execution_surface": he.SURFACE_LIVE,
        "approval": {"token_id": "t", "approver": "local-user",
                     "document_version": HEX, "target_digest": HEX,
                     "edit_digest": HEX, "nonce": "n",
                     "idempotency_key": "k", "single_use": True},
        "training_authorized": False,
    }
    receipt.update(overrides)
    return receipt


def test_a_live_state_changing_receipt_without_approval_is_refused():
    with pytest.raises(he.HarnessEvalError, match="no approval evidence"):
        he.validate_receipt(live_receipt(approval=None))


@pytest.mark.parametrize("field", ["token_id", "approver", "document_version",
                                   "target_digest", "edit_digest", "nonce",
                                   "idempotency_key"])
# single_use is deliberately absent from this list: it is a boolean, so a false
# value means "replayable", which the next test covers with its own message.
def test_every_approval_field_is_load_bearing(field):
    approval = dict(live_receipt()["approval"])
    approval[field] = ""
    with pytest.raises(he.HarnessEvalError, match="missing"):
        he.validate_receipt(live_receipt(approval=approval))


def test_a_replayable_approval_is_refused():
    approval = dict(live_receipt()["approval"], single_use=False)
    with pytest.raises(he.HarnessEvalError, match="single_use must be true"):
        he.validate_receipt(live_receipt(approval=approval))


def test_approval_bindings_must_be_digests_not_labels():
    approval = dict(live_receipt()["approval"], document_version="version-2")
    with pytest.raises(he.HarnessEvalError, match="64 lowercase hex"):
        he.validate_receipt(live_receipt(approval=approval))


def test_a_fixture_executor_cannot_claim_office_live():
    executor_block = dict(live_receipt()["executor"],
                          produces_real_measurements=False)
    with pytest.raises(he.HarnessEvalError, match="cannot claim execution_surface"):
        he.validate_receipt(live_receipt(executor=executor_block))


def test_a_read_only_live_receipt_needs_no_approval():
    """office_read changes nothing, so requiring an approval for it would make
    the approval meaningless by making it universal."""
    he.validate_receipt(live_receipt(
        tool_calls=[{"tool": "office_read"}], approval=None))


def test_an_unknown_execution_surface_is_refused():
    with pytest.raises(he.HarnessEvalError, match="execution_surface must be"):
        he.validate_receipt(live_receipt(execution_surface="probably_office"))


def test_fixture_receipts_report_the_text_surface(experiment, case_set, tmp_path):
    directory = tmp_path / "receipts"
    rhe.run_evaluation(experiment, case_set, hx.FakeOfficeExecutor(), directory)
    receipts = he.load_receipts(directory)
    assert {r["execution_surface"] for r in receipts} == {he.SURFACE_TEXT}
    measurements = he.aggregate(experiment, case_set, receipts, allow_fixture=True)
    assert measurements["execution_surface"] == he.SURFACE_TEXT
    assert measurements["measurement_class"] == "fixture"


def test_mixed_surfaces_cannot_be_one_measurement(experiment, case_set, tmp_path):
    """Reachable only when BOTH executors are real: a fixture run is refused
    earlier, for a different and equally correct reason. The case that matters
    is a document-text adapter and a live one in one measurement."""
    directory = tmp_path / "receipts"
    rhe.run_evaluation(experiment, case_set, hx.FakeOfficeExecutor(), directory)
    receipts = [copy.deepcopy(r) for r in he.load_receipts(directory)]

    approved = copy.deepcopy(case_set)
    approved["approved"] = True
    approved["split"] = "held_out"
    approved["source_class"] = "real_office"
    digest = he.case_set_digest(approved)
    for receipt in receipts:
        receipt["case_set_digest"] = digest
        receipt["executor"]["produces_real_measurements"] = True
    # One arm ran live; the rest ran against a document string.
    receipts[0]["execution_surface"] = he.SURFACE_LIVE
    receipts[0]["approval"] = live_receipt()["approval"]

    with pytest.raises(he.HarnessEvalError, match="mix execution surfaces"):
        he.aggregate(experiment, approved, receipts)


def test_training_is_never_authorized_on_the_live_path():
    receipt = live_receipt()
    assert receipt["training_authorized"] is False
    with pytest.raises(he.HarnessEvalError, match="must be false"):
        he.validate_receipt(live_receipt(training_authorized=True))


def test_no_socket_is_opened_anywhere_on_this_path(monkeypatch):
    import socket

    def forbidden(*args, **kwargs):
        raise AssertionError("the live protocol must not open a socket here")

    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    result = executor().execute(live_request())
    assert result.approval["single_use"] is True
    he.validate_receipt(live_receipt())
    hv.run_declared("before_action", ["request_schema", "target_location"], {})
