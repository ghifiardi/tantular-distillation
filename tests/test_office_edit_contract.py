"""The cross-repository wire contract, pinned by a literal fixture.

contract/office-edit-protocol.v1.json is byte-identical in this repository and
in ghifiardi/LLM-Indonesia. Both sides test against it and both assert its
digest, so a change made on one side and not mirrored on the other fails here
rather than drifting silently until a live run behaves unexpectedly.

The fixture is LITERAL throughout. A fixture that computed its expectations
from the code would agree with whatever the code happened to do, which is the
one thing a contract test must not do.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import harness_executors as hx

CONTRACT_PATH = ROOT / "contract" / "office-edit-protocol.v1.json"

# The other repository asserts this same literal. If the two ever disagree, one
# of them was edited alone -- which is precisely the failure this pins.
CONTRACT_SHA256 = "adec2e01c85be000461f48914013bf3dabde770e85210ccfe57bf642ce18f102"


@pytest.fixture(scope="module")
def contract():
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def test_the_contract_file_is_the_one_both_repositories_pin():
    digest = hashlib.sha256(CONTRACT_PATH.read_bytes()).hexdigest()
    assert digest == CONTRACT_SHA256, (
        "contract/office-edit-protocol.v1.json changed. Mirror the identical "
        "file into ghifiardi/LLM-Indonesia and update the literal in BOTH "
        "repositories, or the two sides have silently diverged.")


def test_this_client_speaks_the_contract_version(contract):
    assert hx.OFFICE_EDIT_PROTOCOL_VERSION == contract["protocol_version"]


def test_the_routes_match(contract):
    routes = contract["routes"]
    assert hx.PREPARE_PATH == routes["prepare"]
    assert hx.EXECUTE_PATH == routes["execute"]
    assert hx.RESULT_PATH == routes["result"]
    assert hx.DIAGNOSTICS_PATH == routes["diagnostics"]


def test_every_refusal_reason_in_the_contract_is_one_this_client_knows(contract):
    declared = set()
    for section in ("prepare", "execute", "result"):
        for key in ("response_refused", "response_duplicate"):
            block = contract[section].get(key) or {}
            declared.update(block.get("reasons") or [])
            if block.get("reason"):
                declared.add(block["reason"])
    declared.add(contract["transport_errors"]["audit_unwritable"]["reason"])
    unknown = sorted(declared - hx.COMPANION_REFUSALS)
    assert not unknown, (
        f"the companion can return {unknown}, which this client would report as "
        "unrecognised. Teach it the reason or remove it from the contract.")


# --- what the executor actually puts on the wire ----------------------------

HEX = "b" * 64
EDIT = {"find": "Angka lama", "replace": "Angka baru", "occurrence": 1}
DOC = "Angka lama tercatat di sini."
LOCATED = {"matchedText": "Angka lama", "ordinal": 0}


class RecordingTransport:
    """Captures every request and replays scripted responses."""

    def __init__(self, responses=None, applied=None):
        self.posts: list[tuple[str, dict]] = []
        self.responses = responses or {}
        self.applied = applied or {"status": "applied", "steps": 1,
                                   "wall_seconds": 1.0,
                                   "scores": {"capability_pass_rate": True}}

    def post(self, path, payload):
        self.posts.append((path, payload))
        if path in self.responses:
            return self.responses[path]
        if path == hx.PREPARE_PATH:
            return {"ok": True, "token": "tok", "nonce": "n",
                    "protocol_version": 1, "document_version": HEX,
                    "target_digest": HEX, "edit_digest": HEX,
                    "disclosure": {}, "expiresAt": 0}
        return {"ok": True, "idempotency_key": "idem", "nonce": "n",
                "protocol_version": 1, "document_version": HEX,
                "target_digest": HEX, "edit_digest": HEX,
                "approver": "local-user"}

    def apply(self, idempotency_key, case_id):
        return dict(self.applied)


def live_request(case_request=None):
    return hx.ExecutionRequest(
        case={"case_id": "c1",
              "request": case_request or {"edits": [EDIT], "document": DOC,
                                          "located": LOCATED},
              "expected": {"scorers": ["capability_pass_rate"]},
              "expects_state_change": True, "requires_approval": True},
        arm="student_current", model_registry="qwen35-9b-instruct",
        model_id="Qwen/Qwen3.5-9B", model_revision="c" * 40,
        harness_name="h", harness_digest=HEX, prompt_sha256=HEX,
        prompt_verified=True,
        tool_policy={"allow": ["office_read", "office_edit"],
                     "state_change_requires_approval": True},
        verification_policy={"repair_attempts": 0},
        budgets={"max_steps": 4, "max_wall_seconds": 300}, run_id="r")


def run(transport):
    executor = hx.OfficeLiveExecutor(transport, served_model_digest="d" * 64,
                                     companion_boot_id="boot")
    return executor.execute(live_request())


def test_the_prepare_request_carries_exactly_the_contract_fields(contract):
    transport = RecordingTransport()
    run(transport)
    path, payload = transport.posts[0]
    assert path == contract["routes"]["prepare"]
    for field in contract["prepare"]["request"]["required_fields"]:
        assert field in payload, f"prepare request is missing {field}"


def test_the_execute_request_carries_material_and_never_a_digest(contract):
    transport = RecordingTransport()
    run(transport)
    path, payload = transport.posts[1]
    assert path == contract["routes"]["execute"]
    for field in contract["execute"]["request"]["required_fields"]:
        assert field in payload, f"execute request is missing {field}"
    for field in contract["execute"]["request"]["forbidden_fields"]:
        assert field not in payload, (
            f"execute request asserted {field}. "
            + contract["execute"]["request"]["forbidden_reason"])


def test_the_client_posts_exactly_two_requests_and_never_the_result(contract):
    """RESULT_PATH is the PANE's obligation: only it holds an Office handle and
    can say what Word did. A client reporting an outcome it did not observe
    would put a guess into the audit log."""
    transport = RecordingTransport()
    run(transport)
    assert [p for p, _ in transport.posts] == [contract["routes"]["prepare"],
                                               contract["routes"]["execute"]]
    assert contract["routes"]["result"] not in [p for p, _ in transport.posts]


def test_only_one_edit_is_ever_sent(contract):
    assert contract["scope"]["max_edits_per_approval"] == 1
    transport = RecordingTransport()
    result = hx.OfficeLiveExecutor(transport, served_model_digest="d" * 64).execute(
        live_request({"edits": [EDIT, dict(EDIT, replace="lain")],
                      "document": DOC, "located": LOCATED}))
    assert "exactly one edit" in result.error
    assert transport.posts == [], "a batch must be refused before anything is sent"


# --- every declared outcome is interpreted ----------------------------------

@pytest.mark.parametrize("reason", [
    "expired", "unknown_token", "document_changed", "target_moved",
    "edit_changed", "duplicate",
])
def test_each_declared_refusal_is_reported_by_name(reason):
    transport = RecordingTransport(responses={
        hx.EXECUTE_PATH: {"ok": False, "reason": reason}})
    result = run(transport)
    assert reason in result.error
    assert "UNRECOGNISED" not in result.error
    assert result.scores == {}
    assert result.approval is None


def test_a_prepare_refusal_is_reported_at_its_own_stage():
    transport = RecordingTransport(responses={
        hx.PREPARE_PATH: {"ok": False, "reason": "not_located"}})
    result = run(transport)
    assert "refused at prepare" in result.error and "not_located" in result.error
    assert len(transport.posts) == 1, "a refused prepare must not be followed by execute"


def test_an_unknown_refusal_is_flagged_rather_than_passed_through():
    """A companion that grows a new state must not be interpreted by a client
    that has never heard of it."""
    transport = RecordingTransport(responses={
        hx.EXECUTE_PATH: {"ok": False, "reason": "quantum_disapproval"}})
    result = run(transport)
    assert "UNRECOGNISED" in result.error
    assert "quantum_disapproval" in result.error


def test_a_partial_outcome_is_recorded_as_the_status_the_pane_reported():
    for status in ("not_found", "skipped", "error"):
        transport = RecordingTransport(applied={"status": status, "steps": 1,
                                                "wall_seconds": 1.0, "scores": {}})
        result = run(transport)
        assert result.error and status in result.error, (
            "a status other than 'applied' must surface, never be smoothed away")


def test_an_internal_transport_error_becomes_a_failure_not_a_crash():
    class Exploding(RecordingTransport):
        def post(self, path, payload):
            raise ConnectionError("companion went away")

    result = run(Exploding())
    assert "companion prepare failed" in result.error
    assert "ConnectionError" in result.error


def test_a_successful_run_carries_the_full_approval_block(contract):
    result = run(RecordingTransport())
    assert result.error is None
    for field in ("token_id", "approver", "document_version", "target_digest",
                  "edit_digest", "nonce", "idempotency_key", "single_use"):
        assert result.approval[field], f"approval is missing {field}"
    assert result.approval["single_use"] is True


def test_the_diagnostics_identity_requirement_is_enforced(contract):
    assert "digest" in contract["diagnostics"]["response_ok"]["served_model_fields"]
    result = hx.OfficeLiveExecutor(RecordingTransport(),
                                   served_model_digest=None).execute(live_request())
    assert "no served-model digest" in result.error


def test_the_no_retry_obligation_is_stated_and_honoured(contract):
    assert "no_retry_on_execute" in contract["client_obligations"]
    import bridge_client
    import inspect
    source = inspect.getsource(bridge_client.TeacherClient.complete)
    assert "attempts = 1 if mutating" in source
