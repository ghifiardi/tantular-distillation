"""The executor boundary: what actually runs one case under one harness.

The evaluation controller must not import a model SDK. If it did, the thing
that decides whether a measurement is admissible would also be the thing that
produces it, and every gate in src/harness_eval.py would be checking its own
work. So the controller talks to an ``Executor``: a narrow protocol that takes
a fully-specified request and returns a fully-specified result.

Two executors ship here:

  FakeOfficeExecutor   REPLAYS outcomes declared in the case set's `fixture`
                       block. Deterministic, offline, and honest about it:
                       ``produces_real_measurements`` is False, which
                       src/harness_eval.py refuses to aggregate as a product
                       measurement.

  RealOfficeExecutor   REFUSES. The Office harness declares tools, verifiers,
                       an approval policy and companion-process isolation, and
                       nothing in this repository can supply them yet -- which
                       is the same fact configs/harnesses/*.yaml already record
                       as ``trace_generation.supported_by_generate_py: false``.
                       It fails closed rather than degrading to a chat call
                       wearing the harness's name.

NOTHING HERE CALLS A MODEL, opens a socket, or touches a document.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from harness_eval import HarnessEvalError

# Which declared tools change Office state. The harness YAML lists the tools it
# allows but does not say which of them mutate, and the approval gate needs to
# know. Declared here, and UNKNOWN TOOLS COUNT AS STATE-CHANGING: a tool nobody
# classified is exactly the one that should need approval.
READ_ONLY_TOOLS = frozenset({"office_read"})


def changes_state(tool: str) -> bool:
    return tool not in READ_ONLY_TOOLS


# Terminal reasons an execution stopped. Recorded on every receipt so a refusal
# can be told apart from a crash and from an ordinary completion.
TERMINATION_COMPLETED = "completed"
TERMINATION_BUDGET_STEPS = "budget_exceeded_steps"
TERMINATION_BUDGET_WALL = "budget_exceeded_wall_seconds"
TERMINATION_TOOL_NOT_ALLOWED = "tool_not_allowed"
TERMINATION_APPROVAL_MISSING = "approval_not_granted"
TERMINATION_REPAIR_LIMIT = "repair_attempts_exceeded"
TERMINATION_EXECUTOR_ERROR = "executor_error"
TERMINATION_NO_ADAPTER = "no_office_adapter"


@dataclass(frozen=True)
class ExecutionRequest:
    """Everything an executor is allowed to know, and nothing more.

    Deliberately complete: an executor that had to look up the harness itself
    could read a different one than the controller recorded on the receipt.
    """
    case: dict[str, Any]
    arm: str
    model_registry: str
    model_id: str
    model_revision: str
    harness_name: str
    harness_digest: str
    prompt_sha256: str | None
    prompt_verified: bool
    tool_policy: dict[str, Any]
    verification_policy: dict[str, Any]
    budgets: dict[str, Any]
    run_id: str
    repetition: int = 0

    @property
    def case_id(self) -> str:
        return str(self.case["case_id"])


@dataclass
class ExecutionResult:
    """What an executor observed. The controller judges it; it does not judge
    itself -- an executor that could declare its own run policy-compliant would
    make the approval and budget gates advisory."""
    output: Any = None
    tools_offered: list[str] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    approvals: list[dict[str, Any]] = field(default_factory=list)
    before_action: list[dict[str, Any]] = field(default_factory=list)
    after_action: list[dict[str, Any]] = field(default_factory=list)
    repair_attempts: int = 0
    steps: int = 0
    wall_seconds: float = 0.0
    scores: dict[str, bool] = field(default_factory=dict)
    error: str | None = None
    started_at: str = ""
    ended_at: str = ""


@runtime_checkable
class Executor(Protocol):
    """The whole surface the controller depends on."""

    def identity(self) -> dict[str, Any]:
        """{name, version, kind, produces_real_measurements}."""

    def execute(self, request: ExecutionRequest) -> ExecutionResult:
        ...


class FakeOfficeExecutor:
    """Replays the outcomes a case set declares. Never a measurement source.

    It exists so the controller, the receipt schema, the budget and approval
    gates and the aggregation arithmetic can all be tested deterministically
    without an Office adapter, a model, or a network. Its identity says so, and
    src/harness_eval.py refuses to aggregate it as a product measurement unless
    the caller explicitly asks for a labelled fixture artifact.
    """

    name = "fake-office-replay"
    version = "1"

    def identity(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "kind": "fake",
            "produces_real_measurements": False,
            "note": "replays declared fixture outcomes; no model, no Office, "
                    "no network",
        }

    def execute(self, request: ExecutionRequest) -> ExecutionResult:
        fixture = (request.case.get("fixture") or {})
        arms = fixture.get("arms")
        if not isinstance(arms, dict) or request.arm not in arms:
            # A fixture case that says nothing about this arm cannot be replayed.
            # Inventing an outcome here is precisely the fabrication the whole
            # module is built to prevent.
            return ExecutionResult(
                error=f"case {request.case_id!r} declares no fixture outcome for "
                      f"arm {request.arm!r}",
                started_at=self._stamp(request, "start"),
                ended_at=self._stamp(request, "end"),
            )
        declared = arms[request.arm]
        if not isinstance(declared, dict):
            return ExecutionResult(
                error=f"fixture.arms.{request.arm} must be a mapping",
                started_at=self._stamp(request, "start"),
                ended_at=self._stamp(request, "end"),
            )
        scores = declared.get("scores") or {}
        if not isinstance(scores, dict) or \
                not all(isinstance(v, bool) for v in scores.values()):
            return ExecutionResult(
                error=f"fixture.arms.{request.arm}.scores must map metric -> bool",
                started_at=self._stamp(request, "start"),
                ended_at=self._stamp(request, "end"),
            )
        return ExecutionResult(
            output=declared.get("output", f"fixture:{request.case_id}"),
            tools_offered=list(request.tool_policy.get("allow") or []),
            tool_calls=[dict(c) for c in (declared.get("tool_calls") or [])],
            approvals=[dict(a) for a in (declared.get("approvals") or [])],
            before_action=[dict(c) for c in (declared.get("before_action") or [])],
            after_action=[dict(c) for c in (declared.get("after_action") or [])],
            repair_attempts=int(declared.get("repair_attempts", 0)),
            steps=int(declared.get("steps", 1)),
            wall_seconds=float(declared.get("wall_seconds", 1.0)),
            scores=dict(scores),
            error=declared.get("error"),
            started_at=self._stamp(request, "start"),
            ended_at=self._stamp(request, "end"),
        )

    @staticmethod
    def _stamp(request: ExecutionRequest, edge: str) -> str:
        """A deterministic fixture equivalent of a timestamp.

        A wall-clock time would make every receipt differ on every run, so the
        receipt-set digest would never reproduce and the fixture would be
        useless as a regression test. The run id and case identify the slot; the
        edge says which end of it.
        """
        return (f"fixture:{request.run_id}:{request.arm}:{request.case_id}:"
                f"{request.repetition}:{edge}")


class RealOfficeExecutor:
    """Fails closed: this repository has no Office harness adapter.

    Running the declared harness means supplying office_read/office_edit,
    enforcing the approval policy, running request_schema / target_location
    before the action and edit_contract / faithful_edit after it, honouring the
    repair budget, and doing all of it under companion-process or sandbox
    isolation. None of that exists here -- the same gap the harness configs
    record as trace_generation.supported_by_generate_py: false.

    The honest failure is a refusal. A chat call dressed in the harness's name
    would produce receipts that assert every one of those policies ran.
    """

    name = "office-adapter"
    version = "0"

    def identity(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "kind": "real",
            "produces_real_measurements": True,
            "available": False,
        }

    def execute(self, request: ExecutionRequest) -> ExecutionResult:
        raise HarnessEvalError(
            "no Office harness adapter is available. Executing "
            f"{request.harness_name!r} requires a runner that supplies the "
            "declared tools, enforces the approval policy, runs the "
            "before_action and after_action verifiers and honours the repair "
            "budget under the declared isolation. src/generate.py is not that "
            "runner (see trace_generation.supported_by_generate_py: false), and "
            "no other adapter exists in this repository. Build the adapter "
            "before asking for a real measurement; do not substitute the fake "
            "executor, whose results are replayed fixtures.")


EXECUTORS: dict[str, type] = {
    "fake": FakeOfficeExecutor,
    "office": RealOfficeExecutor,
}


def get_executor(name: str) -> Any:
    if name not in EXECUTORS:
        raise HarnessEvalError(
            f"unknown executor {name!r} (have: {sorted(EXECUTORS)})")
    return EXECUTORS[name]()
