"""Cleanup after a gate timeout must itself be bounded.

    ./.venv/bin/python -m pytest tests/test_gate_timeout_cleanup.py -q

The timeout handler exists because "a gate that hangs yields no verdict and
stalls the run". It then called `communicate()` with no timeout, which waits for
EOF on both pipes — and EOF arrives only when EVERY writer has closed, including
a descendant that escaped the group kill holding an inherited write end.

Observed 2026-09-10: a gate run sat in select/poll for 82 minutes on 1.02s of
CPU with no children, and never returned. The handler reproduced the failure it
was written to prevent, in a worse form: the 300s timeout at least ends.

Both tests are POSIX-only and clean up after themselves; the integration test's
detached child self-exits so no orphan survives the run.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import run_gates                                              # noqa: E402


class FakeStream:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class FakeProc:
    """Records how it was waited on, and never blocks."""

    def __init__(self, *, communicate_timeouts: int, wait_timeouts: bool = False):
        self.pid = os.getpid()          # a real pid so getpgid works
        self.args = ["node", "--test", "x"]
        self.returncode = None
        self.stdout, self.stderr, self.stdin = FakeStream(), FakeStream(), None
        self._communicate_timeouts = communicate_timeouts
        self._wait_timeouts = wait_timeouts
        self.communicate_calls: list[float | None] = []
        self.wait_calls: list[float | None] = []
        self.killed = False

    def kill(self):
        self.killed = True

    def communicate(self, timeout=None):
        self.communicate_calls.append(timeout)
        if self._communicate_timeouts > 0:
            self._communicate_timeouts -= 1
            raise subprocess.TimeoutExpired(self.args, timeout or 0)
        return ("", "")

    def wait(self, timeout=None):
        self.wait_calls.append(timeout)
        if self._wait_timeouts:
            raise subprocess.TimeoutExpired(self.args, timeout or 0)
        return 0


def test_cleanup_that_also_times_out_still_returns(monkeypatch, capsys):
    """The case that hung: the group stop leaves a writer holding the pipe."""
    monkeypatch.setattr(run_gates.os, "killpg", lambda *a: None, raising=False)
    proc = FakeProc(communicate_timeouts=1, wait_timeouts=True)

    started = time.monotonic()
    with pytest.raises(SystemExit) as exc:
        run_gates.reap_timed_out_gate(proc, "devServerCancellation.test.mjs", 300)
    assert time.monotonic() - started < 5, "the handler must not block"

    # Exactly one communicate, and it carried a deadline.
    assert proc.communicate_calls == [run_gates.GATE_CLEANUP_TIMEOUT_S]
    assert None not in proc.communicate_calls, "no unbounded communicate()"
    # The direct child was waited for, also with a deadline.
    assert proc.wait_calls == [run_gates.GATE_CLEANUP_TIMEOUT_S]
    assert None not in proc.wait_calls, "no unbounded wait()"
    # Our own pipe handles were released rather than read to EOF.
    assert proc.stdout.closed and proc.stderr.closed

    assert exc.value.code == 2, "the gate still fails closed"
    message = capsys.readouterr().err
    assert "detached descendant may still be running" in message
    assert "nothing is left behind" not in message, \
        "that claim is disproved by the condition being reported"
    assert "cleanup also timed out   yes" in message
    assert "process-group stop sent  yes" in message


def test_a_clean_stop_reports_no_surviving_descendant(monkeypatch, capsys):
    monkeypatch.setattr(run_gates.os, "killpg", lambda *a: None, raising=False)
    proc = FakeProc(communicate_timeouts=0)

    with pytest.raises(SystemExit) as exc:
        run_gates.reap_timed_out_gate(proc, "slow.test.mjs", 300)

    assert proc.communicate_calls == [run_gates.GATE_CLEANUP_TIMEOUT_S]
    assert proc.wait_calls == [], "no second wait when the pipes closed"
    assert exc.value.code == 2
    message = capsys.readouterr().err
    assert "cleanup also timed out   no" in message
    assert "its pipes closed" in message


def test_a_process_that_is_already_gone_falls_back_to_the_child(monkeypatch, capsys):
    def gone(*_a):
        raise ProcessLookupError
    monkeypatch.setattr(run_gates.os, "killpg", gone, raising=False)
    proc = FakeProc(communicate_timeouts=0)

    with pytest.raises(SystemExit):
        run_gates.reap_timed_out_gate(proc, "x.test.mjs", 300)
    assert proc.killed, "fell back to killing the direct child"
    assert "process-group stop sent  no (direct child only)" in capsys.readouterr().err


# --- the real thing: a detached descendant holding the inherited pipe --------

# A parent that forks a child into its OWN session, holding the stdout it
# inherited. Killing the parent's process group therefore does not close the
# pipe — the exact condition that hung the handler.
#
# The child records its pid so the test can wait for that ONE process and prove
# it is gone, and it self-exits well inside the test's own wait. Nothing here
# discovers or signals a process it did not create.
DETACHED = r"""
import os, sys, time
pid_file, hold = sys.argv[1], float(sys.argv[2])
if os.fork() == 0:
    os.setsid()
    with open(pid_file, "w") as handle:
        handle.write(str(os.getpid()))
    time.sleep(hold)
    os._exit(0)
sys.stdout.write("parent-started\n")
sys.stdout.flush()
time.sleep(hold * 4)
"""

CHILD_LIFETIME_S = 5.0
CLEANUP_DEADLINE_S = 2.0


def _still_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@pytest.mark.skipif(os.name != "posix", reason="process groups are POSIX-only")
def test_a_detached_descendant_does_not_hang_the_handler(tmp_path, capsys):
    """End to end, with a real escaped writer.

    The invariant under test is a comparison, not a duration: the handler must
    return BEFORE the detached child exits. If it returned only once the child
    died, a passing assertion would prove nothing — it would just be waiting for
    EOF, which is the bug.

    So the child lives 5s, the cleanup deadline is 2s, and the handler must come
    back inside that gap. The test then waits for that exact pid to disappear,
    so it cannot leave an orphan for a later test to trip over.
    """
    pid_file = tmp_path / "detached.pid"
    script = tmp_path / "detached.py"
    script.write_text(DETACHED, encoding="utf-8")

    proc = subprocess.Popen(
        [sys.executable, str(script), str(pid_file), str(CHILD_LIFETIME_S)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        start_new_session=True)

    detached_pid = None
    try:
        # Wait for the child to announce itself, so the pipe really is held.
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not pid_file.exists():
            time.sleep(0.05)
        if not pid_file.exists():
            proc.kill()
            pytest.skip("the fixture never forked; nothing to reap")
        detached_pid = int(pid_file.read_text())

        original = run_gates.GATE_CLEANUP_TIMEOUT_S
        run_gates.GATE_CLEANUP_TIMEOUT_S = CLEANUP_DEADLINE_S
        try:
            started = time.monotonic()
            with pytest.raises(SystemExit) as exc:
                run_gates.reap_timed_out_gate(proc, "detached.py", 2)
            elapsed = time.monotonic() - started
        finally:
            run_gates.GATE_CLEANUP_TIMEOUT_S = original

        # THE invariant: returned before the writer died, so it cannot have
        # been waiting for EOF.
        assert elapsed < CHILD_LIFETIME_S, (
            f"handler took {elapsed:.1f}s against a {CHILD_LIFETIME_S}s child: "
            "it waited for the detached writer")
        assert exc.value.code == 2
        assert "detached descendant may still be running" in capsys.readouterr().err
    finally:
        # Wait for OUR child to self-exit; signal only that exact pid, and only
        # if it overstays. No process is discovered or guessed at.
        if detached_pid is not None:
            deadline = time.monotonic() + CHILD_LIFETIME_S * 3
            while time.monotonic() < deadline and _still_alive(detached_pid):
                time.sleep(0.1)
            if _still_alive(detached_pid):
                try:
                    os.kill(detached_pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
            assert not _still_alive(detached_pid), \
                "the test must not leave its detached child behind"
        for stream in (proc.stdout, proc.stderr):
            if stream is not None:
                stream.close()
        try:
            proc.kill()
            proc.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            pass
