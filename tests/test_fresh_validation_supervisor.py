"""Linux process-tree regressions for the bounded correctness supervisor.

Set TRAINCHECK_TEST_SUPERVISOR to test an archived CLI against the same external
signal cases. The tests always clean up their own recorded process groups.
"""

import contextlib
import io
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.fresh_validation import run_phase
from tools.fresh_validation.contracts import boolean, integer, read_object, required

DEFAULT_SUPERVISOR = Path(run_phase.__file__).resolve()
SUPERVISOR = (
    Path(os.environ["TRAINCHECK_TEST_SUPERVISOR"])
    if "TRAINCHECK_TEST_SUPERVISOR" in os.environ
    else DEFAULT_SUPERVISOR
)

GRANDCHILD = """import os, signal, sys, time
from pathlib import Path
signal.signal(signal.SIGTERM, signal.SIG_IGN)
Path(sys.argv[1]).write_text(str(os.getpid()))
time.sleep(60)
"""
LEADER = """import subprocess, sys, time
from pathlib import Path
child = subprocess.Popen([sys.executable, '-c', sys.argv[1], sys.argv[2]])
while not Path(sys.argv[2]).exists():
    time.sleep(0.01)
Path(sys.argv[3]).write_text('ready')
time.sleep(60)
"""


def live(pid: int) -> bool:
    """An orphan zombie is no longer running; only its parent can reap it."""
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    except FileNotFoundError:
        return False
    return fields[0] != "Z"


def wait_ready(path: Path, timeout: float = 5) -> None:
    deadline = time.monotonic() + timeout
    while not path.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Child did not become ready: {path}")
        time.sleep(0.01)


@unittest.skipUnless(sys.platform == "linux", "Supervisor inspects Linux /proc")
class TestPhaseSupervisor(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory[str]()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        self.source.mkdir()
        (self.source / "unchanged.py").write_text("VALUE = 1\n")

    def options(self, command: list[str]) -> run_phase.PhaseOptions:
        return run_phase.PhaseOptions(
            source=self.source,
            output=self.root / "phase",
            timeout=20.0,
            memory_gib=1.0,
            trace_mib=1.0,
            command=command,
        )

    def assert_stopped(self, pid: int) -> None:
        deadline = time.monotonic() + 3
        while live(pid) and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertFalse(live(pid), f"Owned child {pid} survived supervisor cleanup")

    def external_signal(self, signum: signal.Signals, expected: str) -> None:
        output = self.root / "phase"
        ready, child_pid_file = self.root / "ready", self.root / "grandchild.pid"
        command = [
            sys.executable,
            str(SUPERVISOR),
            "--source",
            str(self.source),
            "--output",
            str(output),
            "--timeout",
            "30",
            "--memory-gib",
            "1",
            "--trace-mib",
            "1",
            "--",
            sys.executable,
            "-c",
            LEADER,
            GRANDCHILD,
            str(child_pid_file),
            str(ready),
        ]
        with (self.root / "supervisor.log").open("wb") as log:
            process = subprocess.Popen(
                command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
            child_group: int | None = None
            try:
                wait_ready(ready)
                invocation = read_object(output / "invocation.json")
                child_group = integer(required(invocation, "pid"), "child PID")
                descendant = int(child_pid_file.read_text())
                os.kill(process.pid, signum)
                code = process.wait(timeout=8)
                self.assert_stopped(descendant)
                self.assert_stopped(child_group)
                self.assertEqual(code, 1)
                result = read_object(output / "phase-result.json")
                self.assertEqual(required(result, "stop_reason"), expected)
                self.assertTrue(
                    boolean(required(result, "source_unchanged"), "source unchanged")
                )
                self.assertNotEqual(
                    integer(required(result, "exit_code"), "child exit"), 0
                )
            finally:
                # This also cleans the leaked child in the archived buggy CLI.
                if child_group is not None:
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(child_group, signal.SIGKILL)
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=3)

    def test_sigterm_cleans_descendant_after_leader_exits(self) -> None:
        self.external_signal(signal.SIGTERM, "signal_sigterm")

    def test_sigint_cleans_descendant_after_leader_exits(self) -> None:
        self.external_signal(signal.SIGINT, "signal_sigint")

    def test_success_preserves_result_and_restores_handlers(self) -> None:
        previous = (signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGINT))
        with contextlib.redirect_stdout(io.StringIO()):
            result = run_phase.run(self.options([sys.executable, "-c", "pass"]))
        self.assertEqual(result, 0)
        self.assertEqual(
            (signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGINT)),
            previous,
        )
        receipt = read_object(self.root / "phase/phase-result.json")
        self.assertIsNone(required(receipt, "stop_reason"))
        self.assertEqual(integer(required(receipt, "exit_code"), "child exit"), 0)
        self.assertTrue(
            boolean(required(receipt, "source_unchanged"), "source unchanged")
        )

    def test_monitor_exception_kills_term_resistant_child_and_restores_handlers(
        self,
    ) -> None:
        ready = self.root / "child.pid"
        previous = (signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGINT))

        def failed_monitor(group: int) -> tuple[int, list[int]]:
            wait_ready(ready)
            raise ValueError("monitor failed")

        try:
            with patch.object(run_phase, "group_memory", side_effect=failed_monitor):
                with self.assertRaisesRegex(ValueError, "monitor failed"):
                    run_phase.run(
                        self.options([sys.executable, "-c", GRANDCHILD, str(ready)])
                    )
            self.assert_stopped(int(ready.read_text()))
            self.assertEqual(
                (signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGINT)),
                previous,
            )
        finally:
            if ready.exists():
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(int(ready.read_text()), signal.SIGKILL)


if __name__ == "__main__":
    unittest.main()
