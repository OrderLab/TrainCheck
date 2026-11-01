"""Exercise CLI transport/lifecycle with a tiny local executable, not a model."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tools.checkpoint_pilot.accounting import (
    FailedExecutionError,
    IncompleteExecutionError,
    TokenUsage,
)
from tools.checkpoint_pilot.cli_turn import CLISettings, command, run_turn

FAKE_PRELUDE = """
import json
import os
import subprocess
import sys
import time
from pathlib import Path

prompt = sys.stdin.read()
response = Path(sys.argv[sys.argv.index('-o') + 1])
rows = [
    {'type': 'thread.started', 'thread_id': 'session-one'},
    {'type': 'turn.started'},
    {'type': 'future.event', 'detail': 'valid extension'},
    {'type': 'turn.completed', 'usage': {
        'input_tokens': 100, 'cached_input_tokens': 40, 'output_tokens': 20,
    }},
]
"""


def setup(root: Path, body: str) -> tuple[CLISettings, Path, Path]:
    workspace = root / "workspace"
    workspace.mkdir()
    schema = root / "response-schema.json"
    schema.write_text('{"type":"object"}\n')
    executable = root / "fake-cli"
    executable.write_text(f"#!{sys.executable}\n" + FAKE_PRELUDE + body)
    executable.chmod(0o700)
    return (
        CLISettings(executable, "test-model", "high", "priority", schema),
        workspace,
        root / "output",
    )


def read_object(path: Path) -> dict[str, object]:
    value: object = json.loads(path.read_text())
    assert isinstance(value, dict)
    result: dict[str, object] = {}
    for key, item in value.items():
        assert isinstance(key, str)
        result[key] = item
    return result


def number(value: object) -> float:
    assert isinstance(value, (int, float)) and not isinstance(value, bool)
    return float(value)


def test_successful_stream_records_buffered_lines_without_waiting_for_exit(
    tmp_path: Path,
) -> None:
    settings, workspace, output = setup(
        tmp_path,
        """
response.write_text(json.dumps({'decision': 'continue', 'prompt': prompt}))
sys.stderr.write('retained stderr\\n')
sys.stdout.write(''.join(json.dumps(row) + '\\n' for row in rows))
sys.stdout.flush()
time.sleep(0.6)
""",
    )
    receipt, parsed = run_turn(settings, workspace, output, "inspect the curve", 10)
    assert receipt.returncode == 0
    assert receipt.error_type is None and receipt.error_message is None
    assert not receipt.timed_out
    assert parsed.thread_id == "session-one"
    assert parsed.usage == TokenUsage(100, 40, 20)
    assert read_object(output / "response.json")["prompt"] == "inspect the curve"
    assert (output / "stderr.log").read_text() == "retained stderr\n"
    received = [
        read_object_line(line)
        for line in (output / "received.jsonl").read_text().splitlines()
    ]
    assert len(received) == 4
    first = number(received[0]["received_monotonic"])
    last = number(received[-1]["received_monotonic"])
    # All events arrived in one write, well before the producer's delayed exit.
    assert last - first < 0.3
    assert receipt.ended_monotonic - last > 0.4
    assert receipt.started_monotonic <= first <= last <= receipt.ended_monotonic
    assert read_object(output / "receipt.json")["returncode"] == 0


def read_object_line(line: str) -> dict[str, object]:
    value: object = json.loads(line)
    assert isinstance(value, dict)
    result: dict[str, object] = {}
    for key, item in value.items():
        assert isinstance(key, str)
        result[key] = item
    return result


@pytest.mark.parametrize("failure", ["incomplete", "failed", "malformed"])
def test_failure_usage_propagates_and_retains_receipt(
    tmp_path: Path, failure: str
) -> None:
    bodies = {
        "incomplete": "print(json.dumps(rows[0]), flush=True)\n",
        "failed": "print(json.dumps(rows[0]), flush=True)\nprint(json.dumps({'type':'turn.failed','error':{'message':'failed'}}), flush=True)\n",
        "malformed": "rows[-1]['usage']['input_tokens'] = True\nprint('\\n'.join(json.dumps(row) for row in rows), flush=True)\n",
    }
    settings, workspace, output = setup(
        tmp_path, "response.write_text('{}')\n" + bodies[failure]
    )
    error_type: type[ValueError] = {
        "incomplete": IncompleteExecutionError,
        "failed": FailedExecutionError,
        "malformed": ValueError,
    }[failure]
    with pytest.raises(error_type):
        run_turn(settings, workspace, output, "continue", 3)
    receipt = read_object(output / "receipt.json")
    assert receipt["returncode"] == 0
    assert receipt["error_type"] == error_type.__name__
    assert (output / "events.jsonl").stat().st_size > 0
    assert "usage" not in receipt  # Unknown usage was not fabricated as zero.


def test_nonzero_exit_is_failure_even_if_completion_event_exists(
    tmp_path: Path,
) -> None:
    settings, workspace, output = setup(
        tmp_path,
        "response.write_text('{}')\nprint('\\n'.join(json.dumps(row) for row in rows), flush=True)\nsys.exit(7)\n",
    )
    with pytest.raises(RuntimeError, match="CLI decision failed"):
        run_turn(settings, workspace, output, "continue", 3)
    receipt = read_object(output / "receipt.json")
    assert receipt["returncode"] == 7
    assert receipt["error_type"] == "RuntimeError"
    assert receipt["timed_out"] is False


def test_partial_line_cannot_block_past_timeout(tmp_path: Path) -> None:
    settings, workspace, output = setup(
        tmp_path,
        'sys.stdout.write(\'{"type":"thread.started"\')\nsys.stdout.flush()\ntime.sleep(60)\n',
    )
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="CLI decision failed"):
        run_turn(settings, workspace, output, "continue", 2)
    assert time.monotonic() - started < 5
    receipt = read_object(output / "receipt.json")
    assert receipt["timed_out"] is True
    assert isinstance(receipt["returncode"], int) and receipt["returncode"] < 0
    assert (output / "events.jsonl").read_text() == '{"type":"thread.started"'


def test_timeout_stops_child_in_owned_process_group(tmp_path: Path) -> None:
    settings, workspace, output = setup(
        tmp_path,
        """
child = subprocess.Popen([sys.executable, '-c', 'import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)'])
Path('child.pid').write_text(str(child.pid))
print(json.dumps(rows[0]), flush=True)
time.sleep(60)
""",
    )
    with pytest.raises(RuntimeError):
        run_turn(settings, workspace, output, "continue", 2)
    pid = int((workspace / "child.pid").read_text())
    # A briefly unreaped zombie is terminated; it is not a running leaked child.
    result = subprocess.run(
        ["ps", "-o", "stat=", "-p", str(pid)],
        capture_output=True,
        text=True,
        check=False,
    )
    state = result.stdout.strip()
    assert not state or state.startswith("Z"), state
    assert read_object(output / "receipt.json")["timed_out"] is True


def test_spawn_failure_retains_unknown_exit_and_exception(tmp_path: Path) -> None:
    settings, workspace, output = setup(tmp_path, "")
    settings.executable.unlink()
    with pytest.raises(FileNotFoundError):
        run_turn(settings, workspace, output, "continue", 3)
    receipt = read_object(output / "receipt.json")
    assert receipt["returncode"] is None
    assert receipt["error_type"] == "FileNotFoundError"
    assert receipt["timed_out"] is False


def test_resumed_session_identity_must_match(tmp_path: Path) -> None:
    settings, workspace, output = setup(
        tmp_path,
        "response.write_text('{}')\nprint('\\n'.join(json.dumps(row) for row in rows), flush=True)\n",
    )
    with pytest.raises(ValueError, match="different thread"):
        run_turn(settings, workspace, output, "continue", 3, thread_id="different")
    receipt = read_object(output / "receipt.json")
    assert receipt["returncode"] == 0
    assert receipt["error_type"] == "ValueError"


@pytest.mark.parametrize("response", [None, "[]", "{not json}"])
def test_missing_or_malformed_structured_response_fails(
    tmp_path: Path, response: str | None
) -> None:
    body = "print('\\n'.join(json.dumps(row) for row in rows), flush=True)\n"
    if response is not None:
        body += f"response.write_text({response!r})\n"
    settings, workspace, output = setup(tmp_path, body)
    with pytest.raises(ValueError):
        run_turn(settings, workspace, output, "continue", 3)
    receipt = read_object(output / "receipt.json")
    assert receipt["returncode"] == 0
    assert receipt["error_type"] in ("ValueError", "JSONDecodeError")


def test_initial_and_resume_commands_preserve_policy_and_model_config(
    tmp_path: Path,
) -> None:
    settings, workspace, output = setup(tmp_path, "")
    initial = command(settings, workspace, output, None)
    resumed = command(settings, workspace, output, "session-one")
    for arguments in (initial, resumed):
        assert arguments[arguments.index("--sandbox") + 1] == "workspace-write"
        assert 'approval_policy="never"' in arguments
        assert arguments[arguments.index("-m") + 1] == "test-model"
        assert 'model_reasoning_effort="high"' in arguments
        assert 'service_tier="priority"' in arguments
        assert "--ignore-user-config" in arguments
        assert "--dangerously-bypass-approvals-and-sandbox" not in arguments
        assert "--dangerously-bypass-hook-trust" not in arguments
        assert arguments[-1] == "-"
    resume_index = resumed.index("resume")
    assert resumed[resume_index + 1] == "session-one"
    assert resumed.index("-m") > resume_index
    assert "resume" not in initial


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf"), True])
def test_invalid_timeout_is_rejected_before_execution(
    tmp_path: Path, timeout: float
) -> None:
    settings, workspace, output = setup(tmp_path, "")
    with pytest.raises(ValueError, match="timeout_seconds"):
        run_turn(settings, workspace, output, "continue", timeout)
    assert not output.exists()
