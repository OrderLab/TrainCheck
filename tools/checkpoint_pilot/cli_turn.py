"""Run one sandboxed CLI decision turn and retain operator-clock evidence."""

from __future__ import annotations

import json
import math
import os
import selectors
import signal
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO, TextIO

from tools.checkpoint_pilot.accounting import CompletedTurn, parse_exec_jsonl


@dataclass(frozen=True)
class CLISettings:
    executable: Path
    model: str
    reasoning_effort: str
    service_tier: str
    response_schema: Path


@dataclass(frozen=True)
class TurnReceipt:
    started_utc: str
    ended_utc: str
    started_monotonic: float
    ended_monotonic: float
    # None means the process never started or cleanup could not establish exit.
    returncode: int | None
    timed_out: bool
    command: list[str]
    workspace: str
    error_type: str | None
    error_message: str | None


def command(
    settings: CLISettings,
    workspace: Path,
    output: Path,
    thread_id: str | None,
) -> list[str]:
    """Use the same policy/configuration on initial and resumed turns."""
    arguments = [
        str(settings.executable),
        "exec",
        "-C",
        str(workspace.resolve(strict=True)),
        "--sandbox",
        "workspace-write",
        "-c",
        'approval_policy="never"',
    ]
    if thread_id is not None:
        if not thread_id.strip():
            raise ValueError("thread_id must not be empty")
        arguments.extend(["resume", thread_id])
    arguments.extend(
        [
            "--json",
            "--ignore-user-config",
            "--skip-git-repo-check",
            "--disable",
            "multi_agent",
            "-m",
            settings.model,
            "-c",
            f"model_reasoning_effort={json.dumps(settings.reasoning_effort)}",
            "-c",
            f"service_tier={json.dumps(settings.service_tier)}",
            "--output-schema",
            str(settings.response_schema.resolve(strict=True)),
            "-o",
            str(output / "response.json"),
            "-",
        ]
    )
    return arguments


def stop_owned_process(process: subprocess.Popen[bytes]) -> None:
    """Terminate the explicitly created process group and reap its leader."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=3)


def _record_lines(buffer: bytearray, received: TextIO, *, final: bool) -> None:
    """Timestamp complete lines without waiting for another byte on the pipe."""
    while buffer:
        boundary = buffer.find(b"\n")
        if boundary < 0 and not final:
            return
        size = len(buffer) if boundary < 0 else boundary + 1
        line = bytes(buffer[:size]).decode("utf-8")
        del buffer[:size]
        received.write(
            json.dumps(
                {
                    "received_utc": datetime.now(timezone.utc).isoformat(),
                    "received_monotonic": time.monotonic(),
                    "raw_line": line.rstrip("\n"),
                }
            )
            + "\n"
        )
        received.flush()


def _collect_events(
    process: subprocess.Popen[bytes],
    events: BinaryIO,
    received: TextIO,
    deadline: float,
) -> bool:
    """Return timeout status; preserve exact bytes even for a partial last line."""
    assert process.stdout is not None
    descriptor = process.stdout.fileno()
    os.set_blocking(descriptor, False)
    buffer = bytearray()
    with selectors.DefaultSelector() as selector:
        selector.register(descriptor, selectors.EVENT_READ)
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                # A partial line is already preserved in events.jsonl. It cannot
                # be interpreted as a complete event or authoritative usage.
                return True
            for key, _ in selector.select(timeout=min(remaining, 0.5)):
                try:
                    chunk = os.read(descriptor, 65536)
                except BlockingIOError:
                    continue  # Readiness can change before the nonblocking read.
                if not chunk:
                    selector.unregister(key.fileobj)
                    _record_lines(buffer, received, final=True)
                    continue
                events.write(chunk)
                events.flush()
                buffer.extend(chunk)
                _record_lines(buffer, received, final=False)
    return False


def _validate_response(path: Path) -> None:
    if not path.is_file():
        raise ValueError("CLI completed without a structured response")
    # The CLI applies response_schema; the study's response consumer must still
    # validate its required fields. This boundary establishes JSON-object shape.
    value: object = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("CLI structured response must be a JSON object")


def run_turn(
    settings: CLISettings,
    workspace: Path,
    output: Path,
    prompt: str,
    timeout_seconds: float,
    thread_id: str | None = None,
) -> tuple[TurnReceipt, CompletedTurn]:
    """Record raw events and receive times, then require authoritative usage.

    A timeout/failure preserves evidence and raises; callers must report it as
    incomplete, never synthesize zero tokens or a successful decision. The
    operator owns the output directory outside the agent's writable packet.
    """
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(timeout_seconds)
        or not 0 < timeout_seconds <= 1800
    ):
        raise ValueError("timeout_seconds must be in (0, 1800]")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must not be empty")
    workspace = workspace.resolve(strict=True)
    output = output.resolve()
    arguments = command(settings, workspace, output, thread_id)
    output.mkdir(parents=True, exist_ok=False)
    (output / "prompt.txt").write_text(prompt)
    started_utc = datetime.now(timezone.utc).isoformat()
    started = time.monotonic()
    timed_out = False
    process: subprocess.Popen[bytes] | None = None
    error_type: str | None = None
    error_message: str | None = None
    try:
        with (
            (output / "prompt.txt").open("rb") as stdin,
            (output / "stderr.log").open("wb") as stderr,
            (output / "events.jsonl").open("wb") as events,
            (output / "received.jsonl").open("w") as received,
        ):
            process = subprocess.Popen(
                arguments,
                cwd=workspace,
                stdin=stdin,
                stdout=subprocess.PIPE,
                stderr=stderr,
                start_new_session=True,
            )
            timed_out = _collect_events(
                process, events, received, started + timeout_seconds
            )
            if not timed_out:
                remaining = timeout_seconds - (time.monotonic() - started)
                try:
                    process.wait(timeout=max(remaining, 0.001))
                except subprocess.TimeoutExpired:
                    timed_out = True
        if timed_out or process.returncode != 0:
            raise RuntimeError(f"CLI decision failed; evidence retained in {output}")
        parsed = parse_exec_jsonl((output / "events.jsonl").read_text())
        if thread_id is not None and parsed.thread_id != thread_id:
            raise ValueError("Resumed CLI returned a different thread")
        _validate_response(output / "response.json")
    except BaseException as error:
        # Cleanup must also cover interruption; preserve and propagate failure.
        error_type, error_message = type(error).__name__, str(error)
        raise
    finally:
        try:
            if process is not None:
                # Reap the leader and stop any remaining members of the owned
                # process group, including when the leader exited successfully.
                stop_owned_process(process)
        except BaseException as cleanup_error:
            detail = f"Cleanup failed: {type(cleanup_error).__name__}: {cleanup_error}"
            if error_type is None:
                error_type = type(cleanup_error).__name__
                error_message = detail
            else:
                error_message = f"{error_message}; {detail}"
            raise
        finally:
            if process is not None and process.stdout is not None:
                process.stdout.close()
            receipt = TurnReceipt(
                started_utc,
                datetime.now(timezone.utc).isoformat(),
                started,
                time.monotonic(),
                None if process is None else process.returncode,
                timed_out,
                arguments,
                str(workspace),
                error_type,
                error_message,
            )
            (output / "receipt.json").write_text(
                json.dumps(asdict(receipt), indent=2) + "\n"
            )
    return receipt, parsed
