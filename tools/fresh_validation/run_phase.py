"""Run one bounded correctness phase in a fresh OS process (Linux only)."""

import argparse
import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path
from types import FrameType
from typing import Literal, TypedDict


class PhaseOptions(argparse.Namespace):
    source: Path
    output: Path
    timeout: float
    memory_gib: float
    trace_mib: float
    command: list[str]


StopReason = Literal[
    "process_group_rss_budget",
    "timeout",
    "trace_size_budget",
    "signal_sigterm",
    "signal_sigint",
]


class PhaseResult(TypedDict):
    exit_code: int
    stop_reason: StopReason | None
    source_unchanged: bool
    pid: int
    elapsed_seconds_operational_only: float
    peak_process_group_rss_bytes_operational_only: int


def group_memory(group: int) -> tuple[int, list[int]]:
    total = 0
    processes: list[int] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text().rsplit(")", 1)[1].split()
            if int(stat[2]) != group:
                continue
            rss = next(
                int(line.split()[1]) * 1024
                for line in (entry / "status").read_text().splitlines()
                if line.startswith("VmRSS:")
            )
            total += rss
            processes.append(int(entry.name))
        except (FileNotFoundError, ProcessLookupError, StopIteration):
            continue
    return total, processes


def terminate_group(process: subprocess.Popen[bytes]) -> None:
    """Stop the owned session, including descendants whose leader already exited.

    A terminating leader does not prove its process group is empty. Always send
    KILL to any remaining members after a bounded TERM grace period, then reap
    the direct child. A child that cannot be reaped is an explicit failure.
    """
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass  # The owned group has already exited.
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=2)


def run(args: PhaseOptions) -> int:
    """Supervise from the main thread, temporarily handling SIGTERM and SIGINT.

    Cleanup covers handled signals, normal completion and Python exceptions.
    Uncatchable termination such as SIGKILL cannot run user-space cleanup.
    """
    args.output.mkdir(parents=True, exist_ok=False)
    source_hashes = {
        str(p.relative_to(args.source)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in args.source.rglob("*.py")
    }
    env = os.environ.copy()
    env.update(
        PYTHONPATH=str(args.source.resolve()),
        PYTHONHASHSEED="0",
        PYTHONUNBUFFERED="1",
        PYTHONDONTWRITEBYTECODE="1",
        CUDA_VISIBLE_DEVICES="",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
        VECLIB_MAXIMUM_THREADS="1",
    )
    command = args.command[1:] if args.command[0] == "--" else args.command
    started, peak = time.monotonic(), 0
    reason: StopReason | None = None
    signal_reason: StopReason | None = None

    def request_stop(signum: int, frame: FrameType | None) -> None:
        nonlocal signal_reason
        assert signum in (signal.SIGTERM, signal.SIGINT)
        if signal_reason is None:
            signal_reason = (
                "signal_sigterm" if signum == signal.SIGTERM else "signal_sigint"
            )

    # The workload has a separate session. An outer timeout's signal to this
    # supervisor must therefore trigger explicit child-group cleanup, not the
    # default immediate supervisor exit. Signals only request a stop so repeated
    # signals cannot interrupt TERM/KILL/reap midway through cleanup.
    prior_term = signal.signal(signal.SIGTERM, request_stop)
    try:
        prior_int = signal.signal(signal.SIGINT, request_stop)
        try:
            with (args.output / "stdout.log").open("w") as log:
                proc = subprocess.Popen(
                    ["nice", "-n", "10", *command],
                    cwd=args.output,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                try:
                    invocation = {
                        "command": command,
                        "pid": proc.pid,
                        "native_threads": 1,
                        "timeout_seconds": args.timeout,
                        "memory_limit_gib": args.memory_gib,
                        "trace_limit_mib": args.trace_mib,
                        "source_sha256": source_hashes,
                        "harness_sha256": {
                            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in Path(__file__).resolve().parent.glob("*.py")
                        },
                    }
                    (args.output / "invocation.json").write_text(
                        json.dumps(invocation, indent=2) + "\n"
                    )
                    while proc.poll() is None:
                        memory, _ = group_memory(proc.pid)
                        peak = max(peak, memory)
                        if memory > args.memory_gib * 1024**3:
                            reason = "process_group_rss_budget"
                        if time.monotonic() - started > args.timeout:
                            reason = "timeout"
                        if (
                            sum(
                                p.stat().st_size
                                for p in args.output.rglob("trace_API*.log")
                            )
                            > args.trace_mib * 1024**2
                        ):
                            reason = "trace_size_budget"
                        if signal_reason is not None:
                            reason = signal_reason
                        if reason is not None:
                            break
                        time.sleep(0.25)
                finally:
                    terminate_group(proc)
                # A signal may arrive while the leader exits or during cleanup.
                if signal_reason is not None:
                    reason = signal_reason
        finally:
            signal.signal(signal.SIGINT, prior_int)
    finally:
        signal.signal(signal.SIGTERM, prior_term)
    unchanged = all(
        hashlib.sha256((args.source / p).read_bytes()).hexdigest() == h
        for p, h in source_hashes.items()
    )
    assert proc.returncode is not None, "Supervisor must reap its child"
    result: PhaseResult = {
        "exit_code": proc.returncode,
        "stop_reason": reason,
        "source_unchanged": unchanged,
        "pid": proc.pid,
        "elapsed_seconds_operational_only": time.monotonic() - started,
        "peak_process_group_rss_bytes_operational_only": peak,
    }
    (args.output / "phase-result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    return int(proc.returncode != 0 or reason is not None or not unchanged)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--memory-gib", type=float, default=8)
    parser.add_argument("--trace-mib", type=float, default=256)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    options = parser.parse_args(namespace=PhaseOptions())
    if not options.command:
        parser.error("A command is required")
    raise SystemExit(run(options))
