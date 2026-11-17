"""Replay a frozen public packet on two ranks with the withheld numerical grader."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Order mirrors worker.SCENARIOS. Validate both rank receipts against these names.
NAMES = (
    "fixed-regression",
    "exact-uniform",
    "overshoot-and-tail",
    "rank-skew",
    "single-window",
    "one-microbatch-updates",
)
OUTCOMES = {
    "pass",
    "silent_wrong",
    "memory_contract",
    "nonfinite",
    "runtime_error",
    "unsupported",
    "communication_contract",
}


def grade(
    source: Path,
    output: Path,
    device: str,
    timeout: float,
    optional_package: Path | None,
    collective_audit: Path | None = None,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    # Reserve the path before allocating any worker time.
    with output.open("x") as stream:
        json.dump({"status": "running"}, stream)
    with tempfile.TemporaryDirectory(prefix="budget-grader-") as temp:
        scratch = Path(temp)
        for name in ("training.py", "workload.py"):
            shutil.copyfile(source / name, scratch / name)
        shutil.copyfile(Path(__file__).with_name("worker.py"), scratch / "worker.py")
        if collective_audit is not None:
            shutil.copyfile(collective_audit, scratch / "collectives.py")
        package_files: dict[str, str] = {}
        if optional_package is not None:
            if (
                optional_package.name != "traincheck"
                or not (optional_package / "__init__.py").is_file()
            ):
                raise ValueError("Expected a frozen traincheck package directory")
            for file in sorted(optional_package.rglob("*")):
                if file.is_symlink():
                    raise ValueError("Expected regular files in the frozen package")
                if file.is_file() and "__pycache__" not in file.parts:
                    relative = file.relative_to(optional_package)
                    dest = scratch / "traincheck" / relative
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(file, dest)
                    package_files[str(relative)] = hashlib.sha256(
                        dest.read_bytes()
                    ).hexdigest()
        start = time.monotonic()
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=2",
            "worker.py",
            "--receipts",
            str(scratch),
            "--device",
            device,
        ]
        if collective_audit is not None:
            command.append("--audit-communication")
        with (scratch / "worker.log").open("wb") as log:
            proc = subprocess.Popen(
                command,
                cwd=scratch,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            timed_out = False
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
            finally:
                if proc.poll() is None:
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    proc.wait()
        worker_seconds = time.monotonic() - start
        statuses: set[str] = set()
        results: list[dict[str, object]] = []
        for index, name in enumerate(NAMES):
            for rank in (0, 1):
                receipt = scratch / f"{index}-rank{rank}.json"
                if receipt.is_file():
                    row: object = json.loads(receipt.read_text())
                    if (
                        not isinstance(row, dict)
                        or row["name"] != name
                        or row["status"] not in OUTCOMES
                    ):
                        raise ValueError(f"Invalid worker receipt {receipt.name}")
                    status = row["status"]
                    results.append({"name": name, "rank": rank, "result": row})
                else:
                    status = "unobserved"
                    results.append(
                        {
                            "name": name,
                            "rank": rank,
                            "result": {
                                "status": status,
                                "started": receipt.with_suffix(".start").exists(),
                            },
                        }
                    )
                statuses.add(status)
        if "unobserved" in statuses or proc.returncode != 0:
            overall = "timeout" if timed_out else "incomplete_execution"
        else:
            overall = next(
                (
                    x
                    for x in (
                        "runtime_error",
                        "unsupported",
                        "nonfinite",
                        "silent_wrong",
                        "memory_contract",
                        "communication_contract",
                    )
                    if x in statuses
                ),
                "pass",
            )
        result = {
            "status": overall,
            "rank_results": results,
            "worker_seconds": worker_seconds,
            "devices": 2 if device == "cuda" else 0,
            "optional_package_files": package_files,
            "source_sha256": hashlib.sha256(
                (scratch / "training.py").read_bytes()
            ).hexdigest(),
            "exit_code": proc.returncode,
            "log_tail": (scratch / "worker.log")
            .read_bytes()[-12000:]
            .decode(errors="replace"),
        }
        staged = output.with_suffix(output.suffix + ".partial")
        with staged.open("x") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
        staged.replace(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--timeout", type=float, default=90)
    parser.add_argument("--optional-package", type=Path)
    parser.add_argument("--collective-audit", type=Path)
    args = parser.parse_args()
    grade(
        args.source.resolve(),
        args.output.resolve(),
        args.device,
        args.timeout,
        args.optional_package,
        args.collective_audit,
    )


if __name__ == "__main__":
    main()
