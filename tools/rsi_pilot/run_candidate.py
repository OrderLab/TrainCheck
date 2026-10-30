"""Bounded client for the CPU adaptive-reasoning development pilot.

The operator owns the trial manifest, trainer, reference invariants and evaluator.
Solvers supply only a research program interpreted by the upstream TensorProgram.
This client is an experiment harness, not a security boundary against its owner.
"""

import argparse
import fcntl
import hashlib
import json
import math
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.fresh_validation.contracts import (  # noqa: E402
    JSONObject,
    JSONValue,
    boolean,
    integer,
    mapping,
    number,
    read_object,
    required,
    sequence,
    string,
)


def checked_text(record: JSONObject, key: str) -> str:
    value = string(required(record, key), key)
    if not value:
        raise ValueError(f"{key} must be nonempty")
    return value


def program_text(path: Path, workspace: Path) -> str:
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(workspace.resolve(strict=True)):
        raise ValueError("Research program must be inside this solver's workspace")
    raw = resolved.read_bytes()
    if not 0 < len(raw) <= 32768:
        raise ValueError("Research program must contain 1..32768 UTF-8 bytes")
    return raw.decode("utf-8")


def configure(template: str, source: str) -> str:
    """Insert source as a TOML basic string; never interpolate it into shell code."""
    marker = "# PILOT_RESEARCH_SOURCE"
    if template.count(marker) != 1:
        raise ValueError("Frozen profile must contain exactly one source marker")
    # TOML forbids JSON's surrogate-pair escapes for non-BMP Unicode characters.
    return template.replace(
        marker, "research_source = " + json.dumps(source, ensure_ascii=False)
    )


def remote_command(manifest: JSONObject, arguments: list[str]) -> list[str]:
    """All remote arguments are separately shell-quoted for OpenSSH's command."""
    return [
        "ssh",
        "-o",
        "BatchMode=yes",
        checked_text(manifest, "host"),
        shlex.join(arguments),
    ]


def stop_process_group(process: subprocess.Popen[bytes]) -> None:
    """Stop only the session created for this command, including descendants."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass  # The owned process group has already exited.
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        pass
    # A leader may exit on TERM while its descendants ignore it.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def execute(command: list[str], log: Path, timeout: int) -> int:
    with log.open("wb") as handle:
        with subprocess.Popen(
            command, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True
        ) as process:
            try:
                return process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                # GNU timeout also bounds the remote process, independently of
                # local SSH cleanup if the connection disappears.
                stop_process_group(process)
                handle.write(b"\nCLIENT_TIMEOUT\n")
                return 124
            except BaseException:
                stop_process_group(process)
                raise


def checked_execute(command: list[str], log: Path, timeout: int = 60) -> None:
    code = execute(command, log, timeout)
    if code:
        raise subprocess.CalledProcessError(code, command)


def upload(manifest: JSONObject, source: Path, target: str, log: Path) -> None:
    checked_execute(
        [
            "scp",
            "-q",
            "-o",
            "BatchMode=yes",
            str(source),
            checked_text(manifest, "host") + ":" + shlex.quote(target),
        ],
        log,
    )


def download(manifest: JSONObject, source: str, target: Path, log: Path) -> None:
    checked_execute(
        [
            "scp",
            "-q",
            "-o",
            "BatchMode=yes",
            checked_text(manifest, "host") + ":" + shlex.quote(source),
            str(target),
        ],
        log,
    )


def train_arguments(manifest: JSONObject, remote: str) -> list[str]:
    workspace = checked_text(manifest, "remote_workspace")
    # Upstream recipe_hashes resolves its workspace from module __file__, not cwd.
    return [
        "flock",
        "-w",
        "360",
        checked_text(manifest, "remote_compute_lock"),
        "timeout",
        "--signal=TERM",
        "--kill-after=10s",
        "300s",
        "taskset",
        "-c",
        checked_text(manifest, "affinity"),
        "env",
        "PYTHONDONTWRITEBYTECODE=1",
        "PYTHONHASHSEED=0",
        "CUDA_VISIBLE_DEVICES=",
        "OMP_NUM_THREADS=4",
        "MKL_NUM_THREADS=4",
        "OPENBLAS_NUM_THREADS=1",
        "PYTHONPATH=" + checked_text(manifest, "remote_overlay") + ":" + workspace,
        checked_text(manifest, "remote_python"),
        "-m",
        "latent_reasoning.train",
        "--config",
        remote + "/config.toml",
        "--output",
        remote + "/training",
        "--device",
        "cpu",
        "--require-complete",
    ]


def write_record(path: Path, record: JSONObject) -> None:
    """Replace one receipt atomically; callers hold this trial's exclusive lock."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def reserve(trial: Path, manifest: JSONObject, diagnose: bool) -> tuple[int, int]:
    """Reserve attempts before remote work; failed attempts remain charged.

    Missing max_diagnostics explicitly disables diagnostics. A missing usage
    ledger is a new/legacy trial: existing complete run receipts reconstruct its
    reservations. Malformed/missing receipts fail closed instead of resetting it.
    """
    limit = integer(required(manifest, "max_candidates"), "max_candidates")
    if not 1 <= limit <= 3:
        raise ValueError("max_candidates must be between 1 and 3")
    diagnostic_limit = (
        integer(manifest["max_diagnostics"], "max_diagnostics")
        if "max_diagnostics" in manifest
        else 0
    )
    if not 0 <= diagnostic_limit <= 1:
        raise ValueError("max_diagnostics must be 0 or 1")
    ledger_path = trial / "usage.json"
    if ledger_path.exists():
        ledger = read_object(ledger_path)
        candidates = integer(
            required(ledger, "candidates_reserved"), "candidates_reserved"
        )
        diagnostics = integer(
            required(ledger, "diagnostics_reserved"), "diagnostics_reserved"
        )
    else:
        existing = sorted((trial / "runs").glob("candidate-*"))
        candidates = len(existing)
        diagnostics = sum(
            boolean(
                required(read_object(path / "run.json"), "diagnostics_requested"),
                "diagnostics_requested",
            )
            for path in existing
        )
    if candidates < 0 or diagnostics < 0 or diagnostics > candidates:
        raise ValueError("Invalid reservation ledger")
    if candidates >= limit:
        raise ValueError(f"Candidate budget exhausted ({limit})")
    if diagnose and diagnostics >= diagnostic_limit:
        raise ValueError(f"Diagnostic budget exhausted ({diagnostic_limit})")
    candidates += 1
    diagnostics += diagnose
    write_record(
        ledger_path,
        {
            "candidates_reserved": candidates,
            "diagnostics_reserved": diagnostics,
        },
    )
    return candidates, diagnostics


def validate_results(
    results: JSONObject, manifest: JSONObject
) -> tuple[int, int | float, list[JSONValue]]:
    # The upstream result writer disallows nonfinite values everywhere. Validate
    # before assigning payloads to the receipt, so cleanup can always serialize it.
    json.dumps(results, allow_nan=False)
    steps = integer(required(results, "completed_steps"), "completed_steps")
    if steps < 1:
        raise ValueError("Training must report a positive completed step count")
    if "expected_steps" in manifest:
        expected = integer(manifest["expected_steps"], "expected_steps")
        if expected < 1 or steps != expected:
            raise ValueError(
                "Completed steps do not match the frozen trial requirement"
            )
    elapsed = number(required(results, "training_seconds"), "training_seconds")
    if not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError("Training duration must be finite and nonnegative")
    if checked_text(results, "device_backend") != "cpu":
        raise ValueError("This pilot requires CPU results")
    final = sequence(required(results, "final"), "final")
    if not final:
        raise ValueError("Training must return nonempty development evaluation results")
    for entry in final:
        row = mapping(entry, "development result")
        depth = integer(required(row, "loops"), "loops")
        accuracy = number(
            required(row, "macro_accuracy_percent"), "macro_accuracy_percent"
        )
        if depth < 1 or not math.isfinite(accuracy) or not 0 <= accuracy <= 100:
            raise ValueError("Malformed development depth or accuracy")
    # read_object has already validated recursive JSON values; preserve them.
    stored_final = results["final"]
    assert isinstance(stored_final, list)
    return steps, elapsed, stored_final


def validate_diagnostics(report: JSONObject) -> None:
    json.dumps(report, allow_nan=False)
    if integer(required(report, "schema_version"), "schema_version") != 1:
        raise ValueError("Expected agent report schema_version 1")
    if checked_text(report, "outcome") not in {
        "violations",
        "incomplete",
        "no_evidence",
        "active_checks_passed",
    }:
        raise ValueError("Unknown TrainCheck report outcome")
    boolean(required(report, "coverage_incomplete"), "coverage_incomplete")
    counts = mapping(required(report, "counts"), "counts")
    for name in (
        "completed",
        "checker_error",
        "observer_error",
        "unsupported_coverage",
        "observed_active",
        "observed_active_passes",
        "observed_dormant",
        "observed_violations",
        "unknown_outcomes",
        "unobserved_coverage",
        "skipped_variable_checks",
        "completed_decisions",
    ):
        if integer(required(counts, name), name) < 0:
            raise ValueError(f"Negative diagnostic count: {name}")
    provenance = mapping(required(report, "coverage_source"), "coverage_source")
    string(required(provenance, "path"), "coverage_source.path")
    string(required(provenance, "sha256"), "coverage_source.sha256")


def run(trial: Path, source_path: Path, diagnose: bool) -> JSONObject:
    manifest_path = trial / "trial.json"
    manifest = read_object(manifest_path)
    workspace = Path(checked_text(manifest, "workspace"))
    source = program_text(source_path, workspace)
    arm = checked_text(manifest, "arm")
    if arm not in ("control", "traincheck"):
        raise ValueError("Unknown trial arm")
    if diagnose and arm != "traincheck":
        raise ValueError("TrainCheck diagnostics are not enabled for this arm")
    template_path = Path(checked_text(manifest, "profile"))
    template_bytes = template_path.read_bytes()
    if hashlib.sha256(template_bytes).hexdigest() != checked_text(
        manifest, "profile_sha256"
    ):
        raise ValueError("Frozen training profile changed")
    runs = trial / "runs"
    runs.mkdir(exist_ok=True)
    with (trial / "run.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        attempt, diagnostic_attempts = reserve(trial, manifest, diagnose)
        candidate = f"candidate-{attempt:02d}"
        output = runs / candidate
        output.mkdir()
        remote = checked_text(manifest, "remote_trial") + "/" + candidate
        errors: list[JSONValue] = []
        record: JSONObject = {
            "candidate": candidate,
            "arm": arm,
            "diagnostics_requested": diagnose,
            "program_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "profile_sha256": hashlib.sha256(template_bytes).hexdigest(),
            "trial_manifest_sha256": hashlib.sha256(
                manifest_path.read_bytes()
            ).hexdigest(),
            "remote_directory": remote,
            "status": "started",
            "candidates_reserved": attempt,
            "diagnostics_reserved": diagnostic_attempts,
            "errors": errors,
        }
        write_record(output / "run.json", record)
        started = time.monotonic()
        phase = "prepare_local"
        try:
            (output / "program.py").write_text(source)
            (output / "config.toml").write_text(
                configure(template_bytes.decode(), source)
            )
            phase = "prepare_remote"
            checked_execute(
                remote_command(manifest, ["mkdir", "-p", remote]), output / "setup.log"
            )
            phase = "upload_config"
            upload(
                manifest,
                output / "config.toml",
                remote + "/config.toml",
                output / "upload.log",
            )
            phase = "training"
            code = execute(
                remote_command(manifest, train_arguments(manifest, remote)),
                output / "training.log",
                700,
            )
            record["training_exit_code"] = code
            record["status"] = "training_complete" if code == 0 else "training_failed"
            if code:
                errors.append(
                    {
                        "phase": phase,
                        "type": "ProcessExit",
                        "message": f"Training exited {code}; see training.log",
                    }
                )
            if code == 0:
                phase = "download_results"
                download(
                    manifest,
                    remote + "/training/results.json",
                    output / "results.json",
                    output / "download-results.log",
                )
                phase = "validate_results"
                results = read_object(output / "results.json")
                steps, elapsed, final = validate_results(results, manifest)
                record["completed_steps"] = steps
                record["training_seconds"] = elapsed
                record["development_depths"] = final
            if diagnose:
                # This operator-supplied driver performs a short independently
                # rerun trace and offline check. It does not alter the checkpoint.
                phase = "diagnostics"
                # The frozen driver owns the same compute lock and its 240s
                # timeout. Do not nest that lock or charge queue wait to tracing.
                diagnostic = [checked_text(manifest, "diagnostic_driver"), remote]
                code = execute(
                    remote_command(manifest, diagnostic),
                    output / "diagnostics.log",
                    700,
                )
                record["diagnostic_exit_code"] = code
                if code == 0:
                    phase = "download_diagnostics"
                    download(
                        manifest,
                        remote + "/diagnostics/agent_report.json",
                        output / "agent_report.json",
                        output / "download-diagnostics.log",
                    )
                    phase = "read_diagnostics"
                    report = read_object(output / "agent_report.json")
                    validate_diagnostics(report)
                    record["traincheck_report"] = report
                    record["diagnostic_status"] = "complete"
                else:
                    record["diagnostic_status"] = "execution_failed"
                    errors.append(
                        {
                            "phase": phase,
                            "type": "ProcessExit",
                            "message": f"Diagnostics exited {code}; see diagnostics.log",
                        }
                    )
            return record
        except BaseException as exc:
            record["status"] = "client_error"
            errors.append(
                {
                    "phase": phase,
                    "type": type(exc).__name__,
                    "message": str(exc) or repr(exc),
                }
            )
            raise
        finally:
            record["operational_seconds_including_transfer_and_diagnostics"] = (
                time.monotonic() - started
            )
            write_record(output / "run.json", record)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", type=Path, required=True)
    parser.add_argument("--program", type=Path, required=True)
    parser.add_argument("--diagnose", action="store_true")
    args = parser.parse_args()
    result = run(args.trial.resolve(strict=True), args.program, args.diagnose)
    print(json.dumps(result, indent=2, allow_nan=False))
    return int(bool(sequence(required(result, "errors"), "errors")))


if __name__ == "__main__":
    raise SystemExit(main())
