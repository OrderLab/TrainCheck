"""Operator driver for a bounded three-update TrainCheck diagnostic probe.

The launcher must hold the experiment's CPU lock and apply an overall timeout.
Each expensive phase also uses the existing process-group resource supervisor.
No invariant is inferred, modified, or removed during candidate diagnostics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from tools.fresh_validation.contracts import boolean, integer, read_object, required


@dataclass(frozen=True)
class Driver:
    study: Path
    python: Path
    cpus: str = "8,10,12,14"

    @property
    def source(self) -> Path:
        return self.study / "source"

    def phase(
        self,
        output: Path,
        command: Sequence[str],
        *,
        timeout: int,
        allow_checker_failure: bool = False,
    ) -> None:
        command_line = [
            "taskset",
            "-c",
            self.cpus,
            str(self.python),
            str(self.source / "tools/fresh_validation/run_phase.py"),
            "--source",
            str(self.source),
            "--output",
            str(output),
            "--timeout",
            str(timeout),
            "--memory-gib",
            "4",
            "--trace-mib",
            "256",
            "--",
            "env",
            f"PYTHONPATH={self.source}:{self.study / 'overlay'}",
            str(self.python),
            *command,
        ]
        process = subprocess.run(command_line, check=False)
        result = read_object(output / "phase-result.json")
        child_exit = integer(required(result, "exit_code"), "exit_code")
        unchanged = boolean(required(result, "source_unchanged"), "source_unchanged")
        stopped = required(result, "stop_reason")
        # The coverage observer returns 1 for a checker/observer error. Its
        # explicit error records are still summarized; limits and missing output
        # never become an empty or passing report.
        allowed = (0, 1) if allow_checker_failure else (0,)
        if (
            child_exit not in allowed
            or not unchanged
            or stopped is not None
            or process.returncode not in allowed
        ):
            raise RuntimeError(
                f"Diagnostic phase failed: {output.name}; inspect {output}"
            )

    def diagnose(self, candidate: Path) -> Path:
        candidate = candidate.resolve()
        config = candidate / "config.toml"
        if not config.is_file():
            raise FileNotFoundError(f"Candidate configuration missing: {config}")
        invariants = self.study / "reference/invariants.json"
        digest = hashlib.sha256(invariants.read_bytes()).hexdigest()
        output = candidate / "diagnostics"
        output.mkdir(exist_ok=False)
        started = time.monotonic()
        trace_output = output / "collection/result"
        self.phase(
            output / "collection",
            [
                str(self.study / "bin/trace_adaptive.py"),
                "--workspace",
                str(self.study / "adaptive-workspace"),
                "--output",
                str(trace_output),
                "--mode",
                "selective",
                "--invariants",
                str(invariants),
                "--",
                "--config",
                str(config),
                "--steps",
                "3",
                "--pool-size",
                "0",
                "--require-complete",
            ],
            timeout=180,
        )
        traces = sorted((trace_output / "traces").glob("trace_API*.log"))
        if not traces or not all(path.stat().st_size > 0 for path in traces):
            raise RuntimeError(
                "Successful diagnostic collection must produce API records"
            )
        coverage = output / "coverage.json"
        self.phase(
            output / "checking",
            [
                str(self.source / "tools/fresh_validation/check_coverage.py"),
                "--traces",
                *map(str, traces),
                "--invariants",
                str(invariants),
                "--output",
                str(coverage),
            ],
            timeout=90,
            allow_checker_failure=True,
        )
        report = output / "agent_report.json"
        self.phase(
            output / "summary",
            [
                str(self.study / "bin/summarize_check.py"),
                "--coverage",
                str(coverage),
                "--collection-policy",
                str(trace_output / "traces/instr_opts.json"),
                "--output",
                str(report),
            ],
            timeout=20,
        )
        if hashlib.sha256(invariants.read_bytes()).hexdigest() != digest:
            raise RuntimeError("Frozen reference invariants changed during diagnosis")
        manifest = {
            "schema_version": 1,
            "status": "complete",
            "invariants_sha256": digest,
            "candidate_config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
            "elapsed_seconds_including_collection_checking_and_summary": time.monotonic()
            - started,
            "scope": "Three fresh CPU updates, API-only selective collection and offline checking",
            "limitations": [
                "Probe overrides steps to three; research loss receives total_steps=3, not the full-training horizon.",
                "A clean probe does not establish full-run health or improved model quality.",
                "Changed training methods can legitimately violate baseline invariants.",
                "No model-variable tracker or online checker is enabled.",
            ],
        }
        (output / "diagnostic_manifest.json").write_text(
            json.dumps(manifest, indent=2, allow_nan=False) + "\n"
        )
        return report


class Options(argparse.Namespace):
    study: Path
    python: Path
    candidate: Path
    cpus: str


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--cpus", default="8,10,12,14")
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args(namespace=Options())
    # Resolving a venv's Python symlink switches to the base interpreter and
    # loses its installed dependencies; preserve the executable's venv path.
    report = Driver(args.study.resolve(), args.python.absolute(), args.cpus).diagnose(
        args.candidate
    )
    print(json.dumps({"report": str(report)}), flush=True)


if __name__ == "__main__":
    main()
