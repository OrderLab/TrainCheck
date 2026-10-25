"""Collect reference/control/holdout traces and check frozen same-workload rules.

Every training, inference, and checking phase starts a separate bounded process.
Only the reference is used to infer rules. This is a correctness/coverage study;
the supervisor's time and memory observations are operational limits only.
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workload", choices=("mnist", "transformer"), required=True)
    args = parser.parse_args()
    if args.workload == "mnist" and args.data_root is None:
        parser.error("MNIST requires an existing real IDX data cache")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    harness = Path(__file__).resolve().parent
    hashes = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in harness.glob("*.py")
    }
    (args.output / "study-plan.json").write_text(
        json.dumps(
            {
                "workload": args.workload,
                "reference_steps": 10,
                "holdout_steps": 20,
                "reference_seed": 42,
                "holdout_seed": 43,
                "data_steps": 20,
                "reference_inputs_overlap_first_half_of_holdout": True,
                "disabled_relations": ["FunctionLeadRelation", "FunctionCoverRelation"],
                "harness_sha256": hashes,
            },
            indent=2,
        )
        + "\n"
    )

    def phase(name, command):
        subprocess.run(
            [
                str(args.python),
                str(harness / "run_phase.py"),
                "--source",
                str(args.source.resolve()),
                "--output",
                str(args.output / name),
                "--",
                str(args.python),
                *map(str, command),
            ],
            check=True,
        )

    for role, steps, seed, mode in (
        ("reference", 10, 42, "instrumented"),
        ("control", 10, 42, "instrumented"),
        ("holdout", 20, 43, "instrumented"),
        ("baseline-holdout", 20, 43, "baseline"),
    ):
        command = [
            harness / "benchmark_workload.py",
            "--benchmark-root",
            args.benchmark_root.resolve(),
            "--workload",
            args.workload,
            "--steps",
            steps,
            "--data-steps",
            20,
            "--seed",
            seed,
            "--batch-size",
            2,
            "--device",
            "cpu",
            "--mode",
            mode,
            "--output",
            args.output / role,
        ]
        if args.data_root is not None:
            command.extend(["--data-root", args.data_root.resolve()])
        phase(role, command)
    invariants = args.output / "inference/invariants.jsonl"
    traces = sorted((args.output / "reference/traces").glob("trace_API*.log"))
    if not traces:
        raise RuntimeError("No reference API traces")
    phase(
        "inference",
        [
            "-m",
            "traincheck.infer_engine",
            "-t",
            *traces,
            "-o",
            invariants,
            "--backend",
            "pandas",
            "-j",
            1,
            "--disable-relation",
            "FunctionLeadRelation",
            "FunctionCoverRelation",
        ],
    )
    for role in ("reference", "control", "holdout"):
        traces = sorted((args.output / role / "traces").glob("trace_API*.log"))
        phase(
            "check-" + role,
            [
                harness / "check_coverage.py",
                "--traces",
                *traces,
                "--invariants",
                invariants,
                "--output",
                args.output / ("check-" + role) / "coverage.json",
            ],
        )
    traces = sorted((args.output / "holdout/traces").glob("trace_API*.log"))
    phase(
        "check-holdout-direct",
        [
            harness / "check_coverage.py",
            "--traces",
            *traces,
            "--invariants",
            invariants,
            "--observer-off",
            "--output",
            args.output / "check-holdout-direct/coverage.json",
        ],
    )

    phase(
        "prepare-fault",
        [
            harness / "fault_control.py",
            "--traces",
            *traces,
            "--invariants",
            invariants,
            "--healthy-coverage",
            args.output / "check-holdout/coverage.json",
            "--output",
            args.output / "fault",
        ],
    )
    faulty_traces = sorted((args.output / "fault/traces").glob("trace_API*.log"))
    phase(
        "check-fault",
        [
            harness / "check_coverage.py",
            "--traces",
            *faulty_traces,
            "--invariants",
            invariants,
            "--output",
            args.output / "check-fault/coverage.json",
        ],
    )
    subprocess.run(
        [
            str(args.python),
            str(harness / "compare_study.py"),
            "--root",
            str(args.output),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
