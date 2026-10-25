"""Reproduce bounded synthetic TracePandas indexing measurements from a checkout.

Run: python -m tools.benchmarks.trace_pandas_indexes --output measurements.json
No training, cluster access, or saved trace files are required. Each lookup run
includes its first lazy-index construction. The legacy oracle preserves the four
original query methods; supporting runtime types are identical on both sides.
"""

import argparse
import hashlib
import inspect
import json
import logging
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import pandas as pd

import traincheck.trace.trace_pandas as candidate
from tests.fixtures import legacy_trace_pandas as legacy
from tests.test_call_index import pair
from tests.test_context_index import cycle, encode, event


def compare(frame, operation, repetitions):
    seconds = {"legacy": [], "candidate": []}
    expected = None
    for repetition in range(repetitions):
        variants = [("legacy", legacy), ("candidate", candidate)]
        if repetition % 2:
            variants.reverse()
        for name, module in variants:
            events = frame.copy(deep=True)
            started = time.perf_counter()
            value = operation(module, events)
            seconds[name].append(time.perf_counter() - started)
            if expected is None:
                expected = value
            assert value == expected, f"{name} result differs from the legacy oracle"
    medians = {name: statistics.median(values) for name, values in seconds.items()}
    return {
        "elapsed_seconds": seconds,
        "median_seconds": medians,
        "speedup": medians["legacy"] / medians["candidate"],
        "exact_results_equal": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["all", "context", "call"], default="all")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--contexts", type=int, default=500)
    parser.add_argument("--noise", type=int, default=50_000)
    parser.add_argument("--calls", type=int, default=20_000)
    parser.add_argument(
        "--queries", type=int, default=500, help="Pre/post pairs queried"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    for name in ["repetitions", "contexts", "calls", "queries"]:
        if getattr(args, name) < 1:
            parser.error(f"--{name} must be positive")
    if args.noise < 0:
        parser.error("--noise must be nonnegative")
    logging.disable(logging.WARNING)
    results = {}
    if args.mode in ("all", "context"):
        rows = []
        for index in range(args.contexts):
            rows.extend(
                cycle(index * 3, obj=index % 20, name=f"fixtures.Ctx{index % 4}")
            )
        rows.extend(
            event(args.contexts * 3 + index, "math_call") for index in range(args.noise)
        )
        frame = pd.DataFrame(rows)

        def prepare(module, events):
            return encode(
                module.TracePandas(events, truncate_incomplete_func_calls=False)
            )

        results["context"] = {
            "rows": len(frame),
            "contexts": args.contexts,
            "timed_work": "constructor plus context-state serialization",
            **compare(frame, prepare, args.repetitions),
        }
    if args.mode in ("all", "call"):
        frame = pd.DataFrame(
            [
                record
                for index in range(args.calls)
                for record in pair(f"call-{index:08}", start=index * 2)
            ]
        )
        wanted = [
            f"call-{(index * 117) % args.calls:08}" for index in range(args.queries)
        ]

        def lookup(module, events):
            trace = module.TracePandas(events, truncate_incomplete_func_calls=False)
            return [
                (
                    trace.get_pre_func_call_record(call_id),
                    trace.get_post_func_call_record(call_id),
                )
                for call_id in wanted
            ]

        results["call"] = {
            "rows": len(frame),
            "record_queries": args.queries * 2,
            "timed_work": "constructor plus record lookups, including lazy-index construction",
            **compare(frame, lookup, args.repetitions),
        }
    output = {
        "description": "Synthetic fixtures, alternating variant order; no training or inference",
        "repetitions": args.repetitions,
        "runtime": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "candidate_source_sha256": hashlib.sha256(
            Path(inspect.getfile(candidate)).read_bytes()
        ).hexdigest(),
        "results": results,
    }
    rendered = json.dumps(output, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
