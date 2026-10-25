#!/usr/bin/env python3
"""Compare exact pandas filtering with the time-range index on synthetic calls.

Run: python -m tools.benchmarks.time_range_index > results.json
Each trial uses a fresh TracePandas. Construction and the first lazy index build
are included, along with fresh event/record construction for every query. Input
creation, result hashing, and garbage collection are outside the timed region.
Repetitions alternate legacy/candidate order. This isolates window queries; it
is not full inference and does not measure peak process memory.
"""

import argparse
import gc
import hashlib
import inspect
import json
import logging
import os
import platform
import statistics
import time
from pathlib import Path
from typing import Literal, TypedDict

import numpy as np
import pandas as pd

import traincheck.trace.trace_pandas as trace_module
from traincheck.instrumentor.tracer import TraceLineType
from traincheck.trace.trace_pandas import TracePandas

TimingField = Literal[
    "construction_seconds",
    "first_query_seconds",
    "remaining_queries_seconds",
    "total_seconds",
]
TIMING_FIELDS: tuple[TimingField, ...] = (
    "construction_seconds",
    "first_query_seconds",
    "remaining_queries_seconds",
    "total_seconds",
)


class TrialResult(TypedDict):
    repetition: int
    order: int
    variant: str
    construction_seconds: float
    first_query_seconds: float
    remaining_queries_seconds: float
    total_seconds: float
    result_sha256: str


class Measurement(TrialResult, total=False):
    indexed_rows: int
    identity_groups: int


class LegacyRangeTrace(TracePandas):
    def _get_func_call_time_positions(
        self, time_range: object, process_id: object, thread_id: object
    ) -> None:
        return None


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=200000)
    parser.add_argument("--queries", type=int, default=1000)
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    if args.rows < 4 or args.rows % 2 or args.queries < 2 or args.repetitions < 1:
        parser.error("require even rows >= 4, queries >= 2, repetitions >= 1")
    logging.disable(logging.CRITICAL)
    row = np.arange(args.rows, dtype=np.int64)
    call = row // 2
    data = pd.DataFrame(
        {
            "time": row.astype(np.float64),
            "type": np.where(
                row % 2 == 0, TraceLineType.FUNC_CALL_PRE, TraceLineType.FUNC_CALL_POST
            ),
            "process_id": 100 + call % 4,
            "thread_id": 1000 + call % 8,
            "func_call_id": [f"call-{i}" for i in call],
            "function": "fixture.step",
        }
    )
    queries: list[tuple[tuple[float, float], int, int]] = []
    for i in range(args.queries):
        index = (i * 97) % (args.rows // 2)
        queries.append(
            (
                (float(index * 2) - 0.5, float(index * 2) + 1.5),
                100 + index % 4,
                1000 + index % 8,
            )
        )

    def input_hash() -> str:
        return hashlib.sha256(
            pd.util.hash_pandas_object(data, index=True).to_numpy().tobytes()
        ).hexdigest()

    before = input_hash()
    records: list[Measurement] = []
    for repetition in range(args.repetitions):
        variants = [("legacy", LegacyRangeTrace), ("candidate", TracePandas)]
        if repetition % 2:
            variants.reverse()
        for order, (name, cls) in enumerate(variants):
            gc.collect()
            start = time.perf_counter()
            trace = cls(data, truncate_incomplete_func_calls=False)
            constructed = time.perf_counter()
            results = [trace.query_func_call_events_within_time(*queries[0])]
            first = time.perf_counter()
            results.extend(
                trace.query_func_call_events_within_time(*query)
                for query in queries[1:]
            )
            end = time.perf_counter()
            assert all(len(events) == 1 for events in results)
            summary = [
                [
                    {
                        "class": type(event).__name__,
                        "pre": event.pre_record,
                        "post": event.post_record,
                    }
                    for event in events
                ]
                for events in results
            ]
            record: Measurement = {
                "repetition": repetition + 1,
                "order": order + 1,
                "variant": name,
                "construction_seconds": constructed - start,
                "first_query_seconds": first - constructed,
                "remaining_queries_seconds": end - first,
                "total_seconds": end - start,
                "result_sha256": digest(summary),
            }
            if name == "candidate":
                assert trace._func_time_positions is not None
                record["indexed_rows"] = sum(
                    len(positions)
                    for _, positions in trace._func_time_positions.values()
                )
                record["identity_groups"] = len(trace._func_time_positions)
            records.append(record)
    assert input_hash() == before
    assert len({record["result_sha256"] for record in records}) == 1
    medians = {
        name: {
            metric: statistics.median(
                record[metric] for record in records if record["variant"] == name
            )
            for metric in TIMING_FIELDS
        }
        for name in ["legacy", "candidate"]
    }
    print(
        json.dumps(
            {
                "scope": "synthetic function-call window queries, including construction and lazy build; not full inference or peak memory",
                "settings": vars(args),
                "runtime": {
                    "python": platform.python_version(),
                    "platform": platform.platform(),
                    "pandas": pd.__version__,
                    "numpy": np.__version__,
                    "native_threads": {
                        name: os.environ.get(name)
                        for name in [
                            "OMP_NUM_THREADS",
                            "OPENBLAS_NUM_THREADS",
                            "MKL_NUM_THREADS",
                        ]
                    },
                },
                "source_sha256": hashlib.sha256(
                    Path(inspect.getfile(trace_module)).read_bytes()
                ).hexdigest(),
                "input_sha256": before,
                "input_unchanged": True,
                "all_ordered_results_equal": True,
                "measurements": records,
                "medians": medians,
                "median_total_ratio": medians["legacy"]["total_seconds"]
                / medians["candidate"]["total_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
