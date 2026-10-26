#!/usr/bin/env python3
"""Deterministic identity-lookup microbenchmark, with no external trace input.

Run against the candidate checkout, for example:
    python -m tools.benchmarks.identity_cache > results.json
Small smoke test:
    python -m tools.benchmarks.identity_cache --rows 1000 --calls 20

The legacy subclass changes only the two identity methods back to their
pre-cache implementation. Each trial constructs a fresh trace over the same
read-only DataFrame, then makes the same ordered get_meta_vars calls. The first
lookup/cache build is included. Three repetitions alternate execution order.
Input generation, correctness hashing, and garbage collection are not timed.

This measures identity scans, not full inference. The synthetic trace has no
context managers. No RSS/PSS, peak memory, or multiprocessing scaling is measured.
The frame-memory estimate is not whole-process memory. Cache tuples add O(P+T)
stored values; each return still allocates a fresh O(P+T) list. The extra source
reference retains a frame until invalidation and is not a DataFrame copy.
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
import subprocess
import sys
import time
from pathlib import Path
from typing import Literal, TypedDict

import numpy as np
import pandas as pd

import traincheck.trace.trace_pandas as trace_module
from traincheck.trace.trace_pandas import TracePandas

TimingField = Literal["construction_seconds", "queries_seconds", "total_seconds"]
TIMING_FIELDS: tuple[TimingField, ...] = (
    "construction_seconds",
    "queries_seconds",
    "total_seconds",
)


class TrialResult(TypedDict):
    repetition: int
    position: int
    variant: str
    construction_seconds: float
    queries_seconds: float
    total_seconds: float
    result_sha256: str
    none_results: int
    empty_dict_results: int


class Measurement(TrialResult, total=False):
    cached_identity_counts: dict[str, int]
    cache_tuple_containers_shallow_bytes: int


class LegacyIdentityTrace(TracePandas):
    """Original lookup expressions; all other behavior uses the same checkout."""

    def get_process_ids(self) -> list[object]:
        if "process_id" not in self.events.columns:
            return []
        return self.events["process_id"].dropna().unique().tolist()

    def get_thread_ids(self) -> list[object]:
        if "thread_id" not in self.events.columns:
            return []
        return self.events["thread_id"].dropna().unique().tolist()


def synthetic_frame(rows: int) -> pd.DataFrame:
    sequence = np.arange(rows, dtype=np.int64)
    process_ids = ((sequence // 16) % 8 + 100).astype(np.float64)
    thread_ids = (sequence % 16 + 1000).astype(np.float64)
    # NaNs exercise the same dropna work as mixed event traces. They never
    # remove every occurrence of an identity at the minimum supported size.
    process_ids[sequence % 97 == 0] = np.nan
    thread_ids[sequence % 193 == 0] = np.nan
    return pd.DataFrame(
        {"time": sequence, "process_id": process_ids, "thread_id": thread_ids}
    )


def fixed_queries(calls: int) -> list[tuple[int, int, int]]:
    # Six valid, one unknown-process, and one unknown-thread call per cycle.
    # Process/thread presence remains independent, matching get_meta_vars.
    return [
        (
            i,
            -1 if i % 8 == 6 else 100 + i % 8,
            -1 if i % 8 == 7 else 1000 + (i * 3) % 16,
        )
        for i in range(calls)
    ]


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def trial(
    cls: type[TracePandas],
    data: pd.DataFrame,
    queries: list[tuple[int, int, int]],
    *,
    repetition: int,
    position: int,
    variant: str,
) -> Measurement:
    gc.collect()
    begin = time.perf_counter()
    trace = cls(data, truncate_incomplete_func_calls=False)
    constructed = time.perf_counter()
    results = [trace.get_meta_vars(*query) for query in queries]
    finished = time.perf_counter()
    record: Measurement = {
        "repetition": repetition,
        "position": position,
        "variant": variant,
        "construction_seconds": constructed - begin,
        "queries_seconds": finished - constructed,
        "total_seconds": finished - begin,
        "result_sha256": digest(results),
        "none_results": sum(value is None for value in results),
        "empty_dict_results": sum(value == {} for value in results),
    }
    if cls is TracePandas:
        assert trace._identity_values_source is trace.events
        record["cached_identity_counts"] = {
            key: len(values) for key, values in trace._identity_values.items()
        }
        record["cache_tuple_containers_shallow_bytes"] = sum(
            sys.getsizeof(values) for values in trace._identity_values.values()
        )
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=573494)
    parser.add_argument("--calls", type=int, default=1000)
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    if args.rows < 256 or args.calls < 8 or args.repetitions < 1:
        parser.error("require rows >= 256, calls >= 8, repetitions >= 1")
    if not hasattr(TracePandas, "_get_identity_values"):
        parser.error("import a candidate checkout containing the identity cache")

    logging.disable(logging.CRITICAL)
    data = synthetic_frame(args.rows)
    queries = fixed_queries(args.calls)
    input_hash = hashlib.sha256(
        pd.util.hash_pandas_object(data, index=True).to_numpy().tobytes()
    ).hexdigest()
    expected: list[dict[str, object] | None] = [
        None if i % 8 in (6, 7) else {} for i in range(args.calls)
    ]
    expected_hash = digest(expected)
    measurements: list[Measurement] = []
    variants = {"legacy": LegacyIdentityTrace, "candidate": TracePandas}
    for repetition in range(args.repetitions):
        order = ["legacy", "candidate"]
        if repetition % 2:
            order.reverse()
        for position, variant in enumerate(order):
            record = trial(
                variants[variant],
                data,
                queries,
                repetition=repetition + 1,
                position=position + 1,
                variant=variant,
            )
            assert record["result_sha256"] == expected_hash, record
            measurements.append(record)

    # Catch accidental mutation of the shared input by either implementation.
    assert (
        hashlib.sha256(
            pd.util.hash_pandas_object(data, index=True).to_numpy().tobytes()
        ).hexdigest()
        == input_hash
    )
    medians = {
        variant: {
            metric: statistics.median(
                record[metric]
                for record in measurements
                if record["variant"] == variant
            )
            for metric in TIMING_FIELDS
        }
        for variant in variants
    }
    report = {
        "scope": "construction plus repeated identity-checked get_meta_vars calls",
        "settings": vars(args),
        "input": {
            "sha256_pandas_row_hashes": input_hash,
            "query_sha256": digest(queries),
            "dtypes": {key: str(value) for key, value in data.dtypes.items()},
            "dataframe_deep_bytes": int(data.memory_usage(index=True, deep=True).sum()),
            "context_manager_count": 0,
            "process_count": 8,
            "thread_count": 16,
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "host_class": f"local {platform.system()} {platform.machine()} workstation",
            "logical_cpu_count": os.cpu_count(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "native_thread_settings": {
                name: os.environ.get(name)
                for name in (
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                )
            },
            "candidate_source_sha256": hashlib.sha256(
                Path(inspect.getfile(trace_module)).read_bytes()
            ).hexdigest(),
            "candidate_commit": subprocess.check_output(
                [
                    "git",
                    "-C",
                    str(Path(inspect.getfile(trace_module)).resolve().parents[2]),
                    "rev-parse",
                    "HEAD",
                ],
                text=True,
            ).strip(),
            "candidate_source_matches_commit": subprocess.run(
                [
                    "git",
                    "-C",
                    str(Path(inspect.getfile(trace_module)).resolve().parents[2]),
                    "diff",
                    "--quiet",
                    "HEAD",
                    "--",
                    "traincheck/trace/trace_pandas.py",
                ],
                check=False,
            ).returncode
            == 0,
            "legacy_methods_source_sha256": hashlib.sha256(
                inspect.getsource(LegacyIdentityTrace).encode()
            ).hexdigest(),
        },
        "measurements": measurements,
        "medians": medians,
        "median_total_speedup": (
            medians["legacy"]["total_seconds"] / medians["candidate"]["total_seconds"]
        ),
        "correctness": {
            "all_results_equal_expected": True,
            "result_sha256": expected_hash,
            "input_unchanged": True,
        },
        "limitations": [
            "Synthetic identity-lookup microbenchmark, not completed inference.",
            "No context-manager work or trace-file loading; input generation excluded.",
            "No RSS/PSS, peak-memory, or multiprocessing memory-scaling measurement.",
            "Cache stores O(P+T) values and returns fresh O(P+T) lists.",
            "Stable identity values, dtypes, and row order required within one frame.",
        ],
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
