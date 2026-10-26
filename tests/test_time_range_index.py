"""Differential tests against the unchanged pandas filtering path."""

import copy
import logging
import math
import pickle
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from traincheck.instrumentor.tracer import TraceLineType as T
from traincheck.trace.trace_pandas import TracePandas


class LegacyRangeTrace(TracePandas):
    def _get_func_call_time_positions(self, *args):
        return None


def frame(times=(1, 2, 3, 4), ids=("z", "z", "a", "a")):
    n = len(times)
    return pd.DataFrame(
        {
            "time": times,
            "type": [T.FUNC_CALL_PRE, T.FUNC_CALL_POST] * (n // 2),
            "process_id": pd.Series([1] * n, dtype="int64"),
            "thread_id": pd.Series([2] * n, dtype="int64"),
            "func_call_id": ids,
            "function": ["fixture.step"] * n,
            "args": [{} for _ in range(n)],
            "kwargs": [{} for _ in range(n)],
            "exception": ["fixture failure"] * n,
        }
    )


def make(data, cls=TracePandas):
    return cls(data.copy(deep=True), truncate_incomplete_func_calls=False)


def value(v):
    if isinstance(v, dict):
        return {k: value(x) for k, x in v.items()}
    if isinstance(v, list):
        return [value(x) for x in v]
    if isinstance(v, float) and math.isnan(v):
        return (type(v).__name__, "nan")
    return (type(v).__name__, v)


def outcome(trace, bounds, process=1, thread=2):
    try:
        events = trace.query_func_call_events_within_time(bounds, process, thread)
        return ("result", [(type(e).__name__, value(e.__dict__)) for e in events])
    except Exception as error:
        return ("exception", type(error).__name__, str(error))


class IntLike(int):
    pass


class EqualType:
    def __init__(self, text):
        self.text = text

    def __eq__(self, other):
        return self.text == other

    def __hash__(self):
        return hash(self.text)


class TimeRangeIndexTests(unittest.TestCase):
    def setUp(self):
        previous = logging.root.manager.disable
        logging.disable(logging.CRITICAL)
        self.addCleanup(logging.disable, previous)

    def compare(self, data, queries):
        old, new = make(data, LegacyRangeTrace), make(data)
        for query in queries:
            with self.subTest(query=query, dtypes=str(data.dtypes)):
                self.assertEqual(outcome(old, *query), outcome(new, *query))
        return old, new

    def test_strict_bounds_ties_reverse_and_empty(self):
        data = frame((1, 2, 2, 3))
        _, new = self.compare(
            data,
            [
                ((a, b),)
                for a, b in [(0, 4), (1, 3), (1, 2), (2, 3), (2, 2), (4, 0), (9, 10)]
            ],
        )
        self.assertIsNotNone(new._func_time_positions)
        self.assertEqual(
            [
                e.pre_record["func_call_id"]
                for e in new.query_func_call_events_within_time((0, 4), 1, 2)
            ],
            ["a", "z"],
        )

    def test_empty_native_frame_and_extreme_identity_keys(self):
        _, empty = self.compare(frame().iloc[:0], [((0, 5),)])
        self.assertEqual(empty._get_func_call_time_positions((0, 5), 1, 2), [])
        for pid, tid in [(-(2**63), 2**63 - 1), (2**63 - 1, -(2**63))]:
            data = frame()
            data["process_id"] = pd.Series([pid] * 4, dtype="int64")
            data["thread_id"] = pd.Series([tid] * 4, dtype="int64")
            _, trace = self.compare(data, [((0, 5), pid, tid)])
            self.assertEqual(
                trace._get_func_call_time_positions((0, 5), pid, tid), [0, 1, 2, 3]
            )

    def test_float_nan_inf_signed_zero(self):
        data = frame(
            (-math.inf, -0.0, 0.0, math.inf, math.nan, math.nan),
            ("z", "z", "a", "a", "missing", "missing"),
        )
        self.compare(
            data,
            [
                ((a, b),)
                for a, b in [
                    (-math.inf, math.inf),
                    (-1.0, 1.0),
                    (-0.0, 0.0),
                    (math.inf, -math.inf),
                    (math.nan, 9.0),
                ]
            ],
        )

    def test_integer_extremes_and_precision_fallback(self):
        for times in [
            (2**53, 2**53 + 1, 2**53 + 2, 2**53 + 3),
            (-(2**63), -(2**63) + 1, 2**63 - 2, 2**63 - 1),
        ]:
            data = frame(times)
            old, new = self.compare(
                data,
                [
                    ((times[0], times[-1]),),
                    ((float(times[0]), float(times[-1])),),
                    ((-(2**63) - 1, 2**63),),
                ],
            )
            self.assertIsNone(
                new._get_func_call_time_positions(
                    (float(times[0]), float(times[-1])), 1, 2
                )
            )
        data = frame(tuple(float(2**53 + i) for i in range(4)))
        self.compare(
            data, [((2**53 + 1, 2**53 + 3),), ((float(2**53), float(2**53 + 4)),)]
        )

    def test_numeric_query_types_fallback(self):
        data = frame()
        for process, thread in [
            (True, 2),
            (1.0, 2.0),
            (np.int64(1), np.int64(2)),
            (IntLike(1), 2),
            (2**64, 2),
            (None, 2),
        ]:
            _, new = self.compare(data, [((0, 5), process, thread)])
            self.assertIsNone(
                new._get_func_call_time_positions((0, 5), process, thread)
            )
        for bounds in [
            (np.int64(0), np.int64(5)),
            (IntLike(0), 5),
            [0, 5],
            (0,),
            None,
            ("a", "b"),
        ]:
            self.compare(data, [(bounds,)])

    def test_other_processes_threads_and_missing_identity(self):
        data = pd.concat([frame(), frame()], ignore_index=True)
        data.loc[4:, "process_id"] = 3
        data.loc[4:, "thread_id"] = 4
        self.compare(
            data, [((0, 5), pid, tid) for pid, tid in [(1, 2), (3, 4), (1, 4), (99, 2)]]
        )

    def test_unsorted_input_duplicate_labels_preserves_row_order(self):
        data = frame((3, 4, 1, 2))
        data.index = [4, 4, 1, 1]
        self.compare(data, [((0, 5),), ((0, 3),)])
        backwards = data.iloc[[1, 0, 2, 3]].copy()
        self.assertEqual(
            self.compare(backwards, [((0, 5),)])[1].query_func_call_events_within_time(
                (0, 0), 1, 2
            ),
            [],
        )
        self.assertEqual(outcome(make(backwards), (0, 5))[1], "AssertionError")

    def test_duplicate_incomplete_and_exception_records(self):
        for data in [
            frame().iloc[:3],
            pd.concat([frame(), frame().iloc[:1]], ignore_index=True),
        ]:
            self.compare(data, [((0, 5),), ((0, 3),)])
        data = frame()
        data.loc[1, "type"] = T.FUNC_CALL_POST_EXCEPTION
        self.compare(data, [((0, 5),), ((0, 3),)])

    def test_null_mixed_and_categorical_call_ids(self):
        for ids in [
            pd.Series(["z", "z", None, None]),
            pd.Series([1, 1, "a", "a"], dtype=object),
            pd.Categorical(["z", "z", "a", "a"], categories=["z", "a", "unused"]),
            pd.Series(["z", "z", "a", "a"], dtype="string"),
        ]:
            data = frame()
            data["func_call_id"] = ids
            self.compare(data, [((0, 5),), ((9, 10),)])

    def test_time_and_identity_dtype_fallback(self):
        for column in ["time", "process_id", "thread_id"]:
            for dtype in ["object", "Int64", "Float64", "category", "uint64", "int32"]:
                data = frame()
                data[column] = data[column].astype(dtype)
                _, new = self.compare(data, [((0, 5),)])
                self.assertIsNone(new._get_func_call_time_positions((0, 5), 1, 2))
            data = frame()
            data[column] = data[column].astype(float)
            data.loc[0, column] = np.nan
            self.compare(data, [((0, 5),), ((0.0, 5.0),)])

    def test_type_nulls_extensions_and_custom_comparison(self):
        for dtype in ["string", "category"]:
            data = frame()
            data["type"] = data["type"].astype(dtype)
            _, new = self.compare(data, [((0, 5),)])
            self.assertIsNone(new._get_func_call_time_positions((0, 5), 1, 2))
        for kind in [None, np.nan, pd.NA, EqualType(T.FUNC_CALL_PRE)]:
            data = frame()
            data.at[0, "type"] = kind
            self.compare(data, [((0, 5),), ((9, 10),)])

    def test_non_function_events_are_excluded(self):
        data = pd.concat([frame(), frame()], ignore_index=True)
        data.loc[4:, "type"] = "unrelated"
        self.compare(data, [((0, 5),)])

    def test_missing_and_duplicate_columns_keep_exceptions(self):
        for col in ["type", "process_id", "thread_id", "func_call_id"]:
            self.compare(frame().drop(columns=col), [((0, 5),)])
        data = pd.concat([frame(), frame()[["type"]]], axis=1)
        self.compare(data, [((0, 5),)])

    def test_cache_rebuild_on_replacement_and_length_change(self):
        old, new = self.compare(frame(), [((0, 5),)])
        for trace in [old, new]:
            trace.events = frame((10, 11, 12, 13))
        self.assertEqual(outcome(old, (9, 14)), outcome(new, (9, 14)))
        for trace in [old, new]:
            trace.events.drop(index=[2, 3], inplace=True)
        self.assertEqual(outcome(old, (9, 14)), outcome(new, (9, 14)))
        self.assertEqual(new._func_time_length, 2)

    def test_fresh_events_and_record_dicts(self):
        trace = make(frame())
        first = trace.query_func_call_events_within_time((0, 5), 1, 2)
        second = trace.query_func_call_events_within_time((0, 5), 1, 2)
        self.assertIsNot(first[0], second[0])
        self.assertIsNot(first[0].pre_record, second[0].pre_record)
        first[0].pre_record["function"] = "mutated"
        self.assertEqual(second[0].pre_record["function"], "fixture.step")
        self.assertEqual(
            trace.query_func_call_events_within_time((0, 5), 1, 2)[0].pre_record[
                "function"
            ],
            "fixture.step",
        )

    def test_repeated_query_avoids_type_isin_rescan(self):
        trace = make(frame())
        with patch.object(
            pd.Series, "isin", side_effect=AssertionError("unexpected full scan")
        ):
            for _ in range(10):
                self.assertEqual(
                    len(trace.query_func_call_events_within_time((0, 5), 1, 2)), 2
                )

    def test_random_windows_and_permuted_rows(self):
        rng = np.random.default_rng(91)
        data = pd.concat(
            [
                frame(
                    (i * 4 + 1, i * 4 + 2, i * 4 + 3, i * 4 + 4),
                    (f"{i}z", f"{i}z", f"{i}a", f"{i}a"),
                )
                for i in range(20)
            ],
            ignore_index=True,
        )
        for _ in range(10):
            permuted = data.iloc[rng.permutation(len(data))].copy()
            self.compare(
                permuted,
                [((int(a), int(b)),) for a, b in rng.integers(-2, 83, size=(20, 2))],
            )

    def test_pickled_index_retains_frame_and_results(self):
        trace = make(frame())
        expected = outcome(trace, (0, 5))
        restored = pickle.loads(pickle.dumps(trace))
        self.assertIs(restored._func_time_source, restored.events)
        self.assertEqual(outcome(restored, (0, 5)), expected)


if __name__ == "__main__":
    unittest.main()
