"""Identity lookup equivalence during precondition inference."""

import copy
import json
import logging
import pickle
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from traincheck.invariant.precondition import find_precondition_from_single_group
from traincheck.trace.trace_pandas import TracePandas


class LegacyIdentityTrace(TracePandas):
    def get_process_ids(self):
        if "process_id" not in self.events.columns:
            return []
        return self.events["process_id"].dropna().unique().tolist()

    def get_thread_ids(self):
        if "thread_id" not in self.events.columns:
            return []
        return self.events["thread_id"].dropna().unique().tolist()


def make(frame, cls=TracePandas):
    return cls(frame.copy(deep=True), truncate_incomplete_func_calls=False)


def frame(process_ids=(1, 2, 1), thread_ids=(4, 3, 4)):
    return pd.DataFrame(
        {
            "time": range(len(process_ids)),
            "process_id": process_ids,
            "thread_id": thread_ids,
        }
    )


class IdentityCacheTests(unittest.TestCase):
    def setUp(self):
        previous = logging.root.manager.disable
        logging.disable(logging.CRITICAL)
        self.addCleanup(logging.disable, previous)

    def assert_equivalent(self, data):
        old, new = make(data, LegacyIdentityTrace), make(data)
        for _ in range(2):
            for method in ("get_process_ids", "get_thread_ids"):
                expected, actual = getattr(old, method)(), getattr(new, method)()
                self.assertEqual(expected, actual)
                self.assertEqual(list(map(type, expected)), list(map(type, actual)))
        return old, new

    def test_values_types_and_encounter_order(self):
        for values in (
            pd.Series([8, 2, 8, -1], dtype="int64"),
            pd.Series([2**63 + 7, 2**63 + 2, None, 2**63 + 7], dtype="UInt64"),
            pd.Series([8, pd.NA, 2, 8], dtype="Int64"),
            pd.Series([8.5, np.nan, 2.5, 8.5]),
            pd.Series(["z", None, "a", "z"], dtype="string"),
            pd.Series(["z", None, "a", "z"], dtype="category"),
            pd.Series([True, 1, False, 0], dtype=object),
            pd.Series([None, np.nan, pd.NA, None], dtype=object),
        ):
            with self.subTest(dtype=str(values.dtype), values=repr(values.tolist())):
                self.assert_equivalent(frame(values, values))

    def test_randomized_identity_columns(self):
        rng = np.random.default_rng(42)
        for _ in range(80):
            values = rng.integers(-8, 8, size=50).astype(float)
            values[rng.random(50) < 0.2] = np.nan
            self.assert_equivalent(frame(values, values[::-1]))

    def test_empty_and_missing_columns(self):
        for data in (frame([], []), pd.DataFrame({"time": [0]})):
            self.assert_equivalent(data)

    def test_returned_lists_are_independent(self):
        trace = make(frame())
        result = trace.get_process_ids()
        result[:] = [999]
        self.assertEqual(trace.get_process_ids(), [1, 2])
        result = trace.get_thread_ids()
        result.clear()
        self.assertEqual(trace.get_thread_ids(), [4, 3])

    def test_frame_replacement_rebuilds_same_size_cache(self):
        trace = make(frame())
        trace.get_process_ids()
        trace.get_thread_ids()
        trace.events = frame((9, 7, 9), (5, 6, 5))
        self.assertEqual(trace.get_process_ids(), [9, 7])
        self.assertEqual(trace.get_thread_ids(), [5, 6])

    def test_same_frame_length_change_rebuilds_cache(self):
        trace = make(frame())
        trace.get_process_ids()
        trace.get_thread_ids()
        trace.events.loc[3] = [3, 9, 7]
        self.assertEqual(trace.get_process_ids(), [1, 2, 9])
        self.assertEqual(trace.get_thread_ids(), [4, 3, 7])
        trace.events.drop(index=[1, 3], inplace=True)
        self.assertEqual(trace.get_process_ids(), [1])
        self.assertEqual(trace.get_thread_ids(), [4])

    def test_stage_traces_and_instances_have_separate_caches(self):
        data = frame()
        data["meta_vars.stage"] = ["init", "train", "init"]
        trace = make(data)
        trace.get_process_ids()
        stages = trace.get_traces_for_stage()
        self.assertEqual(stages["init"].get_process_ids(), [1])
        self.assertEqual(stages["train"].get_process_ids(), [2])
        self.assertEqual(make(frame((9,), (7,))).get_process_ids(), [9])

    def test_pickled_initialized_cache(self):
        trace = make(frame())
        trace.get_process_ids()
        restored = pickle.loads(pickle.dumps(trace))
        self.assertIs(restored._identity_values_source, restored.events)
        self.assertEqual(restored.get_process_ids(), [1, 2])
        self.assertEqual(restored.get_thread_ids(), [4, 3])

    def test_meta_vars_membership_preserves_independent_id_checks(self):
        old, new = self.assert_equivalent(frame((1, 2), (3, 4)))
        for pid, tid in ((1, 3), (1, 4), (2, 3), (9, 3), (1, 9)):
            self.assertEqual(
                old.get_meta_vars(0, pid, tid), new.get_meta_vars(0, pid, tid)
            )
        self.assertEqual(new.get_meta_vars(0, 1, 4), {})
        self.assertIsNone(new.get_meta_vars(0, 9, 3))

    def test_repeated_meta_vars_queries_only_scan_each_column_once(self):
        trace = make(frame())
        original = pd.Series.unique
        scanned = []

        def unique(series):
            scanned.append(series.name)
            return original(series)

        with patch.object(pd.Series, "unique", unique):
            for _ in range(100):
                self.assertEqual(trace.get_meta_vars(0, 1, 4), {})
        self.assertEqual(scanned, ["process_id", "thread_id"])

    def test_real_precondition_inference_matches_legacy(self):
        old, new = self.assert_equivalent(frame())
        positive = [
            [
                {
                    "time": i,
                    "process_id": 1,
                    "thread_id": 4,
                    "function": "fixtures.step",
                    "meta_vars.phase": "train",
                }
            ]
            for i in range(12)
        ]
        negative = [
            [
                {
                    "time": i,
                    "process_id": 2,
                    "thread_id": 3,
                    "function": "fixtures.step",
                    "meta_vars.phase": "eval",
                }
            ]
            for i in range(8)
        ]
        results = []
        for trace in (old, new):
            pos, neg = copy.deepcopy(positive), copy.deepcopy(negative)
            clauses = find_precondition_from_single_group(pos, neg, [trace])
            results.append(
                (
                    sorted(json.dumps(c.to_dict(), sort_keys=True) for c in clauses),
                    pos,
                    neg,
                )
            )
        self.assertTrue(results[0][0])
        self.assertEqual(*results)


if __name__ == "__main__":
    unittest.main()
