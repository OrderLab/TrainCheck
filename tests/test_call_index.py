"""Call-record index differential tests against frozen legacy query methods."""

import copy
import logging
import unittest

import numpy as np
import pandas as pd

import traincheck.trace.trace_pandas as NEW
from tests.fixtures import legacy_trace_pandas as OLD

PRE = OLD.TraceLineType.FUNC_CALL_PRE
POST = OLD.TraceLineType.FUNC_CALL_POST
EXCEPTION = OLD.TraceLineType.FUNC_CALL_POST_EXCEPTION


def record(call_id, when, kind, *, pid=1, tid=2, stage="train", **extra):
    return {
        "func_call_id": call_id,
        "function": "fixtures.work",
        "time": when,
        "type": kind,
        "process_id": pid,
        "thread_id": tid,
        "args": {"0": {"int": 7}},
        "kwargs": {},
        "return_values": {"int": 9},
        "exception": "fixture exception",
        "meta_vars.stage": stage,
        **extra,
    }


def pair(call_id="call", start=0, **extra):
    return [
        record(call_id, start, PRE, **extra),
        record(call_id, start + 1, POST, **extra),
    ]


def make(module, frame, truncate=False):
    if not isinstance(frame, pd.DataFrame):
        frame = pd.DataFrame(copy.deepcopy(frame))
    else:
        frame = frame.copy(deep=True)
    return module.TracePandas(frame, truncate_incomplete_func_calls=truncate)


def outcome(trace, method, call_id):
    try:
        return ("ok", getattr(trace, method)(call_id))
    except Exception as exc:
        return (type(exc).__name__, str(exc))


class StringAlias:
    def __eq__(self, other):
        return other == "alias"

    def __repr__(self):
        return "StringAlias()"


class CallIndexTests(unittest.TestCase):
    def setUp(self):
        previous = logging.root.manager.disable
        logging.disable(logging.WARNING)
        self.addCleanup(logging.disable, previous)

    def equivalent(self, rows, call_ids, *, truncate=False):
        old, new = [make(module, rows, truncate) for module in [OLD, NEW]]
        for call_id in call_ids:
            for method in ["get_pre_func_call_record", "get_post_func_call_record"]:
                with self.subTest(call_id=repr(call_id), method=method):
                    expected = outcome(old, method, call_id)
                    actual = outcome(new, method, call_id)
                    self.assertEqual(expected, actual)
        return old, new

    def test_exact_pre_post_records_and_row_order(self):
        rows = pair("z", 10) + pair("a", 0, pid=9, tid=8)
        _, new = self.equivalent(rows, ["a", "z", "missing"])
        self.assertEqual(new.get_pre_func_call_record("z"), rows[0])
        self.assertEqual(new.get_post_func_call_record("a"), rows[3])
        self.assertIs(type(new.get_pre_func_call_record("z")["time"]), int)

    def test_exception_post_record(self):
        rows = [record("call", 0, PRE), record("call", 1, EXCEPTION)]
        old, new = self.equivalent(rows, ["call"])
        self.assertEqual(
            old.query_func_call_event("call").__dict__,
            new.query_func_call_event("call").__dict__,
        )

    def test_duplicate_pre_count_assertions(self):
        rows = [record("bad", 0, PRE), record("bad", 1, PRE), record("bad", 2, POST)]
        old, new = self.equivalent(rows, ["bad"])
        self.assertEqual(
            outcome(new, "get_pre_func_call_record", "bad"),
            ("AssertionError", "2 pre call events found for bad, expected 1"),
        )

    def test_duplicate_post_and_exception_count_assertions(self):
        rows = [
            record("bad", 0, PRE),
            record("bad", 1, POST),
            record("bad", 2, EXCEPTION),
        ]
        self.equivalent(rows, ["bad"])

    def test_unqueried_duplicates_do_not_raise_early(self):
        rows = pair("good") + [record("bad", 4, PRE), record("bad", 5, PRE)]
        _, new = self.equivalent(rows, ["good"])
        self.assertIsNotNone(new._func_call_positions)
        self.assertEqual(len(new._func_call_positions["bad"][0]), 2)

    def test_missing_outermost_and_nonoutermost_posts_per_thread(self):
        rows = [
            record("outer", 100, PRE),
            record("inner", 0, PRE),
            record("other_thread", 2, PRE, tid=3),
        ]
        old, new = self.equivalent(rows, ["outer", "inner", "other_thread", "absent"])
        self.assertIsNone(new.get_post_func_call_record("outer"))
        self.assertIsNone(new.get_post_func_call_record("other_thread"))
        self.assertEqual(
            outcome(new, "get_post_func_call_record", "inner"),
            (
                "ValueError",
                "No post call event found for inner, but it is not the outermost function call.",
            ),
        )

    def test_duplicate_dataframe_labels_use_positions(self):
        rows = pair("a") + pair("b", 4)
        self.equivalent(pd.DataFrame(rows, index=[17, 17, 17, 17]), ["a", "b"])

    def test_null_ids_never_use_dictionary_equality(self):
        rows = pair("good")
        for value in [None, float("nan"), pd.NA]:
            rows += pair(value, 5)
        _, new = self.equivalent(rows, ["good", None, float("nan"), pd.NA])
        self.assertIsNotNone(new._func_call_positions)

    def test_numeric_ids_and_custom_equality_fall_back(self):
        for rows, ids in [
            (pair(7), [7, "7"]),
            (pair(StringAlias()), ["alias"]),
            (pair(np.str_("numpy")), ["numpy"]),
            (pair("good") + pair(7, 5), ["good", 7]),
        ]:
            old, new = self.equivalent(rows, ids)
            self.assertIsNone(new._func_call_positions)

    def test_pandas_string_and_unsupported_categorical_dtypes(self):
        for dtype in ["string", "category"]:
            frame = pd.DataFrame(pair("a") + pair("b", 4))
            for column in ["func_call_id", "type"]:
                frame[column] = frame[column].astype(dtype)
            _, new = self.equivalent(frame, ["a", "b"])
            if dtype == "category":
                self.assertIsNone(new._func_call_positions)
            else:
                self.assertIsNotNone(new._func_call_positions)

    def test_null_event_types_are_ignored(self):
        rows = pair("good")
        rows += [record("extra", 3, value) for value in [None, float("nan"), pd.NA]]
        self.equivalent(rows, ["good", "extra"])

    def test_stage_split_indexes_are_independent(self):
        rows = pair("init", stage="init") + pair("train", 5, stage="train")
        rows += [
            record("cross", 10, PRE, stage="init"),
            record("cross", 11, POST, stage="train"),
        ]
        old, new = self.equivalent(rows, ["init", "train", "cross"])
        old_stages = old.get_traces_for_stage()
        new_stages = new.get_traces_for_stage()
        for stage in old_stages:
            self.assertIs(type(old_stages[stage]), OLD.TracePandas)
            self.assertIs(type(new_stages[stage]), NEW.TracePandas)
            for call_id in ["init", "train", "cross"]:
                for method in ["get_pre_func_call_record", "get_post_func_call_record"]:
                    self.assertEqual(
                        outcome(old_stages[stage], method, call_id),
                        outcome(new_stages[stage], method, call_id),
                    )
        self.assertIsNot(
            new._func_call_positions, new_stages["train"]._func_call_positions
        )
        self.assertIsNot(
            new_stages["init"]._func_call_positions,
            new_stages["train"]._func_call_positions,
        )

    def test_top_level_return_dicts_are_fresh(self):
        for module in [OLD, NEW]:
            trace = make(module, pair())
            for method in ["get_pre_func_call_record", "get_post_func_call_record"]:
                first = getattr(trace, method)("call")
                second = getattr(trace, method)("call")
                self.assertIsNot(first, second)
                first["function"] = "mutated"
                first["new_key"] = True
                self.assertEqual(getattr(trace, method)("call"), second)

    def test_index_is_lazy_reused_and_rebuilt_on_frame_replacement(self):
        old, new = [make(module, pair()) for module in [OLD, NEW]]
        self.assertIsNone(new._func_call_positions)
        new.get_pre_func_call_record("call")
        cached = new._func_call_positions
        new.get_post_func_call_record("call")
        self.assertIs(new._func_call_positions, cached)
        for trace in [old, new]:
            trace.events = pd.DataFrame(pair("replacement", 9))
        for call_id in ["call", "replacement"]:
            for method in ["get_pre_func_call_record", "get_post_func_call_record"]:
                self.assertEqual(
                    outcome(old, method, call_id), outcome(new, method, call_id)
                )
        self.assertIsNot(new._func_call_positions, cached)

    def test_in_place_row_count_change_rebuilds(self):
        old, new = [make(module, pair()) for module in [OLD, NEW]]
        new.get_pre_func_call_record("call")
        cached = new._func_call_positions
        for trace in [old, new]:
            trace.events.loc[len(trace.events)] = record("call", 4, PRE)
        self.assertEqual(
            outcome(old, "get_pre_func_call_record", "call"),
            outcome(new, "get_pre_func_call_record", "call"),
        )
        self.assertIsNot(new._func_call_positions, cached)

    def test_index_built_after_incomplete_trailing_call_truncation(self):
        rows = [
            record("outer", 0, PRE),
            record("inner", 1, PRE),
            record("inner", 2, POST),
            record("outer", 10, POST),
            record("trailing", 11, PRE),
        ]
        old, new = self.equivalent(rows, ["outer", "inner", "trailing"], truncate=True)
        self.assertEqual(len(new.events), 4)
        self.assertIs(new._func_call_positions_source, new.events)


if __name__ == "__main__":
    unittest.main(verbosity=2)
