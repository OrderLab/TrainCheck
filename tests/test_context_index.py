"""Differential context-index tests against frozen legacy query methods."""

import inspect
import logging
import random
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import traincheck.trace.trace_pandas as NEW
from tests.fixtures import legacy_trace_pandas as OLD

PRE = OLD.TraceLineType.FUNC_CALL_PRE
POST = OLD.TraceLineType.FUNC_CALL_POST


def event(
    time,
    method,
    *,
    name="fixtures.Ctx",
    obj=1,
    pid=10,
    tid=20,
    kind=None,
    args=float("nan"),
    kwargs=None,
    stage="train",
):
    return {
        "time": time,
        "function": f"{name}.{method}",
        "type": kind or (POST if method == "__enter__" else PRE),
        "obj_id": obj,
        "process_id": pid,
        "thread_id": tid,
        "args": args,
        "kwargs": kwargs or {},
        "meta_vars.stage": stage,
    }


def cycle(start, **kwargs):
    return [
        event(start, "__init__", **kwargs),
        event(start + 1, "__enter__", **kwargs),
        event(start + 2, "__exit__", **kwargs),
    ]


def run(module, frame):
    trace = module.TracePandas(
        frame.copy(deep=True), truncate_incomplete_func_calls=False
    )
    return trace, encode(trace)


def encode(trace):
    states = trace.context_manager_states
    if states is None:
        return None
    return [
        (tuple(key), [state.to_dict() for state in value])
        for key, value in states.items()
    ]


class ContextIndexTests(unittest.TestCase):
    def setUp(self):
        previous = logging.root.manager.disable
        logging.disable(logging.WARNING)
        self.addCleanup(logging.disable, previous)

    def equivalent(self, rows, **frame_options):
        frame = (
            rows
            if isinstance(rows, pd.DataFrame)
            else pd.DataFrame(rows, **frame_options)
        )
        original, expected = run(OLD, frame)
        optimized, actual = run(NEW, frame)
        self.assertEqual(expected, actual)
        # Native scalar types remain suitable for the same JSON serialization.
        for _, states in actual or []:
            for state in states:
                self.assertNotIsInstance(state["start_time"], np.generic)
                self.assertNotIsInstance(state["end_time"], np.generic)
        return original, optimized

    def test_nested_reused_ids_processes_threads_and_class_order(self):
        rows = cycle(0, name="fixtures.B", obj=1)
        rows += cycle(5, name="fixtures.A", obj=1)
        rows += cycle(10, name="fixtures.B", obj=1)
        rows += cycle(1, name="fixtures.B", obj=2)
        rows += cycle(0, name="fixtures.B", obj=1, pid=11)
        rows += cycle(0, name="fixtures.B", obj=1, tid=21)
        old, new = self.equivalent(rows)
        self.assertEqual(
            [s.name for s in new.context_manager_states[OLD.PTID(10, 20)]],
            ["fixtures.B", "fixtures.B", "fixtures.B", "fixtures.A"],
        )
        for when in [0, 1, 2, 6, 11, 12, 100]:
            self.assertEqual(
                [s.to_dict() for s in old.query_active_context_managers(when, 10, 20)],
                [s.to_dict() for s in new.query_active_context_managers(when, 10, 20)],
            )

    def test_first_row_not_nearest_timestamp(self):
        rows = [
            event(0, "__init__"),
            event(8, "__enter__"),
            event(2, "__enter__"),
            event(15, "__exit__"),
            event(9, "__exit__"),
        ]
        _, new = self.equivalent(rows)
        state = next(iter(new.context_manager_states.values()))[0]
        self.assertEqual((state.liveness.start_time, state.liveness.end_time), (8, 15))

    def test_equal_timestamps_strictly_later(self):
        rows = [
            event(5, "__init__"),
            event(5, "__enter__"),
            event(6, "__enter__"),
            event(6, "__exit__"),
            event(7, "__exit__"),
        ]
        _, new = self.equivalent(rows)
        self.assertEqual(
            next(iter(new.context_manager_states.values()))[0].liveness.start_time, 6
        )

    def test_null_keys_do_not_match_each_other(self):
        for field in ["obj_id", "process_id", "thread_id"]:
            for missing in [None, float("nan"), pd.NA]:
                with self.subTest(field=field, missing=missing):
                    rows = cycle(0)
                    for row in rows:
                        row[field] = missing
                    rows += cycle(5, obj=2)
                    self.equivalent(rows)

    def test_md_none_key_matches_original_custom_equality(self):
        from traincheck.trace.types import MD_NONE

        for field in ["obj_id", "process_id", "thread_id"]:
            rows = cycle(0)
            for row in rows:
                row[field] = MD_NONE()
            self.equivalent(rows)

    def test_mixed_md_none_and_null_keys(self):
        from traincheck.trace.types import MD_NONE

        rows = [
            event(0, "__init__", obj=None),
            event(1, "__enter__", obj=MD_NONE()),
            event(2, "__exit__", obj=MD_NONE()),
        ]
        self.equivalent(rows)

    def test_nullable_numeric_keys(self):
        frame = pd.DataFrame(cycle(0) + cycle(5, obj=None) + cycle(10, obj=2))
        for col in ["obj_id", "process_id", "thread_id"]:
            frame[col] = frame[col].astype("Int64")
        self.equivalent(frame)

    def test_missing_pairs_and_wrong_event_types(self):
        rows = [
            event(0, "__init__"),
            event(1, "__enter__", kind=PRE),
            event(2, "__exit__"),
        ]
        rows += [event(4, "__init__", obj=2), event(5, "__enter__", obj=2)]
        rows += [event(6, "__enter__", obj=3), event(7, "__exit__", obj=3)]
        rows += cycle(10, obj=4)
        self.equivalent(rows)

    def test_infinite_missing_and_nullable_timestamps(self):
        rows = [
            event(float("nan"), "__init__", obj=2),
            event(0, "__init__"),
            event(float("nan"), "__enter__"),
            event(1, "__enter__"),
            event(float("inf"), "__exit__"),
        ]
        self.equivalent(rows)
        frame = pd.DataFrame(rows)
        frame["time"] = frame["time"].astype("Float64")
        for module in [OLD, NEW]:
            with self.assertRaises(TypeError):
                run(module, frame)

    def test_large_integer_timestamps_preserve_precision(self):
        for dtype in ["int64", "uint64", "Int64", "UInt64", "float64", "object"]:
            for start in [2**53, 2**60, 2**63 - 4]:
                with self.subTest(dtype=dtype, start=start):
                    frame = pd.DataFrame(cycle(start))
                    frame["time"] = frame["time"].astype(dtype)
                    self.equivalent(frame)
        for start in [2**53, 2**60, 2**63 - 8]:
            # Object dtype retains a float init query against exact integers.
            frame = pd.DataFrame(cycle(start))
            frame["time"] = pd.Series(
                [float(start), start + 1, start + 2], dtype=object
            )
            self.equivalent(frame)

    def test_duplicate_dataframe_indexes_and_mixed_key_types(self):
        rows = cycle(0, obj="1") + cycle(5, obj=1) + cycle(10, obj=True)
        self.equivalent(pd.DataFrame(rows, index=[4] * len(rows)))

    def test_no_context_or_no_function(self):
        self.equivalent(pd.DataFrame({"time": [0]}))
        self.equivalent([event(0, "unrelated")])
        self.equivalent([event(0, "__enter__")])

    def test_ignore_autograd_contexts(self):
        rows = cycle(0, name="torch.autograd.grad_mode.enable_grad")
        rows += cycle(3, name="torch.autograd.profiler.record_function")
        rows += cycle(6)
        _, new = self.equivalent(rows)
        self.assertEqual(len(next(iter(new.context_manager_states.values()))), 1)

    def test_stage_split_rebuild_and_cache(self):
        rows = cycle(0, stage="init") + cycle(3, stage="train")
        rows += [
            event(10, "__init__", obj=2, stage="init"),
            event(11, "__enter__", obj=2, stage="train"),
            event(12, "__exit__", obj=2, stage="train"),
        ]
        old, new = self.equivalent(rows)
        expected = old.get_traces_for_stage()
        actual = new.get_traces_for_stage()
        self.assertEqual(list(expected), list(actual))
        for stage in expected:
            self.assertIs(type(expected[stage]), OLD.TracePandas)
            self.assertIs(type(actual[stage]), NEW.TracePandas)
            self.assertEqual(encode(expected[stage]), encode(actual[stage]))
        self.assertIs(new.get_traces_for_stage(), actual)
        cached = new.context_manager_states
        self.assertIs(new._index_context_manager_meta_vars(), cached)

    def test_bindings_and_signature_lookup_remain_per_init(self):
        rows = cycle(0, args={"0": {"object": {}}, "1": {"int": 7}})
        rows += cycle(5, args={"0": {"object": {}}, "1": {"int": 9}})
        signatures = [
            inspect.Signature(
                [
                    inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD),
                    inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD),
                ]
            )
            for name in ["first", "second"]
        ]
        outputs = []
        for module in [OLD, NEW]:
            with patch.object(
                module, "load_signature_from_class_method_name", side_effect=signatures
            ) as loader:
                outputs.append(run(module, pd.DataFrame(rows))[1])
                self.assertEqual(loader.call_count, 2)
        self.assertEqual(*outputs)

    def test_randomized_row_order_and_reused_identity_oracle(self):
        for seed in range(80):
            rng = random.Random(seed)
            rows = []
            for _ in range(20):
                rows.extend(
                    cycle(
                        rng.randrange(20),
                        obj=rng.randrange(3),
                        pid=rng.randrange(2),
                        tid=rng.randrange(2),
                        name=rng.choice(["fixtures.A", "fixtures.B"]),
                    )
                )
            rng.shuffle(rows)
            with self.subTest(seed=seed):
                self.equivalent(rows)


if __name__ == "__main__":
    unittest.main(verbosity=2)
