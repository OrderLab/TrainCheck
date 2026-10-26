"""Regression coverage for tracer self-reentry and concurrent buffer flushing."""

import copy
import json
import threading
import unittest
from queue import Queue
from unittest.mock import patch

from traincheck.instrumentor import dumper, tracer


def wrapped(fn, name="outer", scan_proxy=False):
    def call(*args, **kwargs):
        return tracer.function_wrapper(
            fn,
            name,
            False,
            scan_proxy,
            False,
            True,
            None,
            True,
            None,
            False,
            {},
            False,
            *args,
            **kwargs,
        )

    return call


class TracingTests(unittest.TestCase):
    def setUp(self):
        self.records = []
        self.patches = [
            patch.object(tracer.config, "DISABLE_WRAPPER", False),
            patch.object(
                tracer,
                "dump_trace_API",
                side_effect=lambda r: self.records.append(copy.deepcopy(r)),
            ),
        ]
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)

    def test_serializer_reentry_does_not_trace_itself(self):
        helper = wrapped(lambda x: x, "serializer_helper")
        with patch.object(
            tracer, "var_to_serializable", side_effect=lambda x: helper(x)
        ):
            self.assertEqual(wrapped(lambda x: x + 1)(3), 4)
        self.assertEqual([r["function"] for r in self.records], ["outer", "outer"])

    def test_encoding_reentry_and_nested_guard_restoration(self):
        helper = wrapped(lambda: "encoded", "encoder_helper")

        def emit(record):
            helper()
            self.records.append(copy.deepcopy(record))

        with (
            patch.object(tracer, "dump_trace_API", side_effect=emit),
            patch.object(tracer, "var_to_serializable", side_effect=copy.deepcopy),
        ):
            wrapped(lambda x: x)(1)
            with tracer.suppress_tracing():
                tracer.to_dict_return_value(1)
                self.assertTrue(tracer.tracing_suppressed())
            self.assertFalse(tracer.tracing_suppressed())
        self.assertEqual([r["function"] for r in self.records], ["outer", "outer"])

    def test_nested_user_calls_and_mutable_snapshots(self):
        inner = wrapped(lambda x: x.append(2), "inner")

        def outer(x):
            inner(x)
            return x

        with patch.object(tracer, "var_to_serializable", side_effect=copy.deepcopy):
            self.assertEqual(wrapped(outer)([1]), [1, 2])
        self.assertEqual(
            [r["function"] for r in self.records], ["outer", "inner", "inner", "outer"]
        )
        self.assertEqual(self.records[0]["args"][0], [1])
        self.assertEqual(self.records[-1]["return_values"], [1, 2])
        self.assertEqual(
            self.records[0]["func_call_id"], self.records[-1]["func_call_id"]
        )
        self.assertEqual(
            self.records[1]["func_call_id"], self.records[2]["func_call_id"]
        )

    def test_real_tensor_conversion_preserves_pre_post_shape(self):
        import torch

        value = torch.ones(2)

        def resize(tensor):
            tensor.resize_(3)
            return tensor

        self.assertIs(wrapped(resize)(value), value)
        before = next(iter(self.records[0]["args"][0].values()))
        after = next(iter(self.records[-1]["return_values"].values()))
        self.assertEqual(before["shape"], (2,))
        self.assertEqual(after["shape"], (3,))
        self.assertEqual(self.records[0]["meta_vars"], self.records[-1]["meta_vars"])

    def test_generator_application_calls_remain_visible(self):
        inner = wrapped(lambda: 7, "inner")

        def values():
            yield inner()

        with patch.object(tracer, "var_to_serializable", side_effect=lambda x: repr(x)):
            self.assertEqual(wrapped(lambda x: list(x), scan_proxy=True)(values()), [7])
        self.assertEqual(
            [r["function"] for r in self.records], ["inner", "inner", "outer", "outer"]
        )
        self.assertFalse(tracer.tracing_suppressed())

    def test_conversion_exception_restores_guard(self):
        with patch.object(
            tracer, "var_to_serializable", side_effect=ValueError("conversion")
        ):
            with self.assertRaisesRegex(ValueError, "conversion"):
                tracer.to_dict_args_kwargs((1,), {})
        with patch.object(tracer, "var_to_serializable", side_effect=copy.deepcopy):
            wrapped(lambda x: x)(1)
        self.assertEqual(len(self.records), 2)

    def test_frozen_framework_exception_preserved(self):
        from dataclasses import dataclass

        @dataclass(frozen=True, slots=True)
        class FrameworkSignal(Exception):
            reason: str

        error = FrameworkSignal("cache bypass")

        def fail():
            raise error

        try:
            wrapped(fail)()
        except FrameworkSignal as caught:
            self.assertIs(caught, error)
        else:
            self.fail("Original exception was swallowed")
        self.assertFalse(tracer.tracing_suppressed())

    def test_original_exception_preserved(self):
        error = ValueError("original")

        def fail():
            raise error

        with self.assertRaises(ValueError) as caught:
            wrapped(fail)()
        self.assertIs(caught.exception, error)
        self.assertEqual(
            self.records[-1]["type"], tracer.TraceLineType.FUNC_CALL_POST_EXCEPTION
        )
        self.assertEqual(
            self.records[0]["func_call_id"], self.records[-1]["func_call_id"]
        )

    def test_conversion_suppression_is_thread_local(self):
        entered, release = threading.Event(), threading.Event()

        def convert(x):
            if x == "blocked":
                entered.set()
                if not release.wait(5):
                    raise TimeoutError("test release")
            return x

        with patch.object(tracer, "var_to_serializable", side_effect=convert):
            thread = threading.Thread(
                target=tracer.to_dict_args_kwargs, args=(("blocked",), {})
            )
            thread.start()
            try:
                self.assertTrue(entered.wait(5))
                self.assertEqual(wrapped(lambda x: x)(3), 3)
            finally:
                release.set()
                thread.join(5)
        self.assertEqual(len(self.records), 2)


class BufferTests(unittest.TestCase):
    def make_buffer(self, queue):
        # Drive the periodic thread's flush explicitly, so scheduling is controlled.
        with patch.object(threading.Thread, "start"):
            return dumper.TraceBuffer(
                lambda: queue, buffer_size=100, flush_interval=3600
            )

    def test_concurrent_flush_exactly_once(self):
        entered, release, second_done = (
            threading.Event(),
            threading.Event(),
            threading.Event(),
        )

        class BlockingQueue(Queue):
            def put(self, item):
                if not entered.is_set():
                    entered.set()
                    if not release.wait(5):
                        raise TimeoutError("test release")
                super().put(item)

        queue = BlockingQueue()
        buffer = self.make_buffer(queue)
        buffer.add_trace({"id": 1})
        first = threading.Thread(target=buffer._flush)

        def flush_again():
            buffer._flush()
            second_done.set()

        second = threading.Thread(target=flush_again)
        first.start()
        try:
            self.assertTrue(entered.wait(5))
            second.start()
            second_done.wait(0.1)
        finally:
            release.set()
            first.join(5)
            second.join(5)
        rows = []
        while not queue.empty():
            rows.extend(json.loads(row) for row in queue.get_nowait().splitlines())
        self.assertEqual(rows, [{"id": 1}])

    def test_concurrent_producer_and_flush_order_and_snapshot(self):
        queue = Queue()
        buffer = self.make_buffer(queue)
        done = threading.Event()

        def flush():
            while not done.is_set():
                buffer._flush()

        thread = threading.Thread(target=flush)
        thread.start()
        try:
            item = {"id": 0}
            for i in range(2000):
                item["id"] = i
                buffer.add_trace(item)
        finally:
            done.set()
            thread.join(5)
        buffer._flush()
        rows = []
        while not queue.empty():
            rows.extend(
                json.loads(row)["id"] for row in queue.get_nowait().splitlines()
            )
        self.assertEqual(rows, list(range(2000)))


if __name__ == "__main__":
    unittest.main()
