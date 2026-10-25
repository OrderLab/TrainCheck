"""Failure contracts for recorder state and buffered writes."""

import threading
import unittest
from queue import Queue
from unittest.mock import patch

from traincheck.instrumentor import dumper, tracer


class TracingContractTests(unittest.TestCase):
    def test_application_scope_restores_suppression_after_exception(self) -> None:
        error = ValueError("application failure")
        with tracer.suppress_tracing():
            with self.assertRaises(ValueError) as caught:
                with tracer.application_tracing():
                    self.assertFalse(tracer.tracing_suppressed())
                    raise error
            self.assertIs(caught.exception, error)
            self.assertTrue(tracer.tracing_suppressed())
        self.assertFalse(tracer.tracing_suppressed())

    def test_new_thread_initializes_independent_state(self) -> None:
        observations: list[bool] = []

        def observe() -> None:
            observations.append(tracer.tracing_suppressed())
            with tracer.suppress_tracing():
                observations.append(tracer.tracing_suppressed())
            observations.append(tracer.tracing_suppressed())

        with tracer.suppress_tracing():
            thread = threading.Thread(target=observe)
            thread.start()
            thread.join(5)
            self.assertFalse(thread.is_alive())
            self.assertTrue(tracer.tracing_suppressed())
        self.assertEqual(observations, [False, True, False])
        self.assertFalse(tracer.tracing_suppressed())

    def test_failed_enqueue_retains_batch_and_propagates(self) -> None:
        queue: Queue[str] = Queue()
        with patch.object(threading.Thread, "start"):
            buffer = dumper.TraceBuffer(lambda: queue, buffer_size=100)
        buffer.add_trace({"id": 1})
        error = RuntimeError("queue unavailable")
        with patch.object(queue, "put", side_effect=error):
            with self.assertRaises(RuntimeError) as caught:
                buffer._flush()
        self.assertIs(caught.exception, error)
        buffer._flush()
        self.assertEqual(queue.get_nowait(), '{"id":1}')
        self.assertTrue(queue.empty())
        buffer._flush()
        self.assertTrue(queue.empty())

    def test_serialization_failure_preserves_existing_batch(self) -> None:
        queue: Queue[str] = Queue()
        with patch.object(threading.Thread, "start"):
            buffer = dumper.TraceBuffer(lambda: queue, buffer_size=100)
        buffer.add_trace({"id": 1})
        error = TypeError("unsupported record")
        with patch.object(dumper, "serialize", side_effect=error):
            with self.assertRaises(TypeError) as caught:
                buffer.add_trace({"id": object()})
        self.assertIs(caught.exception, error)
        buffer._flush()
        self.assertEqual(queue.get_nowait(), '{"id":1}')
        self.assertTrue(queue.empty())


if __name__ == "__main__":
    unittest.main()
