"""Worker settings are required and validated before any process state changes."""

import unittest
from contextlib import ExitStack
from unittest.mock import patch

import traincheck.config.config as config
import traincheck.parallel as parallel
import traincheck.utils as utils
from traincheck.trace.trace import Trace


class WorkerConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch.multiple(config, **parallel.config_snapshot()))
        self.original_traces: list[Trace] = []
        stack.enter_context(
            patch.object(parallel, "_WORKER_TRACES", self.original_traces)
        )
        stack.enter_context(patch.object(utils, "_suppress_inner_progress", False))

    def assert_rejected_unchanged(
        self, snapshot: dict[str, object], error: type[Exception]
    ) -> None:
        before = parallel.config_snapshot()
        with self.assertRaises(error):
            parallel.worker_init([], snapshot)
        self.assertEqual(parallel.config_snapshot(), before)
        self.assertIs(parallel.get_worker_traces(), self.original_traces)
        self.assertFalse(utils._suppress_inner_progress)

    def test_all_fields_are_required(self) -> None:
        for field in parallel.config_snapshot():
            snapshot: dict[str, object] = dict(parallel.config_snapshot())
            del snapshot[field]
            with self.subTest(field=field):
                self.assert_rejected_unchanged(snapshot, KeyError)

    def test_unknown_fields_are_rejected(self) -> None:
        snapshot: dict[str, object] = dict(parallel.config_snapshot())
        snapshot["UNRECOGNIZED_SETTING"] = 1
        self.assert_rejected_unchanged(snapshot, ValueError)

    def test_malformed_fields_leave_all_state_unchanged(self) -> None:
        invalid: dict[str, list[object]] = {
            "ENABLE_PRECOND_SAMPLING": [1, None, "true"],
            "PRECOND_SAMPLING_THRESHOLD": [True, 1.5, "100"],
            "MAX_EXAMPLE_LIST_SIZE": [True, -1, 2.5, "10"],
            "EXAMPLE_SAMPLING_SEED": [object(), {}, []],
        }
        for field, values in invalid.items():
            for value in values:
                snapshot: dict[str, object] = dict(parallel.config_snapshot())
                snapshot[field] = value
                with self.subTest(field=field, value=value):
                    self.assert_rejected_unchanged(snapshot, ValueError)

    def test_supported_seeds_and_uncapped_or_zero_capacity(self) -> None:
        for seed in (None, 9, -8, 1.5, "seed", b"seed", bytearray(b"seed")):
            for capacity in (None, 0, 7):
                snapshot: dict[str, object] = {
                    "ENABLE_PRECOND_SAMPLING": False,
                    "PRECOND_SAMPLING_THRESHOLD": 0,
                    "MAX_EXAMPLE_LIST_SIZE": capacity,
                    "EXAMPLE_SAMPLING_SEED": seed,
                }
                traces: list[Trace] = []
                parallel.worker_init(traces, snapshot)
                self.assertEqual(parallel.config_snapshot(), snapshot)
                self.assertIs(parallel.get_worker_traces(), traces)
                self.assertTrue(utils._suppress_inner_progress)


if __name__ == "__main__":
    unittest.main()
