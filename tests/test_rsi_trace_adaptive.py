"""Behavioral contracts for the narrow, unchanged-source RSI trace window."""

from __future__ import annotations

import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
from types import FrameType, FunctionType

import torch

from tools.rsi_pilot.trace_adaptive import (
    StepWindow,
    locate_window,
    run,
    state_value,
    tensor_evidence,
)

SOURCE = """def helper():
    return 3

def main(optimizer, fail=False):
    completed = 0
    for step in range(1, 3):
        helper()
        optimizer.zero_grad()
        loss = torch.tensor(float(step))
        helper()
        if fail:
            raise ValueError('original failure')
        completed = step
        helper()
    return completed
"""


class FakeOptimizer:
    def __init__(self, flags: list[bool]) -> None:
        self.flags = flags
        self.observed: list[bool] = []

    def zero_grad(self) -> None:
        self.observed.append(self.flags[0])


class TestWindow(unittest.TestCase):
    def test_restricts_window_preserves_return_and_captures_outside(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "train.py"
            source.write_text(SOURCE)
            lines = locate_window(
                source, hashlib.sha256(source.read_bytes()).hexdigest()
            )
            namespace: dict[str, object] = {"torch": torch}
            exec(compile(SOURCE, str(source), "exec"), namespace)
            entry = namespace["main"]
            assert isinstance(entry, FunctionType)
            flags = [False]
            snapshots: list[int] = []

            def enable() -> None:
                flags[0] = True

            def disable() -> None:
                flags[0] = False

            def snapshot(frame: FrameType) -> dict[str, object]:
                self.assertFalse(flags[0])
                step: object = frame.f_locals["step"]
                assert isinstance(step, int)
                snapshots.append(step)
                return {"step": step}

            window = StepWindow(entry.__code__, lines, enable, disable, snapshot)
            optimizer = FakeOptimizer(flags)
            prior = sys.gettrace()
            try:
                sys.settrace(window)
                result: object = entry(optimizer)
            finally:
                sys.settrace(prior)
                window.close()
            self.assertEqual(result, 2)
            self.assertEqual(optimizer.observed, [True, True])
            self.assertEqual(snapshots, [1, 1, 2, 2])
            self.assertEqual([row["loss"] for row in window.records], [1.0, 2.0])
            self.assertFalse(window.failed)
            self.assertFalse(flags[0])
            self.assertIsNone(window(sys._getframe(), "line", None))

    def test_exception_propagates_and_disarms(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "train.py"
            source.write_text(SOURCE)
            lines = locate_window(
                source, hashlib.sha256(source.read_bytes()).hexdigest()
            )
            namespace: dict[str, object] = {"torch": torch}
            exec(compile(SOURCE, str(source), "exec"), namespace)
            entry = namespace["main"]
            assert isinstance(entry, FunctionType)
            flags = [False]

            def enable() -> None:
                flags[0] = True

            def disable() -> None:
                flags[0] = False

            def snapshot(frame: FrameType) -> dict[str, object]:
                return {}

            window = StepWindow(entry.__code__, lines, enable, disable, snapshot)
            prior = sys.gettrace()
            try:
                sys.settrace(window)
                with self.assertRaisesRegex(ValueError, "original failure"):
                    entry(FakeOptimizer(flags), True)
            finally:
                sys.settrace(prior)
                window.close()
            self.assertTrue(window.failed)
            self.assertFalse(flags[0])
            self.assertEqual(window.records, [])

    def test_source_drift_and_duplicate_anchors_fail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "train.py"
            source.write_text(SOURCE)
            with self.assertRaisesRegex(ValueError, "pinned source hash"):
                locate_window(source, "0" * 64)
            duplicate = SOURCE.replace(
                "optimizer.zero_grad()",
                "optimizer.zero_grad()\n        optimizer.zero_grad()",
            )
            source.write_text(duplicate)
            with self.assertRaisesRegex(ValueError, "unique"):
                locate_window(source, hashlib.sha256(source.read_bytes()).hexdigest())

    def test_foreign_hook_is_not_replaced(self) -> None:
        def prior(frame: FrameType, event: str, arg: object) -> None:
            return None

        old = sys.gettrace()
        try:
            sys.settrace(prior)
            with self.assertRaisesRegex(RuntimeError, "existing Python trace hook"):
                run(Path("missing"), Path("unused"), "plain", None, [])
            self.assertIs(sys.gettrace(), prior)
        finally:
            sys.settrace(old)

    def test_adapter_owned_arguments_and_policy_contract(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly for selective"):
            run(Path("missing"), Path("unused"), "selective", None, [])
        with self.assertRaisesRegex(ValueError, "owns upstream"):
            run(Path("missing"), Path("unused"), "plain", None, ["--device=cuda"])

    def test_snapshot_copies_values_and_preserves_container_kinds(self) -> None:
        value = torch.tensor([1.0, 2.0])
        before = tensor_evidence(value)
        copied = state_value({0: (value,), "group": [True, None]})
        value.add_(1)
        self.assertNotEqual(before, tensor_evidence(value))
        self.assertEqual(
            copied,
            {
                "kind": "dict",
                "items": [
                    [
                        0,
                        {
                            "kind": "tuple",
                            "items": [{"kind": "tensor", "value": before}],
                        },
                    ],
                    ["group", {"kind": "list", "items": [True, None]}],
                ],
            },
        )
        with self.assertRaisesRegex(TypeError, "state keys"):
            state_value({(1, 2): 3})
        with self.assertRaisesRegex(TypeError, "Unsupported"):
            state_value(object())


if __name__ == "__main__":
    unittest.main()
