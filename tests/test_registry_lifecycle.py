"""Registry sampling and model-finalization contracts."""

import gc
import os
import subprocess
import sys
import unittest
import weakref
from pathlib import Path
from unittest.mock import patch

from torch import nn

import traincheck.config.config as config
from traincheck.instrumentor.proxy_wrapper import subclass
from traincheck.instrumentor.proxy_wrapper.proxy_registry import VarRegistry
from traincheck.instrumentor.proxy_wrapper.subclass import (
    ProxyParameter,
    proxy_parameter,
)


class RecordingVariable:
    def __init__(
        self, name: str, calls: list[tuple[str, str]], error: Exception | None = None
    ) -> None:
        self.name = name
        self.calls = calls
        self.error = error

    def dump_trace(self, phase: str, dump_loc: str | None) -> None:
        self.calls.append((self.name, phase))
        if self.error is not None:
            raise self.error


class RegistryLifecycleTests(unittest.TestCase):
    def test_cyclic_model_is_retired_after_collection(self) -> None:
        registry = VarRegistry()
        with (
            patch.object(subclass, "get_global_registry", return_value=registry),
            patch.object(config, "DISABLE_WRAPPER", True),
        ):
            model = nn.Linear(2, 2)
            setattr(model, "saved_forward", model.forward)
            proxy_parameter(model, parent_name="model", should_dump_trace=False)
            reference = weakref.ref(model)
            parameter = model.weight
            del model
            gc.collect()
            self.assertIsNone(reference())
            self.assertEqual(registry.registry, {})
            assert isinstance(parameter, ProxyParameter)
            parameter.register_object()
            self.assertEqual(registry.registry, {})

    def test_gc_during_both_sampling_modes_finishes_and_preserves_shared_owner(
        self,
    ) -> None:
        probe = Path(__file__).parent / "fixtures" / "registry_gc_probe.py"
        root = Path(__file__).resolve().parents[1]
        for mode in ("sample", "modified"):
            with self.subTest(mode=mode):
                result = subprocess.run(
                    [sys.executable, str(probe), mode],
                    cwd=root,
                    text=True,
                    capture_output=True,
                    timeout=20,
                    check=False,
                    env={**os.environ, "PYTHONPATH": str(root)},
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn(f"OK {mode}", result.stdout)

    def test_nested_sampling_and_reentrant_updates_preserve_live_stale_status(
        self,
    ) -> None:
        probe = Path(__file__).parent / "fixtures" / "registry_gc_probe.py"
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, str(probe), "reentrant"],
            cwd=root,
            text=True,
            capture_output=True,
            timeout=20,
            check=False,
            env={**os.environ, "PYTHONPATH": str(root)},
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("OK reentrant", result.stdout)

    def test_missing_configuration_remains_an_error_without_stale_changes(self) -> None:
        registry = VarRegistry()
        calls: list[tuple[str, str]] = []
        registry.add_var(RecordingVariable("variable", calls), "variable", "kind")
        with self.assertRaises(AttributeError):
            registry.dump_sample()
        with self.assertRaises(AttributeError):
            registry.dump_modified()
        self.assertEqual(calls, [])
        self.assertFalse(registry.registry["variable"].stale)

    def test_sampling_preserves_order_filter_and_stale_flags(self) -> None:
        registry = VarRegistry()
        calls: list[tuple[str, str]] = []
        first = RecordingVariable("first", calls)
        second = RecordingVariable("second", calls)
        filtered = RecordingVariable("filtered", calls)
        registry.add_var(first, "first", "selected")
        registry.add_var(second, "second", "selected")
        registry.add_var(filtered, "filtered", "other")
        registry.dump_sample(dump_config={"selected": True})
        self.assertEqual(calls, [("first", "sample"), ("second", "sample")])
        self.assertTrue(registry.registry["first"].stale)
        self.assertTrue(registry.registry["second"].stale)
        self.assertFalse(registry.registry["filtered"].stale)
        registry.dump_modified(dump_config={"selected": True})
        self.assertEqual(len(calls), 2)
        registry.add_var(first, "first", "selected")
        registry.dump_modified(dump_config={"selected": True})
        self.assertEqual(calls[-1], ("first", "selective-sample"))
        self.assertEqual(len(calls), 3)

    def test_sampling_propagates_error_without_marking_later_entries_stale(
        self,
    ) -> None:
        for mode in ("sample", "modified"):
            with self.subTest(mode=mode):
                registry = VarRegistry()
                calls: list[tuple[str, str]] = []
                error = RuntimeError("dump failed")
                registry.add_var(
                    RecordingVariable("first", calls, error), "first", "kind"
                )
                registry.add_var(RecordingVariable("later", calls), "later", "kind")
                with self.assertRaises(RuntimeError) as caught:
                    if mode == "sample":
                        registry.dump_sample(dump_config={"kind": True})
                    else:
                        registry.dump_modified(dump_config={"kind": True})
                self.assertIs(caught.exception, error)
                self.assertTrue(registry.registry["first"].stale)
                self.assertFalse(registry.registry["later"].stale)
                self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
