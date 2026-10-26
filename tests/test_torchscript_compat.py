"""Builtin-registration contracts and fresh-process bias-GELU regressions."""

import argparse
import copy
import gc
import json
import os
import subprocess
import sys
import tempfile
import unittest
import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from traincheck.instrumentor import torchscript_compat
from traincheck.instrumentor.torchscript_compat import (
    _REGISTERED_WRAPPERS,
    preserve_torchscript_builtin,
)


def bias_gelu(bias: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """Real-source-file fixture for the tanh-based fused bias-GELU expression."""
    x = bias + value
    return (
        x * 0.5 * (1.0 + torch.tanh(0.7978845608028654 * x * (1.0 + 0.044715 * x * x)))
    )


def evaluate(function):
    bias = torch.linspace(-0.7, 0.9, 8, dtype=torch.float64, requires_grad=True)
    value = (
        torch.linspace(-2.3, 2.1, 24, dtype=torch.float64)
        .reshape(3, 8)
        .requires_grad_()
    )
    output = function(bias, value)
    grad_bias, grad_value = torch.autograd.grad(output.square().sum(), (bias, value))
    return {
        "output": output.detach().tolist(),
        "grad_bias": grad_bias.tolist(),
        "grad_value": grad_value.tolist(),
    }


def probe(mode: str) -> dict[str, object]:
    """Run one control in its own interpreter so JIT caches cannot leak."""
    from traincheck.instrumentor import tracer

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    records: list[dict[str, object]] = []
    registry = torch.jit._builtins
    original = torch.tanh
    result: dict[str, object] = {
        "mode": mode,
        "torch": torch.__version__,
        "initial_registry_initialized": registry._builtin_table is not None,
    }
    if mode.endswith("warm"):
        result["original_operator_before_wrapping"] = registry._find_builtin(original)

    with (
        patch.object(tracer.config, "DISABLE_WRAPPER", False),
        patch.object(
            tracer,
            "dump_trace_API",
            side_effect=lambda record: records.append(copy.deepcopy(record)),
        ),
    ):
        if mode != "baseline":
            if mode == "integrated-fixed-warm":
                wrapped = tracer.wrapper(
                    original,
                    False,
                    False,
                    False,
                    dump_args=True,
                    dump_ret=True,
                    handle_proxy=False,
                )
            else:
                # Disable only registration to reproduce the former wrapper;
                # the helper controls then apply the production helper directly.
                with patch.object(
                    tracer,
                    "preserve_torchscript_builtin",
                    return_value={"status": "not-registered", "operator": None},
                ):
                    wrapped = tracer.wrapper(
                        original,
                        False,
                        False,
                        False,
                        dump_args=True,
                        dump_ret=True,
                        handle_proxy=False,
                    )
                if mode.startswith("fixed"):
                    result["candidate"] = preserve_torchscript_builtin(
                        original, wrapped, torch
                    )
            torch.tanh = wrapped
        try:
            result["eager"] = evaluate(bias_gelu)
            result["eager_records"] = copy.deepcopy(records)
            records.clear()
            scripted = torch.jit.script(bias_gelu)
            result["graph_has_aten_tanh"] = "aten::tanh" in str(scripted.graph)
            result["scripted"] = evaluate(scripted)
            result["scripted_records"] = copy.deepcopy(records)
            result["script_compilation"] = "passed"
        except Exception as error:
            result["script_compilation"] = "failed"
            result["exception_type"] = type(error).__name__
            result["exception"] = str(error)
        finally:
            result["current_operator_after_compilation"] = registry._find_builtin(
                torch.tanh
            )
            torch.tanh = original
    return result


class BuiltinCompatibilityTests(unittest.TestCase):
    def fake_torch(self, operator):
        table, events = {}, []

        def find(function):
            events.append(("find", function))
            return operator

        def register(function, mapped_operator):
            events.append(("register", function, mapped_operator))
            table[id(function)] = mapped_operator

        builtins = SimpleNamespace(_find_builtin=find, _register_builtin=register)
        return SimpleNamespace(jit=SimpleNamespace(_builtins=builtins)), table, events

    def test_copies_exact_original_mapping(self):
        fake, table, events = self.fake_torch("aten::tanh")

        def original():
            pass

        def wrapped():
            pass

        result = preserve_torchscript_builtin(original, wrapped, fake)
        self.assertEqual(result, {"status": "preserved", "operator": "aten::tanh"})
        self.assertEqual(
            events, [("find", original), ("register", wrapped, "aten::tanh")]
        )
        self.assertEqual(table[id(wrapped)], "aten::tanh")

    def test_unregistered_function_is_unchanged(self):
        fake, table, events = self.fake_torch(None)
        original = lambda: None
        wrapped = lambda: None
        self.assertEqual(
            preserve_torchscript_builtin(original, wrapped, fake)["status"],
            "not-registered",
        )
        self.assertEqual(table, {})
        self.assertEqual(events, [("find", original)])
        self.assertNotIn(wrapped, _REGISTERED_WRAPPERS)

    def test_unavailable_private_api_is_noop(self):
        original = lambda: None
        wrapped = lambda: None
        missing_register = SimpleNamespace(
            jit=SimpleNamespace(
                _builtins=SimpleNamespace(_find_builtin=lambda f: "aten::tanh")
            )
        )
        for fake in [SimpleNamespace(), missing_register]:
            self.assertEqual(
                preserve_torchscript_builtin(original, wrapped, fake)["status"],
                "unsupported-private-api",
            )
        self.assertNotIn(wrapped, _REGISTERED_WRAPPERS)

    def test_registered_wrapper_cannot_leave_stale_reusable_id(self):
        fake, table, events = self.fake_torch("aten::tanh")
        original = lambda: None
        wrapped = lambda: None
        weak = weakref.ref(wrapped)
        address = id(wrapped)
        preserve_torchscript_builtin(original, wrapped, fake)
        events.clear()
        del wrapped
        gc.collect()
        self.assertIsNotNone(weak())
        self.assertEqual(table[address], "aten::tanh")
        self.assertEqual(_REGISTERED_WRAPPERS[weak()], "aten::tanh")


class TorchScriptWrapperTests(unittest.TestCase):
    def test_fresh_process_bias_gelu_semantics_and_eager_traces(self):
        registry = getattr(getattr(torch, "jit", None), "_builtins", None)
        if not all(
            callable(getattr(registry, name, None))
            for name in ["_find_builtin", "_register_builtin"]
        ):
            self.skipTest("PyTorch lacks the private builtin-registration APIs")

        source = Path(torchscript_compat.__file__).resolve().parents[2]
        environment = os.environ | {
            "PYTHONPATH": str(source),
            "PYTHONDONTWRITEBYTECODE": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "CUDA_VISIBLE_DEVICES": "",
        }
        modes = [
            "baseline",
            "buggy-unprimed",
            "buggy-warm",
            "fixed-unprimed",
            "fixed-warm",
            "integrated-fixed-warm",
        ]
        results = {}
        with tempfile.TemporaryDirectory(prefix="traincheck-torchscript-") as temporary:
            for mode in modes:
                output = Path(temporary) / (mode + ".json")
                completed = subprocess.run(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--probe",
                        mode,
                        "--output",
                        str(output),
                    ],
                    cwd=source,
                    env=environment,
                    capture_output=True,
                    text=True,
                    timeout=90,
                )
                self.assertEqual(
                    completed.returncode, 0, completed.stdout + completed.stderr
                )
                results[mode] = json.loads(output.read_text())

        baseline = results["baseline"]
        checks = 0
        for mode in modes[:-1]:
            result = results[mode]
            with self.subTest(mode=mode):
                self.assertEqual(result["eager"], baseline["eager"])
                checks += 1
                if mode.startswith("buggy"):
                    self.assertEqual(result["script_compilation"], "failed")
                    self.assertEqual(result["exception_type"], "TypeError")
                    self.assertIn("builtin_function_or_method", result["exception"])
                    self.assertIsNone(result["current_operator_after_compilation"])
                else:
                    self.assertEqual(result["script_compilation"], "passed")
                    self.assertTrue(result["graph_has_aten_tanh"])
                    self.assertEqual(result["scripted"], baseline["scripted"])
                    self.assertEqual(
                        result["current_operator_after_compilation"], "aten::tanh"
                    )
                    self.assertEqual(result["scripted_records"], [])
                    checks += 1
                if mode != "baseline":
                    self.assert_paired_eager_record(result)
                    checks += 1
        integrated = results["integrated-fixed-warm"]
        self.assertEqual(integrated["script_compilation"], "passed")
        self.assertTrue(integrated["graph_has_aten_tanh"])
        self.assertEqual(integrated["scripted"], baseline["scripted"])
        self.assertEqual(integrated["eager"], baseline["eager"])
        self.assertEqual(integrated["scripted_records"], [])
        self.assertEqual(integrated["current_operator_after_compilation"], "aten::tanh")
        self.assert_paired_eager_record(integrated)
        checks += 1
        self.assertEqual(checks, 13)

    def assert_paired_eager_record(self, result):
        records = result["eager_records"]
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["type"], "function_call (pre)")
        self.assertEqual(records[1]["type"], "function_call (post)")
        self.assertEqual(records[0]["func_call_id"], records[1]["func_call_id"])
        self.assertEqual(records[0]["function"], records[1]["function"])
        self.assertIn("tanh", records[0]["function"])
        self.assertTrue(records[0]["args"])
        self.assertTrue(records[1]["return_values"])


if __name__ == "__main__":
    if "--probe" in sys.argv:
        parser = argparse.ArgumentParser()
        parser.add_argument("--probe", required=True)
        parser.add_argument("--output", required=True, type=Path)
        arguments = parser.parse_args()
        arguments.output.write_text(json.dumps(probe(arguments.probe)) + "\n")
    else:
        unittest.main()
