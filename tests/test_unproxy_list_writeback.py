"""Regression tests for issue #176: list arguments under unproxying.

``torch.optim.SGD.step`` builds ``momentum_buffer_list``, passes it to the
functional ``sgd``, which stores each newly created buffer into that list, and
then copies the buffers into ``optimizer.state``. Unproxying copied the list, so
the writes went into the copy and momentum was silently reset every step.
Unproxying in place instead also replaced the proxies in the caller's ``params``
list, so ``optimizer.state`` became keyed by unproxied tensors.

The contract tested here: unproxying copies every list, and only list slots of
registered APIs that the callee replaced are written back to the caller.
"""

import importlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from collections.abc import Callable
from pathlib import Path
from unittest.mock import patch

import torch

from traincheck.instrumentor import tracer
from traincheck.instrumentor.proxy_wrapper import list_writeback
from traincheck.instrumentor.proxy_wrapper.list_writeback import ListWritebackError
from traincheck.instrumentor.proxy_wrapper.proxy_basics import (
    is_proxied,
    unproxy_arg,
    unproxy_func,
)
from traincheck.utils import typename

ROOT = Path(__file__).resolve().parents[1]
PROBE = Path(__file__).parent / "fixtures" / "sgd_momentum_probe.py"


class MarkerProxy:
    """Carries the attributes ``is_proxied`` and ``unproxy_arg`` rely on."""

    def __init__(self, obj: object) -> None:
        self.__dict__["is_traincheck_proxied_obj"] = True
        self.__dict__["_obj"] = obj


def assert_no_proxies(*values: object) -> None:
    for value in values:
        if type(value) in (list, tuple):
            assert_no_proxies(*value)  # type: ignore[arg-type]
        elif is_proxied(value):
            raise AssertionError(f"callee received a proxy: {value!r}")


def fill_missing(params: list[object], buffers: list[object]) -> int:
    assert_no_proxies(params, buffers)
    for index, buffer in enumerate(buffers):
        if buffer is None:
            buffers[index] = torch.full((1,), float(index))
    return len(params)


def fill_keyword_only(params: list[object], *, buffers: list[object]) -> None:
    buffers[0] = torch.zeros(1)


def fill_optional(params: list[object], buffers: list[object] | None = None) -> None:
    if buffers is not None:
        buffers[0] = torch.zeros(1)


def fill_variadic(params: list[object], *buffers: object) -> None:
    pass


def grow(params: list[object], buffers: list[object]) -> None:
    buffers.append(torch.zeros(1))


def grow_then_raise(params: list[object], buffers: list[object]) -> None:
    buffers.append(torch.zeros(1))
    raise ValueError("callee failed")


CALLS: list[str] = []


def fill_two(first: list[object], second: list[object]) -> None:
    CALLS.append("fill_two")
    first[0] = torch.zeros(1)


def fill_first_grow_second(first: list[object], second: list[object]) -> None:
    first[0] = torch.zeros(1)
    second.append(torch.zeros(1))


def fill_and_alias(
    params: list[object], buffers: list[object], box: dict[str, list[object]]
) -> None:
    buffers[0] = torch.zeros(1)
    box["buffers"][0] = torch.ones(1)


def fill_both_references(
    params: list[object], buffers: list[object], box: dict[str, list[object]]
) -> None:
    value = torch.zeros(1)
    buffers[0] = value
    box["buffers"][0] = value


class CalleeError(Exception):
    pass


def fill_then_raise(params: list[object], buffers: list[object]) -> None:
    buffers[0] = torch.zeros(1)
    raise CalleeError("callee failed after replacing a slot")


class WritebackTestCase(unittest.TestCase):
    def register(self, function: Callable[..., object], *parameters: str) -> None:
        patcher = patch.dict(
            list_writeback.LIST_WRITEBACK_ARGUMENTS,  # type: ignore[arg-type]
            {typename(function): frozenset(parameters)},
        )
        patcher.start()
        self.addCleanup(patcher.stop)


class UnproxyCopiesListsTests(WritebackTestCase):
    def test_list_argument_is_copied_and_caller_keeps_proxies(self) -> None:
        tensor = torch.zeros(1)
        proxy = MarkerProxy(tensor)
        values = [proxy, [MarkerProxy(tensor)]]
        unproxied = unproxy_arg(values)
        self.assertIsNot(unproxied, values)
        self.assertIs(unproxied[0], tensor)
        self.assertIs(unproxied[1][0], tensor)
        self.assertIs(values[0], proxy)
        self.assertTrue(is_proxied(values[1][0]))

    def test_unregistered_api_does_not_write_back(self) -> None:
        proxy = MarkerProxy(torch.zeros(1))
        params: list[object] = [proxy]
        buffers: list[object] = [None]
        unproxy_func(fill_missing)(params, buffers)
        self.assertEqual(buffers, [None])
        self.assertIs(params[0], proxy)


class RegisteredListWritebackTests(WritebackTestCase):
    def setUp(self) -> None:
        self.param = MarkerProxy(torch.zeros(1))
        self.kept = MarkerProxy(torch.ones(1))
        self.params: list[object] = [self.param]
        self.buffers: list[object] = [None, self.kept]

    def assert_written_back(self) -> None:
        self.assertIsInstance(self.buffers[0], torch.Tensor)
        self.assertIs(self.buffers[1], self.kept)
        self.assertIs(self.params[0], self.param)

    def test_positional_list_slots_are_written_back(self) -> None:
        self.register(fill_missing, "buffers")
        wrapped = unproxy_func(fill_missing, api_name=typename(fill_missing))
        self.assertEqual(wrapped(self.params, self.buffers), 1)
        self.assert_written_back()

    def test_keyword_list_slots_are_written_back(self) -> None:
        self.register(fill_missing, "buffers")
        unproxy_func(fill_missing)(self.params, buffers=self.buffers)
        self.assert_written_back()

    def test_keyword_only_parameter_is_written_back(self) -> None:
        self.register(fill_keyword_only, "buffers")
        unproxy_func(fill_keyword_only)(self.params, buffers=self.buffers)
        self.assert_written_back()

    def test_slots_replaced_before_an_exception_are_written_back(self) -> None:
        self.register(fill_then_raise, "buffers")
        with self.assertRaises(CalleeError) as raised:
            unproxy_func(fill_then_raise)(self.params, self.buffers)
        self.assertEqual(str(raised.exception), "callee failed after replacing a slot")
        self.assert_written_back()

    def test_length_change_is_an_error(self) -> None:
        self.register(grow, "buffers")
        with self.assertRaisesRegex(ListWritebackError, "'buffers'"):
            unproxy_func(grow)(self.params, self.buffers)
        self.assertEqual(self.buffers, [None, self.kept])

    def test_length_change_during_an_exception_keeps_the_original(self) -> None:
        self.register(grow_then_raise, "buffers")
        with self.assertRaises(ListWritebackError) as raised:
            unproxy_func(grow_then_raise)(self.params, self.buffers)
        self.assertIsInstance(raised.exception.__cause__, ValueError)

    def test_unknown_registered_parameter_fails_when_wrapping(self) -> None:
        self.register(fill_missing, "momentum_buffer_list")
        with self.assertRaisesRegex(ListWritebackError, "no parameter"):
            unproxy_func(fill_missing)

    def test_variadic_registered_parameter_fails_when_wrapping(self) -> None:
        self.register(fill_variadic, "buffers")
        with self.assertRaisesRegex(ListWritebackError, "variadic"):
            unproxy_func(fill_variadic)

    def test_same_list_for_two_registered_parameters_is_an_error(self) -> None:
        self.register(fill_two, "first", "second")
        CALLS.clear()
        with self.assertRaisesRegex(ListWritebackError, "same list"):
            unproxy_func(fill_two)(self.buffers, self.buffers)
        self.assertEqual(CALLS, [])
        self.assertEqual(self.buffers, [None, self.kept])

    def test_failed_write_back_writes_no_slot(self) -> None:
        self.register(fill_first_grow_second, "first", "second")
        first: list[object] = [None]
        with self.assertRaisesRegex(ListWritebackError, "'second'"):
            unproxy_func(fill_first_grow_second)(first, [None])
        self.assertEqual(first, [None])

    def test_slot_changed_through_another_reference_is_an_error(self) -> None:
        self.register(fill_and_alias, "buffers")
        box = {"buffers": self.buffers}
        with self.assertRaisesRegex(ListWritebackError, "another reference"):
            unproxy_func(fill_and_alias)(self.params, self.buffers, box)
        # The caller's list is as the callee left it through the alias.
        written = self.buffers[0]
        assert isinstance(written, torch.Tensor)
        self.assertTrue(torch.equal(written, torch.ones(1)))

    def test_same_object_through_both_references_is_written_back(self) -> None:
        self.register(fill_both_references, "buffers")
        box = {"buffers": self.buffers}
        unproxy_func(fill_both_references)(self.params, self.buffers, box)
        self.assert_written_back()

    def test_registered_argument_that_is_not_a_list_is_unproxied_normally(
        self,
    ) -> None:
        self.register(fill_optional, "buffers")
        unproxy_func(fill_optional)(self.params)
        tensor = torch.ones(1)
        with self.assertRaises(TypeError):
            # The callee receives the unproxied tuple and cannot assign into it.
            unproxy_func(fill_optional)(self.params, (MarkerProxy(tensor),))


class InstalledTorchTests(unittest.TestCase):
    def test_registry_matches_the_installed_library(self) -> None:
        for api_name, parameters in list_writeback.LIST_WRITEBACK_ARGUMENTS.items():
            with self.subTest(api_name=api_name):
                module_name, attribute = api_name.rsplit(".", 1)
                function = getattr(importlib.import_module(module_name), attribute)
                self.assertEqual(typename(function), api_name)
                positions = list_writeback.list_writeback_positions(function, api_name)
                assert positions is not None
                self.assertEqual(set(positions), set(parameters))

    def test_functional_sgd_buffers_reach_the_caller(self) -> None:
        sgd = importlib.import_module("torch.optim.sgd").sgd
        wrapped = unproxy_func(sgd)
        parameter = torch.nn.Parameter(torch.zeros(3))
        gradient = torch.tensor([1.0, 2.0, 3.0])
        parameter_proxy = MarkerProxy(parameter)
        params: list[object] = [parameter_proxy]
        buffers: list[object] = [None]
        options = dict(
            weight_decay=0.0,
            momentum=0.9,
            lr=0.1,
            dampening=0.0,
            nesterov=False,
            maximize=False,
            foreach=False,
        )
        with torch.no_grad():
            wrapped(params, [MarkerProxy(gradient)], buffers, **options)
        first = buffers[0]
        assert isinstance(first, torch.Tensor)
        self.assertTrue(torch.equal(first, gradient))
        self.assertIs(params[0], parameter_proxy)

        with torch.no_grad():
            wrapped(params, [MarkerProxy(gradient)], buffers, **options)
        self.assertIs(buffers[0], first)
        self.assertTrue(torch.allclose(first, gradient * 1.9))

    def test_sgd_momentum_survives_an_unproxied_functional_sgd(self) -> None:
        def train(wrap: bool) -> list[torch.Tensor]:
            torch.manual_seed(0)
            model = torch.nn.Linear(3, 1)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
            sgd_module = importlib.import_module("torch.optim.sgd")
            replacement = unproxy_func(sgd_module.sgd) if wrap else sgd_module.sgd
            with patch.object(sgd_module, "sgd", replacement):
                for _ in range(3):
                    optimizer.zero_grad()
                    model(torch.randn(4, 3)).sum().backward()
                    optimizer.step()
            self.assertEqual(len(optimizer.state), 2)
            return [optimizer.state[p]["momentum_buffer"] for p in model.parameters()]

        for wrapped, plain in zip(train(True), train(False)):
            self.assertTrue(torch.equal(wrapped, plain))


class TracerWritebackTests(WritebackTestCase):
    def setUp(self) -> None:
        self.records: list[dict[str, object]] = []
        for patcher in (
            patch.object(tracer.config, "DISABLE_WRAPPER", False),
            patch.object(tracer, "dump_trace_API", side_effect=self.records.append),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.param = MarkerProxy(torch.zeros(1))
        self.params: list[object] = [self.param]
        self.buffers: list[object] = [None]

    def call_function_wrapper(self, function: Callable[..., object]) -> object:
        return tracer.function_wrapper(
            function,
            typename(function),
            False,
            False,
            False,
            False,
            None,
            False,
            None,
            False,
            None,
            True,
            False,
            None,
            self.params,
            self.buffers,
        )

    def test_function_wrapper_writes_back_registered_slots(self) -> None:
        self.register(fill_missing, "buffers")
        self.assertEqual(self.call_function_wrapper(fill_missing), 1)
        self.assertIsInstance(self.buffers[0], torch.Tensor)
        self.assertIs(self.params[0], self.param)
        self.assertEqual(len(self.records), 2)

    def test_function_wrapper_writes_back_before_reraising(self) -> None:
        self.register(fill_then_raise, "buffers")
        with self.assertRaises(CalleeError):
            self.call_function_wrapper(fill_then_raise)
        self.assertIsInstance(self.buffers[0], torch.Tensor)
        self.assertEqual(
            self.records[-1]["exception_msg"], "callee failed after replacing a slot"
        )

    def test_function_wrapper_copies_unregistered_lists(self) -> None:
        self.call_function_wrapper(fill_missing)
        self.assertEqual(self.buffers, [None])
        self.assertIs(self.params[0], self.param)

    def test_function_wrapper_records_a_registry_mismatch_as_the_exception(
        self,
    ) -> None:
        self.register(fill_missing, "momentum_buffer_list")
        with self.assertRaises(ListWritebackError):
            self.call_function_wrapper(fill_missing)
        self.assertEqual(
            [record["type"] for record in self.records],
            [
                tracer.TraceLineType.FUNC_CALL_PRE,
                tracer.TraceLineType.FUNC_CALL_POST_EXCEPTION,
            ],
        )
        self.assertEqual(self.buffers, [None])

    def test_wrapper_rejects_a_registry_mismatch_when_wrapping(self) -> None:
        self.register(fill_missing, "momentum_buffer_list")
        for disable_dump in (False, True):
            with self.subTest(disable_dump=disable_dump):
                with patch.object(
                    tracer, "is_funcs_to_be_unproxied", return_value=True
                ):
                    with self.assertRaisesRegex(ListWritebackError, "no parameter"):
                        tracer.wrapper(
                            fill_missing, False, False, False, disable_dump=disable_dump
                        )

    def wrap_without_dumping(
        self, function: Callable[..., object]
    ) -> Callable[..., object]:
        with patch.object(tracer, "is_funcs_to_be_unproxied", return_value=True):
            return tracer.wrapper(function, False, False, False, disable_dump=True)

    def test_dump_disabled_wrapper_writes_back_registered_slots(self) -> None:
        self.register(fill_missing, "buffers")
        self.assertEqual(
            self.wrap_without_dumping(fill_missing)(self.params, self.buffers), 1
        )
        self.assertIsInstance(self.buffers[0], torch.Tensor)
        self.assertIs(self.params[0], self.param)

    def test_dump_disabled_wrapper_copies_unregistered_lists(self) -> None:
        self.wrap_without_dumping(fill_missing)(self.params, self.buffers)
        self.assertEqual(self.buffers, [None])
        self.assertEqual(self.records, [])


def needs_cuda_probe_workaround() -> bool:
    """Default instrumentation probes CUDA state, which fails without a driver."""
    return torch.backends.cuda.is_built() and not torch.cuda.is_available()


class CollectedTrainingMatchesPlainTraining(unittest.TestCase):
    """SGD momentum, optimizer state and checkpoints under model tracking."""

    def run_probe(
        self,
        directory: Path,
        collector_arguments: list[str] | None,
        without_writeback: bool = False,
    ) -> dict[str, object]:
        # The runner writes a copy of the script next to the original and sets
        # PYTHONPATH to that directory, so run a copy outside the source tree.
        script = directory / PROBE.name
        shutil.copyfile(PROBE, script)
        report = directory / "report.json"
        env = {
            **os.environ,
            "PYTHONPATH": str(ROOT),
            "TRAINCHECK_SGD_PROBE_OUT": str(report),
        }
        if without_writeback:
            env["TRAINCHECK_SGD_PROBE_WITHOUT_WRITEBACK"] = "1"
        if collector_arguments is None:
            command = [sys.executable, str(script)]
        else:
            command = [
                sys.executable,
                "-m",
                "traincheck.collect_trace",
                "--pyscript",
                str(script),
                "--output-dir",
                str(directory / "trace"),
                *collector_arguments,
            ]
        result = subprocess.run(
            command,
            cwd=directory,
            env=env,
            text=True,
            capture_output=True,
            timeout=600,
            check=False,
        )
        output = result.stdout + result.stderr
        program_output = directory / "trace" / "program_output.txt"
        if program_output.exists():
            output += program_output.read_text(encoding="utf-8", errors="replace")
        self.assertEqual(result.returncode, 0, output[-4000:])
        self.assertTrue(report.exists(), output[-4000:])
        loaded = json.loads(report.read_text(encoding="utf-8"))
        assert isinstance(loaded, dict)
        return loaded

    def sgd_observations(self, trace_directory: Path) -> int:
        pattern = re.compile(r'"dump_loc":\s*"torch\.optim\.sgd\.sgd"')
        return sum(
            len(pattern.findall(path.read_text(encoding="utf-8", errors="replace")))
            for path in trace_directory.iterdir()
            if path.suffix in (".json", ".log")
        )

    def check(self, extra_arguments: list[str]) -> None:
        with tempfile.TemporaryDirectory() as plain_directory:
            plain = self.run_probe(Path(plain_directory), None)
        self.assertIsNone(plain.pop("traincheck_file"))
        self.assertEqual(plain["state_count"], plain["parameter_count"])
        self.assertNotIn(None, plain["momentum_buffers"])  # type: ignore[operator]
        for style in ("proxy", "subclass"):
            with self.subTest(style=style), tempfile.TemporaryDirectory() as directory:
                collected = self.run_probe(
                    Path(directory),
                    [
                        "--models-to-track",
                        "model",
                        "--model-tracker-style",
                        style,
                        *extra_arguments,
                    ],
                )
                # The training process must have used this checkout, which the
                # runner's PYTHONPATH does not ensure; and the observer around
                # functional sgd must have run, or the comparison would not
                # exercise the unproxying path at all.
                traincheck_file = collected.pop("traincheck_file")
                assert isinstance(traincheck_file, str)
                self.assertTrue(
                    Path(traincheck_file).resolve().is_relative_to(ROOT),
                    f"{traincheck_file} is not in {ROOT}; install this checkout",
                )
                self.assertGreater(self.sgd_observations(Path(directory) / "trace"), 0)
                self.assertEqual(collected, plain)
            # Control: with the registry emptied inside the training process the
            # same run must lose momentum, which shows that this style unproxies
            # functional sgd and that the registry entry is what restores it.
            with self.subTest(style=style, control="without write-back"):
                with tempfile.TemporaryDirectory() as directory:
                    control = self.run_probe(
                        Path(directory),
                        [
                            "--models-to-track",
                            "model",
                            "--model-tracker-style",
                            style,
                            *extra_arguments,
                        ],
                        without_writeback=True,
                    )
                self.assertEqual(
                    control["momentum_buffers"], [None] * len(plain["parameters"])  # type: ignore[arg-type]
                )
                self.assertNotEqual(control["losses"], plain["losses"])

    def test_model_tracking_without_api_instrumentation(self) -> None:
        self.check(["--modules-to-instr", "torch.nn", "torch.optim"])

    @unittest.skipIf(
        needs_cuda_probe_workaround(),
        "default instrumentation initializes CUDA, which needs a driver",
    )
    def test_model_tracking_with_default_instrumentation(self) -> None:
        self.check([])


if __name__ == "__main__":
    unittest.main()
