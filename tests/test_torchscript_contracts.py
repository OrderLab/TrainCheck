"""Explicit unsupported, malformed, and failed TorchScript boundary contracts."""

import threading
import unittest
import warnings
from types import SimpleNamespace
from typing import Callable
from unittest.mock import patch

from traincheck.instrumentor import torchscript_compat, tracer
from traincheck.instrumentor.torchscript_compat import preserve_torchscript_builtin


def original(value: int = 1) -> int:
    return value + 1


def candidate(value: int = 1) -> int:
    return original(value)


def framework(find: object, register: object) -> SimpleNamespace:
    return SimpleNamespace(
        jit=SimpleNamespace(
            _builtins=SimpleNamespace(_find_builtin=find, _register_builtin=register)
        )
    )


class TorchScriptContractTests(unittest.TestCase):
    def test_present_noncallable_callback_is_rejected(self) -> None:
        def find(function: Callable[..., object]) -> str:
            return "aten::tanh"

        def register(function: Callable[..., object], operator: str) -> None:
            self.fail("Invalid capabilities must not register a wrapper")

        malformed_callbacks: tuple[object, ...] = (None, 17, "callable", [])
        for malformed in malformed_callbacks:
            for name, fake in (
                ("_find_builtin", framework(malformed, register)),
                ("_register_builtin", framework(find, malformed)),
            ):
                with self.subTest(name=name, malformed=malformed):
                    with self.assertRaisesRegex(TypeError, name):
                        preserve_torchscript_builtin(original, candidate, fake)

    def test_missing_callback_does_not_hide_an_invalid_present_callback(self) -> None:
        fake = SimpleNamespace(
            jit=SimpleNamespace(_builtins=SimpleNamespace(_register_builtin=None))
        )
        with self.assertRaisesRegex(TypeError, "_register_builtin"):
            preserve_torchscript_builtin(original, candidate, fake)

    def test_malformed_operator_is_rejected_before_registration(self) -> None:
        registered: list[str] = []

        def register(function: Callable[..., object], operator: str) -> None:
            registered.append(operator)

        malformed_operators: tuple[object, ...] = ("", 1, False, [], {})
        for malformed in malformed_operators:

            def find(function: Callable[..., object]) -> object:
                return malformed

            with self.subTest(operator=malformed):
                with self.assertRaisesRegex(TypeError, "nonempty string or None"):
                    preserve_torchscript_builtin(
                        original, candidate, framework(find, register)
                    )
        self.assertEqual(registered, [])
        self.assertNotIn(candidate, torchscript_compat._REGISTERED_WRAPPERS)

    def test_find_failure_propagates_unchanged(self) -> None:
        error = RuntimeError("registry lookup failed")

        def find(function: Callable[..., object]) -> str | None:
            raise error

        def register(function: Callable[..., object], operator: str) -> None:
            self.fail("Failed lookup must not register a wrapper")

        with self.assertRaises(RuntimeError) as caught:
            preserve_torchscript_builtin(original, candidate, framework(find, register))
        self.assertIs(caught.exception, error)
        self.assertNotIn(candidate, torchscript_compat._REGISTERED_WRAPPERS)

    def test_registration_failure_propagates_without_retention(self) -> None:
        error = RuntimeError("registry update failed")

        def find(function: Callable[..., object]) -> str:
            return "aten::tanh"

        def register(function: Callable[..., object], operator: str) -> None:
            raise error

        with self.assertRaises(RuntimeError) as caught:
            preserve_torchscript_builtin(original, candidate, framework(find, register))
        self.assertIs(caught.exception, error)
        self.assertNotIn(candidate, torchscript_compat._REGISTERED_WRAPPERS)

    def test_wrapper_exposes_unsupported_and_warns_once(self) -> None:
        with (
            patch.object(tracer, "torch", SimpleNamespace()),
            patch.object(torchscript_compat, "_UNSUPPORTED_WARNING_EMITTED", False),
            warnings.catch_warnings(record=True) as captured,
        ):
            warnings.simplefilter("always")
            first = tracer.wrapper(
                original, False, False, False, disable_dump=True, handle_proxy=False
            )
            second = tracer.wrapper(
                original, False, False, False, disable_dump=True, handle_proxy=False
            )
            self.assertEqual(first(3), 4)
            self.assertEqual(second(8), 9)
        self.assertEqual(len(captured), 1)
        self.assertIn(
            "Eager instrumentation remains available", str(captured[0].message)
        )
        for wrapped in (first, second):
            # Function metadata is deliberately public to instrumentation users.
            self.assertEqual(
                getattr(wrapped, "_traincheck_torchscript_builtin"),
                {"status": "unsupported-private-api", "operator": None},
            )

    def test_unsupported_warning_is_once_across_threads(self) -> None:
        with (
            patch.object(torchscript_compat, "_UNSUPPORTED_WARNING_EMITTED", False),
            warnings.catch_warnings(record=True) as captured,
        ):
            warnings.simplefilter("always")
            workers = [
                threading.Thread(target=torchscript_compat.warn_unsupported_torchscript)
                for _ in range(8)
            ]
            for worker in workers:
                worker.start()
            for worker in workers:
                worker.join(5)
                self.assertFalse(worker.is_alive())
        self.assertEqual(len(captured), 1)

    def test_wrapper_propagates_metadata_failure(self) -> None:
        def find(function: Callable[..., object]) -> object:
            return 3

        def register(function: Callable[..., object], operator: str) -> None:
            self.fail("Malformed operator must not be registered")

        with patch.object(tracer, "torch", framework(find, register)):
            with self.assertRaisesRegex(TypeError, "nonempty string or None"):
                tracer.wrapper(
                    original, False, False, False, disable_dump=True, handle_proxy=False
                )


if __name__ == "__main__":
    unittest.main()
