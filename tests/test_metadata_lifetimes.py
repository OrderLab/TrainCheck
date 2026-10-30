"""Native metadata bindings preserve observable temporary-object lifetimes."""

from __future__ import annotations

import gc
import sys
import types
import unittest
import weakref
from contextlib import ExitStack
from typing import Literal, TypeAlias
from unittest.mock import patch

import torch

from traincheck.instrumentor import dumper

LifetimeCase: TypeAlias = Literal[
    "previous_value", "previous_type", "receiver_reference", "function_reference"
]


def _generic_binding(receiver: object, name: str) -> None:
    # A plain function adds no retained call arguments, unlike a mock object.
    return None


def _lifetime_case(case: LifetimeCase) -> dict[str, object]:
    tensor = torch.ones(1)
    refs: list[weakref.ReferenceType[object]] = []

    class Ephemeral:
        pass

    def prepare(receiver: object) -> object:
        value = Ephemeral()
        refs.append(weakref.ref(value))
        return value

    def prepare_class(receiver: object) -> object:
        class TemporaryClass:
            pass

        refs.append(weakref.ref(TemporaryClass))
        return TemporaryClass()

    def observe_alive(receiver: object) -> bool:
        return refs[-1]() is not None

    def observe_class(receiver: object) -> bool:
        gc.collect()
        return refs[-1]() is not None

    def observe_count(receiver: object) -> int:
        return sys.getrefcount(receiver)

    def observe_function(receiver: object) -> bool:
        delattr(torch.Tensor, "metadata_lifetime_method")
        return refs[-1]() is not None

    def method(receiver: object) -> None:
        pass

    attrs: dict[str, object]
    config: dict[str, object]
    if case == "previous_value":
        attrs = {
            "metadata_lifetime_prepare": property(prepare),
            "metadata_lifetime_observe": property(observe_alive),
        }
        config = {
            "metadata_lifetime_prepare": None,
            "add": None,
            "metadata_lifetime_observe": None,
        }
    elif case == "previous_type":
        attrs = {
            "metadata_lifetime_prepare": property(prepare_class),
            "metadata_lifetime_observe": property(observe_class),
        }
        config = {
            "metadata_lifetime_prepare": None,
            "add": None,
            "metadata_lifetime_observe": None,
        }
    elif case == "receiver_reference":
        attrs = {"metadata_lifetime_observe": property(observe_count)}
        config = {"add": None, "metadata_lifetime_observe": None}
    else:
        assert case == "function_reference"
        refs.append(weakref.ref(method))
        attrs = {
            "metadata_lifetime_method": method,
            "metadata_lifetime_observe": property(observe_function),
        }
        config = {"metadata_lifetime_method": None, "metadata_lifetime_observe": None}
    names = list(attrs)
    for name in names:
        if hasattr(torch.Tensor, name):
            raise RuntimeError(f"Lifetime fixture attribute already exists: {name}")
    try:
        for name, value in attrs.items():
            setattr(torch.Tensor, name, value)
        # Only class storage and the converter's real binding may retain method.
        attrs.clear()
        del value
        del method
        return dumper.convert_var_to_dict(tensor, False, config)
    finally:
        for name in names:
            if name in vars(torch.Tensor):
                delattr(torch.Tensor, name)


class MetadataLifetimeTests(unittest.TestCase):
    def setUp(self) -> None:
        contexts = ExitStack()
        self.addCleanup(contexts.close)
        for name in (
            "skip_attrs_due_to_errs",
            "skip_type_due_to_errs",
            "skip_type_due_to_recursion",
        ):
            contexts.enter_context(patch.object(dumper, name, {}))

    def _assert_lifetime_parity(self, case: LifetimeCase) -> dict[str, object]:
        with patch.object(dumper, "tensor_method_binding", _generic_binding):
            expected = _lifetime_case(case)
        actual = _lifetime_case(case)
        self.assertEqual(actual, expected)
        self.assertEqual(list(actual), list(expected))
        return expected

    def test_previous_attribute_is_released_before_later_property(self) -> None:
        expected = self._assert_lifetime_parity("previous_value")
        self.assertIs(expected["metadata_lifetime_observe"], False)

    def test_previous_attribute_type_is_released_before_later_property(self) -> None:
        expected = self._assert_lifetime_parity("previous_type")
        self.assertIs(expected["metadata_lifetime_observe"], False)

    def test_bound_method_retains_receiver_during_later_property(self) -> None:
        expected = self._assert_lifetime_parity("receiver_reference")
        count = expected["metadata_lifetime_observe"]
        self.assertIs(type(count), int)
        assert isinstance(count, int)
        self.assertGreater(count, 0)

    def test_bound_method_retains_removed_function_during_later_property(self) -> None:
        expected = self._assert_lifetime_parity("function_reference")
        self.assertIs(expected["metadata_lifetime_observe"], True)

    def test_function_instance_get_override_is_not_descriptor_protocol(self) -> None:
        calls: list[str] = []

        def method(receiver: object) -> None:
            pass

        def instance_get(*args: object, **kwargs: object) -> int:
            calls.append("instance __get__")
            return 17

        setattr(method, "__get__", instance_get)
        name = "metadata_lifetime_descriptor"
        if hasattr(torch.Tensor, name):
            raise RuntimeError(f"Lifetime fixture attribute already exists: {name}")
        with patch.object(torch.Tensor, name, method, create=True):
            tensor = torch.ones(1)
            config: dict[str, object] = {name: None}
            with patch.object(dumper, "tensor_method_binding", _generic_binding):
                expected = dumper.convert_var_to_dict(tensor, False, config)
            actual = dumper.convert_var_to_dict(tensor, False, config)
            self.assertEqual(expected, {})
            self.assertEqual(actual, expected)
            binding = dumper.tensor_method_binding(tensor, name)
            self.assertIsInstance(binding, types.MethodType)
            assert isinstance(binding, types.MethodType)
            self.assertIs(binding.__func__, method)
            self.assertIs(binding.__self__, tensor)
            self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
