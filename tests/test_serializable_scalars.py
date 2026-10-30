"""Compatibility boundaries for skipping redundant exact-scalar JSON probes."""

import json
import math
import sys
import unittest
from collections.abc import Iterator
from unittest.mock import patch

import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from traincheck.config import config
from traincheck.instrumentor import dumper


class WatchedList(list[int]):
    def __init__(self) -> None:
        super().__init__([1, 2])
        self.iterations = 0

    def __iter__(self) -> Iterator[int]:
        self.iterations += 1
        return super().__iter__()


class MetadataDictionary(dict[str, int]):
    @property
    def status(self) -> str:
        return "kept"


class EqualityTrap(type):
    def __eq__(self, other: object) -> bool:
        raise AssertionError("Scalar type guards must not invoke class equality")

    def __hash__(self) -> int:
        raise AssertionError("Scalar type guards must not hash user classes")


class CustomInteger(int, metaclass=EqualityTrap):
    pass


class SerializableScalarTests(unittest.TestCase):
    def setUp(self) -> None:
        compile_guard = patch.object(config, "USE_TORCH_COMPILE", False)
        compile_guard.start()
        self.addCleanup(compile_guard.stop)

    def test_exact_scalar_values_keep_runtime_labels_and_original_objects(self) -> None:
        values: tuple[tuple[str, object], ...] = (
            ("NoneType", None),
            ("bool", True),
            ("bool", False),
            ("int", -(2**63)),
            ("int", 2**63 - 1),
            ("float", 1.25),
            ("str", "雪\ud800"),
        )
        for name, value in values:
            with self.subTest(name=name, value=value):
                result = dumper.var_to_serializable(value)
                self.assertEqual(list(result), [name])
                self.assertIs(result[name], value)

    def test_nonfinite_and_negative_zero_are_not_replaced_or_coerced(self) -> None:
        for value in (float("nan"), float("inf"), float("-inf"), -0.0):
            result = dumper.var_to_serializable(value)
            self.assertIs(result["float"], value)
            self.assertEqual(
                dumper.serialize(result), dumper.serialize({"float": value})
            )
        result_zero = dumper.var_to_serializable(-0.0)["float"]
        assert isinstance(result_zero, float)
        self.assertEqual(math.copysign(1, result_zero), -1)

    def test_large_integers_outside_fast_path_still_keep_value(self) -> None:
        for value in (-(2**63) - 1, 2**63, 2**128):
            self.assertIs(dumper.var_to_serializable(value)["int"], value)

    def test_huge_integer_preserves_configured_json_value_error(self) -> None:
        # Python exposes this optional security setting on current 3.10+ patch
        # releases. Older interpreters without it cannot exercise this contract.
        get_limit: object = getattr(sys, "get_int_max_str_digits", None)
        set_limit: object = getattr(sys, "set_int_max_str_digits", None)
        if not callable(get_limit) or not callable(set_limit):
            self.skipTest("Interpreter has no configurable integer digit limit")
        previous: object = get_limit()
        assert isinstance(previous, int)
        try:
            set_limit(640)
            value = 10**1000
            with self.assertRaises(ValueError) as expected:
                json.dumps(value)
            with self.assertRaises(ValueError) as actual:
                dumper.var_to_serializable(value)
            self.assertEqual(str(actual.exception), str(expected.exception))
        finally:
            set_limit(previous)

    def test_custom_scalar_class_equality_and_hash_are_not_invoked(self) -> None:
        value = CustomInteger(7)
        result = dumper.var_to_serializable(value)
        self.assertEqual(len(result), 1)
        self.assertIs(next(iter(result.values())), value)
        self.assertTrue(next(iter(result)).endswith("CustomInteger"))

    def test_list_subclass_keeps_json_iterator_side_effect(self) -> None:
        value = WatchedList()
        result = dumper.var_to_serializable(value)
        self.assertEqual(value.iterations, 1)
        self.assertIs(next(iter(result.values())), value)

    def test_dictionary_subclass_keeps_selected_metadata_conversion(self) -> None:
        value = MetadataDictionary(hidden=3)
        result = dumper.var_to_serializable(value, {"status": None})
        self.assertEqual(next(iter(result.values())), {"status": "kept"})

    def test_mixed_containers_keep_identity_and_nested_contents(self) -> None:
        value: list[object] = [1, "雪", None, {"flag": True}, (2.0, False)]
        result = dumper.var_to_serializable(value)
        self.assertIs(result["list"], value)
        self.assertEqual(json.dumps(result), json.dumps({"list": value}))

    def test_cyclic_container_still_raises_at_conversion(self) -> None:
        value: list[object] = []
        value.append(value)
        with self.assertRaisesRegex(ValueError, "Circular reference"):
            dumper.var_to_serializable(value)

    def test_existing_compile_gate_runs_before_scalar_shortcut(self) -> None:
        # The existing compile-mode heuristic treats objects without a device
        # as fake. This optimization must preserve that established gate.
        with patch.object(config, "USE_TORCH_COMPILE", True):
            self.assertEqual(
                dumper.var_to_serializable(3),
                {"builtins.int": {"fake": True, "shape": None, "dtype": None}},
            )

    def test_actual_fake_tensor_keeps_dedicated_metadata_path(self) -> None:
        with FakeTensorMode() as mode:
            value = mode.from_tensor(torch.empty(2, 3))
        with patch.object(config, "USE_TORCH_COMPILE", True):
            result = dumper.var_to_serializable(value)
        self.assertEqual(
            next(iter(result.values())),
            {"fake": True, "shape": (2, 3), "dtype": "torch.float32"},
        )

    def test_primitive_keeps_ignoring_uninterpreted_dump_configuration(self) -> None:
        value = 42
        self.assertIs(dumper.var_to_serializable(value, object())["int"], value)


if __name__ == "__main__":
    unittest.main()
