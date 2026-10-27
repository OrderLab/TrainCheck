"""Tensor metadata fast paths preserve dynamic lookup and warning contracts."""

import types
import unittest
import warnings
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from unittest.mock import patch

import torch

from traincheck.instrumentor import dumper


class TensorSubclass(torch.Tensor):
    def __getattribute__(self, name: str) -> object:
        if name == "add":
            return 91
        return super().__getattribute__(name)


class NativeMetadataMethodTests(unittest.TestCase):
    def setUp(self) -> None:
        contexts = ExitStack()
        self.addCleanup(contexts.close)
        for name in (
            "skip_attrs_due_to_errs",
            "skip_type_due_to_errs",
            "skip_type_due_to_recursion",
        ):
            contexts.enter_context(patch.object(dumper, name, {}))

    def test_exact_tensors_preserve_values_identity_order_and_state(self) -> None:
        for value in (torch.ones(2, 3), torch.nn.Parameter(torch.ones(2, 3))):
            with self.subTest(receiver=type(value)):
                stable = torch.zeros(1)
                value.__dict__["payload"] = stable
                config: dict[str, object] = {
                    "add": None,
                    "dtype": None,
                    "backward": None,
                    "shape": None,
                    "payload": None,
                    "requires_grad": None,
                }
                rng_before = torch.get_rng_state().clone()
                state_before = value.detach().clone()
                with patch.object(
                    dumper, "is_discardable_tensor_method", return_value=False
                ):
                    expected = dumper.convert_var_to_dict(value, False, config)
                actual = dumper.convert_var_to_dict(value, False, config)
                self.assertEqual(actual, expected)
                self.assertEqual(list(actual), list(expected))
                self.assertEqual(actual["_TRAINCHECK_payload_ID"], id(stable))
                self.assertTrue(torch.equal(rng_before, torch.get_rng_state()))
                self.assertTrue(torch.equal(state_before, value))

    def test_instance_shadow_is_live_across_conversions(self) -> None:
        value = torch.ones(1)
        config: dict[str, object] = {"add": None}
        self.assertEqual(dumper.convert_var_to_dict(value, False, config), {})
        value.__dict__["add"] = 12
        self.assertEqual(dumper.convert_var_to_dict(value, False, config), {"add": 12})
        value.__dict__["add"] = 13
        self.assertEqual(dumper.convert_var_to_dict(value, False, config), {"add": 13})
        del value.__dict__["add"]
        self.assertEqual(dumper.convert_var_to_dict(value, False, config), {})

    def test_property_can_shadow_a_later_method(self) -> None:
        def prepare(value: torch.Tensor) -> int:
            value.__dict__["add"] = 17
            return 1

        with patch.object(torch.Tensor, "prepare", property(prepare), create=True):
            value = torch.ones(1)
            self.assertEqual(
                dumper.convert_var_to_dict(
                    value, False, {"prepare": None, "add": None}
                ),
                {"prepare": 1, "add": 17},
            )

    def test_property_can_replace_a_later_class_method(self) -> None:
        def prepare(value: torch.Tensor) -> int:
            setattr(torch.Tensor, "add", 19)
            return 1

        # The outer patch restores the original method even if conversion fails.
        with patch.object(torch.Tensor, "add", torch.Tensor.add):
            with patch.object(torch.Tensor, "prepare", property(prepare), create=True):
                self.assertEqual(
                    dumper.convert_var_to_dict(
                        torch.ones(1), False, {"prepare": None, "add": None}
                    ),
                    {"prepare": 1, "add": 19},
                )

    def test_custom_getattribute_and_subclasses_retain_lookup(self) -> None:
        def lookup(value: torch.Tensor, name: str) -> object:
            if name == "add":
                return 23
            return object.__getattribute__(value, name)

        value = torch.ones(1)
        with patch.object(torch.Tensor, "__getattribute__", lookup):
            self.assertEqual(
                dumper.convert_var_to_dict(value, False, {"add": None}), {"add": 23}
            )
        subclass = value.as_subclass(TensorSubclass)
        self.assertEqual(
            dumper.convert_var_to_dict(subclass, False, {"add": None}), {"add": 91}
        )

    def test_foreign_descriptor_keeps_failed_attribute_backoff(self) -> None:
        with patch.object(torch.Tensor, "foreign", str.upper, create=True):
            self.assertEqual(
                dumper.convert_var_to_dict(torch.ones(1), False, {"foreign": None}), {}
            )
        self.assertIn("foreign", dumper.skip_attrs_due_to_errs[str(torch.Tensor)])

    def test_named_traincheck_callable_is_retained(self) -> None:
        # Any substring match is retained by the existing metadata classifier.
        name = "prefix_TRAINCHECK_callback"
        with patch.object(torch.Tensor, name, torch.Tensor.add, create=True):
            value = torch.ones(1)
            result = dumper.convert_var_to_dict(value, False, {name: None})
            self.assertEqual(list(result), [name])
            callback = result[name]
            self.assertIsInstance(callback, types.BuiltinFunctionType)
            assert isinstance(callback, types.BuiltinFunctionType)
            self.assertIs(callback.__self__, value)

    def test_retained_callable_types_are_not_skipped(self) -> None:
        value = torch.ones(1)
        retained = dumper.primitive_types | {
            types.BuiltinFunctionType,
            types.MethodType,
        }
        with patch.object(dumper, "primitive_types", retained):
            result = dumper.convert_var_to_dict(
                value, False, {"add": None, "backward": None}
            )
        self.assertEqual(list(result), ["add", "backward"])
        for callback in result.values():
            self.assertIsInstance(
                callback, (types.BuiltinFunctionType, types.MethodType)
            )

    def test_property_warning_mutation_does_not_leak_to_next_attribute(self) -> None:
        visits: list[str] = []

        def first(value: torch.Tensor) -> object:
            visits.append("first")
            warnings.warn("first", UserWarning)
            warnings.simplefilter("error", RuntimeWarning)
            return len

        def later(value: torch.Tensor) -> int:
            visits.append("later")
            warnings.warn("later", RuntimeWarning)
            return 29

        with patch.object(torch.Tensor, "first", property(first), create=True):
            with patch.object(torch.Tensor, "later", property(later), create=True):
                with warnings.catch_warnings(record=True) as observed:
                    warnings.simplefilter("always")
                    filters = list(warnings.filters)
                    result = dumper.convert_var_to_dict(
                        torch.ones(1),
                        False,
                        {"first": None, "add": None, "later": None},
                    )
                    self.assertEqual(warnings.filters, filters)
                    self.assertEqual(observed, [])
        self.assertEqual(result, {"later": 29})
        self.assertEqual(visits, ["first", "later"])

    def test_concurrent_receivers_do_not_share_shadow_state(self) -> None:
        def convert(index: int) -> dict[str, object]:
            value = torch.ones(1)
            if index % 2:
                value.__dict__["add"] = index
            return dumper.convert_var_to_dict(value, False, {"add": None})

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(convert, range(20)))
        self.assertEqual(results, [{"add": i} if i % 2 else {} for i in range(20)])


if __name__ == "__main__":
    unittest.main()
