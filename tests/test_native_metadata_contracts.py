"""Adversarial lookup and configuration contracts for tensor method skipping."""

import json
import subprocess
import sys
import textwrap
import types
import unittest
from contextlib import ExitStack
from unittest.mock import patch

import torch

from traincheck.instrumentor import dumper


class NativeMetadataContractTests(unittest.TestCase):
    def setUp(self) -> None:
        contexts = ExitStack()
        self.addCleanup(contexts.close)
        for name in (
            "skip_attrs_due_to_errs",
            "skip_type_due_to_errs",
            "skip_type_due_to_recursion",
        ):
            contexts.enter_context(patch.object(dumper, name, {}))

    def test_parameter_metaclass_properties_are_never_evaluated(self) -> None:
        visits: list[str] = []

        def namespace(owner: object) -> object:
            visits.append("namespace")
            raise AssertionError("Metaclass namespace property must not execute")

        def inheritance(owner: object) -> object:
            visits.append("inheritance")
            raise AssertionError("Metaclass MRO property must not execute")

        original = type(torch.nn.Parameter)
        replacement: type[object] = type(
            "SpoofedParameterMeta",
            (original,),
            {"__dict__": property(namespace), "__mro__": property(inheritance)},
        )
        value = torch.nn.Parameter(torch.ones(1))
        try:
            setattr(torch.nn.Parameter, "__class__", replacement)
            result = dumper.convert_var_to_dict(
                value, False, {"backward": None, "requires_grad": None}
            )
        finally:
            setattr(torch.nn.Parameter, "__class__", original)
        self.assertEqual(result, {"requires_grad": True})
        self.assertEqual(visits, [])

    def test_changed_parameter_inheritance_observes_new_descriptor(self) -> None:
        # Restoring __bases__ creates another MRO tuple; isolate this global class
        # mutation so later tests still exercise the ordinary Parameter fast path.
        fixture = textwrap.dedent(
            """            import json
            import torch
            from unittest.mock import patch
            from traincheck.instrumentor import dumper

            class ReplacementBase(torch.Tensor):
                pass

            def replacement(value: torch.Tensor) -> int:
                return 37

            setattr(ReplacementBase, "add", property(replacement))
            value = torch.nn.Parameter(torch.ones(1))
            original: object = type.__getattribute__(torch.nn.Parameter, "__bases__")
            try:
                setattr(torch.nn.Parameter, "__bases__", (ReplacementBase,))
                with patch.object(
                    dumper, "is_discardable_tensor_method", return_value=False
                ):
                    baseline = dumper.convert_var_to_dict(value, False, {"add": None})
                result = dumper.convert_var_to_dict(value, False, {"add": None})
            finally:
                setattr(torch.nn.Parameter, "__bases__", original)
            print(json.dumps({"baseline": baseline, "candidate": result}))
            """
        )
        completed = subprocess.run(
            [sys.executable, "-c", fixture],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(
            json.loads(completed.stdout),
            {"baseline": {"add": 37}, "candidate": {"add": 37}},
        )

    def test_instance_dictionary_key_equality_matches_baseline(self) -> None:
        visits: list[object] = []

        class CollidingKey:
            def __hash__(self) -> int:
                return hash("add")

            def __eq__(self, other: object) -> bool:
                visits.append(other)
                return False

        value = torch.ones(1)
        storage: object = object.__getattribute__(value, "__dict__")
        assert isinstance(storage, dict)
        dict.__setitem__(storage, CollidingKey(), 1)
        with patch.object(dumper, "is_discardable_tensor_method", return_value=False):
            expected = dumper.convert_var_to_dict(value, False, {"add": None})
        expected_visits = list(visits)
        visits.clear()
        actual = dumper.convert_var_to_dict(value, False, {"add": None})
        self.assertEqual(actual, expected)
        self.assertTrue(expected_visits)
        self.assertTrue(all(name == "add" for name in expected_visits))
        self.assertEqual(visits, expected_visits)

    def test_earlier_getter_can_replace_lookup_hook(self) -> None:
        visits: list[str] = []

        def lookup(value: torch.Tensor, name: str) -> object:
            visits.append(name)
            if name == "add":
                return 41
            return object.__getattribute__(value, name)

        def prepare(value: torch.Tensor) -> int:
            setattr(torch.Tensor, "__getattribute__", lookup)
            return 1

        value = torch.ones(1)
        with patch.object(torch.Tensor, "__getattribute__", object.__getattribute__):
            with patch.object(torch.Tensor, "prepare", property(prepare), create=True):
                result = dumper.convert_var_to_dict(
                    value, False, {"prepare": None, "add": None}
                )
        self.assertEqual(result, {"prepare": 1, "add": 41})
        self.assertEqual(visits, ["add"])

    def test_replaced_method_failure_retains_backoff(self) -> None:
        visits: list[str] = []

        def failure(value: torch.Tensor) -> object:
            visits.append("add")
            raise ValueError("metadata unavailable")

        value = torch.ones(1)
        with patch.object(torch.Tensor, "add", property(failure)):
            first = dumper.convert_var_to_dict(value, False, {"add": None})
            second = dumper.convert_var_to_dict(value, False, {"add": None})
        self.assertEqual(first, {})
        self.assertEqual(second, {})
        self.assertEqual(visits, ["add"])
        self.assertIn("add", dumper.skip_attrs_due_to_errs[str(torch.Tensor)])

    def test_retained_type_set_subclass_preserves_membership_calls(self) -> None:
        visits: list[object] = []

        class ObservedTypes(set[type[object]]):
            def __contains__(self, item: object) -> bool:
                visits.append(item)
                return super().__contains__(item)

        retained = ObservedTypes(dumper.primitive_types)
        retained.add(types.BuiltinFunctionType)
        value = torch.ones(1)
        with patch.object(dumper, "primitive_types", retained):
            with patch.object(
                dumper, "is_discardable_tensor_method", return_value=False
            ):
                expected = dumper.convert_var_to_dict(value, False, {"add": None})
            expected_visits = list(visits)
            visits.clear()
            actual = dumper.convert_var_to_dict(value, False, {"add": None})
        self.assertEqual(actual, expected)
        self.assertEqual(expected_visits, [types.BuiltinFunctionType])
        self.assertEqual(visits, expected_visits)

    def test_nonempty_receiver_preserves_primitive_metaclass_comparisons(self) -> None:
        visits: list[object] = []

        class CollidingMeta(type):
            def __hash__(cls) -> int:
                return hash(types.BuiltinFunctionType)

            def __eq__(cls, other: object) -> bool:
                visits.append(other)
                return False

        class ConfiguredType(metaclass=CollidingMeta):
            pass

        retained: set[type[object]] = set(dumper.primitive_types)
        retained.add(ConfiguredType)
        value = torch.ones(1)
        value.__dict__["unrelated"] = 1
        visits.clear()
        with patch.object(dumper, "primitive_types", retained):
            with patch.object(
                dumper, "is_discardable_tensor_method", return_value=False
            ):
                expected = dumper.convert_var_to_dict(value, False, {"add": None})
            expected_visits = list(visits)
            visits.clear()
            actual = dumper.convert_var_to_dict(value, False, {"add": None})
        self.assertEqual(actual, expected)
        self.assertTrue(expected_visits)
        self.assertTrue(
            all(
                value_type is types.BuiltinFunctionType
                for value_type in expected_visits
            )
        )
        self.assertEqual(visits, expected_visits)


if __name__ == "__main__":
    unittest.main()
