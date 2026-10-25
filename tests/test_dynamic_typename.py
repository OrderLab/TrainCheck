"""Runtime type names must tolerate dynamic metadata without calling it."""

import unittest

import torch

from traincheck.utils import typename


class DynamicMetadata:
    __module__ = "fixtures.dynamic"

    def __getattr__(self, name: str) -> object:
        if name in {"__qualname__", "__name__"}:
            return lambda: None
        raise AttributeError(name)


class UnsafeTruthValue:
    def __bool__(self) -> bool:
        raise AssertionError("Metadata truth value must not be evaluated")


class VisibleType:
    pass


class ReportedClass:
    __module__ = "fixtures.dynamic"

    def __getattribute__(self, name: str) -> object:
        if name == "__class__":
            return VisibleType
        attribute: object = super().__getattribute__(name)
        return attribute


class DynamicTypenameTests(unittest.TestCase):
    def test_dynamic_callable_metadata_uses_class_name(self) -> None:
        for runtime in [False, True]:
            self.assertEqual(
                typename(DynamicMetadata(), is_runtime=runtime),
                "fixtures.dynamic.DynamicMetadata",
            )

    def test_non_string_name_values_fall_back_without_truth_testing(self) -> None:
        for value in [object(), 17, ["name"], UnsafeTruthValue()]:

            class Metadata:
                __module__ = "fixtures.dynamic"

                def __getattr__(self, name: str) -> object:
                    if name == "__name__":
                        return value
                    raise AttributeError(name)

            for runtime in [False, True]:
                self.assertEqual(
                    typename(Metadata(), is_runtime=runtime),
                    "fixtures.dynamic.Metadata",
                )

    def test_valid_string_names_still_take_precedence(self) -> None:
        class Metadata:
            __module__ = "fixtures.dynamic"

            def __getattr__(self, name: str) -> object:
                if name == "__qualname__":
                    return "Outer.call"
                if name == "__name__":
                    raise AssertionError("Qualified name should take precedence")
                raise AttributeError(name)

        self.assertEqual(typename(Metadata()), "fixtures.dynamic.Outer.call")

    def test_valid_unqualified_name_is_preserved(self) -> None:
        value = DynamicMetadata()
        setattr(value, "__name__", "exported")
        self.assertEqual(typename(value), "fixtures.dynamic.exported")

    def test_reported_class_fallback_is_preserved(self) -> None:
        self.assertEqual(typename(ReportedClass()), "fixtures.dynamic.VisibleType")

    def test_standard_functions_descriptors_tensors_and_parameters(self) -> None:
        self.assertEqual(typename(len), "len")
        self.assertEqual(typename(str.upper), "str.upper")
        tensor = torch.zeros(1)
        self.assertEqual(typename(tensor), tensor.type())
        self.assertEqual(typename(torch.nn.Parameter(tensor)), "torch.nn.Parameter")

    def test_invalid_reported_class_name_retains_final_assertion(self) -> None:
        class InvalidName:
            def __getattribute__(self, name: str) -> object:
                if name == "__module__":
                    return "fixtures.dynamic"
                if name in {"__qualname__", "__name__"}:
                    return None
                if name == "__class__":
                    return InvalidClassName()
                raise AttributeError(name)

        class InvalidClassName:
            __name__ = 17

        for runtime in (False, True):
            with self.assertRaisesRegex(
                AssertionError, "prefix and last_name should be str"
            ):
                typename(InvalidName(), is_runtime=runtime)

    def test_unexpected_metadata_failure_propagates(self) -> None:
        error = ValueError("metadata unavailable")

        class FailingMetadata:
            def __getattr__(self, name: str) -> object:
                if name == "__qualname__":
                    raise error
                raise AttributeError(name)

        with self.assertRaises(ValueError) as caught:
            typename(FailingMetadata())
        self.assertIs(caught.exception, error)

    def test_real_pytorch_dynamic_ops_wrapper(self) -> None:
        from torch._inductor.virtualized import ops

        self.assertEqual(typename(ops), "torch._inductor.virtualized.OpsWrapper")
        self.assertEqual(
            typename(ops, is_runtime=True),
            "torch._inductor.virtualized.OpsWrapper",
        )


if __name__ == "__main__":
    unittest.main()
