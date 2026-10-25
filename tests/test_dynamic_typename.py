"""Runtime type names must tolerate dynamic metadata without calling it."""

import unittest

import torch

from traincheck.utils import typename


class DynamicMetadata:
    __module__ = "fixtures.dynamic"

    def __getattr__(self, name):
        if name in {"__qualname__", "__name__"}:
            return lambda: None
        raise AttributeError(name)


class UnsafeTruthValue:
    def __bool__(self):
        raise AssertionError("Metadata truth value must not be evaluated")


class VisibleType:
    pass


class ReportedClass:
    __module__ = "fixtures.dynamic"

    @property
    def __class__(self):
        return VisibleType


class DynamicTypenameTests(unittest.TestCase):
    def test_dynamic_callable_metadata_uses_class_name(self):
        for runtime in [False, True]:
            self.assertEqual(
                typename(DynamicMetadata(), is_runtime=runtime),
                "fixtures.dynamic.DynamicMetadata",
            )

    def test_non_string_name_values_fall_back_without_truth_testing(self):
        for value in [object(), 17, ["name"], UnsafeTruthValue()]:

            class Metadata:
                __module__ = "fixtures.dynamic"

                def __getattr__(self, name):
                    if name == "__name__":
                        return value
                    raise AttributeError(name)

            for runtime in [False, True]:
                self.assertEqual(
                    typename(Metadata(), is_runtime=runtime),
                    "fixtures.dynamic.Metadata",
                )

    def test_valid_string_names_still_take_precedence(self):
        class Metadata:
            __module__ = "fixtures.dynamic"

            def __getattr__(self, name):
                if name == "__qualname__":
                    return "Outer.call"
                if name == "__name__":
                    raise AssertionError("Qualified name should take precedence")
                raise AttributeError(name)

        self.assertEqual(typename(Metadata()), "fixtures.dynamic.Outer.call")

    def test_valid_unqualified_name_is_preserved(self):
        value = DynamicMetadata()
        value.__name__ = "exported"
        self.assertEqual(typename(value), "fixtures.dynamic.exported")

    def test_reported_class_fallback_is_preserved(self):
        self.assertEqual(typename(ReportedClass()), "fixtures.dynamic.VisibleType")

    def test_standard_functions_descriptors_tensors_and_parameters(self):
        self.assertEqual(typename(len), "len")
        self.assertEqual(typename(str.upper), "str.upper")
        tensor = torch.zeros(1)
        self.assertEqual(typename(tensor), tensor.type())
        self.assertEqual(typename(torch.nn.Parameter(tensor)), "torch.nn.Parameter")

    def test_real_pytorch_dynamic_ops_wrapper(self):
        from torch._inductor.virtualized import ops

        self.assertEqual(typename(ops), "torch._inductor.virtualized.OpsWrapper")
        self.assertEqual(
            typename(ops, is_runtime=True),
            "torch._inductor.virtualized.OpsWrapper",
        )


if __name__ == "__main__":
    unittest.main()
