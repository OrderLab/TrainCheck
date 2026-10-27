"""Immutable namespace caching must preserve live application class lookup."""

import types
import unittest
from collections.abc import Mapping
from unittest.mock import patch

import torch

from traincheck.instrumentor import dumper, metadata_methods


class ImmutableMetadataSuffixTests(unittest.TestCase):
    def test_metaclass_cannot_spoof_immutable_type_flags(self) -> None:
        visits: list[str] = []

        class SpoofedFlags(type):
            @property
            def __flags__(cls) -> int:
                visits.append("flags")
                return 1 << 8

        class Mutable(metaclass=SpoofedFlags):
            marker = 1

        mro, namespaces = metadata_methods._class_layout(Mutable)
        mutable, suffix = metadata_methods._immutable_suffix(mro, namespaces)
        self.assertEqual(visits, [])
        self.assertIs(mutable[0], namespaces[0])
        self.assertNotIn("marker", suffix)
        Mutable.marker = 2
        self.assertEqual(mutable[0]["marker"], 2)

    def test_mutable_middle_and_head_remain_live(self) -> None:
        class Middle(list[object]):
            middle_marker = 1

        class Head(Middle):
            head_marker = 2

        mro, namespaces = metadata_methods._class_layout(Head)
        mutable, suffix = metadata_methods._immutable_suffix(mro, namespaces)
        self.assertIs(mutable[0], namespaces[0])
        self.assertIs(mutable[1], namespaces[1])
        self.assertNotIn("middle_marker", suffix)
        self.assertNotIn("head_marker", suffix)
        Middle.middle_marker = 3
        self.assertEqual(mutable[1]["middle_marker"], 3)
        setattr(Head, "inserted", 4)
        self.assertEqual(mutable[0]["inserted"], 4)
        delattr(Head, "inserted")
        self.assertNotIn("inserted", mutable[0])

    def assert_uncached_binding(
        self,
        mutable: tuple[Mapping[object, object], ...],
        suffix: Mapping[str, object],
    ) -> None:
        self.assertIs(mutable, metadata_methods._TENSOR_NAMESPACES)
        self.assertEqual(suffix, {})
        tensor = torch.ones(1)
        with patch.object(metadata_methods, "_TENSOR_MUTABLE", mutable):
            with patch.object(metadata_methods, "_TENSOR_SUFFIX", suffix):
                binding = metadata_methods.tensor_method_binding(tensor, "add")
                assert binding is not None
                self.assertIs(binding.__self__, tensor)

    def test_other_runtime_uses_original_live_namespaces(self) -> None:
        with patch.object(
            metadata_methods.platform, "python_implementation", return_value="PyPy"
        ):
            mutable, suffix = metadata_methods._immutable_suffix(
                metadata_methods._TENSOR_MRO, metadata_methods._TENSOR_NAMESPACES
            )
        self.assert_uncached_binding(mutable, suffix)

    def test_absent_native_flags_uses_original_live_namespaces(self) -> None:
        namespace: dict[str, object] = dict(vars(type))
        del namespace["__flags__"]
        with patch.object(metadata_methods, "_TYPE_NAMESPACE", namespace):
            mutable, suffix = metadata_methods._immutable_suffix(
                metadata_methods._TENSOR_MRO, metadata_methods._TENSOR_NAMESPACES
            )
        self.assert_uncached_binding(mutable, suffix)

    def test_malformed_present_flags_is_not_absent(self) -> None:
        visits: list[object] = []

        def application_flags(value: object) -> int:
            visits.append(value)
            return 1 << 8

        for malformed in (None, object(), property(application_flags)):
            with self.subTest(metadata_type=type(malformed)):
                namespace: dict[str, object] = dict(vars(type))
                namespace["__flags__"] = malformed
                with patch.object(metadata_methods, "_TYPE_NAMESPACE", namespace):
                    with self.assertRaises(AssertionError):
                        metadata_methods._immutable_suffix(
                            metadata_methods._TENSOR_MRO,
                            metadata_methods._TENSOR_NAMESPACES,
                        )
        self.assertEqual(visits, [])

    def test_replacing_and_restoring_inherited_method_is_observed(self) -> None:
        tensor = torch.ones(1)
        original = metadata_methods.tensor_method_binding(tensor, "add")
        assert original is not None
        self.assertIs(original.__self__, tensor)

        def replacement(value: torch.Tensor) -> int:
            return 37

        with patch.object(torch.Tensor, "add", property(replacement)):
            self.assertIsNone(metadata_methods.tensor_method_binding(tensor, "add"))
            self.assertEqual(
                dumper.convert_var_to_dict(tensor, False, {"add": None}), {"add": 37}
            )
        restored = metadata_methods.tensor_method_binding(tensor, "add")
        assert restored is not None
        self.assertIs(restored.__self__, tensor)
        self.assertIs(type(restored), type(original))

    def test_new_head_function_binding_and_deletion_are_observed(self) -> None:
        tensor = torch.ones(1)

        def inserted(value: torch.Tensor) -> int:
            return 41

        name = "traincheck_native_suffix_test_method"
        self.assertIsNone(metadata_methods.tensor_method_binding(tensor, name))
        with patch.object(torch.Tensor, name, inserted, create=True):
            binding = metadata_methods.tensor_method_binding(tensor, name)
            assert isinstance(binding, types.MethodType)
            self.assertIs(binding.__func__, inserted)
            self.assertIs(binding.__self__, tensor)
        self.assertIsNone(metadata_methods.tensor_method_binding(tensor, name))

    def test_earlier_property_can_shadow_cached_base_method(self) -> None:
        tensor = torch.ones(1)

        def replacement(value: torch.Tensor) -> int:
            return 43

        def prepare(value: torch.Tensor) -> int:
            setattr(torch.Tensor, "add", property(replacement))
            return 1

        with patch.object(torch.Tensor, "add", torch.Tensor.add):
            with patch.object(torch.Tensor, "prepare", property(prepare), create=True):
                result = dumper.convert_var_to_dict(
                    tensor, False, {"prepare": None, "add": None}
                )
        self.assertEqual(result, {"prepare": 1, "add": 43})


if __name__ == "__main__":
    unittest.main()
