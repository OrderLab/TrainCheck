"""Record contracts for metadata classification, including custom type behavior."""

import unittest
import warnings
from contextlib import ExitStack
from unittest.mock import patch

import torch

from traincheck.instrumentor import dumper


class AttributeHolder:
    def __init__(self, value: object) -> None:
        self.value = value


class CustomParameter(torch.nn.Parameter):
    pass


class CustomTensor(torch.Tensor):
    pass


class ChangingClass:
    """An object allowed by Python's isinstance protocol, not an actual tensor."""

    def __init__(self, error: Exception | None = None) -> None:
        self.reads = 0
        self.error = error

    def __getattribute__(self, name: str) -> object:
        if name == "__class__":
            self.reads += 1
            if self.reads == 1:
                return object
            if self.error is not None:
                raise self.error
            return torch.nn.Parameter
        return super().__getattribute__(name)


class MetadataClassificationTests(unittest.TestCase):
    def setUp(self) -> None:
        contexts = ExitStack()
        self.addCleanup(contexts.close)
        for name in (
            "skip_attrs_due_to_errs",
            "skip_type_due_to_errs",
            "skip_type_due_to_recursion",
        ):
            contexts.enter_context(patch.object(dumper, name, {}))

    def test_callable_attributes_keep_metadata_values_and_order(self) -> None:
        def function() -> None:
            return None

        callables: tuple[object, ...] = (
            function,
            len,
            self.setUp,
            object().__str__,
            object.__str__,
            str.upper,
            dict.__dict__["fromkeys"],
        )
        for callback in callables:
            with self.subTest(callback_type=type(callback)):
                holder = AttributeHolder(callback)
                holder.__dict__["_TRAINCHECK_callback"] = callback
                holder.__dict__["name"] = "observed"
                holder.__dict__["dtype"] = torch.float32
                holder.__dict__["shape"] = torch.Size([2, 3])
                record = dumper.convert_var_to_dict(
                    holder,
                    include_tensor_data=False,
                    dump_config={
                        "value": None,
                        "name": None,
                        "_TRAINCHECK_callback": None,
                        "dtype": None,
                        "shape": None,
                    },
                )
                self.assertEqual(
                    list(record), ["name", "_TRAINCHECK_callback", "dtype", "shape"]
                )
                self.assertEqual(record["name"], "observed")
                self.assertIs(record["_TRAINCHECK_callback"], callback)
                self.assertEqual(record["dtype"], "torch.float32")
                self.assertEqual(record["shape"], (2, 3))

    def test_tensor_and_parameter_subclasses_keep_identity_and_values(self) -> None:
        flagged = torch.tensor([3.0, 4.0])
        flagged.__dict__["_is_param"] = True
        values = (
            torch.tensor([3.0, 4.0]),
            torch.tensor([3.0, 4.0]).as_subclass(CustomTensor),
            torch.nn.Parameter(torch.tensor([3.0, 4.0])),
            CustomParameter(torch.tensor([3.0, 4.0])),
            flagged,
        )
        with patch.dict(
            dumper.tensor_dump_format,
            dump_tensor_hash=False,
            dump_tensor_stats=False,
            dump_tensor_full=True,
        ):
            for value in values:
                with self.subTest(tensor_type=type(value)):
                    holder = AttributeHolder(value)
                    metadata = dumper.convert_var_to_dict(
                        holder, False, {"value": None}
                    )
                    self.assertEqual(metadata, {"_TRAINCHECK_value_ID": id(value)})
                    before = dumper.convert_var_to_dict(holder, True, {"value": None})
                    self.assertEqual(
                        before,
                        {"_TRAINCHECK_value_ID": id(value), "value": [3.0, 4.0]},
                    )
                    self.assertEqual(list(before), ["_TRAINCHECK_value_ID", "value"])
                    with torch.no_grad():
                        value.add_(1)
                    after = dumper.convert_var_to_dict(holder, True, {"value": None})
                    self.assertEqual(
                        after,
                        {"_TRAINCHECK_value_ID": id(value), "value": [4.0, 5.0]},
                    )
                    self.assertEqual(before["value"], [3.0, 4.0])

    def test_dynamic_class_keeps_parameter_fallback(self) -> None:
        value = ChangingClass()
        record = dumper.convert_var_to_dict(
            AttributeHolder(value), False, {"value": None}
        )
        self.assertEqual(record, {"_TRAINCHECK_value_ID": id(value)})
        self.assertEqual(value.reads, 2)

    def test_dynamic_class_exception_still_propagates(self) -> None:
        error = ValueError("dynamic class failed")
        value = ChangingClass(error)
        with self.assertRaises(ValueError) as caught:
            dumper.convert_var_to_dict(AttributeHolder(value), False, {"value": None})
        self.assertIs(caught.exception, error)

    def test_custom_metaclass_does_not_add_classification_side_effects(self) -> None:
        visits: list[str] = []

        class Metaclass(type):
            def __hash__(cls) -> int:
                visits.append("hash")
                if len(visits) > 1:
                    raise ValueError("unexpected repeated type hashing")
                return type.__hash__(cls)

        class Value(metaclass=Metaclass):
            pass

        self.assertEqual(
            dumper.convert_var_to_dict(
                AttributeHolder(Value()), False, {"value": None}
            ),
            {},
        )
        self.assertEqual(visits, ["hash"])

    def test_getters_warning_restoration_and_failed_attribute_backoff(self) -> None:
        visits: list[str] = []

        class Getters:
            @property
            def callback(self) -> object:
                visits.append("callback")
                warnings.warn("metadata warning", UserWarning)
                warnings.simplefilter("error", RuntimeWarning)
                return len

            @property
            def broken(self) -> object:
                visits.append("broken")
                raise ValueError("missing metadata")

            @property
            def value(self) -> int:
                visits.append("value")
                warnings.warn("later warning", RuntimeWarning)
                return 7

        with warnings.catch_warnings(record=True) as observed:
            warnings.simplefilter("always")
            filters = list(warnings.filters)
            holder = Getters()
            config = {"callback": None, "broken": None, "value": None}
            self.assertEqual(
                dumper.convert_var_to_dict(holder, False, config), {"value": 7}
            )
            self.assertEqual(
                dumper.convert_var_to_dict(holder, False, config), {"value": 7}
            )
            self.assertEqual(warnings.filters, filters)
            self.assertEqual(observed, [])
        self.assertEqual(visits, ["callback", "broken", "value", "callback", "value"])

    def test_dict_subclass_keeps_dynamic_keys_and_missing_fields(self) -> None:
        class Metadata(dict[str, object]):
            pass

        value = Metadata(first=1, callback=len)
        self.assertEqual(dumper.convert_var_to_dict(value, False), {"first": 1})
        value["second"] = 2
        result = dumper.convert_var_to_dict(value, False)
        self.assertEqual(result, {"first": 1, "second": 2})
        self.assertEqual(list(result), ["first", "second"])
        self.assertEqual(
            dumper.convert_var_to_dict(value, False, {"missing": None}), {}
        )


if __name__ == "__main__":
    unittest.main()
