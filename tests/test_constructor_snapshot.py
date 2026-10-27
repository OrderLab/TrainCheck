"""Constructor tracing must not inspect an uninitialized receiver dynamically."""

import functools
import inspect
import json
import logging
import random
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Callable, NoReturn
from unittest.mock import patch

import torch

import traincheck.config.config as config
from traincheck.instrumentor import dumper, tracer
from traincheck.instrumentor.constructor_snapshot import constructor_receiver_snapshot
from traincheck.trace.trace_pandas import read_trace_file_Pandas
from traincheck.trace.utils import bind_args_kwargs_to_signature

PACKET_TYPE = "torch._ops.OpOverloadPacket"
SNAPSHOT_FIELD = "_TRAINCHECK_CONSTRUCTOR_SNAPSHOT"


class UnexpectedGetter(BaseException):
    """Bypass legacy serializer exception swallowing to fail the regression fast."""


class ConstructorSnapshotTests(unittest.TestCase):
    def setUp(self) -> None:
        self.records: list[dict[str, object]] = []
        destination = SimpleNamespace(add_trace=self.capture)
        patches = ExitStack()
        self.addCleanup(patches.close)
        patches.enter_context(
            patch.object(
                dumper,
                "THREAD_DATA",
                SimpleNamespace(
                    api_trace_buffer=destination, var_trace_buffer=destination
                ),
            )
        )
        patches.enter_context(patch.object(config, "DISABLE_WRAPPER", False))
        patches.enter_context(patch.object(config, "INSTR_OPTS", None))
        patches.enter_context(patch.object(tracer, "IS_INSTRUMENTING", False))

    def capture(self, record: dict[str, object]) -> None:
        value: object = json.loads(dumper.serialize(record))
        assert isinstance(value, dict)
        validated: dict[str, object] = {}
        for key, item in value.items():
            assert isinstance(key, str)
            validated[key] = item
        self.records.append(validated)

    def wrap(
        self, constructor: Callable[..., object], *, scan_proxy: bool = False
    ) -> Callable[..., object]:
        return tracer.wrapper(
            constructor,
            is_bound_method=True,
            scan_proxy_in_args=scan_proxy,
            dump_stack_trace=False,
            handle_proxy=False,
            is_constructor=True,
        )

    def snapshot(self, encoded: dict[str, object]) -> dict[str, object]:
        self.assertEqual(len(encoded), 1)
        packet = next(iter(encoded.values()))
        assert isinstance(packet, dict)
        snapshot = packet[SNAPSHOT_FIELD]
        assert isinstance(snapshot, dict)
        result: dict[str, object] = {}
        for key, value in snapshot.items():
            assert isinstance(key, str)
            result[key] = value
        return result

    def test_constructor_keeps_pre_post_and_other_arguments_without_getter_reads(
        self,
    ) -> None:
        source = torch.ops.aten.add
        packet = object.__new__(torch._ops.OpOverloadPacket)
        object.__getattribute__(packet, "__dict__")["existing_value"] = 7

        def forbidden_getter(receiver: object) -> object:
            raise UnexpectedGetter("receiver property accessed before initialization")

        with patch.object(
            torch._ops.OpOverloadPacket, "op", property(forbidden_getter)
        ):
            result = self.wrap(torch._ops.OpOverloadPacket.__init__)(
                packet, "aten::add", "add", source._op, source._overload_names
            )
        self.assertIsNone(result)
        self.assertEqual(str(packet), "aten.add")
        self.assertEqual(packet.existing_value, 7)
        self.assertEqual(
            [r["type"] for r in self.records],
            [tracer.TraceLineType.FUNC_CALL_PRE, tracer.TraceLineType.FUNC_CALL_POST],
        )
        pre, post = self.records
        self.assertEqual(pre["obj_id"], id(packet))
        self.assertEqual(pre["func_call_id"], post["func_call_id"])
        args = pre["args"]
        assert isinstance(args, dict)
        self.assertEqual(args["1"], dumper.var_to_serializable("aten::add"))
        self.assertEqual(args["2"], dumper.var_to_serializable("add"))
        self.assertEqual(args["3"], dumper.var_to_serializable(source._op))
        self.assertEqual(args["4"], dumper.var_to_serializable(source._overload_names))
        snapshot = self.snapshot(args["0"])
        self.assertEqual(snapshot["phase"], "before_init")
        self.assertEqual(snapshot["coverage"], "stored_attributes_only")
        self.assertEqual(snapshot["object_id"], id(packet))
        self.assertEqual(
            snapshot["attributes"], {"existing_value": {"kind": "value", "value": 7}}
        )

        # Existing readers preserve the marker and bind every original argument.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace_API.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in self.records) + "\n")
            trace = read_trace_file_Pandas(
                str(path), truncate_incomplete_func_calls=False
            )
            call_id = pre["func_call_id"]
            assert isinstance(call_id, str)
            loaded = trace.get_pre_func_call_record(call_id)
            bound = bind_args_kwargs_to_signature(
                loaded["args"],
                loaded["kwargs"],
                inspect.signature(torch._ops.OpOverloadPacket.__init__),
            )
            self.assertEqual(bound.get_arg("qualified_op_name"), args["1"])
            self.assertEqual(
                bound.get_arg("self")[PACKET_TYPE][SNAPSHOT_FIELD]["coverage"],
                "stored_attributes_only",
            )

    def test_constructor_failure_retains_pre_exception_and_original_error(self) -> None:
        packet = object.__new__(torch._ops.OpOverloadPacket)
        original_error = RuntimeError("constructor failed")
        calls: list[object] = []

        @functools.wraps(torch._ops.OpOverloadPacket.__init__)
        def failing(receiver: object, value: int) -> None:
            calls.extend([receiver, value])
            raise original_error

        with self.assertRaises(RuntimeError) as result:
            self.wrap(failing)(packet, 9)
        self.assertIs(result.exception, original_error)
        self.assertIs(calls[0], packet)
        self.assertEqual(calls[1], 9)
        self.assertEqual(
            [r["type"] for r in self.records],
            [
                tracer.TraceLineType.FUNC_CALL_PRE,
                tracer.TraceLineType.FUNC_CALL_POST_EXCEPTION,
            ],
        )
        self.assertEqual(self.records[1]["exception_msg"], "constructor failed")
        self.assertEqual(
            self.records[0]["func_call_id"], self.records[1]["func_call_id"]
        )

    def test_initialized_receiver_is_snapshotted_on_reinitialization(self) -> None:
        source = torch.ops.aten.add
        packet = torch._ops.OpOverloadPacket(
            "aten::add", "add", source._op, source._overload_names
        )
        snapshot = self.snapshot(constructor_receiver_snapshot(packet))
        self.assertEqual(snapshot["object_id"], id(packet))
        attributes = snapshot["attributes"]
        assert isinstance(attributes, dict)
        self.assertEqual(
            attributes["_qualified_op_name"], {"kind": "value", "value": "aten::add"}
        )
        arguments = (packet, "name", 3)
        ordinary = tracer.to_dict_args_kwargs(arguments, {"flag": True})
        constructor = tracer.to_dict_args_kwargs(
            arguments, {"flag": True}, is_constructor=True
        )
        self.assertEqual(ordinary["kwargs"], constructor["kwargs"])
        for index in (1, 2):
            self.assertEqual(ordinary["args"][index], constructor["args"][index])
        self.assertNotEqual(ordinary["args"][0], constructor["args"][0])

    def test_stored_none_false_and_opaque_references_are_distinct(self) -> None:
        packet = object.__new__(torch._ops.OpOverloadPacket)
        opaque = object()
        object.__getattribute__(packet, "__dict__").update(
            {"unset": None, "flag": False, "opaque": opaque}
        )
        snapshot = self.snapshot(constructor_receiver_snapshot(packet))
        self.assertEqual(
            snapshot["attributes"],
            {
                "unset": {"kind": "value", "value": None},
                "flag": {"kind": "value", "value": False},
                "opaque": {
                    "kind": "object_reference",
                    "object_id": id(opaque),
                    "type": "builtins.object",
                },
            },
        )

    def test_selective_config_retains_explicit_unavailable_fields(self) -> None:
        packet = object.__new__(torch._ops.OpOverloadPacket)
        object.__getattribute__(packet, "__dict__").update({"first": 1, "second": 2})
        encoded = tracer.to_dict_args_kwargs(
            (packet, "unchanged"),
            {},
            {"0": {"first": True, "op": True}},
            is_constructor=True,
        )
        self.assertEqual(set(encoded["args"]), {0})
        snapshot = self.snapshot(encoded["args"][0])
        self.assertEqual(snapshot["requested_attributes"], ["first", "op"])
        self.assertEqual(snapshot["unavailable_attributes"], ["op"])
        self.assertEqual(
            snapshot["attributes"], {"first": {"kind": "value", "value": 1}}
        )

    def test_malformed_state_and_selector_are_rejected(self) -> None:
        packet = object.__new__(torch._ops.OpOverloadPacket)
        for config_value in (1, {1: True}):
            with self.subTest(config_value=config_value), self.assertRaises(TypeError):
                constructor_receiver_snapshot(packet, config_value)
        object.__getattribute__(packet, "__dict__")[1] = "malformed name"
        with self.assertRaisesRegex(TypeError, "attribute names must be strings"):
            constructor_receiver_snapshot(packet)

    def test_metaclass_properties_are_not_evaluated_for_receiver_or_values(
        self,
    ) -> None:
        def forbidden(receiver: object) -> object:
            raise UnexpectedGetter("metaclass property executed")

        metaclass = type(
            "MetadataFailure",
            (type,),
            {
                "__module__": property(forbidden),
                "__dict__": property(forbidden),
                "__mro__": property(forbidden),
            },
        )
        receiver_type = metaclass("Receiver", (), {"__module__": "fixture"})
        opaque_type = metaclass("Opaque", (), {"__module__": "fixture"})
        receiver = receiver_type()
        opaque = opaque_type()
        object.__setattr__(receiver, "opaque", opaque)
        encoded = constructor_receiver_snapshot(receiver)
        self.assertEqual(set(encoded), {"fixture.Receiver"})
        self.assertEqual(
            self.snapshot(encoded)["attributes"],
            {
                "opaque": {
                    "kind": "object_reference",
                    "object_id": id(opaque),
                    "type": "fixture.Opaque",
                }
            },
        )

    def test_nonfinite_stored_values_survive_json_distinct_from_none(self) -> None:
        packet = object.__new__(torch._ops.OpOverloadPacket)
        object.__getattribute__(packet, "__dict__").update(
            {
                "missing": None,
                "nan": float("nan"),
                "positive": float("inf"),
                "negative": float("-inf"),
                "finite": 1.5,
            }
        )
        encoded = constructor_receiver_snapshot(packet)
        decoded: object = json.loads(dumper.serialize(encoded))
        assert isinstance(decoded, dict)
        self.assertEqual(
            self.snapshot(decoded)["attributes"],
            {
                "missing": {"kind": "value", "value": None},
                "nan": {"kind": "nonfinite_float", "value": "nan"},
                "positive": {"kind": "nonfinite_float", "value": "+inf"},
                "negative": {"kind": "nonfinite_float", "value": "-inf"},
                "finite": {"kind": "value", "value": 1.5},
            },
        )

    def test_proxy_scanning_does_not_inspect_constructor_receiver(self) -> None:
        class Hostile:
            def __init__(self, value: int) -> None:
                object.__setattr__(self, "value", value)

            def __getattribute__(self, name: str) -> object:
                raise UnexpectedGetter("receiver dynamically inspected")

        receiver = object.__new__(Hostile)
        wrapped = tracer.wrapper(
            Hostile.__init__,
            is_bound_method=True,
            scan_proxy_in_args=True,
            dump_stack_trace=False,
            handle_proxy=True,
            is_constructor=True,
        )
        wrapped(receiver, 5)
        self.assertEqual(object.__getattribute__(receiver, "value"), 5)
        self.assertEqual(len(self.records), 2)
        args = self.records[0]["args"]
        assert isinstance(args, dict)
        self.assertEqual(self.snapshot(args["0"])["object_id"], id(receiver))

    def test_generic_constructor_skips_properties_and_keeps_application_effects(
        self,
    ) -> None:
        calls: list[int] = []

        class Model:
            def __init__(self, width: int) -> None:
                calls.append(width)
                self.width = width

            @property
            def initialized_width(self) -> int:
                raise UnexpectedGetter("constructor property was evaluated")

        receiver = object.__new__(Model)
        result = self.wrap(Model.__init__)(receiver, 12)
        self.assertIsNone(result)
        self.assertEqual(receiver.width, 12)
        self.assertEqual(calls, [12])
        self.assertEqual(len(self.records), 2)
        args = self.records[0]["args"]
        assert isinstance(args, dict)
        snapshot = self.snapshot(args["0"])
        self.assertEqual(snapshot["attributes"], {})
        self.assertEqual(snapshot["dictionary_status"], "captured")
        self.assertEqual(args["1"], dumper.var_to_serializable(12))

    def test_snapshot_does_not_call_dynamic_lookup_dir_or_repr(self) -> None:
        class Hostile:
            def __getattribute__(self, name: str) -> object:
                raise UnexpectedGetter("__getattribute__ executed")

            def __getattr__(self, name: str) -> object:
                raise UnexpectedGetter("__getattr__ executed")

            def __dir__(self) -> list[str]:
                raise UnexpectedGetter("__dir__ executed")

            def __repr__(self) -> str:
                raise UnexpectedGetter("__repr__ executed")

        receiver = object.__new__(Hostile)
        object.__setattr__(receiver, "value", 4)
        snapshot = self.snapshot(constructor_receiver_snapshot(receiver))
        self.assertEqual(snapshot["object_id"], id(receiver))
        self.assertEqual(
            snapshot["attributes"], {"value": {"kind": "value", "value": 4}}
        )

    def test_shadowed_dictionary_accessor_is_explicitly_unavailable(self) -> None:
        def forbidden(receiver: object) -> object:
            raise UnexpectedGetter("__dict__ property executed")

        receiver_type = type(
            "ShadowedDictionary", (), {"__dict__": property(forbidden)}
        )
        receiver = receiver_type()
        object.__setattr__(receiver, "stored", 5)
        snapshot = self.snapshot(
            constructor_receiver_snapshot(receiver, {"stored": True})
        )
        self.assertEqual(snapshot["dictionary_status"], "unavailable")
        self.assertEqual(snapshot["attributes"], {})
        self.assertEqual(snapshot["unavailable_attributes"], ["stored"])

    def test_inherited_native_dictionary_accessor_bypasses_shadowing_property(
        self,
    ) -> None:
        class Base:
            pass

        def forbidden(receiver: object) -> object:
            raise UnexpectedGetter("shadowing __dict__ property executed")

        receiver_type = type("Shadow", (Base,), {"__dict__": property(forbidden)})
        receiver = receiver_type()
        object.__setattr__(receiver, "stored", 5)
        snapshot = self.snapshot(constructor_receiver_snapshot(receiver))
        self.assertEqual(snapshot["dictionary_status"], "captured")
        self.assertEqual(
            snapshot["attributes"], {"stored": {"kind": "value", "value": 5}}
        )

    def test_native_dict_iteration_does_not_call_dict_subclass_override(self) -> None:
        class HostileDict(dict[str, object]):
            def items(self) -> NoReturn:
                raise UnexpectedGetter("overridden dictionary items executed")

        class Model:
            pass

        receiver = Model()
        object.__setattr__(receiver, "__dict__", HostileDict({"value": 8}))
        snapshot = self.snapshot(constructor_receiver_snapshot(receiver))
        self.assertEqual(
            snapshot["attributes"], {"value": {"kind": "value", "value": 8}}
        )

    def test_metaclass_equality_is_not_used_for_scalar_classification(self) -> None:
        class HostileMeta(type):
            def __eq__(self, other: object) -> bool:
                raise UnexpectedGetter("metaclass equality executed")

        class Opaque(metaclass=HostileMeta):
            pass

        receiver = SimpleNamespace(value=Opaque())
        snapshot = self.snapshot(constructor_receiver_snapshot(receiver))
        attributes = snapshot["attributes"]
        assert isinstance(attributes, dict)
        value = attributes["value"]
        assert isinstance(value, dict)
        self.assertEqual(value["kind"], "object_reference")
        self.assertEqual(value["object_id"], id(receiver.value))

    def test_slots_preserve_populated_unset_inherited_and_private_storage(self) -> None:
        class Base:
            __slots__ = ("base", "__private")

        class Child(Base):
            __slots__ = ("child", "pending")

        receiver = object.__new__(Child)
        object.__setattr__(receiver, "base", 3)
        object.__setattr__(receiver, "_Base__private", 9)
        object.__setattr__(receiver, "child", False)
        snapshot = self.snapshot(constructor_receiver_snapshot(receiver))
        self.assertEqual(snapshot["dictionary_status"], "absent")
        self.assertEqual(snapshot["attributes"], {})
        slots = snapshot["slots"]
        assert isinstance(slots, list)
        self.assertCountEqual(
            slots,
            [
                {
                    "owner_type": f"{Base.__module__}.{Base.__qualname__}",
                    "name": "base",
                    "value": {"kind": "value", "value": 3},
                },
                {
                    "owner_type": f"{Base.__module__}.{Base.__qualname__}",
                    "name": "_Base__private",
                    "value": {"kind": "value", "value": 9},
                },
                {
                    "owner_type": f"{Child.__module__}.{Child.__qualname__}",
                    "name": "child",
                    "value": {"kind": "value", "value": False},
                },
            ],
        )
        self.assertEqual(
            snapshot["unset_slots"],
            [
                {
                    "owner_type": f"{Child.__module__}.{Child.__qualname__}",
                    "name": "pending",
                }
            ],
        )

    def test_duplicate_slots_and_dictionary_field_are_not_overwritten(self) -> None:
        class Base:
            __slots__ = ("value",)

        class Child(Base):
            __slots__ = ("value", "__dict__")

        receiver = object.__new__(Child)
        Base.__dict__["value"].__set__(receiver, 1)
        Child.__dict__["value"].__set__(receiver, 2)
        object.__getattribute__(receiver, "__dict__")["value"] = 3
        snapshot = self.snapshot(
            constructor_receiver_snapshot(receiver, {"value": True})
        )
        self.assertEqual(
            snapshot["attributes"], {"value": {"kind": "value", "value": 3}}
        )
        slots = snapshot["slots"]
        assert isinstance(slots, list)
        self.assertCountEqual(
            slots,
            [
                {
                    "owner_type": f"{Base.__module__}.{Base.__qualname__}",
                    "name": "value",
                    "value": {"kind": "value", "value": 1},
                },
                {
                    "owner_type": f"{Child.__module__}.{Child.__qualname__}",
                    "name": "value",
                    "value": {"kind": "value", "value": 2},
                },
            ],
        )
        self.assertEqual(snapshot["unavailable_attributes"], [])

    def test_slot_storage_is_read_without_evaluating_shadowing_property(self) -> None:
        class Base:
            __slots__ = ("value",)

        def forbidden(receiver: object) -> object:
            raise UnexpectedGetter("slot shadowing property executed")

        receiver_type = type(
            "Shadow", (Base,), {"__slots__": (), "value": property(forbidden)}
        )
        receiver: object = object.__new__(receiver_type)
        Base.__dict__["value"].__set__(receiver, 7)
        snapshot = self.snapshot(constructor_receiver_snapshot(receiver))
        self.assertEqual(
            snapshot["slots"],
            [
                {
                    "owner_type": f"{Base.__module__}.{Base.__qualname__}",
                    "name": "value",
                    "value": {"kind": "value", "value": 7},
                }
            ],
        )

    def test_selective_unset_slot_is_distinct_from_missing_and_stored_none(
        self,
    ) -> None:
        class Model:
            __slots__ = ("present", "pending", "excluded")

        receiver = object.__new__(Model)
        object.__setattr__(receiver, "present", None)
        object.__setattr__(receiver, "excluded", 4)
        snapshot = self.snapshot(
            constructor_receiver_snapshot(
                receiver, {"present": True, "pending": True, "unknown": True}
            )
        )
        self.assertEqual(snapshot["unavailable_attributes"], ["pending", "unknown"])
        self.assertEqual(
            snapshot["slots"],
            [
                {
                    "owner_type": f"{Model.__module__}.{Model.__qualname__}",
                    "name": "present",
                    "value": {"kind": "value", "value": None},
                }
            ],
        )
        self.assertEqual(
            snapshot["unset_slots"],
            [
                {
                    "owner_type": f"{Model.__module__}.{Model.__qualname__}",
                    "name": "pending",
                }
            ],
        )

    def test_empty_selector_and_absent_dictionary_are_distinct(self) -> None:
        receiver = SimpleNamespace(value=1)
        selected = self.snapshot(constructor_receiver_snapshot(receiver, {}))
        self.assertEqual(selected["dictionary_status"], "captured")
        self.assertEqual(selected["requested_attributes"], [])
        self.assertEqual(selected["attributes"], {})
        absent = self.snapshot(constructor_receiver_snapshot(object()))
        self.assertEqual(absent["dictionary_status"], "absent")
        self.assertIsNone(absent["requested_attributes"])
        self.assertEqual(absent["attributes"], {})
        self.assertEqual(absent["slots"], [])

    def test_constructor_preserves_rng_and_does_not_mutate_pre_snapshot(self) -> None:
        class Model:
            def __init__(self, initial: int) -> None:
                self.initial = initial
                self.python_draw = random.random()
                self.torch_draw = float(torch.rand(()))

            @property
            def destructive(self) -> float:
                return random.random() + float(torch.rand(()))

        python_state = random.getstate()
        torch_state = torch.get_rng_state().clone()
        self.addCleanup(random.setstate, python_state)
        self.addCleanup(torch.set_rng_state, torch_state)
        plain = Model(2)
        expected_python = random.getstate()
        expected_torch = torch.get_rng_state().clone()
        random.setstate(python_state)
        torch.set_rng_state(torch_state)
        receiver = object.__new__(Model)
        object.__setattr__(receiver, "initial", 1)
        self.wrap(Model.__init__)(receiver, 2)
        self.assertEqual(receiver.__dict__, plain.__dict__)
        self.assertEqual(random.getstate(), expected_python)
        self.assertTrue(torch.equal(torch.get_rng_state(), expected_torch))
        self.assertEqual(len(self.records), 2)
        args = self.records[0]["args"]
        assert isinstance(args, dict)
        self.assertEqual(
            self.snapshot(args["0"])["attributes"],
            {"initial": {"kind": "value", "value": 1}},
        )

    def test_base_constructor_records_partially_initialized_subclass_state(
        self,
    ) -> None:
        class Base:
            def __init__(self) -> None:
                self.base = 9

        class Child(Base):
            def __init__(self) -> None:
                self.child = 7
                super().__init__()

        with (
            patch.object(Base, "__init__", self.wrap(Base.__init__)),
            patch.object(Child, "__init__", self.wrap(Child.__init__)),
        ):
            receiver = Child()
        self.assertEqual(receiver.child, 7)
        self.assertEqual(receiver.base, 9)
        self.assertEqual(
            [row["type"] for row in self.records],
            [
                tracer.TraceLineType.FUNC_CALL_PRE,
                tracer.TraceLineType.FUNC_CALL_PRE,
                tracer.TraceLineType.FUNC_CALL_POST,
                tracer.TraceLineType.FUNC_CALL_POST,
            ],
        )
        outer_args = self.records[0]["args"]
        inner_args = self.records[1]["args"]
        assert isinstance(outer_args, dict)
        assert isinstance(inner_args, dict)
        self.assertEqual(self.snapshot(outer_args["0"])["attributes"], {})
        self.assertEqual(
            self.snapshot(inner_args["0"])["attributes"],
            {"child": {"kind": "value", "value": 7}},
        )
        self.assertEqual(
            self.records[0]["func_call_id"], self.records[3]["func_call_id"]
        )
        self.assertEqual(
            self.records[1]["func_call_id"], self.records[2]["func_call_id"]
        )

    def test_malformed_static_type_metadata_is_rejected_without_formatting(
        self,
    ) -> None:
        class MisleadingString(str):
            def __str__(self) -> str:
                raise UnexpectedGetter("module string subclass formatted")

        for module in (1, MisleadingString("fixture")):
            with self.subTest(module_type=type(module).__name__):
                receiver_type = type("BadModule", (), {"__module__": module})
                with self.assertRaises(TypeError):
                    constructor_receiver_snapshot(receiver_type())

    def test_native_constructor_without_signature_still_snapshots_and_executes(
        self,
    ) -> None:
        receiver: dict[str, int] = {}
        with patch.object(
            tracer.inspect,
            "signature",
            side_effect=ValueError("native signature unavailable"),
        ):
            wrapped = self.wrap(dict.__init__)
        self.assertIsNone(wrapped(receiver, {"value": 7}))
        self.assertEqual(receiver, {"value": 7})
        self.assertEqual(len(self.records), 2)
        self.assertEqual(self.records[0]["obj_id"], id(receiver))
        args = self.records[0]["args"]
        assert isinstance(args, dict)
        snapshot = self.snapshot(args["0"])
        self.assertEqual(snapshot["object_id"], id(receiver))
        self.assertEqual(snapshot["dictionary_status"], "absent")
        self.assertEqual(
            self.records[0]["func_call_id"], self.records[1]["func_call_id"]
        )

    def test_malformed_python_constructor_signature_is_not_silently_ignored(
        self,
    ) -> None:
        def initialize(receiver: object) -> None:
            raise AssertionError("malformed signature must fail during wrapping")

        with patch.object(initialize, "__signature__", object(), create=True):
            with self.assertRaises(TypeError):
                self.wrap(initialize)
        original_error = ValueError("malformed Python signature")
        with patch.object(tracer.inspect, "signature", side_effect=original_error):
            with self.assertRaises(ValueError) as caught:
                self.wrap(initialize)
        self.assertIs(caught.exception, original_error)
        self.assertEqual(self.records, [])

    def test_keyword_receiver_preserves_argument_order_and_identity(self) -> None:
        class Model:
            def __init__(receiver, value: int, *, enabled: bool) -> None:
                object.__setattr__(receiver, "value", value)
                object.__setattr__(receiver, "enabled", enabled)

            def __getattribute__(self, name: str) -> object:
                raise UnexpectedGetter("keyword receiver dynamically inspected")

        receiver = object.__new__(Model)
        with patch.object(tracer, "is_funcs_to_be_unproxied", return_value=True):
            wrapped = tracer.wrapper(
                Model.__init__,
                is_bound_method=True,
                scan_proxy_in_args=True,
                dump_stack_trace=False,
                handle_proxy=True,
                is_constructor=True,
            )
        wrapped(value=5, receiver=receiver, enabled=False)
        self.assertEqual(object.__getattribute__(receiver, "value"), 5)
        self.assertIs(object.__getattribute__(receiver, "enabled"), False)
        self.assertEqual(len(self.records), 2)
        pre = self.records[0]
        self.assertEqual(pre["obj_id"], id(receiver))
        self.assertEqual(pre["args"], {})
        kwargs = pre["kwargs"]
        assert isinstance(kwargs, dict)
        self.assertEqual(list(kwargs), ["value", "receiver", "enabled"])
        self.assertEqual(kwargs["value"], dumper.var_to_serializable(5))
        self.assertEqual(kwargs["enabled"], dumper.var_to_serializable(False))
        self.assertEqual(self.snapshot(kwargs["receiver"])["object_id"], id(receiver))

    def test_keyword_metaclass_receiver_captures_class_namespace(self) -> None:
        class Meta(type):
            def __init__(
                cls,
                name: str,
                bases: tuple[type[object], ...],
                namespace: dict[str, object],
            ) -> None:
                type.__init__(cls, name, bases, namespace)
                type.__setattr__(cls, "initialized", True)

            @property
            def dangerous(cls) -> object:
                raise UnexpectedGetter("metaclass receiver property evaluated")

        receiver = type.__new__(
            Meta, "Subject", (), {"__module__": "fixture", "seed": 3}
        )
        self.wrap(Meta.__init__, scan_proxy=True)(
            cls=receiver, name="Subject", bases=(), namespace={"seed": 3}
        )
        self.assertIs(type.__getattribute__(receiver, "initialized"), True)
        self.assertEqual(len(self.records), 2)
        self.assertEqual(self.records[0]["obj_id"], id(receiver))
        kwargs = self.records[0]["kwargs"]
        assert isinstance(kwargs, dict)
        snapshot = self.snapshot(kwargs["cls"])
        self.assertEqual(snapshot["dictionary_status"], "captured")
        attributes = snapshot["attributes"]
        assert isinstance(attributes, dict)
        self.assertEqual(attributes["seed"], {"kind": "value", "value": 3})
        self.assertNotIn("initialized", attributes)

    def test_missing_receiver_keeps_original_typeerror_and_exception_pair(self) -> None:
        class Model:
            def __init__(receiver, value: int) -> None:
                raise AssertionError("missing receiver must prevent constructor body")

        # The callable interface deliberately permits an invalid call so its
        # original runtime TypeError can be compared with the traced call.
        original: Callable[..., object] = Model.__init__
        with self.assertRaises(TypeError) as plain:
            original(value=5)
        with self.assertRaises(TypeError) as traced:
            self.wrap(original, scan_proxy=True)(value=5)
        self.assertEqual(str(traced.exception), str(plain.exception))
        self.assertEqual(
            [row["type"] for row in self.records],
            [
                tracer.TraceLineType.FUNC_CALL_PRE,
                tracer.TraceLineType.FUNC_CALL_POST_EXCEPTION,
            ],
        )
        self.assertIsNone(self.records[0]["obj_id"])
        self.assertEqual(
            self.records[0]["func_call_id"], self.records[1]["func_call_id"]
        )
        self.assertEqual(self.records[1]["exception_msg"], str(plain.exception))

    def test_metaclass_initializer_snapshots_class_namespace_without_getters(
        self,
    ) -> None:
        class Meta(type):
            def __init__(
                cls,
                name: str,
                bases: tuple[type[object], ...],
                namespace: dict[str, object],
            ) -> None:
                type.__init__(cls, name, bases, namespace)
                type.__setattr__(cls, "initialized", True)

            def __getattribute__(cls, name: str) -> object:
                raise UnexpectedGetter("class receiver dynamically inspected")

            @property
            def dangerous(cls) -> object:
                raise UnexpectedGetter("class receiver property evaluated")

        with (
            patch.object(config, "MODEL_TRACKER_STYLE", "subclass"),
            patch.object(
                tracer,
                "get_instrumentation_logger_for_process",
                return_value=logging.getLogger(__name__),
            ),
        ):
            instrumentor = tracer.Instrumentor(
                Meta,
                scan_proxy_in_args=True,
                use_full_instr=True,
                sampling_interval=0,
                warm_up_steps=0,
            )
            instrumentor._instrument_module(Meta, set(), True, 0)
        receiver = Meta("Subject", (), {"__module__": "fixture", "seed": 8})
        self.assertIs(type.__getattribute__(receiver, "initialized"), True)
        self.assertEqual(len(self.records), 2)
        self.assertEqual(self.records[0]["obj_id"], id(receiver))
        args = self.records[0]["args"]
        assert isinstance(args, dict)
        snapshot = self.snapshot(args["0"])
        self.assertEqual(snapshot["dictionary_status"], "captured")
        attributes = snapshot["attributes"]
        assert isinstance(attributes, dict)
        self.assertEqual(attributes["seed"], {"kind": "value", "value": 8})
        self.assertNotIn("initialized", attributes)
        self.assertEqual(
            self.records[0]["func_call_id"], self.records[1]["func_call_id"]
        )

    def test_instrumenter_distinguishes_class_initializer_from_same_named_function(
        self,
    ) -> None:
        class Model:
            def __init__(receiver, value: int) -> None:
                receiver.value = value

            @property
            def dangerous(self) -> object:
                raise UnexpectedGetter("class constructor property read")

        def __init__(self: object) -> None:
            return None

        module = ModuleType(__name__)
        setattr(module, "Model", Model)
        setattr(module, "__init__", __init__)
        with (
            patch.object(config, "MODEL_TRACKER_STYLE", "subclass"),
            patch.object(
                tracer,
                "get_instrumentation_logger_for_process",
                return_value=logging.getLogger(__name__),
            ),
        ):
            instrumentor = tracer.Instrumentor(
                module,
                scan_proxy_in_args=False,
                use_full_instr=True,
                sampling_interval=0,
                warm_up_steps=0,
            )
            instrumentor._instrument_module(module, set(), True, 0)
        receiver = Model(5)
        standalone = vars(module)["__init__"]
        assert callable(standalone)
        standalone(SimpleNamespace(value=6))
        self.assertEqual(receiver.value, 5)
        self.assertEqual(len(self.records), 4)
        constructor_args = self.records[0]["args"]
        standalone_args = self.records[2]["args"]
        assert isinstance(constructor_args, dict)
        assert isinstance(standalone_args, dict)
        self.assertEqual(self.snapshot(constructor_args["0"])["attributes"], {})
        self.assertEqual(
            standalone_args["0"], dumper.var_to_serializable(SimpleNamespace(value=6))
        )


if __name__ == "__main__":
    unittest.main()
