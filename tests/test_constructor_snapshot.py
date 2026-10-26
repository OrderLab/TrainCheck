"""Constructor tracing must not inspect an uninitialized receiver dynamically."""

import functools
import inspect
import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from typing import Callable
from unittest.mock import patch

import torch

import traincheck.config.config as config
from traincheck.instrumentor import dumper, tracer
from traincheck.instrumentor.constructor_snapshot import constructor_receiver_snapshot
from traincheck.trace.trace_pandas import read_trace_file_Pandas
from traincheck.trace.utils import bind_args_kwargs_to_signature

CONSTRUCTOR = "torch._ops.OpOverloadPacket.__init__"
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

    def wrap(self, constructor: Callable[..., object]) -> Callable[..., object]:
        return tracer.wrapper(
            constructor,
            is_bound_method=True,
            scan_proxy_in_args=False,
            dump_stack_trace=False,
            handle_proxy=False,
        )

    def snapshot(self, encoded: dict[str, object] | None) -> dict[str, object]:
        assert encoded is not None
        packet = encoded[PACKET_TYPE]
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

    def test_initialized_packet_and_other_arguments_keep_existing_encoding(
        self,
    ) -> None:
        packet = torch.ops.aten.add
        self.assertIsNone(constructor_receiver_snapshot(CONSTRUCTOR, packet))
        arguments = (packet, "name", 3)
        before = tracer.to_dict_args_kwargs(arguments, {"flag": True})
        after = tracer.to_dict_args_kwargs(
            arguments, {"flag": True}, function_name=CONSTRUCTOR
        )
        self.assertEqual(dumper.serialize(dict(before)), dumper.serialize(dict(after)))
        self.assertIsNone(constructor_receiver_snapshot("different.__init__", object()))
        self.assertIsNone(constructor_receiver_snapshot(CONSTRUCTOR, object()))

    def test_stored_none_false_and_opaque_references_are_distinct(self) -> None:
        packet = object.__new__(torch._ops.OpOverloadPacket)
        opaque = object()
        object.__getattribute__(packet, "__dict__").update(
            {"unset": None, "flag": False, "opaque": opaque}
        )
        snapshot = self.snapshot(constructor_receiver_snapshot(CONSTRUCTOR, packet))
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
            function_name=CONSTRUCTOR,
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
                constructor_receiver_snapshot(CONSTRUCTOR, packet, config_value)
        object.__getattribute__(packet, "__dict__")[1] = "malformed name"
        with self.assertRaisesRegex(TypeError, "attribute names must be strings"):
            constructor_receiver_snapshot(CONSTRUCTOR, packet)

    def test_opaque_type_metadata_failure_propagates(self) -> None:
        def unavailable_module(receiver: object) -> str:
            raise RuntimeError("type metadata unavailable")

        metaclass = type(
            "MetadataFailure", (type,), {"__module__": property(unavailable_module)}
        )
        opaque_type = metaclass("Opaque", (), {})
        packet = object.__new__(torch._ops.OpOverloadPacket)
        object.__getattribute__(packet, "__dict__")["opaque"] = opaque_type()
        with self.assertRaisesRegex(RuntimeError, "type metadata unavailable"):
            constructor_receiver_snapshot(CONSTRUCTOR, packet)

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
        encoded = constructor_receiver_snapshot(CONSTRUCTOR, packet)
        assert encoded is not None
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


if __name__ == "__main__":
    unittest.main()
