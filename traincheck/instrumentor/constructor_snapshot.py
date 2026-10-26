"""Explicit stored-state snapshots for receivers that are not initialized yet."""

import math
from collections.abc import Mapping
from typing import Literal, TypeAlias, TypedDict

import torch

_PACKET_CONSTRUCTOR = "torch._ops.OpOverloadPacket.__init__"
_PACKET_TYPE = "torch._ops.OpOverloadPacket"
_SNAPSHOT_FIELD = "_TRAINCHECK_CONSTRUCTOR_SNAPSHOT"

Scalar: TypeAlias = str | int | float | bool | None


class StoredValue(TypedDict):
    kind: Literal["value"]
    value: Scalar


class StoredReference(TypedDict):
    kind: Literal["object_reference"]
    object_id: int
    type: str


class StoredNonfiniteFloat(TypedDict):
    kind: Literal["nonfinite_float"]
    value: Literal["nan", "+inf", "-inf"]


StoredAttribute: TypeAlias = StoredValue | StoredReference | StoredNonfiniteFloat


class ConstructorSnapshot(TypedDict):
    phase: Literal["before_init"]
    coverage: Literal["stored_attributes_only"]
    object_id: int
    requested_attributes: list[str] | None
    unavailable_attributes: list[str]
    attributes: dict[str, StoredAttribute]


def _stored_value(value: object) -> StoredAttribute:
    if value is None or type(value) in (str, int, float, bool):
        # The explicit isinstance check narrows the exact built-in types above.
        assert value is None or isinstance(value, (str, int, float, bool))
        if isinstance(value, float) and not math.isfinite(value):
            label: Literal["nan", "+inf", "-inf"] = (
                "nan" if math.isnan(value) else "+inf" if value > 0 else "-inf"
            )
            return {"kind": "nonfinite_float", "value": label}
        return {"kind": "value", "value": value}

    value_type = type(value)
    # Read class metadata without invoking an instance's dynamic lookup or repr.
    module: object = type.__getattribute__(value_type, "__module__")
    qualified_name: object = type.__getattribute__(value_type, "__qualname__")
    if not isinstance(module, str) or not isinstance(qualified_name, str):
        raise TypeError("Stored attribute type metadata must contain string names")
    return {
        "kind": "object_reference",
        "object_id": id(value),
        "type": f"{module}.{qualified_name}",
    }


def constructor_receiver_snapshot(
    function_name: str, receiver: object, dump_config: object = None
) -> dict[str, object] | None:
    """Return an explicit partial snapshot only for an uninitialized packet.

    None means the normal serializer applies. Full instrumentation observes
    OpOverloadPacket.__init__ before PyTorch sets _op and _qualified_op_name.
    Its ``op`` property and dynamic name lookup recurse when either is absent.
    This adapter keeps the receiver and its identity in the PRE record without
    running those getters. Stored scalars retain their values; opaque values are
    clearly marked identity references, not claims about their internal state.
    Nonfinite floats use tagged names so JSON cannot turn them into null.

    Initialized packets, other arguments, other constructors, and subclasses
    retain their existing serialization. This is not a fallback after a getter
    fails and does not suppress application errors. The caller supplies the
    positional receiver used by PyTorch. The tracer's existing inability to
    identify a bound receiver supplied only as a keyword is outside this adapter.
    """
    if function_name != _PACKET_CONSTRUCTOR:
        return None
    if type(receiver) is not torch._ops.OpOverloadPacket:
        return None

    state: object = object.__getattribute__(receiver, "__dict__")
    if not isinstance(state, dict):
        raise TypeError("OpOverloadPacket stored state must be a dictionary")
    if "_op" in state and "_qualified_op_name" in state:
        return None

    requested: list[str] | None = None
    if dump_config is not None:
        if not isinstance(dump_config, Mapping):
            raise TypeError("Constructor receiver dump configuration must be a mapping")
        requested = []
        for name in dump_config:
            if not isinstance(name, str):
                raise TypeError("Requested constructor attribute names must be strings")
            requested.append(name)

    attributes: dict[str, StoredAttribute] = {}
    for name, value in state.items():
        if not isinstance(name, str):
            raise TypeError("OpOverloadPacket stored attribute names must be strings")
        if requested is None or name in requested:
            attributes[name] = _stored_value(value)
    snapshot: ConstructorSnapshot = {
        "phase": "before_init",
        "coverage": "stored_attributes_only",
        "object_id": id(receiver),
        "requested_attributes": requested,
        "unavailable_attributes": (
            [name for name in requested if name not in state]
            if requested is not None
            else []
        ),
        "attributes": attributes,
    }
    return {_PACKET_TYPE: {_SNAPSHOT_FIELD: snapshot}}
