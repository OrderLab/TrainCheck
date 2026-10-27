"""Constructor PRE snapshots without evaluating application attributes."""

import math
import types
from collections.abc import Iterable, Mapping
from typing import Literal, TypeAlias, TypedDict

_SNAPSHOT_FIELD = "_TRAINCHECK_CONSTRUCTOR_SNAPSHOT"
# Immutable built-in type namespace; no application class metadata is cached.
_TYPE_NAMESPACE: Mapping[str, object] = vars(type)

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


class SlotIdentity(TypedDict):
    owner_type: str
    name: str


class StoredSlot(SlotIdentity):
    value: StoredAttribute


class ConstructorSnapshot(TypedDict):
    phase: Literal["before_init"]
    coverage: Literal["stored_attributes_only"]
    object_id: int
    dictionary_status: Literal["captured", "absent", "unavailable"]
    requested_attributes: list[str] | None
    unavailable_attributes: list[str]
    attributes: dict[str, StoredAttribute]
    slots: list[StoredSlot]
    unset_slots: list[SlotIdentity]


def _type_metadata(cls: type[object], name: str) -> object:
    descriptor = _TYPE_NAMESPACE[name]
    assert isinstance(
        descriptor, (types.GetSetDescriptorType, types.MemberDescriptorType)
    )
    return descriptor.__get__(cls)


def _type_name(cls: type[object]) -> str:
    # Invoke type's own native descriptors, bypassing custom metaclass
    # __getattribute__ AND properties with these names. type.__getattribute__
    # alone still invokes such properties.
    module: object = _type_metadata(cls, "__module__")
    qualified_name: object = _type_metadata(cls, "__qualname__")
    module_type, name_type = type(module), type(qualified_name)
    if module_type is not str or name_type is not str:
        raise TypeError("Stored attribute type metadata must contain string names")
    assert isinstance(module, str) and isinstance(qualified_name, str)
    return f"{module}.{qualified_name}"


def _stored_value(value: object) -> StoredAttribute:
    value_type = type(value)
    # Identity comparisons also avoid custom metaclass __eq__ implementations.
    if (
        value is None
        or value_type is str
        or value_type is int
        or value_type is float
        or value_type is bool
    ):
        assert value is None or isinstance(value, (str, int, float, bool))
        if isinstance(value, float) and not math.isfinite(value):
            label: Literal["nan", "+inf", "-inf"] = (
                "nan" if math.isnan(value) else "+inf" if value > 0 else "-inf"
            )
            return {"kind": "nonfinite_float", "value": label}
        return {"kind": "value", "value": value}
    return {
        "kind": "object_reference",
        "object_id": id(value),
        "type": _type_name(value_type),
    }


def constructor_receiver_snapshot(
    receiver: object, dump_config: object = None
) -> dict[str, object]:
    """Capture stored state at entry to any selected instance initializer.

    The caller identifies __init__ using the owning class, not a name heuristic.
    Reinitialization and base-class initializers use the same PRE representation.
    Read native dictionary/slot storage only: never evaluate properties, custom
    descriptors, dir(), repr(), or receiver/metaclass attribute lookup hooks.
    Opaque values are identity references, not recursive snapshots. In particular
    tensor references do not claim tensor-content or hash coverage.

    Dictionary entries and slots are separate because inherited slots (and the
    dictionary) can contain distinct values with the same name. Unset slots and
    inaccessible dictionary storage are explicit. Native extension storage not
    exposed by these descriptors is outside stored_attributes_only coverage.
    No metadata is cached: each call observes current class and instance storage.
    """
    requested: list[str] | None = None
    if dump_config is not None:
        if not isinstance(dump_config, Mapping):
            raise TypeError("Constructor receiver dump configuration must be a mapping")
        requested = []
        for name in dump_config:
            name_type = type(name)
            if name_type is not str:
                raise TypeError("Requested constructor attribute names must be strings")
            requested.append(name)

    receiver_type = type(receiver)
    mro: object = _type_metadata(receiver_type, "__mro__")
    assert isinstance(mro, tuple)
    dictionary_offset: object = _type_metadata(receiver_type, "__dictoffset__")
    assert isinstance(dictionary_offset, int)
    dictionary_status: Literal["captured", "absent", "unavailable"] = (
        "unavailable" if dictionary_offset else "absent"
    )
    attributes: dict[str, StoredAttribute] = {}
    slots: list[StoredSlot] = []
    unset_slots: list[SlotIdentity] = []
    available: set[str] = set()

    for owner in mro:
        assert isinstance(owner, type)
        namespace: object = _type_metadata(owner, "__dict__")
        assert isinstance(namespace, types.MappingProxyType)
        for name, descriptor in namespace.items():
            name_type = type(name)
            if name_type is not str:
                raise TypeError("Class attribute names must be strings")
            descriptor_type = type(descriptor)
            if (
                name == "__dict__"
                and dictionary_status != "captured"
                and (
                    descriptor_type is types.GetSetDescriptorType
                    or descriptor_type is types.MemberDescriptorType
                )
                and descriptor.__name__ == name
                and descriptor.__objclass__ is owner
            ):
                state: object = descriptor.__get__(receiver, receiver_type)
                # A metaclass initializer receives a class: its native stored
                # namespace is a mappingproxy. Instance dictionaries may be dict
                # subclasses, whose overridden methods must not run here.
                entries: Iterable[tuple[object, object]]
                if isinstance(state, dict):
                    entries = dict.items(state)
                elif isinstance(state, types.MappingProxyType):
                    entries = state.items()
                else:
                    raise TypeError(
                        "Constructor stored state must be a dictionary or class namespace"
                    )
                for field, value in entries:
                    field_type = type(field)
                    if field_type is not str:
                        raise TypeError("Stored attribute names must be strings")
                    assert isinstance(field, str)
                    available.add(field)
                    if requested is None or field in requested:
                        attributes[field] = _stored_value(value)
                dictionary_status = "captured"
            elif (
                descriptor_type is types.MemberDescriptorType
                and descriptor.__name__ == name
                and descriptor.__objclass__ is owner
                and name not in ("__dict__", "__weakref__")
                and (requested is None or name in requested)
            ):
                owner_name = _type_name(owner)
                try:
                    slot_value: object = descriptor.__get__(receiver, receiver_type)
                except AttributeError:
                    # Native slot descriptors raise AttributeError for unset storage.
                    unset_slots.append({"owner_type": owner_name, "name": name})
                else:
                    available.add(name)
                    slots.append(
                        {
                            "owner_type": owner_name,
                            "name": name,
                            "value": _stored_value(slot_value),
                        }
                    )

    snapshot: ConstructorSnapshot = {
        "phase": "before_init",
        "coverage": "stored_attributes_only",
        "object_id": id(receiver),
        "dictionary_status": dictionary_status,
        "requested_attributes": requested,
        "unavailable_attributes": (
            [name for name in requested if name not in available]
            if requested is not None
            else []
        ),
        "attributes": attributes,
        "slots": slots,
        "unset_slots": unset_slots,
    }
    return {_type_name(receiver_type): {_SNAPSHOT_FIELD: snapshot}}
