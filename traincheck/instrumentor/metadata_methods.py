"""Bind verified native tensor methods without per-attribute warning contexts."""

import platform
import types
from collections.abc import Mapping
from typing import NamedTuple, TypeAlias

import torch

_TYPE_NAMESPACE: Mapping[str, object] = vars(type)
_OBJECT_GETATTRIBUTE = vars(object)["__getattribute__"]
_ABSENT = object()
_TENSOR_TYPE = torch.Tensor
_PARAMETER_TYPE = torch.nn.Parameter

BoundTensorMethod: TypeAlias = (
    types.MethodType | types.BuiltinFunctionType | types.MethodWrapperType
)


def _type_descriptors() -> tuple[
    types.MemberDescriptorType | types.GetSetDescriptorType,
    types.GetSetDescriptorType,
]:
    mro = _TYPE_NAMESPACE["__mro__"]
    namespace = _TYPE_NAMESPACE["__dict__"]
    # CPython 3.12 exposes type.__mro__ as a getset descriptor; 3.10/3.11
    # expose a member descriptor. Both read native storage without user hooks.
    assert isinstance(mro, (types.MemberDescriptorType, types.GetSetDescriptorType))
    assert isinstance(namespace, types.GetSetDescriptorType)
    return mro, namespace


_MRO_DESCRIPTOR, _NAMESPACE_DESCRIPTOR = _type_descriptors()


def _class_layout(
    cls: type[object],
) -> tuple[tuple[object, ...], tuple[Mapping[object, object], ...]]:
    # Type's native descriptors bypass custom metaclass properties as well as
    # custom __getattribute__. These mapping proxies remain live as attributes
    # change. A changed MRO is detected by identity before using any proxy.
    mro: object = _MRO_DESCRIPTOR.__get__(cls)
    assert isinstance(mro, tuple)
    namespaces: list[Mapping[object, object]] = []
    for owner in mro:
        assert isinstance(owner, type)
        namespace: object = _NAMESPACE_DESCRIPTOR.__get__(owner)
        assert isinstance(namespace, types.MappingProxyType)
        namespaces.append(namespace)
    return mro, tuple(namespaces)


_TENSOR_MRO, _TENSOR_NAMESPACES = _class_layout(_TENSOR_TYPE)
_PARAMETER_MRO, _PARAMETER_NAMESPACES = _class_layout(_PARAMETER_TYPE)


class _ImmutableSuffix(NamedTuple):
    namespaces: tuple[Mapping[object, object], ...]
    has_getattr: bool
    object_getattribute: bool


def _immutable_suffix(
    mro: tuple[object, ...], namespaces: tuple[Mapping[object, object], ...]
) -> tuple[tuple[Mapping[object, object], ...], _ImmutableSuffix]:
    """Keep namespaces live and cache only immutable lookup-hook facts.

    CPython's Py_TPFLAGS_IMMUTABLETYPE (1 << 8) disallows attribute changes
    through the Python type API. Read that flag through type's native descriptor,
    bypassing metaclass hooks. Other runtimes or an absent capability keep every
    namespace live. Present malformed native metadata remains an internal error.

    Retain only existing instrumenter-owned namespace views and booleans. Even
    immutable descriptors and their names must not acquire extra cache references:
    application properties can observe their reference counts. Each descriptor
    is still read from its original namespace. An unchanged MRO is required before
    callers use the facts; no mutable class lookup decision is cached.
    """
    assert len(mro) == len(namespaces)
    empty = _ImmutableSuffix((), False, False)
    if platform.python_implementation() != "CPython":
        return namespaces, empty
    flags_descriptor = _TYPE_NAMESPACE.get("__flags__", _ABSENT)
    if flags_descriptor is _ABSENT:
        return namespaces, empty
    assert isinstance(
        flags_descriptor, (types.GetSetDescriptorType, types.MemberDescriptorType)
    )
    offset = len(mro)
    for index in range(len(mro) - 1, -1, -1):
        owner = mro[index]
        assert isinstance(owner, type)
        flags: object = flags_descriptor.__get__(owner)
        flags_type = type(flags)
        assert flags_type is int
        assert isinstance(flags, int)
        if not flags & (1 << 8):
            break
        offset = index
    immutable = namespaces[offset:]
    # A custom dictionary key can execute equality during hook lookup. Such a
    # namespace keeps the original live path rather than moving those calls.
    for namespace in immutable:
        for name in namespace:
            name_type = type(name)
            if name_type is not str:
                return namespaces, empty
    getattribute: object = _ABSENT
    has_getattr = False
    for namespace in immutable:
        if "__getattr__" in namespace:
            has_getattr = True
        if getattribute is _ABSENT:
            getattribute = namespace.get("__getattribute__", _ABSENT)
    return namespaces[:offset], _ImmutableSuffix(
        immutable, has_getattr, getattribute is _OBJECT_GETATTRIBUTE
    )


_TENSOR_MUTABLE, _TENSOR_SUFFIX = _immutable_suffix(_TENSOR_MRO, _TENSOR_NAMESPACES)
_PARAMETER_MUTABLE, _PARAMETER_SUFFIX = _immutable_suffix(
    _PARAMETER_MRO, _PARAMETER_NAMESPACES
)


def _native_tensor_dictionary() -> types.GetSetDescriptorType | None:
    # Absence/replacement before import disables this optional fast path. A
    # genuine native descriptor remains valid for its owner after a user replaces
    # the public __dict__ attribute; invoking it reads actual storage directly.
    candidate = _TENSOR_NAMESPACES[0].get("__dict__", _ABSENT)
    candidate_type = type(candidate)
    if candidate_type is not types.GetSetDescriptorType:
        return None
    assert isinstance(candidate, types.GetSetDescriptorType)
    if candidate.__name__ == "__dict__" and candidate.__objclass__ is _TENSOR_TYPE:
        return candidate
    return None


_TENSOR_DICTIONARY = _native_tensor_dictionary()


def tensor_method_binding(receiver: object, name: str) -> BoundTensorMethod | None:
    """Bind a native method, or return None to use the original guarded getter.

    This is deliberately restricted to exact Tensor/Parameter instances with
    empty native dictionaries. Subclasses, custom lookup, altered inheritance,
    instance state and non-method descriptors retain the original getter path.
    Each decision rereads current namespaces and dictionary storage, including
    after an earlier getter mutates the receiver or its class. No application
    property, descriptor or lookup hook is invoked while deciding.

    Return the real binding and leave classification to the caller: even an
    unrecorded method keeps its receiver/function alive until the next attribute
    replaces it. Skipping the binding would change observable object lifetimes.
    """
    name_type = type(name)
    if name_type is not str or "_TRAINCHECK" in name:
        return None
    receiver_type = type(receiver)
    if receiver_type is _TENSOR_TYPE:
        mro, namespaces = _TENSOR_MRO, _TENSOR_MUTABLE
        suffix = _TENSOR_SUFFIX
    elif receiver_type is _PARAMETER_TYPE:
        mro, namespaces = _PARAMETER_MRO, _PARAMETER_MUTABLE
        suffix = _PARAMETER_SUFFIX
    else:
        return None
    if _TENSOR_DICTIONARY is None or _MRO_DESCRIPTOR.__get__(receiver_type) is not mro:
        return None

    getattribute: object = _ABSENT
    descriptor: object = _ABSENT
    for namespace in namespaces:
        if "__getattr__" in namespace:
            return None
        if getattribute is _ABSENT:
            getattribute = namespace.get("__getattribute__", _ABSENT)
        if descriptor is _ABSENT:
            descriptor = namespace.get(name, _ABSENT)
    if suffix.has_getattr:
        return None
    if getattribute is _ABSENT and suffix.object_getattribute:
        getattribute = _OBJECT_GETATTRIBUTE
    if descriptor is _ABSENT:
        for namespace in suffix.namespaces:
            descriptor = namespace.get(name, _ABSENT)
            if descriptor is not _ABSENT:
                break
    if getattribute is not _OBJECT_GETATTRIBUTE:
        return None

    descriptor_type = type(descriptor)
    if (
        descriptor_type is not types.FunctionType
        and descriptor_type is not types.MethodDescriptorType
        and descriptor_type is not types.WrapperDescriptorType
        and descriptor_type is not types.ClassMethodDescriptorType
    ):
        return None
    if descriptor_type is not types.FunctionType:
        assert isinstance(
            descriptor,
            (
                types.MethodDescriptorType,
                types.WrapperDescriptorType,
                types.ClassMethodDescriptorType,
            ),
        )
        descriptor_owner = descriptor.__objclass__
        for owner in mro:
            if descriptor_owner is owner:
                break
        else:
            # A foreign descriptor such as str.upper would fail to bind, and
            # the original getter must retain its failure/backoff behavior.
            return None

    state: object = _TENSOR_DICTIONARY.__get__(receiver, receiver_type)
    assert isinstance(state, dict)
    # Even membership checks on a nonempty dictionary can run user key equality.
    # Falling back here preserves lookup multiplicity, shadowing and side effects.
    if dict.__len__(state) != 0:
        return None
    # Call the descriptor TYPE's native slot. A Python function may shadow its
    # own __get__ attribute, which normal descriptor lookup never invokes.
    binding: object
    if isinstance(descriptor, types.FunctionType):
        binding = types.FunctionType.__get__(descriptor, receiver, receiver_type)
    elif isinstance(descriptor, types.MethodDescriptorType):
        binding = types.MethodDescriptorType.__get__(
            descriptor, receiver, receiver_type
        )
    elif isinstance(descriptor, types.WrapperDescriptorType):
        binding = types.WrapperDescriptorType.__get__(
            descriptor, receiver, receiver_type
        )
    else:
        assert isinstance(descriptor, types.ClassMethodDescriptorType)
        binding = types.ClassMethodDescriptorType.__get__(
            descriptor, receiver, receiver_type
        )
    assert isinstance(
        binding, (types.MethodType, types.BuiltinFunctionType, types.MethodWrapperType)
    )
    return binding
