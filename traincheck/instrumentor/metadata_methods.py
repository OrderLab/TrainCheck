"""Recognize native tensor method bindings discarded by metadata conversion."""

import types
from collections.abc import Mapping

import torch

_TYPE_NAMESPACE: Mapping[str, object] = vars(type)
_OBJECT_GETATTRIBUTE = vars(object)["__getattribute__"]
_ABSENT = object()
_TENSOR_TYPE = torch.Tensor
_PARAMETER_TYPE = torch.nn.Parameter


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


def is_discardable_tensor_method(
    receiver: object, name: str, retained_types: set[type[object]]
) -> bool:
    """Whether normal lookup can only bind an otherwise discarded method.

    This is deliberately restricted to exact Tensor/Parameter instances with
    empty native dictionaries. Subclasses, custom lookup, altered inheritance,
    instance state and non-method descriptors retain the original getter path.
    Each decision rereads current namespaces and dictionary storage, including
    after an earlier getter mutates the receiver or its class. No application
    property, descriptor or lookup hook is invoked while deciding.

    ``retained_types`` is the converter's live primitive-type configuration: a
    bound method promoted to primitive data must still be recorded. Names with
    TrainCheck metadata anywhere in them are likewise never discarded.
    """
    name_type = type(name)
    retained_set_type = type(retained_types)
    if name_type is not str or "_TRAINCHECK" in name or retained_set_type is not set:
        return False
    receiver_type = type(receiver)
    if receiver_type is _TENSOR_TYPE:
        mro, namespaces = _TENSOR_MRO, _TENSOR_NAMESPACES
    elif receiver_type is _PARAMETER_TYPE:
        mro, namespaces = _PARAMETER_MRO, _PARAMETER_NAMESPACES
    else:
        return False
    if _TENSOR_DICTIONARY is None or _MRO_DESCRIPTOR.__get__(receiver_type) is not mro:
        return False

    getattribute: object = _ABSENT
    descriptor: object = _ABSENT
    for namespace in namespaces:
        if "__getattr__" in namespace:
            return False
        if getattribute is _ABSENT:
            getattribute = namespace.get("__getattribute__", _ABSENT)
        if descriptor is _ABSENT:
            descriptor = namespace.get(name, _ABSENT)
    if getattribute is not _OBJECT_GETATTRIBUTE:
        return False

    descriptor_type = type(descriptor)
    if descriptor_type is types.FunctionType:
        bound_type: type[object] = types.MethodType
    elif descriptor_type is types.MethodDescriptorType:
        bound_type = types.BuiltinFunctionType
    elif descriptor_type is types.WrapperDescriptorType:
        bound_type = types.MethodWrapperType
    elif descriptor_type is types.ClassMethodDescriptorType:
        bound_type = types.BuiltinFunctionType
    else:
        return False
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
            return False

    state: object = _TENSOR_DICTIONARY.__get__(receiver, receiver_type)
    assert isinstance(state, dict)
    # Even membership checks on a nonempty dictionary can run user key equality.
    # Falling back here preserves lookup multiplicity, shadowing and side effects.
    if dict.__len__(state) != 0:
        return False
    # An ordinary membership test could invoke custom metaclass equality, then
    # invoke it again after falling back. Snapshot the exact set without user
    # iteration hooks, inspect entries without hashing/comparing them, and leave
    # custom metaclass configurations to the original classifier unchanged.
    for retained in tuple(retained_types):
        if type(retained) is not type or retained is bound_type:
            return False
    return True
