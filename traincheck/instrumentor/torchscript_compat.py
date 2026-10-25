"""Preserve TorchScript operator identity for instrumented builtin functions."""

import threading
import warnings
from typing import Callable, Literal, TypedDict


class PreservedBuiltin(TypedDict):
    status: Literal["preserved"]
    operator: str


class UnregisteredBuiltin(TypedDict):
    status: Literal["not-registered"]
    operator: None


class UnsupportedBuiltinAPI(TypedDict):
    status: Literal["unsupported-private-api"]
    operator: None


BuiltinPreservation = PreservedBuiltin | UnregisteredBuiltin | UnsupportedBuiltinAPI

# TorchScript indexes by id only. Keep registered wrappers alive, like PyTorch's
# own _builtin_ops list, so a later unrelated object cannot reuse a stale ID.
# These callables are retained/passed through, never invoked by this module.
_REGISTERED_WRAPPERS: dict[Callable[..., object], str] = {}
_MISSING = object()
_UNSUPPORTED_WARNING_LOCK = threading.Lock()
_UNSUPPORTED_WARNING_EMITTED = False


def warn_unsupported_torchscript() -> None:
    """Report missing private support once per process, including always filters."""
    global _UNSUPPORTED_WARNING_EMITTED
    with _UNSUPPORTED_WARNING_LOCK:
        if not _UNSUPPORTED_WARNING_EMITTED:
            warnings.warn(
                "TrainCheck cannot preserve TorchScript builtin identity: this "
                "PyTorch version lacks the private builtin-registration API. "
                "Eager instrumentation remains available.",
                RuntimeWarning,
                stacklevel=2,
            )
            _UNSUPPORTED_WARNING_EMITTED = True


def preserve_torchscript_builtin(
    original: Callable[..., object],
    wrapped: Callable[..., object],
    torch_module: object,
) -> BuiltinPreservation:
    """Preserve a builtin mapping before publishing wrapped into its module.

    TorchScript compiles the existing operator directly and does not run Python
    recorder hooks inside its graph. Eager calls still execute the wrapper.
    Missing private APIs are unsupported; present malformed APIs or metadata
    are errors. Exceptions from framework callbacks propagate unchanged.
    """
    # Missing capabilities are expected across PyTorch versions. A sentinel
    # distinguishes absence from present-but-invalid None callback attributes.
    jit: object = getattr(torch_module, "jit", _MISSING)
    builtins: object = getattr(jit, "_builtins", _MISSING)
    find_builtin: object = getattr(builtins, "_find_builtin", _MISSING)
    register_builtin: object = getattr(builtins, "_register_builtin", _MISSING)
    for name, callback in (
        ("_find_builtin", find_builtin),
        ("_register_builtin", register_builtin),
    ):
        if callback is not _MISSING and not callable(callback):
            raise TypeError(f"torch.jit._builtins.{name} must be callable")
    if find_builtin is _MISSING or register_builtin is _MISSING:
        return {"status": "unsupported-private-api", "operator": None}
    # These checks narrow dynamic framework attributes after the missing case.
    assert callable(find_builtin) and callable(register_builtin)
    # _find_builtin also initializes the lazy table before the first replacement.
    operator: object = find_builtin(original)
    if operator is None:
        return {"status": "not-registered", "operator": None}
    if not isinstance(operator, str) or not operator:
        raise TypeError(
            "TorchScript builtin operator must be a nonempty string or None"
        )
    register_builtin(wrapped, operator)
    _REGISTERED_WRAPPERS[wrapped] = operator
    return {"status": "preserved", "operator": operator}
