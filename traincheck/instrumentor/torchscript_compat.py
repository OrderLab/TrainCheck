"""Preserve TorchScript operator identity for instrumented builtin functions."""

# TorchScript indexes by id only. Keep registered wrappers alive, like PyTorch's
# own _builtin_ops list, so a later unrelated object cannot reuse a stale ID.
_REGISTERED_WRAPPERS = {}


def preserve_torchscript_builtin(original, wrapped, torch_module):
    """Call before publishing wrapped into its module, while original is visible.

    TorchScript compiles the existing operator directly and does not run Python
    recorder hooks inside its graph. Eager calls still execute the wrapper.
    Private API feature detection leaves unsupported versions unchanged.
    """
    jit = getattr(torch_module, "jit", None)
    builtins = getattr(jit, "_builtins", None)
    find_builtin = getattr(builtins, "_find_builtin", None)
    register_builtin = getattr(builtins, "_register_builtin", None)
    if not callable(find_builtin) or not callable(register_builtin):
        return {"status": "unsupported-private-api", "operator": None}
    # _find_builtin also initializes the lazy table before the first replacement.
    operator = find_builtin(original)
    if operator is None:
        return {"status": "not-registered", "operator": None}
    register_builtin(wrapped, operator)
    _REGISTERED_WRAPPERS[wrapped] = operator
    return {"status": "preserved", "operator": operator}
