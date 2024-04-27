import torch


def typename(o):
    if isinstance(o, torch.Tensor):
        return o.type()
    module = ""
    class_name = ""
    if (
        hasattr(o, "__module__")
        and o.__module__ != "builtins"
        and o.__module__ != "__builtin__"
        and o.__module__ is not None
    ):
        module = o.__module__ + "."
    if hasattr(o, "__qualname__") and isinstance(
        o.__qualname__, str
    ):  # the instance here is for the case when __qualname__ is _ClassNamespace
        class_name = o.__qualname__
    elif hasattr(o, "__name__"):
        class_name = o.__name__
    else:
        class_name = o.__class__.__name__
    assert isinstance(module, str) and isinstance(class_name, str)
    return module + class_name
