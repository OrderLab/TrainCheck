import torch


def safe_getattr(obj, attr, default=None):
    """Safely get the attribute of an object.
    try except is necessary as some objects (e.g. cuBLASModule in PyTorch) might have custom __getattr__
    method that raises an exception when accessing certain attributes.
    """
    try:
        return getattr(obj, attr, default)
    except AssertionError:
        return default


def typename(o):
    if isinstance(o, torch.Tensor):
        return o.type()

    module = safe_getattr(o, "__module__", "")
    if module in ["buitins", "__builtin__", None]:
        module = ""

    class_name = safe_getattr(o, "__qualname__", "")
    if not isinstance(
        class_name, str
    ):  # the instance here is for the case when __qualname__ is _ClassNamespace
        class_name = ""

    if not class_name:
        class_name = safe_getattr(o, "__name__", "")
    if not class_name:
        class_name = safe_getattr(o, "__class__", type(o)).__name__

    assert isinstance(module, str) and isinstance(class_name, str)

    return f"{module}.{class_name}" if module else class_name