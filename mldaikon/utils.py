import logging
import sys
import threading
import time
import traceback
import uuid
from importlib.machinery import ModuleSpec

import torch

PROCESS_UUID = uuid.uuid4().hex


def safe_getattr(obj, attr, default=None):
    """Safely get the attribute of an object.
    try except is necessary as some objects (e.g. cuBLASModule in PyTorch) might have custom __getattr__
    method that raises an exception when accessing certain attributes.
    """
    try:
        return getattr(obj, attr, default)
    except Exception as e:
        if isinstance(e, AssertionError):
            return default
        if isinstance(e, RuntimeError):
            if (
                str(e)
                in "RuntimeError: Tried to instantiate class '__qualname__.__qualname__', but it does not exist! Ensure that it is registered via torch::class_"
            ):
                return default
        raise


def typename(o):
    if isinstance(o, torch.nn.Parameter):
        return "torch.nn.Parameter"
    if isinstance(o, torch.Tensor):
        return o.type()
    module = safe_getattr(o, "__module__", "")
    if isinstance(module, ModuleSpec):
        # handle the case when module is a ModuleSpec object
        module = module.name
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
    assert isinstance(module, str) and isinstance(
        class_name, str
    ), f"module and class_name should be str, but got {module} and {class_name} for {o}"
    return f"{module}.{class_name}" if module else class_name


def handle_excepthook(typ, message, stack):
    """Custom exception handler

    Print detailed stack information with local variables
    """
    logger = logging.getLogger("mldaikon")

    if issubclass(typ, KeyboardInterrupt):
        sys.__excepthook__(typ, message, stack)
        return

    stack_info = traceback.StackSummary.extract(
        traceback.walk_tb(stack), capture_locals=True
    ).format()
    logger.critical("An exception occured: %s: %s.", typ, message)
    for i in stack_info:
        logger.critical(i.encode().decode("unicode-escape"))

    # re-raise the exception so that vscode debugger can catch it and give useful information
    raise typ(message) from None


def thread_excepthook(args):
    """Exception notifier for threads"""
    logger = logging.getLogger("threading")

    exc_type = args.exc_type
    exc_value = args.exc_value
    exc_traceback = args.exc_traceback
    _ = args.thread
    if issubclass(exc_type, KeyboardInterrupt):
        threading.__excepthook__(args)
        return

    stack_info = traceback.StackSummary.extract(
        traceback.walk_tb(exc_traceback), capture_locals=True
    ).format()
    logger.critical("An exception occured: %s: %s.", exc_type, exc_value)
    for i in stack_info:
        logger.critical(i.encode().decode("unicode-escape"))

    # re-raise the exception so that vscode debugger can catch it and give useful information
    raise exc_type(exc_value) from None


def register_custom_excepthook():
    sys.excepthook = handle_excepthook
    threading.excepthook = thread_excepthook


def get_timestamp_ns():
    """Get the current timestamp, guaranteed to be unique in a single program run"""
    return time.monotonic_ns()


def get_unique_id():
    """Get a unique id a single program run"""
    return f"{PROCESS_UUID}_{get_timestamp_ns()}"
