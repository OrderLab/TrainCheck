import logging
import re
import sys
import threading
import time
import traceback
import uuid
from importlib.machinery import ModuleSpec
from typing import Any

import pandas as pd
import torch

THREAD_LOCAL = threading.local()


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
        if isinstance(e, ModuleNotFoundError):
            return default
        raise


def typename(o, is_runtime=False):
    if isinstance(o, torch.nn.Parameter):
        return "torch.nn.Parameter"
    if isinstance(o, torch.Tensor):
        return o.type()
    prefix = safe_getattr(o, "__module__", "")
    if isinstance(prefix, ModuleSpec):
        # handle the case when prefix is a ModuleSpec object
        prefix = prefix.name
    if prefix in ["builtins", "__builtin__", None]:
        prefix = ""
    is_class_name_qualname = True
    last_name = safe_getattr(o, "__qualname__", "")
    if not isinstance(
        last_name, str
    ):  # the instance here is for the case when __qualname__ is _ClassNamespace
        last_name = ""
        is_class_name_qualname = False
    if not last_name:
        last_name = safe_getattr(o, "__name__", "")
        is_class_name_qualname = False
    if not last_name:
        last_name = safe_getattr(o, "__class__", type(o)).__name__
        is_class_name_qualname = False
    # Handle the typename logic for method descriptors. The `is_class_name_qualname` flag is used as a heuristic to determine if `o` is likely a method descriptor.
    # During instrumentation (when `is_runtime` is False), we perform a `str(o)` check to identify the correct typename for method descriptors.
    # At runtime (when `is_runtime` is True), this check is avoided as it is too expensive for argument/return value serialization.
    if not is_runtime and not prefix and is_class_name_qualname:
        match = re.match(r"<method '(\w+)' of '([\w\.]+)' objects>", str(o))
        if match:
            last_name, prefix = match.groups()
    assert isinstance(prefix, str) and isinstance(
        last_name, str
    ), f"prefix and last_name should be str, but got {prefix} and {last_name} for {o}"
    return f"{prefix}.{last_name}" if prefix else last_name


def handle_excepthook(typ, message, stack):
    """Custom exception handler

    Print detailed stack information with local variables
    """
    logger = logging.getLogger("traincheck")

    if issubclass(typ, KeyboardInterrupt):
        sys.__excepthook__(typ, message, stack)
        return

    stack_info = traceback.StackSummary.extract(
        traceback.walk_tb(stack), capture_locals=True
    ).format()
    logger.critical("An exception occurred: %s: %s.", typ, message)
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
    logger.critical("An exception occurred: %s: %s.", exc_type, exc_value)
    for i in stack_info:
        logger.critical(i.encode().decode("unicode-escape"))

    # re-raise the exception so that vscode debugger can catch it and give useful information
    raise exc_type(exc_value) from None


def register_custom_excepthook(add_file_handler=False):
    if add_file_handler:
        file_handler = logging.FileHandler("debug.log")
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)
        # add to the "traincheck" logger and the "threading" logger
        logging.getLogger("traincheck").addHandler(file_handler)
        logging.getLogger("threading").addHandler(file_handler)
    sys.excepthook = handle_excepthook
    threading.excepthook = thread_excepthook


def get_timestamp_ns():
    """Get the current timestamp, guaranteed to be unique in a single program run"""
    return time.monotonic_ns()


def get_unique_id():
    if not hasattr(THREAD_LOCAL, "PROCESS_UUID"):
        THREAD_LOCAL.PROCESS_UUID = uuid.uuid4().hex
    """Get a unique id a single program run"""
    return f"{THREAD_LOCAL.PROCESS_UUID}_{get_timestamp_ns()}"


def safe_isnan(value: Any) -> bool:
    if value is pd.NA:
        return True

    return isinstance(value, float) and pd.isna(value)
