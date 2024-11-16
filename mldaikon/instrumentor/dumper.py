import datetime
import json  # consider using ORJSON for better performance?
import logging
import os
import threading
from queue import Empty, Queue

import torch

from mldaikon.instrumentor.types import PTID
from mldaikon.proxy_wrapper.proxy_config import (
    attribute_black_list,
    primitive_types,
    tensor_dump_format,
)

if torch.cuda.is_available():
    from mldaikon.proxy_wrapper.hash import tensor_hash

from mldaikon.proxy_wrapper.utils import print_debug
from mldaikon.utils import typename

DEBUG = os.environ.get("ML_DAIKON_DEBUG", False)


# per process & thread logging
stop_event = threading.Event()
monitoring_thread = None
trace_API_dumper_queues: dict[PTID, Queue] = {}
trace_VAR_dumper_queues: dict[PTID, Queue] = {}

# per process logging
instrumentation_loggers: dict[int, logging.Logger] = {}


def monitor_main_thread(main_thread, stop_event):
    main_thread.join()  # Wait for the main thread to finish
    print("Main thread has finished or encountered an exception")
    stop_event.set()  # Signal the logging threads to stop


def trace_dumper(task_queue: Queue, trace_file_name: str, stop_event: threading.Event):
    with open(trace_file_name, "w") as f:
        while True:
            try:
                trace = task_queue.get(timeout=0.5)
            except Empty:
                if stop_event.is_set():
                    break
                continue
            f.write(f"{trace}\n")
            task_queue.task_done()


def get_trace_API_dumper_queue():
    global monitoring_thread
    if monitoring_thread is None:
        monitoring_thread = threading.Thread(
            target=monitor_main_thread, args=(threading.main_thread(), stop_event)
        )
        monitoring_thread.start()

    pid = os.getpid()
    tid = threading.current_thread().ident

    ptid = PTID(pid, tid)
    if ptid in trace_API_dumper_queues:
        return trace_API_dumper_queues[ptid]

    output_dir = os.getenv("ML_DAIKON_OUTPUT_DIR")
    assert (
        output_dir is not None
    ), "ML_DAIKON_OUTPUT_DIR is not set, examine the instrumented code to see if os.environ['ML_DAIKON_OUTPUT_DIR'] is set in the main function"

    trace_queue = Queue()
    trace_file_name = f"trace_API_{pid}_{tid}.log"
    trace_file_full_path = os.path.join(output_dir, trace_file_name)
    log_thread = threading.Thread(
        target=trace_dumper, args=(trace_queue, trace_file_full_path, stop_event)
    )
    log_thread.start()

    trace_API_dumper_queues[ptid] = trace_queue
    return trace_queue


def get_trace_VAR_dumper_queue():
    global monitoring_thread
    if monitoring_thread is None:
        monitoring_thread = threading.Thread(
            target=monitor_main_thread, args=(threading.main_thread(), stop_event)
        )
        monitoring_thread.start()

    pid = os.getpid()
    tid = threading.current_thread().ident

    ptid = PTID(pid, tid)
    if ptid in trace_VAR_dumper_queues:
        return trace_VAR_dumper_queues[ptid]

    output_dir = os.getenv("ML_DAIKON_OUTPUT_DIR")
    assert (
        output_dir is not None
    ), "ML_DAIKON_OUTPUT_DIR is not set, examine the instrumented code to see if os.environ['ML_DAIKON_OUTPUT_DIR'] is set in the main function"

    trace_queue = Queue()
    trace_file_name = f"trace_VAR_{pid}_{tid}.log"
    trace_file_full_path = os.path.join(output_dir, trace_file_name)
    log_thread = threading.Thread(
        target=trace_dumper, args=(trace_queue, trace_file_full_path, stop_event)
    )
    log_thread.start()

    trace_VAR_dumper_queues[ptid] = trace_queue
    return trace_queue


def dump_trace_API(trace: dict):
    """add a timestamp (unix) to the trace and dump it to the trace log file"""
    trace_queue = get_trace_API_dumper_queue()
    trace["time"] = datetime.datetime.now().timestamp()
    trace_queue.put(
        json.dumps(trace)
    )  # this is additional copying important, as the trace is mutable and we don't want subsequent changes to affect the trace


def dump_trace_VAR(trace: dict):
    """add a timestamp (unix) to the trace and dump it to the trace log file"""
    trace_queue = get_trace_VAR_dumper_queue()
    if "time" not in trace:
        trace["time"] = datetime.datetime.now().timestamp()
    trace_queue.put(json.dumps(trace))


def get_instrumentation_logger_for_process():
    pid = os.getpid()
    output_dir = os.getenv("ML_DAIKON_OUTPUT_DIR")
    assert (
        output_dir is not None
    ), "ML_DAIKON_OUTPUT_DIR is not set, examine the instrumented code to see if os.environ['ML_DAIKON_OUTPUT_DIR'] is set in the main function"

    if pid in instrumentation_loggers:
        return instrumentation_loggers[pid]

    logger = logging.getLogger(f"instrumentation_{pid}")
    if DEBUG:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    log_file = f"instrumentation_{pid}.log"
    file_handler = logging.FileHandler(os.path.join(output_dir, log_file))
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(file_handler)
    instrumentation_loggers[pid] = logger
    return logger


def tensor_stats(tensor: torch.Tensor):
    min = float(tensor.min().item())
    max = float(tensor.max().item())
    mean = float(tensor.mean().item())
    std = float(tensor.std().item())
    shape = tuple(int(x) for x in tensor.size())
    return {
        "min": min,
        "max": max,
        "mean": mean,
        "std": std,
        "shape": shape,
    }


def dump_tensor(value):
    param_list = None
    if isinstance(value, torch.Tensor):
        if tensor_dump_format["dump_tensor_stats"]:
            param_list = tensor_stats(value)
        elif tensor_dump_format["dump_tensor_hash"]:
            if not torch.cuda.is_available():
                raise Exception(
                    "CUDA is not available, cannot dump tensor hash, please set '--tensor-dump-format' to 'full' or 'stats'."
                )
            try:
                # perform tensor hash a deep copy of the tensor
                param_list = tensor_hash(value, with_parallel=True, with_cuda=True)
            except Exception as e:
                print_debug(
                    f"Failed to dump tensor hash, error: {e}, fullback to cpu hashing."
                )
                param_list = tensor_hash(value, with_parallel=True, with_cuda=False)
        elif tensor_dump_format["dump_tensor_full"]:
            param_list = value.detach().flatten().tolist()
        else:
            raise ValueError(
                "Invalid tensor dump format, please set '--tensor-dump-format' to 'full', 'stats' or 'hash'."
            )

    return param_list


def convert_var_to_dict(var, include_tensor_data=True) -> dict:
    result: dict[str, object | str] = {}
    # currently only dump primitive types, tensors and nn.Module

    try:
        attr_names = [name for name in dir(var) if not name.startswith("__")]
    except Exception as e:
        get_instrumentation_logger_for_process().debug(
            f"Failed to get attributes of object type {type(var)}, skipping it. Error: {e}."
        )
        return result

    for attr_name in attr_names:
        # don't track the attr_name starts with a _ (private variable)
        if attr_name.startswith("_") and not attr_name.startswith("_ML_DAIKON"):
            continue

        if attr_name in attribute_black_list:
            continue
        try:
            attr = getattr(var, attr_name)
            if type(attr) in primitive_types:
                result[attr_name] = attr

            elif include_tensor_data and isinstance(attr, torch.Tensor):
                result[attr_name] = dump_tensor(attr)

            elif include_tensor_data and isinstance(attr, torch.nn.parameter.Parameter):
                result[attr_name] = attr.__class__.__name__ + "(Parameter)"
                result[attr_name] = dump_tensor(attr.data)

            elif include_tensor_data and isinstance(attr, torch.nn.Module):
                result[attr_name] = attr.__class__.__name__ + "(nn.Module)"
                # dump out all tensors inside the nn.Module
                for name, param in attr.named_parameters():
                    result[attr_name] += f"\n{name}: {dump_tensor(param)}"  # type: ignore

            # if attr_name == "grad_fn":  # FIXME: ad-hoc
            #     assert attr is None or callable(
            #         attr
            #     ), f"grad_fn should be None or callable, but got {attr}"
            # result[attr_name] = typename(attr) if attr is not None else None

            elif isinstance(attr, torch.dtype):
                # result[attr_name] = typename(attr)
                result[attr_name] = str(attr)
            elif isinstance(attr, torch.Size):
                result[attr_name] = tuple(attr)
            elif "_ML_DAIKON" in attr_name:
                # should always be serializable, so blindly assign here.
                result[attr_name] = attr

        except Exception as e:  # noqa
            print_debug(
                lambda: f"Failed to get attribute {attr_name} of object type {type(var)}, skipping it. Error: {e}."  # noqa
            )
            continue
    if include_tensor_data and "data" not in result and isinstance(var, torch.Tensor):
        raise ValueError(
            f"Failed to dump tensor data of tensor {var}, please turn on debugging mode and see the debugging log."
        )
    return result


def var_to_serializable(obj) -> dict[str, object]:
    """Convert any object to a serializable dictionary.

    Note that this function does not dump the `data` attribute of a tensor.
    If you want to dump the `data` attribute of a tensor, use `convert_var_to_dict` and set `include_tensor_data=True`.
    """

    try:
        json.dumps({"foo": obj})
        return {typename(obj): obj}
    except TypeError:
        if isinstance(obj, torch.dtype):
            return {typename(obj): str(obj)}
        elif isinstance(obj, torch.Size):
            return {typename(obj): tuple(obj)}
        try:
            var_dict = convert_var_to_dict(obj, include_tensor_data=False)
            return {typename(obj): var_dict}
        except RecursionError:
            logger = logging.getLogger(__name__)
            logger.warning(
                f"Recursion detected when converting object to dict. Probably due to a issue in the __getattr__ method of the object. Object type: {type(obj)}."
            )
            return {str(type(obj)): None}
        # assert var_dict, f"Failed to convert object {obj} to dict."
