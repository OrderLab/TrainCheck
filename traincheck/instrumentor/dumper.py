import json
import logging
import os
import threading
import time
from queue import Empty, Queue
from typing import Hashable

import orjson
import torch

from traincheck.config.config import (
    BUFFER_SIZE,
    FLUSH_INTERVAL,
    RECURSION_ERR_THRESHOLD,
    TYPE_ERR_THRESHOLD,
)

# if torch.cuda.is_available():
from traincheck.proxy_wrapper.hash import tensor_hash
from traincheck.proxy_wrapper.proxy_config import (
    attribute_black_list,
    primitive_types,
    tensor_dump_format,
)
from traincheck.utils import get_timestamp_ns, typename

DEBUG = os.environ.get("ML_DAIKON_DEBUG", False)
THREAD_DATA = threading.local()
IS_CUDA_AVAILABLE = torch.cuda.is_available()

# per process & thread logging
stop_event = threading.Event()
monitoring_thread = None

# per process logging
instrumentation_loggers: dict[int, logging.Logger] = {}

# this is a global variable to store the attributes that cannot be accessed due to errors, so that we don't try to access them again and waste time.
skip_attrs_due_to_errs: dict[str, set[str | Hashable]] = {}
skip_type_due_to_errs: dict[type, int] = {}
skip_type_due_to_recursion: dict[type, int] = {}

logger = logging.getLogger(__name__)


def serialize(obj_dict: dict[str, object | str]) -> str:
    try:
        return orjson.dumps(obj_dict).decode("utf-8")
    except Exception:
        # if orjson fails (e.g. cannot handle ints larger than 64-bit), fallback to json
        return json.dumps(obj_dict)


def monitor_main_thread(main_thread, stop_event):
    main_thread.join()  # Wait for the main thread to finish
    print("Main thread has finished or encountered an exception")
    print("Flushing all buffers to the trace log file")
    stop_event.set()  # Signal the logging threads to stop


def trace_dumper(task_queue: Queue, trace_file_name: str, stop_event: threading.Event):
    with open(trace_file_name, "w") as f:
        while True:
            try:
                trace = task_queue.get(
                    timeout=FLUSH_INTERVAL * 2
                )  # wait for 2x the flush interval, this is an arbitrary number, as long as it is larger than the flush interval, it should be fine.
            except Empty:
                if stop_event.is_set():
                    print("Trace dumper thread has stopped.")
                    break
                continue
            f.write(f"{trace}\n")
            task_queue.task_done()
    print("Trace dumper thread has finished normally...")


def get_trace_API_dumper_queue():
    global THREAD_DATA
    global monitoring_thread
    if monitoring_thread is None:
        monitoring_thread = threading.Thread(
            target=monitor_main_thread, args=(threading.main_thread(), stop_event)
        )
        monitoring_thread.start()

    if hasattr(THREAD_DATA, "API_trace_dumper_queue"):
        return THREAD_DATA.API_trace_dumper_queue

    pid = os.getpid()
    tid = threading.get_ident()

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

    THREAD_DATA.API_trace_dumper_queue = trace_queue
    return trace_queue


def get_trace_VAR_dumper_queue():
    global THREAD_DATA
    global monitoring_thread
    if monitoring_thread is None:
        monitoring_thread = threading.Thread(
            target=monitor_main_thread, args=(threading.main_thread(), stop_event)
        )
        monitoring_thread.start()

    if hasattr(THREAD_DATA, "VAR_trace_dumper_queue"):
        return THREAD_DATA.VAR_trace_dumper_queue

    pid = os.getpid()
    tid = threading.current_thread().ident

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
    THREAD_DATA.VAR_trace_dumper_queue = trace_queue

    return trace_queue


class TraceBuffer:
    """Buffer traces and flush them to the trace log file periodically or when the buffer is full.

    Caveat: **This class must be instantiated PER THREAD*, as the buffer is not thread-safe.
    Also, the queue must be created at the same time as the buffer (as did in the __init__ method).
    Previously, the queue was dynamic getted from the queue_getter method but it caused a mysterious bug,
    where a trace file contains traces of different thread ids and process ids.
    """

    def __init__(
        self, queue_getter, buffer_size=BUFFER_SIZE, flush_interval=FLUSH_INTERVAL
    ):
        self.queue = queue_getter()
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval
        self.buffer = []
        self.last_flush_time = time.time()
        self.flush_thread = threading.Thread(target=self._flush_periodically)
        self.flush_thread.daemon = True
        self.flush_thread.start()

    def add_trace(self, trace):
        self.buffer.append(
            serialize(trace)
        )  # TODO: serialization step cannot be buffered rn as trace dicts might get modified
        if (
            len(self.buffer) >= self.buffer_size
            or (time.time() - self.last_flush_time) >= self.flush_interval
        ):
            self._flush()

    def _flush(self):
        if not self.buffer:
            return
        # serialize all traces in the buffer
        self.queue.put("\n".join(self.buffer))
        self.buffer.clear()
        self.last_flush_time = time.time()

    def _flush_periodically(self):
        while not stop_event.is_set():
            time.sleep(self.flush_interval)
            self._flush()
        self._flush()  # flush any remaining traces before exiting


# THREAD_DATA.api_trace_buffer = TraceBuffer(get_trace_API_dumper_queue)
# THREAD_DATA.var_trace_buffer = TraceBuffer(get_trace_VAR_dumper_queue)


def dump_trace_API(trace: dict):
    """add a timestamp (unix) to the trace and dump it to the trace log file"""
    trace["time"] = get_timestamp_ns()
    if not hasattr(THREAD_DATA, "api_trace_buffer"):
        THREAD_DATA.api_trace_buffer = TraceBuffer(get_trace_API_dumper_queue)

    THREAD_DATA.api_trace_buffer.add_trace(trace)


def dump_trace_VAR(trace: dict):
    """add a timestamp (unix) to the trace and dump it to the trace log file"""
    if "time" not in trace:
        trace["time"] = get_timestamp_ns()
    if not hasattr(THREAD_DATA, "var_trace_buffer"):
        THREAD_DATA.var_trace_buffer = TraceBuffer(get_trace_VAR_dumper_queue)

    THREAD_DATA.var_trace_buffer.add_trace(trace)


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
    if hasattr(tensor, "traincheck_tensor_stats"):
        return tensor.traincheck_tensor_stats
    min = float(tensor.min().item())
    max = float(tensor.max().item())
    mean = float(tensor.mean().item())
    std = float(tensor.std().item())
    shape = tuple(int(x) for x in tensor.size())
    result = {
        "min": min,
        "max": max,
        "mean": mean,
        "std": std,
        "shape": shape,
    }
    tensor.traincheck_tensor_stats = result  # type: ignore
    return result


def dump_tensor(value):
    param_list = None
    if isinstance(value, torch.Tensor):
        if tensor_dump_format["dump_tensor_stats"]:
            param_list = tensor_stats(value)
        elif tensor_dump_format["dump_tensor_hash"]:
            if value.is_cuda:
                param_list = tensor_hash(value, with_parallel=True, with_cuda=True)
            else:
                # TODO: support quick hashing methods for MPS tensors
                param_list = tensor_hash(value, with_parallel=True, with_cuda=False)

        elif tensor_dump_format["dump_tensor_full"]:
            param_list = value.detach().flatten().tolist()
        else:
            raise ValueError(
                "Invalid tensor dump format, please set '--tensor-dump-format' to 'full', 'stats' or 'hash'."
            )

    return param_list


class NOT_FOUND:
    pass


def safe_getattr(obj, attr_name):
    try:
        attr = getattr(obj, attr_name, NOT_FOUND)
        if attr is NOT_FOUND:
            if issubclass(type(obj), dict):
                attr = dict.get(obj, attr_name, NOT_FOUND)
        return attr
    except Exception:
        return NOT_FOUND


def convert_var_to_dict(var, include_tensor_data=True, dump_config=None) -> dict:
    """
    TODO: variables can be nested and thus this should be a recursive function (dump_config supports this).
    But this function is not recursive yet, so it only dumps the first level of attributes of the variable.

    Args:
        var: the variable to be converted to a dictionary.
        include_tensor_data: whether to include the data of a tensor in the dictionary (introduces a lot of overhead).
        dump_config: a dictionary that specifies which attributes to dump for the variable. If None, all attributes will be dumped.
    """

    result: dict[str, object | str] = {}
    # currently only dump primitive types, tensors and nn.Module
    if dump_config is None:
        if (
            type(var) in skip_type_due_to_errs
            and skip_type_due_to_errs[type(var)] > TYPE_ERR_THRESHOLD
        ):
            return result
        try:
            attr_names: list[str] = [
                name for name in dir(var) if not name.startswith("__")
            ]  # dir() won't get called on primitive vars (look at the logic below checking for primitive types) whose dump_config is always None, so no need to check for primitive types here.
            if issubclass(type(var), dict):
                attr_names += [k for k in var.keys() if isinstance(k, str)]
        except Exception as e:
            if type(var) not in skip_type_due_to_errs:
                skip_type_due_to_errs[type(var)] = 0
            skip_type_due_to_errs[type(var)] += 1
            get_instrumentation_logger_for_process().debug(
                f"Failed to get attributes of object type {type(var)}, skipping it. Error: {e}."
            )
            return result
    else:
        # selective instrumentation mode
        attr_names = list(dump_config.keys())

    var_type = str(type(var))

    for attr_name in attr_names:
        if (
            isinstance(attr_name, str)
            and attr_name.startswith("_")
            and not attr_name.startswith("_ML_DAIKON")
        ):
            continue

        if attr_name in attribute_black_list:
            continue

        if (
            var_type in skip_attrs_due_to_errs
            and attr_name in skip_attrs_due_to_errs[var_type]
        ):
            continue

        attr = safe_getattr(var, attr_name)
        if attr is NOT_FOUND:
            logger.warning(
                f"Failed to get attribute {attr_name} of object type {type(var)}, skipping it for all following dumps for this attribute."
            )
            if var_type not in skip_attrs_due_to_errs:
                skip_attrs_due_to_errs[var_type] = set()
            skip_attrs_due_to_errs[var_type].add(attr_name)
            continue

        attr_name = str(attr_name)
        if type(attr) in primitive_types:
            result[attr_name] = attr

        elif isinstance(attr, torch.Tensor):
            result[f"_ML_DAIKON_{attr_name}_ID"] = id(attr)
            if include_tensor_data:
                result[attr_name] = dump_tensor(attr)

        elif isinstance(attr, torch.nn.parameter.Parameter):
            result[f"_ML_DAIKON_{attr_name}_ID"] = id(attr)
            if include_tensor_data:
                result[attr_name] = dump_tensor(attr.data)

        elif include_tensor_data and isinstance(attr, torch.nn.Module):
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

    if include_tensor_data and "data" not in result and isinstance(var, torch.Tensor):
        raise ValueError(
            f"Failed to dump tensor data of tensor {var}, please turn on debugging mode and see the debugging log."
        )
    return result


def obj_to_serializable(obj, dump_config=None) -> dict[str, object]:
    if (
        type(obj) in skip_type_due_to_recursion
        and skip_type_due_to_recursion[type(obj)] > RECURSION_ERR_THRESHOLD
    ):
        return {str(type(obj)): None}

    if isinstance(obj, torch.dtype):
        return {typename(obj, is_runtime=True): str(obj)}
    elif isinstance(obj, torch.Size):
        return {typename(obj, is_runtime=True): tuple(obj)}
    try:
        var_dict = convert_var_to_dict(
            obj, include_tensor_data=False, dump_config=dump_config
        )
        return {typename(obj, is_runtime=True): var_dict}
    except RecursionError:
        skip_type_due_to_recursion[type(obj)] = (
            skip_type_due_to_recursion.get(type(obj), 0) + 1
        )
        logger.warning(
            f"Recursion detected when converting object to dict. Probably due to a issue in the __getattr__ method of the object. Object type: {type(obj)}."
        )
        return {str(type(obj)): None}


def var_to_serializable(obj, dump_config=None) -> dict[str, object]:
    """Convert any object to a serializable dictionary.

    Note that this function is largely a wrapper of convert_var_to_dict to add some heuristics about how to dump a few types of objects.
      and it does not dump the `data` attribute of a tensor.
    If you want to dump the `data` attribute of a tensor, use `convert_var_to_dict` and set `include_tensor_data=True`.
    """

    if issubclass(type(obj), dict) and type(obj) != dict:  # noqa E721
        return obj_to_serializable(obj, dump_config=dump_config)

    try:
        json.dumps(
            obj
        )  # HACK: using json instead of to check if obj is serializable as it always raises an exception if obj is not serializable, orjson may or may not raise an exception for unknown reasons.
        return {typename(obj, is_runtime=True): obj}
    except (TypeError, AttributeError):
        return obj_to_serializable(obj, dump_config=dump_config)
        # assert var_dict, f"Failed to convert object {obj} to dict."
