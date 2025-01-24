import datetime
import functools
import importlib
import inspect
import json
import logging
import os
import threading
import time
import traceback
import types
import uuid
from typing import Any, Callable, Optional

import torch
import torch.utils

import mldaikon.config.config as config  # needed to allow for change of values after import
from mldaikon.config.config import (
    INSTR_MODULES_TO_SKIP,
    TRAIN_STEP_NAMES,
    WRAP_WITHOUT_DUMP,
)
from mldaikon.instrumentor.caches import cache_meta_vars, meta_vars
from mldaikon.instrumentor.dumper import (
    convert_var_to_dict,
    dump_trace_API,
    dump_trace_VAR,
    get_instrumentation_logger_for_process,
    var_to_serializable,
)
from mldaikon.instrumentor.replace_functions import (
    funcs_to_be_replaced,
    is_funcs_to_be_unproxied,
)
from mldaikon.instrumentor.types import PTID
from mldaikon.proxy_wrapper.proxy_basics import is_proxied, unproxy_func
from mldaikon.proxy_wrapper.proxy_config import enable_C_level_observer
from mldaikon.utils import typename

_instancemethod_t = type(torch._C._distributed_c10d.ProcessGroup.broadcast)

METRIC_INSTRUMENTED_FUNC_LIST: dict[str, list[str]] = {"dump": [], "no_dump": []}

IS_INSTRUMENTING = False

DISABLE_WRAPPER = False

# for prompt generation tasks using the transformers library (see mldaikon/developer/instr_stage_annotation.py:annotate_answer_start_token_ids)
GENERATE_START_TOKEN_ID: None | int = None
GENERATE_START_TOKEN_ID_INCLUDE_START_TOKEN = False

COLLECT_OVERHEAD_METRICS = os.environ.get("COLLECT_OVERHEAD_METRICS", "0") == "1"


class TraceLineType:
    FUNC_CALL_PRE = "function_call (pre)"
    FUNC_CALL_POST = "function_call (post)"
    FUNC_CALL_POST_EXCEPTION = "function_call (post) (exception)"
    STATE_CHANGE = "state_change"


def is_c_level_function(original_function):
    return not hasattr(original_function, "__code__")


def get_meta_vars() -> dict:
    """HACK: this function is a hack to get the meta_vars from the related frames

    it should only get
    """

    frame = inspect.currentframe()

    all_frame_vars: dict[str, object] = {}
    # get the file name list inside the repo
    while frame is not None:
        if "mldaikon" in frame.f_code.co_filename:
            frame = frame.f_back
            continue

        frame_vars = frame.f_locals

        file_full_path = frame.f_code.co_filename
        # can we also get the function we are inside?
        func_name = frame.f_code.co_name

        if "/site-packages/" in file_full_path:
            file_full_path = file_full_path.split("/site-packages/")[1]
        file_full_path = file_full_path.strip("/home/")

        new_frame_vars = {}
        for name, value in frame_vars.items():
            for step_var_name in TRAIN_STEP_NAMES:
                if step_var_name.lower() in name.lower():
                    if isinstance(value, (int, float)):
                        if name in all_frame_vars:
                            name = f"{name}_2"
                        new_frame_vars[name] = value
                        break
                    # elif not callable(value):
                    #     print("YUXUAN DEBUGGING: ", name, value, type(value))
        frame = frame.f_back
        # frame_vars = {
        #     name: value
        #     for name, value in frame_vars.items()
        #     # Ziming: only dump primitive types, block the var name on the black list
        #     if isinstance(value, (int, float, str, bool))
        #     and (
        #         not name.startswith("__")
        #         and "mldaikon" not in name
        #         and name not in META_VARS_FORBID_LIST
        #     )
        # }

        full_path = f"{file_full_path}:{func_name}"
        if new_frame_vars:
            if full_path not in all_frame_vars:
                all_frame_vars[full_path] = new_frame_vars
            else:
                print("YUXUAN DEBUGGING: RECURSIVE CALL DETECTED", full_path)
                while f"{full_path}_2" in all_frame_vars:
                    full_path = f"{full_path}_2"
                all_frame_vars[f"{full_path}_2"] = new_frame_vars

        # update the meta_vars with the current frame_vars
        all_frame_vars.update(meta_vars)
    return all_frame_vars


def should_dump_trace(
    cond_dump: bool,
    ptid: PTID | None,
    key,
    meta_vars: dict[str, Any] | None,
    meta_vars_targets: list[str] | None,
    update_cache: bool = True,
) -> bool:
    """Determine if trace dumping should be enabled for this particular function call
    - cond_dump (bool): whether conditional dumping should be enabled at all, if False, always return True
    - ptid (PTID): process and thread id, if None, will be inferred from the current process and thread
    - key (str): a unique key that can identify the function or the variable to be dumped
    - meta_vars (dict[str, Any]): the current meta_vars, if None, will be inferred from the current frame
    - meta_vars_targets (list[str]|None): a subset of keys in meta_vars that should be used to determine if the trace should be dumped
    - update_cache (bool): whether to update the cache_meta_vars with the current meta_vars if the trace should be dumped

    If True is returned, cache_meta_vars will be updated with the current meta_vars
    """
    global IS_INSTRUMENTING
    if IS_INSTRUMENTING:
        # don't dump anything during instrumentation
        return False

    if not cond_dump:
        return True

    # conditional dumping logic
    if ptid is None:
        tid = threading.current_thread().ident
        assert tid is not None, "threading.current_thread().ident is None"
        ptid = PTID(os.getpid(), tid)

    if meta_vars is None:
        meta_vars = get_meta_vars()

    prev_meta_vars = cache_meta_vars[ptid][key]
    if not prev_meta_vars:
        if update_cache:
            cache_meta_vars[ptid][key] = meta_vars
        return True

    prev_targets = prev_meta_vars
    targets = meta_vars
    if meta_vars_targets:
        prev_targets = {
            k: v for k, v in prev_meta_vars.items() if k in meta_vars_targets
        }
        targets = {k: v for k, v in meta_vars.items() if k in meta_vars_targets}

    # only if the meta_vars have changed, we will dump the trace
    if prev_targets != targets:
        if update_cache:
            cache_meta_vars[ptid][key] = meta_vars
        return True

    return False


def to_dict_args_kwargs(args, kwargs) -> dict:
    global DISABLE_WRAPPER
    DISABLE_WRAPPER = True
    result = {
        "args": [var_to_serializable(arg) for arg in args],
        "kwargs": {k: var_to_serializable(v) for k, v in kwargs.items()},
    }
    DISABLE_WRAPPER = False
    return result


def to_dict_return_value(result) -> dict | list[dict]:
    global DISABLE_WRAPPER
    DISABLE_WRAPPER = True
    result_dict: dict | list[dict]
    if isinstance(result, tuple):
        result_dict = [var_to_serializable(r) for r in result]
    else:
        result_dict = var_to_serializable(result)

    DISABLE_WRAPPER = False
    return result_dict


def global_wrapper(
    original_function,
    is_bound_method,
    is_builtin,
    scan_proxy_in_args,
    dump_stack_trace,
    cond_dump,
    dump_args,
    dump_ret,
    handle_proxy,
    *args,
    **kwargs,
):
    """Instrumentation for APIs

    Pre-call Phase
    1. Log the pre-call information
    2. Unproxy the arguments if the function is a C level function -- Proxy objects passed to built-in functions will cause segfault
    3. Add additional 'observer' (monitoring whether the input arguments have changed after the function call) to the function if specified

    Call Phase
    1. Calls the original function
    2. If an exception is raised, log the exception and re-raise it

    Post-call Phase
    1. Log the post-call information
    """

    global DISABLE_WRAPPER
    if DISABLE_WRAPPER:
        return original_function(*args, **kwargs)

    ENTER_PERF_TIME = time.perf_counter()

    logger = logging.getLogger(__name__)

    func_call_id = uuid.uuid4().hex
    thread_id = threading.current_thread().ident
    process_id = os.getpid()
    func_name = typename(original_function)
    pre_meta_vars = get_meta_vars()

    # determine at runtime whether to dump the trace
    is_dumping = should_dump_trace(
        cond_dump,
        PTID(process_id, thread_id),
        f"API_{func_name}",  # any key that can uniquely identify the function or the variable to be dumped
        meta_vars=pre_meta_vars,
        meta_vars_targets=None,  # can be used to restrain the meta_vars to a subset of keys
        update_cache=True,
    )

    if not is_dumping:
        if handle_proxy:
            return core_wrapper(
                original_function, is_builtin, handle_proxy, *args, **kwargs
            )
        else:
            return original_function(
                *args, **kwargs
            )  # avoid additional function call as core_wrapper only handles proxy

    pre_record = {
        "func_call_id": func_call_id,
        "thread_id": thread_id,
        "process_id": process_id,
        "meta_vars": pre_meta_vars,
        "type": TraceLineType.FUNC_CALL_PRE,
        "function": func_name,
        "is_bound_method": is_bound_method,
        "obj_id": None if not is_bound_method else id(args[0]),
    }

    if dump_stack_trace:
        pre_record["stack_trace"] = traceback.format_stack()

    if scan_proxy_in_args:
        proxy_in_args = []

        def find_proxy_in_args(args):
            for i, arg in enumerate(args):
                if is_proxied(arg):
                    proxy_in_args.append(arg)
                elif type(arg) in [list, tuple]:
                    find_proxy_in_args(arg)
                elif isinstance(arg, types.GeneratorType) and not isinstance(
                    arg, tuple
                ):
                    arg_list = list(arg)
                    args[i] = iter(arg_list)
                    find_proxy_in_args(arg_list)

        args = list(args)
        find_proxy_in_args(args)
        args = tuple(args)

        if proxy_in_args:
            if "proxy_obj_names" not in pre_record:
                pre_record["proxy_obj_names"] = []
            for proxy in proxy_in_args:
                pre_record["proxy_obj_names"].append(
                    [proxy.__dict__["var_name"], type(proxy._obj).__name__]
                )
    if dump_args:
        dict_args_kwargs = to_dict_args_kwargs(args, kwargs)
        pre_record["args"] = dict_args_kwargs["args"]
        pre_record["kwargs"] = dict_args_kwargs["kwargs"]
    dump_trace_API(pre_record)

    if handle_proxy:
        if enable_C_level_observer and is_builtin:
            from mldaikon.proxy_wrapper.proxy_observer import (
                add_observer_to_func,  # import here to avoid circular import
            )

            original_function = add_observer_to_func(
                original_function, cond_dump=cond_dump, unproxy=True
            )
        elif is_funcs_to_be_unproxied(original_function):
            original_function = unproxy_func(
                original_function, inspect_torch_module=True
            )
        elif is_builtin:
            # proxy objects being passed to backend will cause seg fault: TODO: replace with unproxy func
            original_function = unproxy_func(original_function)

    try:
        ORIG_ENTER_PERF_TIME = time.perf_counter()
        result = original_function(*args, **kwargs)
        ORIG_EXIT_PERF_TIME = time.perf_counter()
    except Exception as e:
        ORIG_EXIT_PERF_TIME = time.perf_counter()
        dump_trace_API(
            {
                "func_call_id": func_call_id,
                "thread_id": thread_id,
                "process_id": process_id,
                "meta_vars": pre_meta_vars,
                "type": TraceLineType.FUNC_CALL_POST_EXCEPTION,
                "function": func_name,
                # "args": [f"{arg}" for arg in args],
                # "kwargs": [f"{k}={v}" for k, v in kwargs.items()],
                "exception": typename(e),
                "exception_msg": str(e),
                "is_bound_method": is_bound_method,
                "obj_id": None if not is_bound_method else id(args[0]),
            },
        )
        logger.error(f"Error in {func_name}: {type(e)} {e}")
        EXIT_PERF_TIME = time.perf_counter()
        (
            print(
                f"WRAPPER TIME: {func_name},{ORIG_EXIT_PERF_TIME - ORIG_ENTER_PERF_TIME},{EXIT_PERF_TIME - ENTER_PERF_TIME}"
            )
            if COLLECT_OVERHEAD_METRICS
            else None
        )
        raise e
    pre_record.pop("args", None)
    pre_record.pop("kwargs", None)
    post_record = (
        pre_record.copy()
    )  # copy the pre_record (though we don't actually need to copy anything)
    post_record["type"] = TraceLineType.FUNC_CALL_POST
    post_record["meta_vars"] = pre_meta_vars

    result_to_dump = result

    # if the current function name is transformers.generate, then we will dump the response tokens only, let's see.
    # a concrete name: "transformers.models.whisper.modeling_whisper.WhisperForConditionalGeneration.generate"
    # we want a pattern that abstracts the specific model name
    pattern = "transformers.models.*.generate"
    # find matches in the pattern
    import re

    if (
        GENERATE_START_TOKEN_ID is not None
        and re.match(pattern, func_name)
        and isinstance(result, torch.Tensor)
    ):
        print(f"Found match for {func_name}")
        # the first dimension is the batch size, and each corresponds to a separate response, let's try to match the batch size with the start token ids first
        response_starting_indices = []
        for i in range(result.size(0)):
            # try to find the match of the start token ids in the response
            response = result[i]
            # Find all indices where the start_token_id matches
            matches = (response == GENERATE_START_TOKEN_ID).nonzero(as_tuple=True)[0]
            if len(matches) == 0:
                # No occurrences found
                print(
                    f"start_token_id ({GENERATE_START_TOKEN_ID}) not found in response {i}"
                )
                start_index = -1  # Handle case where token is not found
            elif len(matches) > 1:
                # Multiple occurrences found, raise an error
                raise ValueError(
                    f"Multiple occurrences of start_token_id ({GENERATE_START_TOKEN_ID}) found in response {i}: {matches.tolist()}"
                )
            else:
                # Single occurrence found, get the index
                start_index = matches.item()
                if not GENERATE_START_TOKEN_ID_INCLUDE_START_TOKEN:
                    start_index += 1

            response_starting_indices.append(start_index)

        # compute the length of each response
        response_lengths = []
        for i in range(result.size(0)):
            response = result[i]
            start_index = response_starting_indices[i]
            if start_index == -1:
                response_lengths.append(0)
            else:
                response_lengths.append(response.size(0) - start_index)

        result_to_dump = result.detach()
        setattr(
            result_to_dump,
            "_ML_DAIKON_RESPONSE_STARTING_INDICES",
            response_starting_indices,
        )
        setattr(result_to_dump, "_ML_DAIKON_RESPONSE_LENGTHS", response_lengths)

        print(response_starting_indices)
        print(response_lengths)
    if dump_ret:
        post_record["return_values"] = to_dict_return_value(result_to_dump)
    dump_trace_API(post_record)

    EXIT_PERF_TIME = time.perf_counter()
    (
        print(
            f"WRAPPER TIME: {func_name},{ORIG_EXIT_PERF_TIME - ORIG_ENTER_PERF_TIME},{EXIT_PERF_TIME - ENTER_PERF_TIME}"
        )
        if COLLECT_OVERHEAD_METRICS
        else None
    )
    return result


def core_wrapper(original_function, is_builtin, handle_proxy, *args, **kwargs):
    """same as global_wrapper but without the logging, will have lower overhead than global_wrapper
    We use this wrapper on the functions that are not helpful for invariant inference,  but still needs to be instrumented to handle proxy classes
    """
    global DISABLE_WRAPPER
    if DISABLE_WRAPPER:
        return original_function(*args, **kwargs)

    if handle_proxy and is_builtin:
        original_function = unproxy_func(original_function)
    return original_function(*args, **kwargs)


def wrapper(
    original_function,
    is_bound_method,
    scan_proxy_in_args,
    dump_stack_trace,
    cond_dump,
    disable_dump=False,
    dump_args=True,
    dump_ret=True,
    handle_proxy=True,
):
    is_builtin = is_c_level_function(original_function)

    # determine statically whether to dump the trace
    if not disable_dump:
        METRIC_INSTRUMENTED_FUNC_LIST["dump"].append(typename(original_function))

        @functools.wraps(original_function)
        def wrapped(*args, **kwargs):
            return global_wrapper(  # the wrapper cannot be invoked with named parameters as *args has to be after the named parameters
                original_function,
                is_bound_method,
                is_builtin,
                scan_proxy_in_args,
                dump_stack_trace,
                cond_dump,
                dump_args,
                dump_ret,
                handle_proxy,
                *args,
                **kwargs,
            )

    else:
        METRIC_INSTRUMENTED_FUNC_LIST["no_dump"].append(typename(original_function))

        if handle_proxy:

            @functools.wraps(original_function)
            def wrapped(*args, **kwargs):
                return core_wrapper(
                    original_function, is_builtin, handle_proxy, *args, **kwargs
                )

        else:
            return original_function

    wrapped._ml_daikon_original_function = original_function
    wrapped._ml_daikon_instrumented = True
    return wrapped


def safe_serialize(obj):
    """Include custom serialization logic to handle parameters that cannot be serialized by json.dumps"""
    try:
        if isinstance(obj, torch.Tensor):
            return f"Tensor(shape={obj.size()}, dtype={obj.dtype})"
        return json.dumps(obj)
    except TypeError:
        return str(type(obj))


# https://stackoverflow.com/a/63851681/9201239
def get_all_subclasses(cls):
    subclass_list = []

    def recurse(cl):
        for subclass in cl.__subclasses__():
            subclass_list.append(subclass)
            recurse(subclass)

    recurse(cls)
    return set(subclass_list)


def log_instrumentation_progress(
    depth: int,
    msg: str,
    attr: object | None,
    attr_name: str | None,
    pymodule: types.ModuleType | type,
):
    if attr_name is None:
        attr_name = ""
    get_instrumentation_logger_for_process().info(
        f"Depth: {depth}, {msg}: {attr_name}, {typename(attr) if attr is not None else 'attr not provided'}, {typename(pymodule)}"
    )


modules_or_cls_id_instrumented = set()


def mark_module_or_cls_as_visited(module: object):
    # not using a flag here as some classes do not allow setting attributes or flags
    # the goal of marking the module as visited is to avoid cycles in the module graph
    modules_or_cls_id_instrumented.add(id(module))


def is_module_or_cls_instrumented(module: object) -> bool:
    return id(module) in modules_or_cls_id_instrumented


def is_API_instrumented(obj: Callable) -> bool:
    # APIs has to be marked with a flag as ids will be changed after instrumentation, and also having the same id would mean that the object is not instrumented (e.g. multiple references to the same object)
    try:
        # we cannot use hasattr as it would trigger the __getattr__ method of the object, and can lead to exceptions at https://github.com/pytorch/pytorch/blob/main/torch/_ops.py#L1029-L1031
        return obj.__dict__.get("_ml_daikon_instrumented", False)
    except Exception:
        # a wrapped API would have __dict__ and have the flag
        return False


def is_API_bound_method(obj: Callable) -> bool:
    """We will see if the object will be a bound method or not. If the object is a bound method, we will return True, else False"""
    logger = logging.getLogger(__name__)
    signature = None

    # handle the case where the object is already a bound method, theoretically, this should not happen
    if inspect.ismethod(obj):
        logger.warning(f"Object is already a bound method: {obj}")
        return True

    # handle the case where the object is a method not instantiated yet, e.g. torch.optim.Adam.step is a method, but not a bound method yet
    try:
        signature = inspect.signature(obj)
    except (
        ValueError
    ) as e:  # inspect.signature raises ValueError if no signature is found, TypeError if obj is not a callable
        logger.debug(f"Error in inspect.signature: {e}")
        return False
    param_names = list(signature.parameters.keys())
    return len(param_names) > 0 and "self" == param_names[0]


def get_module_path_from_file_path(file_path: str, root_module: str) -> str | None:
    # import root_module and get root module
    if (
        not file_path.endswith(".py")
        or not os.path.exists(file_path)
        or f"/{root_module}/" not in file_path
    ):
        return None
    # get the path of the module from the file path
    path_after_root_module = file_path.split(f"/{root_module}/")[1].split(".py")[0]
    module_path = f"{root_module}.{path_after_root_module}".replace("/", ".")
    return module_path


class Instrumentor:
    def __init__(
        self,
        target: (
            types.ModuleType
            | type
            | types.FunctionType
            | types.BuiltinFunctionType
            | types.BuiltinMethodType
        ),
        scan_proxy_in_args: bool,
        use_full_instr: bool,
        funcs_to_instr: Optional[list[str]] = None,
        API_dump_stack_trace: bool = False,
        cond_dump: bool = False,
    ):
        """
        Instruments the specified target with additional tracing functionality.

        Args:
            target:
                The module, class, or function to instrument.
                Note: Instrumenting functions is not supported; calling this will do nothing.
            scan_proxy_in_args (bool):
                Whether to scan the arguments of the function for proxy objects.
                Enabling this will allow the instrumentor to log the proxy objects in the function arguments,
                which can be useful to establish the causal relationship between the proxy objects and the function calls.
                Enabling this leads to a mild 2% overhead on 84911.
            use_full_instr (bool):
                Whether to dump trace for all APIs. If False, APIs in certain modules deemed to be not important (e.g. `jit` in `torch`) will not have trace being dumped.
                Refer to WRAP_WITHOUT_DUMP in config.py for the list of functions/modules that will have the dump disabled.
            funcs_to_instr (Optional[List[Callable]]):
                An optional list of functions that are of interest for invariant inference.
                If provided, all functions not in this list will be instrumented with dump disabled,
                and the functions in this list will be instrumented with dump enabled. NOTE: If this list is provided, use_full_str must be set to False. WRAP_WITHOUT_DUMP will be ignored.
            API_dump_stack_trace (bool):
                Whether to dump the stack trace of the function call. Enabling this will add the stack trace to the trace log.
            cond_dump (bool):
                Whether to dump the trace conditionally. If True, the trace will only be dumped if meta_vars have changed since the last call of this particular function.
                This might cause additional overhead (cpu and memory) as the meta_vars will be compared with the previous call, and meta_vars will have to be cached in memory.

        Indirectly, at initialization, the instrumentor will also load the instr_opts.json file if it exists.
        This file is automatically generated by the `collect_trace` script when `--invariants` is provided.
        The user should not need to interact with this file directly.

        """

        self.instrumenting = True
        if isinstance(target, types.ModuleType):
            self.root_module = target.__name__.split(".")[0]
        elif inspect.isclass(target):
            self.root_module = target.__module__.split(".")[0]
        elif callable(target):
            get_instrumentation_logger_for_process().warning(
                f"""Unsupported target {target}. This instrumentor does not support function, 
                due to inability to swap the original function with the wrapper function 
                in the namespace. However, you can use the wrapper function directly by 
                setting 
                    `func = wrapper(func)`
                """
            )
            self.instrumenting = False
        else:
            get_instrumentation_logger_for_process().warning(
                f"Unsupported target {target}. This instrumentor only supports module, class."
            )
            self.instrumenting = False
        self.instrumented_count = 0
        self.target = target
        self.scan_proxy_in_args = scan_proxy_in_args
        self.use_full_instr = use_full_instr
        self.funcs_to_instr = funcs_to_instr
        self.API_dump_stack_trace = API_dump_stack_trace
        self.cond_dump = cond_dump
        self.instr_opts: None | dict[str, dict[str, dict[str, bool]]] = None

        if self.funcs_to_instr is not None and self.use_full_instr:
            get_instrumentation_logger_for_process().fatal(
                "Invariants are provided but use_full_instr is True. Selective instrumentation cannot be done. Please remove the `--use-full-instr` flag or remove the invariants"
            )
            raise ValueError(
                "Invariants are provided but use_full_instr is True. Selective instrumentation cannot be done. Please remove the `--use-full-instr` flag or remove the invariants"
            )

        if self.funcs_to_instr is not None:
            get_instrumentation_logger_for_process().info(
                f"Functions of interest for invariant inference: {self.funcs_to_instr}"
            )

        # discover if instr_opts.json is present
        instr_opts_path = config.INSTR_OPTS_FILE
        print("instr_opts_path: ", instr_opts_path)
        get_instrumentation_logger_for_process().info(
            f"Checking instr_opts at {instr_opts_path}"
        )
        if os.path.exists(instr_opts_path):
            print(f"Loading instr_opts from {instr_opts_path}")
            with open(instr_opts_path, "r") as f:
                instr_opts = json.load(f)
                self.instr_opts = instr_opts
            get_instrumentation_logger_for_process().info(
                f"Loaded instr_opts: {json.dumps(instr_opts, indent=4)}"
            )

    def instrument(self) -> int:
        if not self.instrumenting:
            return 0

        global IS_INSTRUMENTING
        IS_INSTRUMENTING = True
        visited_file_paths: set[str] = set()

        first_pass_instrumented_count = 0
        get_instrumentation_logger_for_process().info(
            "First pass: Recursive scan of the module"
        )
        assert isinstance(self.target, (types.ModuleType, type)), "Invalid target"
        first_pass_instrumented_count += self._instrument_module(
            self.target, visited_file_paths, True, 0
        )
        get_instrumentation_logger_for_process().info(
            "Files scanned %s", "\n".join(sorted(visited_file_paths))
        )
        get_instrumentation_logger_for_process().info(
            "First pass instrumented %d functions", first_pass_instrumented_count
        )

        get_instrumentation_logger_for_process().info(
            "Second pass: Direct instrumentation of the files"
        )
        second_pass_instrumented_count = 0
        for file_path in sorted(visited_file_paths):
            module_path = get_module_path_from_file_path(file_path, self.root_module)
            if module_path is None or "__init__" in module_path:
                get_instrumentation_logger_for_process().info(
                    f"Skipping file {file_path}"
                )
                continue

            get_instrumentation_logger_for_process().info(
                f"Instrumenting module {module_path}"
            )

            pymodule = importlib.import_module(module_path)
            second_pass_instrumented_count += self._instrument_module(
                pymodule,
                visited_file_paths,
                False,
                0,
            )
        get_instrumentation_logger_for_process().info(
            "Second pass instrumented %d functions", second_pass_instrumented_count
        )

        self.instrumented_count = (
            first_pass_instrumented_count + second_pass_instrumented_count
        )

        # sort the instrumented functions by their name
        METRIC_INSTRUMENTED_FUNC_LIST["dump"] = sorted(
            METRIC_INSTRUMENTED_FUNC_LIST["dump"]
        )
        METRIC_INSTRUMENTED_FUNC_LIST["no_dump"] = sorted(
            METRIC_INSTRUMENTED_FUNC_LIST["no_dump"]
        )

        # dump the instrumented functions
        get_instrumentation_logger_for_process().info(
            "Functions instrumented with trace dumping enabled:\n%s",
            "\n".join(METRIC_INSTRUMENTED_FUNC_LIST["dump"]),
        )
        get_instrumentation_logger_for_process().info(
            "Functions instrumented with trace dumping disabled:\n%s",
            "\n".join(METRIC_INSTRUMENTED_FUNC_LIST["no_dump"]),
        )

        # do some simple checking for correctness:
        # 1. if funcs_to_instr is provided, then METRIC_INSTRUMENTED_FUNC_LIST["dump"] should be equal to funcs_to_instr
        if self.funcs_to_instr is not None:
            # assert set(METRIC_INSTRUMENTED_FUNC_LIST["dump"]) == set(
            #     self.funcs_to_instr
            # ), f"METRIC_INSTRUMENTED_FUNC_LIST['dump'] != funcs_to_instr, diff: {set(METRIC_INSTRUMENTED_FUNC_LIST['dump']) ^ set(self.funcs_to_instr)}"
            assert set(METRIC_INSTRUMENTED_FUNC_LIST["dump"]).issubset(
                set(self.funcs_to_instr)
            ), f"Actual functions being instrumented are not a subset of the functions required by the provided invariants, diff: {set(METRIC_INSTRUMENTED_FUNC_LIST['dump']) ^ set(self.funcs_to_instr)}"

            if set(METRIC_INSTRUMENTED_FUNC_LIST["dump"]) != set(self.funcs_to_instr):
                get_instrumentation_logger_for_process().warning(
                    f"Not all functions required by the provided invariants are instrumented (e.g. due to transfering ), some invariants might not be active at all, funcs not instrumented: {set(METRIC_INSTRUMENTED_FUNC_LIST['dump']) ^ set(self.funcs_to_instr)}"
                )  # TODO: report a number of functions not instrumented and thus the invariants that will not be active

        IS_INSTRUMENTING = False
        return self.instrumented_count

    def _should_skip_module_or_cls(self, pymodule: object) -> str | None:
        module_or_cls = "class" if inspect.isclass(pymodule) else "module"

        if typename(pymodule) in INSTR_MODULES_TO_SKIP:
            return f"Skipping {module_or_cls} as it is in INSTR_MODULES_TO_SKIP"

        for modules_to_skip_prefix in INSTR_MODULES_TO_SKIP:
            if typename(pymodule).startswith(modules_to_skip_prefix):
                return f"Skipping {module_or_cls} as it is in INSTR_MODULES_TO_SKIP"

        if not typename(pymodule).startswith(self.root_module):
            return f"Skipping {module_or_cls} as it does not belong to the target"

        return None

    def _should_skip_instr_attr(self, attr_name: str, pymodule: object) -> str | None:
        # 1. skip attrs with no objects (e.g. __abstractmethods__ and C extension functions)
        attr = pymodule.__dict__.get(attr_name, None)
        if attr is None:
            # try getting it in case it is a descriptor (almost certainly will be)
            try:
                attr = getattr(pymodule, attr_name)
                if not (
                    config.INSTR_DESCRIPTORS and "method_descriptor" in str(type(attr))
                ):
                    # print("TRIGGERED", attr_name)
                    attr = None
            except Exception:
                pass

        if attr is None:
            return "Skipping attribute as it is None"

        # 2. Skip if the attribute is already instrumented
        if is_API_instrumented(attr):
            return "Skipping attribute as it is already instrumented"

        if type(attr).__name__ == "_OpNamespace":
            return "Skipping attribute as it is _OpNamespace (calling typename on it will raise an exception)"

        # 3. Instrumenting inspect.getfile lead to --> TypeError: module, class, method, function, traceback, frame, or code object was expected, got builtin_function_or_method"
        if "getfile" in attr_name:  # cannot handle getfile correctly
            return "Skipping attribute as it is getfile"

        # 3. Skip magic methods except __init__ and __call__ # TODO: try if __init__ and __call__ can be instrumented
        if (
            attr_name.startswith("__")
            and attr_name.endswith("__")
            and attr_name not in ["__init__", "__call__", "__enter__", "__exit__"]
        ):
            return "Skipping magic functions"

        # print("attr_name: ", attr_name)
        if "_ClassNamespace" in repr(attr):
            return "Skipping attribute as it is _ClassNamespace and getting the qualname will raise an exception"

        attr_full_name = typename(attr)
        # 4. Skip if the attribute is in INSTR_MODULES_TO_SKIP | MANUAL CONFIG
        if attr_full_name in INSTR_MODULES_TO_SKIP:
            return "Skipping attribute as it is one of INSTR_MODULES_TO_SKIP"

        # 5. Skip if the attribute is in modules_to_skip_prefix | MANUAL CONFIG
        for modules_to_skip_prefix in INSTR_MODULES_TO_SKIP:
            if attr_full_name.startswith(modules_to_skip_prefix):
                return "Skipping attribute as it is in INSTR_MODULES_TO_SKIP"

        # 6. Skip if the attribute does not belong to the target root module
        if not attr_full_name.startswith(self.root_module) and not (
            config.INSTR_DESCRIPTORS
            and ("method_descriptor" in attr_full_name or "Tensor" in attr_full_name)
        ):
            # builtin methods in torch.Tensor's qualname does not start with torch for some reason
            return "Skipping attribute as it does not belong to the root module"

        return None

    def should_disable_dump(self, attr) -> bool:
        """Check if the dump should be disabled for the attribute.
        If use_full_instr is True, then the dump will not be disabled.
        If funcs_to_instr is provided, then the dump will be disabled for all functions except the ones in funcs_to_instr.
        If the attribute is in WRAP_WITHOUT_DUMP, then the dump will be disabled. Otherwise, the dump will not be disabled.
        """

        if self.use_full_instr:
            return False

        if self.funcs_to_instr is not None:
            if typename(attr) in self.funcs_to_instr:
                return False
            return True

        logger = logging.getLogger(__name__)
        attr_name = typename(attr)
        for wrap_without_dump_module in WRAP_WITHOUT_DUMP:
            if attr_name.startswith(wrap_without_dump_module):
                logger.debug(
                    f"Skipping dump for {attr_name} as it is in WRAP_WITHOUT_DUMP {wrap_without_dump_module}"
                )
                return True
        return False

    def get_wrapped_function(self, func_obj: Callable) -> Callable:
        """Get the wrapped function for the provided function object"""
        used_proxy = True  # TODO: dump instr_opts when doing full instr as well so we can determine whether to handle proxy based on the specific instrumentation args
        if self.instr_opts is not None:
            used_proxy = (
                "model_tracker_style" in self.instr_opts
                and self.instr_opts["model_tracker_style"] == "proxy"
            )
            func_name = typename(func_obj)
            if func_name not in self.instr_opts["funcs_instr_opts"]:
                return wrapper(
                    func_obj,
                    is_bound_method=None,
                    scan_proxy_in_args=None,
                    dump_stack_trace=None,
                    cond_dump=None,
                    disable_dump=True,
                    handle_proxy=used_proxy,
                )

            instr_opts = self.instr_opts["funcs_instr_opts"][func_name]
            return wrapper(
                func_obj,
                is_bound_method=is_API_bound_method(func_obj),
                scan_proxy_in_args=instr_opts["scan_proxy_in_args"],
                disable_dump=self.should_disable_dump(func_obj),
                dump_stack_trace=self.API_dump_stack_trace,
                cond_dump=self.cond_dump,
                dump_args=instr_opts["dump_args"],
                dump_ret=instr_opts["dump_ret"],
                handle_proxy=used_proxy,
            )

        return wrapper(
            func_obj,
            is_bound_method=is_API_bound_method(func_obj),
            scan_proxy_in_args=self.scan_proxy_in_args,
            disable_dump=self.should_disable_dump(func_obj),
            dump_stack_trace=self.API_dump_stack_trace,
            cond_dump=self.cond_dump,
            handle_proxy=used_proxy,
        )

    def _instrument_module(
        self,
        pymodule: types.ModuleType | type,
        visited_file_paths: set,
        recurse_into_sub_module: bool,
        depth,
    ):
        target_name = pymodule.__name__

        if not recurse_into_sub_module and inspect.ismodule(pymodule):
            # not recurse_into_sub_module means that we are in the second pass, and we are directly instrumenting the module
            # we should not skip the module even if it is already instrumented as the first pass might have skipped private functions
            pass
        else:
            if is_module_or_cls_instrumented(pymodule):
                module_or_cls = "class" if inspect.isclass(pymodule) else "module"
                get_instrumentation_logger_for_process().info(
                    f"Depth: {depth}, Skipping {module_or_cls}: {target_name}, Reason: Already instrumented"
                )
                return 0

        # if pymodule in instrumented_modules or pymodule in skipped_modules:
        if reason := self._should_skip_module_or_cls(pymodule):
            get_instrumentation_logger_for_process().info(
                f"Depth: {depth}, Skipping module: {target_name}, Reason: {reason}"
            )
            return 0

        get_instrumentation_logger_for_process().info(
            f"Depth: {depth}, Instrumenting module: {target_name}"
        )

        mark_module_or_cls_as_visited(pymodule)

        count_wrapped = 0
        for attr_name in dir(pymodule):
            attr = pymodule.__dict__.get(attr_name)
            if attr is None:
                # try access this attribute to handle lazy loading
                try:
                    attr = getattr(pymodule, attr_name)
                except Exception as e:
                    get_instrumentation_logger_for_process().debug(
                        f"Depth: {depth}, lazy loading failed for attribute: {attr_name}, Module: {target_name}: {e}"
                    )

            if reason := self._should_skip_instr_attr(attr_name, pymodule):
                get_instrumentation_logger_for_process().debug(
                    f"Depth: {depth}, Skipping attribute: {attr_name}, Reason: {reason}, Module: {target_name}"
                )
                continue

            try:
                file_path = inspect.getsourcefile(attr)  # type: ignore
                if file_path is not None:
                    visited_file_paths.add(file_path)
            except Exception:
                pass

            if isinstance(
                attr, (types.FunctionType, types.BuiltinFunctionType, _instancemethod_t)
            ) or (
                config.INSTR_DESCRIPTORS and "method_descriptor" in str(type(attr))
            ):  # instrumented with potential accuracy issues as descriptor-controlled method access might change what to return based on given information, but is needed to get tensor method invocations
                assert callable(attr), f"{attr} is not callable"
                assert not (
                    recurse_into_sub_module and is_API_instrumented(attr)
                ), f"{attr} is already instrumented"
                if not recurse_into_sub_module and is_API_instrumented(attr):
                    log_instrumentation_progress(
                        depth,
                        "Skipping function as it is already instrumented",
                        attr,
                        attr_name,
                        pymodule,
                    )
                    continue
                log_instrumentation_progress(
                    depth, "Instrumenting function", attr, attr_name, pymodule
                )

                if typename(attr) in funcs_to_be_replaced:
                    get_instrumentation_logger_for_process().info(
                        f"Replacing function {typename(attr)} with funcs_to_be_replaced[typename(attr)]"
                    )
                    attr = funcs_to_be_replaced[typename(attr)]

                wrapped = self.get_wrapped_function(attr)
                try:
                    setattr(pymodule, attr_name, wrapped)
                except Exception as e:
                    # handling immutable types and attrs that have no setters
                    log_instrumentation_progress(
                        depth,
                        f"Skipping function due to error: {e}",
                        attr,
                        attr_name,
                        pymodule,
                    )
                    continue
                count_wrapped += 1
            elif inspect.isclass(attr):
                log_instrumentation_progress(
                    depth, "Recursing into class", attr, attr_name, pymodule
                )
                count_wrapped += self._instrument_module(
                    attr,
                    visited_file_paths,
                    recurse_into_sub_module,
                    depth + 1,
                )
            elif recurse_into_sub_module and isinstance(attr, types.ModuleType):
                log_instrumentation_progress(
                    depth, "Recursing into module", attr, attr_name, pymodule
                )
                count_wrapped += self._instrument_module(
                    attr,
                    visited_file_paths,
                    recurse_into_sub_module,
                    depth + 1,
                )
            else:
                log_instrumentation_progress(
                    depth, "Not instrumenting", attr, attr_name, pymodule
                )

        log_instrumentation_progress(
            depth,
            f"Finished instrumenting module with {count_wrapped} functions wrapped",
            None,
            None,
            pymodule,
        )
        return count_wrapped


class VarSampler:
    """
    Tracker for the state of a variable. This variable itself cannot be reassigned, i.e. var.attr = new_value is allowed but not var = new_var.

    Currently only suports torch models.

    The difference of this class with StatefulVarObserver is that this class does not keep track of the previous state of the variable.
    Only the current state is dumped during each observation, regardless of whether the state has changed or not.
    """

    def __init__(self, var, dump_tensor_hash: bool = True):
        # Get the current thread object
        if isinstance(var, list):
            assert (
                len(var) == 1
            ), "Currently only supports single variable, please use multiple observers for multiple variables."
            var = var[0]
        assert isinstance(var, torch.nn.Module), "Currently only supports torch models."
        self.var = var

        """DANGEROUS: This param_version tracking is used to track the version of the parameters, so that we can skip the parameters that have not changed.
            However, the `_version` attribute is only bumped when inplace ops (ones with a `_` suffix) like `add_` are called. This means this trick only
            applies to model parameters which should be updated inplace for memory efficiency. 

            However, this trick will not apply to any other variables that are not updated inplace. For example, if you have a variable `x` and you do `x = x + 1`,
            the `_version` of `x` will not be updated and the observer will not be able to detect the change.

            **Many of the activations and intermediate tensors are not updated inplace, so this observer will not be able to detect the changes in those tensors.**
        """
        self.param_versions = {}  # type: ignore
        timestamp = datetime.datetime.now().timestamp()

        curr_meta_vars = get_meta_vars()
        for param in self._get_state_copy():
            attributes = param["attributes"]
            if dump_tensor_hash:
                from mldaikon.proxy_wrapper.hash import tensor_hash

                for attr_name, attr in attributes.items():
                    if isinstance(attr, torch.Tensor):
                        attributes[f"{attr_name}_hash"] = tensor_hash(attr)

            dump_trace_VAR(
                {
                    "var_name": param["name"],
                    "var_type": param["type"],
                    "process_id": os.getpid(),
                    "thread_id": threading.current_thread().ident,
                    "meta_vars": curr_meta_vars,
                    "type": TraceLineType.STATE_CHANGE,
                    "attributes": attributes,
                    "time": timestamp,
                }
            )

    def _get_state_copy(self):
        state_copy = []
        for name, param in self.var.named_parameters():
            self.param_versions[name] = param._version
            state_copy.append(
                {
                    "name": name,
                    "type": typename(param),
                    "attributes": convert_var_to_dict(param),
                }
            )
        return state_copy

    def dump_sample(self):
        """The function is called to observe the state of the model. Each call to this function will
        1. Get the current state of the model
        2. Log the state
        """

        timestamp = datetime.datetime.now().timestamp()

        curr_meta_vars = get_meta_vars()
        for param in self._get_state_copy():
            dump_trace_VAR(
                {
                    "var_name": param["name"],
                    "var_type": param["type"],  # FIXME: hardcoding the type for now
                    "process_id": os.getpid(),
                    "thread_id": threading.current_thread().ident,
                    "meta_vars": curr_meta_vars,
                    "type": TraceLineType.STATE_CHANGE,
                    "attributes": param["attributes"],
                    "time": timestamp,
                }
            )

    def register_hook(self, optimizer: torch.optim.Optimizer):
        # register a post step hook to observe the state of the model after each step
        def hook(optimizer, *args, **kwargs):
            self.dump_sample()

        optimizer.register_step_post_hook(hook)
