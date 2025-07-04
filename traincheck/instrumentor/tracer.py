import functools
import importlib
import inspect
import os
import threading
import time
import traceback
import types
from typing import Callable, Optional

import torch

import traincheck.config.config as config  # needed to allow for change of values after import
from traincheck.config.config import (
    INSTR_MODULES_TO_SKIP,
    SKIP_INSTR_APIS,
    WRAP_WITHOUT_DUMP,
    WRAP_WITHOUT_DUMP_WHITELIST,
)
from traincheck.instrumentor.caches import meta_vars
from traincheck.instrumentor.dumper import (
    convert_var_to_dict,
    dump_trace_API,
    dump_trace_VAR,
    get_instrumentation_logger_for_process,
    var_to_serializable,
)
from traincheck.instrumentor.replace_functions import (
    funcs_to_be_replaced,
    is_funcs_to_be_unproxied,
)
from traincheck.proxy_wrapper.proxy_basics import is_proxied, unproxy_func
from traincheck.proxy_wrapper.proxy_config import enable_C_level_observer
from traincheck.proxy_wrapper.proxy_registry import get_global_registry
from traincheck.utils import get_timestamp_ns, get_unique_id, typename

_instancemethod_t = type(torch._C._distributed_c10d.ProcessGroup.broadcast)

METRIC_INSTRUMENTED_FUNC_LIST: dict[str, list[str]] = {"dump": [], "no_dump": []}

IS_INSTRUMENTING = False

DISABLE_WRAPPER = False

# for prompt generation tasks using the transformers library (see traincheck/developer/instr_stage_annotation.py:annotate_answer_start_token_ids)
GENERATE_START_TOKEN_ID: None | int = None
GENERATE_START_TOKEN_ID_INCLUDE_START_TOKEN = False

COLLECT_OVERHEAD_METRICS = os.environ.get("COLLECT_OVERHEAD_METRICS", "0") == "1"


THREAD_DATA = threading.local()


class TraceLineType:
    FUNC_CALL_PRE = "function_call (pre)"
    FUNC_CALL_POST = "function_call (post)"
    FUNC_CALL_POST_EXCEPTION = "function_call (post) (exception)"
    STATE_CHANGE = "state_change"


def get_process_thread_id() -> tuple[int, int]:
    global THREAD_DATA
    # If the ID isn't cached yet, fetch and store it in this thread's local storage
    if not hasattr(THREAD_DATA, "thread_id"):
        THREAD_DATA.thread_id = threading.get_ident()

    if not hasattr(THREAD_DATA, "process_id"):
        THREAD_DATA.process_id = os.getpid()

    return THREAD_DATA.process_id, THREAD_DATA.thread_id


def is_c_level_function(original_function):
    return not hasattr(original_function, "__code__")


def get_meta_vars() -> dict:
    """Deprecated: use meta_vars directly"""
    return meta_vars


def increment_step_if_needed(func_obj, func_name, is_bound_method, args):
    """Increment the global step if
    - the function is torch.optim.Optimizer.step"""
    if not is_bound_method:
        return

    obj = args[0]

    if func_name.endswith(".step"):
        # if the function is a bound method and the object is an instance of torch.optim.Optimizer
        if isinstance(obj, torch.optim.Optimizer):
            meta_vars[
                "step"
            ] += 1  # TODO: what if the users have annotated their own step function?
            return True


def to_dict_args_kwargs(args, kwargs, dump_args_config=None) -> dict:
    global DISABLE_WRAPPER
    DISABLE_WRAPPER = True
    if dump_args_config is None:
        result = {
            "args": {i: var_to_serializable(arg) for i, arg in enumerate(args)},
            "kwargs": {k: var_to_serializable(v) for k, v in kwargs.items()},
        }
    else:
        result = {"args": {}, "kwargs": {}}
        args_dicts = {}
        kwargs_dicts = {}
        for i, arg in enumerate(args):
            if str(i) in dump_args_config:
                args_dicts[i] = var_to_serializable(arg, dump_args_config[str(i)])

        for k, v in kwargs.items():
            if k in dump_args_config:
                kwargs_dicts[k] = var_to_serializable(v, dump_args_config[k])

        result["args"] = args_dicts
        result["kwargs"] = kwargs_dicts
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
    original_function: Callable,
    original_function_name: str,
    is_bound_method: bool,
    is_builtin: bool,
    scan_proxy_in_args: bool,
    dump_stack_trace: bool,
    dump_args: bool,
    dump_args_config,
    dump_ret: bool,
    dump_ret_config,
    handle_proxy: bool,
    trigger_proxy_state_dump: bool,
    proxy_state_dump_config: dict,
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

    # if "step" in original_function_name and not "scheduler" in original_function_name:
    #     print("step function called" + original_function_name)
    #     print(trigger_proxy_state_dump)
    #     print(proxy_state_dump_config)
    #     exit(1)

    global DISABLE_WRAPPER
    global PROCESS_ID

    if DISABLE_WRAPPER:
        return original_function(*args, **kwargs)

    if COLLECT_OVERHEAD_METRICS:
        ENTER_PERF_TIME = time.perf_counter()

    func_call_id = get_unique_id()
    process_id, thread_id = get_process_thread_id()
    increment_step_if_needed(
        original_function, original_function_name, is_bound_method, args
    )

    pre_meta_vars = get_meta_vars()

    if IS_INSTRUMENTING:
        return original_function(
            *args, **kwargs
        )  # don't instrument while instrumenting

    pre_record = {
        "func_call_id": func_call_id,
        "thread_id": thread_id,
        "process_id": process_id,
        "meta_vars": pre_meta_vars,
        "type": TraceLineType.FUNC_CALL_PRE,
        "function": original_function_name,
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

        args = list(args)  # type: ignore[assignment]
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
        dict_args_kwargs = to_dict_args_kwargs(args, kwargs, dump_args_config)
        pre_record["args"] = dict_args_kwargs["args"]
        pre_record["kwargs"] = dict_args_kwargs["kwargs"]
    dump_trace_API(pre_record)

    if handle_proxy and trigger_proxy_state_dump:
        """Mimicking the behavior the observer wrapper: pre-observe"""
        get_global_registry().dump_only_modified(
            dump_loc=original_function_name, dump_config=proxy_state_dump_config
        )

    if handle_proxy:
        if enable_C_level_observer and is_builtin:
            from traincheck.proxy_wrapper.proxy_observer import (
                add_observer_to_func,  # import here to avoid circular import
            )

            original_function = add_observer_to_func(original_function, unproxy=True)
        elif is_funcs_to_be_unproxied(original_function):
            original_function = unproxy_func(
                original_function, inspect_torch_module=True
            )
        elif is_builtin:
            # proxy objects being passed to backend will cause seg fault: TODO: replace with unproxy func
            original_function = unproxy_func(original_function)

    try:
        if COLLECT_OVERHEAD_METRICS:
            ORIG_ENTER_PERF_TIME = time.perf_counter()
        result = original_function(*args, **kwargs)
        if COLLECT_OVERHEAD_METRICS:
            ORIG_EXIT_PERF_TIME = time.perf_counter()
    except Exception as e:
        if COLLECT_OVERHEAD_METRICS:
            ORIG_EXIT_PERF_TIME = time.perf_counter()

        if handle_proxy and trigger_proxy_state_dump:
            """Mimicking the behavior the observer wrapper: post-observe"""
            get_global_registry().dump_only_modified(
                dump_loc=original_function_name, dump_config=proxy_state_dump_config
            )

        dump_trace_API(
            {
                "func_call_id": func_call_id,
                "thread_id": thread_id,
                "process_id": process_id,
                "meta_vars": pre_meta_vars,
                "type": TraceLineType.FUNC_CALL_POST_EXCEPTION,
                "function": original_function_name,
                "exception": typename(e, is_runtime=True),
                "exception_msg": str(e),
                "is_bound_method": is_bound_method,
                "obj_id": None if not is_bound_method else id(args[0]),
            },
        )

        if COLLECT_OVERHEAD_METRICS:
            EXIT_PERF_TIME = time.perf_counter()
            print(
                f"WRAPPER TIME: {original_function_name},{ORIG_EXIT_PERF_TIME - ORIG_ENTER_PERF_TIME},{EXIT_PERF_TIME - ENTER_PERF_TIME}"
            )
        raise e

    if handle_proxy and trigger_proxy_state_dump:
        get_global_registry().dump_only_modified(
            dump_loc=original_function_name, dump_config=proxy_state_dump_config
        )

    post_record = {
        "func_call_id": func_call_id,
        "thread_id": thread_id,
        "process_id": process_id,
        "meta_vars": pre_meta_vars,
        "type": TraceLineType.FUNC_CALL_POST,
        "function": original_function_name,
        "is_bound_method": is_bound_method,
        "obj_id": None if not is_bound_method else id(args[0]),
    }

    result_to_dump = result

    # if the current function name is transformers.generate, then we will dump the response tokens only, let's see.
    # a concrete name: "transformers.models.whisper.modeling_whisper.WhisperForConditionalGeneration.generate"
    # we want a pattern that abstracts the specific model name
    pattern = "transformers.models.*.generate"
    # find matches in the pattern
    import re

    if (
        GENERATE_START_TOKEN_ID is not None
        and re.match(pattern, original_function_name)
        and isinstance(result, torch.Tensor)
    ):
        print(f"Found match for {original_function_name}")
        # the first dimension is the batch size, and each corresponds to a separate response, let's try to match the batch size with the start token ids first
        response_starting_indices = []
        for i in range(result.size(0)):
            # try to find the match of the start token ids in the response
            response = result[i]
            # Find all indices where the start_token_id matches
            matches = (response == GENERATE_START_TOKEN_ID).nonzero(as_tuple=True)[0]
            indexes = matches.tolist()
            if len(indexes) == 0:
                # No occurrences found
                print(
                    f"start_token_id ({GENERATE_START_TOKEN_ID}) not found in response {i}"
                )
                start_index = -1  # Handle case where token is not found
            elif len(indexes) > 1:
                # Multiple occurrences found, raise an error
                raise ValueError(
                    f"Multiple occurrences of start_token_id ({GENERATE_START_TOKEN_ID}) found in response {i}: {matches.tolist()}"
                )
            else:
                # Single occurrence found, get the index
                start_index = indexes[0]
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

    if COLLECT_OVERHEAD_METRICS:
        EXIT_PERF_TIME = time.perf_counter()
        print(
            f"WRAPPER TIME: {original_function_name},{ORIG_EXIT_PERF_TIME - ORIG_ENTER_PERF_TIME},{EXIT_PERF_TIME - ENTER_PERF_TIME}"
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
    disable_dump=False,
    dump_args=True,
    dump_args_config=None,
    dump_ret=True,
    dump_ret_config=None,
    handle_proxy=True,
    trigger_proxy_state_dump=False,
    proxy_state_dump_config=None,
):
    is_builtin = is_c_level_function(original_function)
    original_function_name = typename(original_function)
    # determine statically whether to dump the trace
    if not disable_dump:
        METRIC_INSTRUMENTED_FUNC_LIST["dump"].append(original_function_name)

        @functools.wraps(original_function)
        def wrapped(*args, **kwargs):
            return global_wrapper(  # the wrapper cannot be invoked with named parameters as *args has to be after the named parameters
                original_function,
                original_function_name,
                is_bound_method,
                is_builtin,
                scan_proxy_in_args,
                dump_stack_trace,
                dump_args,
                dump_args_config,
                dump_ret,
                dump_ret_config,
                handle_proxy,
                trigger_proxy_state_dump,
                proxy_state_dump_config,
                *args,
                **kwargs,
            )

    else:
        METRIC_INSTRUMENTED_FUNC_LIST["no_dump"].append(original_function_name)
        if handle_proxy:

            @functools.wraps(original_function)
            def wrapped(*args, **kwargs):
                return core_wrapper(
                    original_function, is_builtin, handle_proxy, *args, **kwargs
                )

        else:
            return original_function

    wrapped._traincheck_original_function = original_function
    wrapped._traincheck_instrumented = True
    wrapped._traincheck_dump_disabled = disable_dump
    return wrapped


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
        return obj.__dict__.get("_traincheck_instrumented", False)
    except Exception:
        # a wrapped API would have __dict__ and have the flag
        return False


def is_API_bound_method(obj: Callable) -> bool:
    """We will see if the object will be a bound method or not. If the object is a bound method, we will return True, else False"""
    logger = get_instrumentation_logger_for_process()

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
        self.instr_opts = config.load_instr_opts()

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

        # 7. Skip if the user has specified to skip the attribute
        if attr_full_name in SKIP_INSTR_APIS:
            return "Skipping attribute as it is in SKIP_INSTR_APIS"

        return None

    def should_disable_dump(self, attr) -> bool:
        """Check if the dump should be disabled for the attribute.
        If use_full_instr is True, then the dump will not be disabled.
        If funcs_to_instr is provided, then the dump will be disabled for all functions except the ones in funcs_to_instr.
        If the attribute is in WRAP_WITHOUT_DUMP, then the dump will be disabled. Otherwise, the dump will not be disabled.
        """

        logger = get_instrumentation_logger_for_process()

        if self.use_full_instr:
            return False

        if self.funcs_to_instr is not None:
            if typename(attr) in self.funcs_to_instr:
                return False
            return True

        attr_name = typename(attr)
        for wrap_without_dump_module in WRAP_WITHOUT_DUMP:
            whitelist_modules = [
                w
                for w in WRAP_WITHOUT_DUMP_WHITELIST
                if w.startswith(wrap_without_dump_module)
            ]
            if attr_name.startswith(wrap_without_dump_module) and not any(
                attr_name.startswith(w) for w in whitelist_modules
            ):
                logger.debug(
                    f"Skipping dump for {attr_name} as it is in WRAP_WITHOUT_DUMP {wrap_without_dump_module}"
                )
                return True
        return False

    def get_wrapped_function(self, func_obj: Callable) -> Callable:
        """Get the wrapped function for the provided function object,
        based on the instrumentation options provided in instr_opts.json.
        """
        used_proxy = True  # TODO: dump instr_opts when doing full instr as well so we can determine whether to handle proxy based on the specific instrumentation args
        if self.instr_opts is not None:
            used_proxy = self.instr_opts.model_tracker_style == "proxy"
            func_name = typename(func_obj)
            if func_name not in self.instr_opts.funcs_instr_opts:
                return wrapper(
                    func_obj,
                    is_bound_method=None,
                    scan_proxy_in_args=None,
                    dump_stack_trace=None,
                    disable_dump=True,
                    handle_proxy=used_proxy,
                )

            func_instr_opt = self.instr_opts.funcs_instr_opts[func_name]
            return wrapper(
                func_obj,
                is_bound_method=is_API_bound_method(func_obj),
                scan_proxy_in_args=func_instr_opt["scan_proxy_in_args"],
                disable_dump=self.should_disable_dump(func_obj),
                dump_stack_trace=self.API_dump_stack_trace,
                dump_args=func_instr_opt["dump_args"],
                dump_args_config=(
                    func_instr_opt["dump_args_config"]
                    if "dump_args_config" in func_instr_opt
                    else None
                ),  # TODO: refactor this existence check | None indicates that everything should be dumped
                dump_ret=func_instr_opt["dump_ret"],
                dump_ret_config=(
                    func_instr_opt["dump_ret_config"]
                    if "dump_ret_config" in func_instr_opt
                    else None
                ),
                handle_proxy=used_proxy,
                trigger_proxy_state_dump=self.instr_opts.disable_proxy_dumping
                and len(func_instr_opt["var_types_to_track"]) > 0,
                proxy_state_dump_config=func_instr_opt["var_types_to_track"],
            )

        return wrapper(
            func_obj,
            is_bound_method=is_API_bound_method(func_obj),
            scan_proxy_in_args=self.scan_proxy_in_args,
            disable_dump=self.should_disable_dump(func_obj),
            dump_stack_trace=self.API_dump_stack_trace,
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
                    continue

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
                msg = "Not instrumenting"
                if (
                    "method_descriptor" in str(type(attr))
                    and not config.INSTR_DESCRIPTORS
                ):
                    msg = "Not instrumenting because it is a method descriptor and config.INSTR_DESCRIPTORS is False"
                log_instrumentation_progress(depth, msg, attr, attr_name, pymodule)

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

    def __init__(self, var, var_name: str):
        # Get the current thread object
        if isinstance(var, list):
            assert (
                len(var) == 1
            ), "Currently only supports single variable, please use multiple observers for multiple variables."
            var = var[0]
        assert isinstance(var, torch.nn.Module), "Currently only supports torch models."
        self.var = var
        self.var_name = var_name

        """DANGEROUS: This param_version tracking is used to track the version of the parameters, so that we can skip the parameters that have not changed.
            However, the `_version` attribute is only bumped when inplace ops (ones with a `_` suffix) like `add_` are called. This means this trick only
            applies to model parameters which should be updated inplace for memory efficiency. 

            However, this trick will not apply to any other variables that are not updated inplace. For example, if you have a variable `x` and you do `x = x + 1`,
            the `_version` of `x` will not be updated and the observer will not be able to detect the change.

            **Many of the activations and intermediate tensors are not updated inplace, so this observer will not be able to detect the changes in those tensors.**
        """
        self.param_versions = {}  # type: ignore
        timestamp = get_timestamp_ns()

        curr_meta_vars = get_meta_vars()
        for param in self._get_state_copy():
            attributes = param["attributes"]

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
            param_global_name = self.var_name + "." + name
            self.param_versions[param_global_name] = param._version
            state_copy.append(
                {
                    "name": param_global_name,
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

        timestamp = get_timestamp_ns()

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
