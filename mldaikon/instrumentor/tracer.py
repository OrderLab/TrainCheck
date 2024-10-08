import datetime
import functools
import importlib
import inspect
import json
import logging
import os
import threading
import traceback
import types
import uuid
from typing import Any, Callable, Optional

import torch
import torch.utils

from mldaikon.config.config import INSTR_MODULES_TO_SKIP, WRAP_WITHOUT_DUMP
from mldaikon.instrumentor.caches import cache_meta_vars
from mldaikon.instrumentor.dumper import (
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
from mldaikon.proxy_wrapper.proxy_config import (
    disable_proxy_class,
    enable_C_level_observer,
)
from mldaikon.utils import typename

disable_proxy_class = disable_proxy_class

_instancemethod_t = type(torch._C._distributed_c10d.ProcessGroup.broadcast)

METRIC_INSTRUMENTED_FUNC_LIST: dict[str, list[str]] = {"dump": [], "no_dump": []}

IS_INSTRUMENTING = False


class TraceLineType:
    FUNC_CALL_PRE = "function_call (pre)"
    FUNC_CALL_POST = "function_call (post)"
    FUNC_CALL_POST_EXCEPTION = "function_call (post) (exception)"
    STATE_CHANGE = "state_change"


def is_c_level_function(original_function):
    return not hasattr(original_function, "__code__")


def get_meta_vars() -> dict:
    return {}


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
    result = {
        "args": [var_to_serializable(arg) for arg in args],
        "kwargs": {k: var_to_serializable(v) for k, v in kwargs.items()},
    }
    return result


def to_dict_return_value(result) -> dict | list[dict]:
    if isinstance(result, tuple):
        return [var_to_serializable(r) for r in result]
    return var_to_serializable(result)


def global_wrapper(
    original_function,
    is_bound_method,
    is_builtin,
    scan_proxy_in_args,
    dump_stack_trace,
    cond_dump,
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
        # logger.debug(f"Skipping dump for {func_name} as meta_vars have not changed")
        # print(f"Skipping dump for {func_name} as meta_vars have not changed")
        return core_wrapper(original_function, is_builtin, *args, **kwargs)

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
            for proxy in proxy_in_args:
                pre_record["proxy_obj_names"].append(
                    [proxy.__dict__["var_name"], type(proxy._obj).__name__]
                )

    dict_args_kwargs = to_dict_args_kwargs(args, kwargs)
    pre_record["args"] = dict_args_kwargs["args"]
    pre_record["kwargs"] = dict_args_kwargs["kwargs"]
    dump_trace_API(pre_record)
    if enable_C_level_observer and is_builtin:
        from mldaikon.proxy_wrapper.proxy_observer import (  # import here to avoid circular import
            add_observer_to_func,
        )

        original_function = add_observer_to_func(
            original_function, cond_dump=cond_dump, unproxy=True
        )
    elif is_funcs_to_be_unproxied(original_function):
        original_function = unproxy_func(original_function, inspect_torch_module=True)
    elif is_builtin:
        # proxy objects being passed to backend will cause seg fault: TODO: replace with unproxy func
        original_function = unproxy_func(original_function)

    try:
        result = original_function(*args, **kwargs)
    except Exception as e:
        dump_trace_API(
            {
                "func_call_id": func_call_id,
                "thread_id": thread_id,
                "process_id": process_id,
                "meta_vars": get_meta_vars(),
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
        raise e
    pre_record.pop("args")
    pre_record.pop("kwargs")
    post_record = (
        pre_record.copy()
    )  # copy the pre_record (though we don't actually need to copy anything)
    post_record["type"] = TraceLineType.FUNC_CALL_POST
    post_record["meta_vars"] = get_meta_vars()
    post_record["return_values"] = to_dict_return_value(result)
    dump_trace_API(post_record)

    return result


def core_wrapper(original_function, is_builtin, *args, **kwargs):
    """same as global_wrapper but without the logging, will have lower overhead than global_wrapper
    We use this wrapper on the functions that are not helpful for invariant inference,  but still needs to be instrumented to handle proxy classes
    """
    if is_builtin:
        original_function = unproxy_func(original_function)
    return original_function(*args, **kwargs)


def wrapper(
    original_function,
    is_bound_method,
    scan_proxy_in_args,
    dump_stack_trace,
    cond_dump,
    disable_dump=False,
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
                *args,
                **kwargs,
            )

    else:
        METRIC_INSTRUMENTED_FUNC_LIST["no_dump"].append(typename(original_function))

        @functools.wraps(original_function)
        def wrapped(*args, **kwargs):
            return core_wrapper(original_function, is_builtin, *args, **kwargs)

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
        funcs_of_inv_interest: Optional[list[str]] = None,
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
            funcs_of_inv_interest (Optional[List[Callable]]):
                An optional list of functions that are of interest for invariant inference.
                If provided, all functions not in this list will be instrumented with dump disabled,
                and the functions in this list will be instrumented with dump enabled. NOTE: If this list is provided, use_full_str must be set to False. WRAP_WITHOUT_DUMP will be ignored.
            API_dump_stack_trace (bool):
                Whether to dump the stack trace of the function call. Enabling this will add the stack trace to the trace log.
            cond_dump (bool):
                Whether to dump the trace conditionally. If True, the trace will only be dumped if meta_vars have changed since the last call of this particular function.
                This might cause additional overhead (cpu and memory) as the meta_vars will be compared with the previous call, and meta_vars will have to be cached in memory.
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
        self.funcs_of_inv_interest = funcs_of_inv_interest
        self.API_dump_stack_trace = API_dump_stack_trace
        self.cond_dump = cond_dump

        if self.funcs_of_inv_interest is not None and self.use_full_instr:
            get_instrumentation_logger_for_process().fatal(
                "Invariants are provided but use_full_instr is True. Selective instrumentation cannot be done. Please remove the `--use-full-instr` flag or remove the invariants"
            )
            raise ValueError(
                "Invariants are provided but use_full_instr is True. Selective instrumentation cannot be done. Please remove the `--use-full-instr` flag or remove the invariants"
            )

        if self.funcs_of_inv_interest is not None:
            get_instrumentation_logger_for_process().info(
                f"Functions of interest for invariant inference: {self.funcs_of_inv_interest}"
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
        # 1. if funcs_of_inv_interest is provided, then METRIC_INSTRUMENTED_FUNC_LIST["dump"] should be equal to funcs_of_inv_interest
        if self.funcs_of_inv_interest is not None:
            # assert set(METRIC_INSTRUMENTED_FUNC_LIST["dump"]) == set(
            #     self.funcs_of_inv_interest
            # ), f"METRIC_INSTRUMENTED_FUNC_LIST['dump'] != funcs_of_inv_interest, diff: {set(METRIC_INSTRUMENTED_FUNC_LIST['dump']) ^ set(self.funcs_of_inv_interest)}"
            assert set(METRIC_INSTRUMENTED_FUNC_LIST["dump"]).issubset(
                set(self.funcs_of_inv_interest)
            ), f"Actual functions being instrumented are not a subset of the functions required by the provided invariants, diff: {set(METRIC_INSTRUMENTED_FUNC_LIST['dump']) ^ set(self.funcs_of_inv_interest)}"

            if set(METRIC_INSTRUMENTED_FUNC_LIST["dump"]) != set(
                self.funcs_of_inv_interest
            ):
                get_instrumentation_logger_for_process().warning(
                    f"Not all functions required by the provided invariants are instrumented (e.g. due to transfering ), some invariants might not be active at all, funcs not instrumented: {set(METRIC_INSTRUMENTED_FUNC_LIST['dump']) ^ set(self.funcs_of_inv_interest)}"
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
            return "Skipping attribute as it is None"

        # 2. Skip if the attribute is already instrumented
        if is_API_instrumented(attr):
            return "Skipping attribute as it is already instrumented"

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

        # 4. Skip if the attribute is in INSTR_MODULES_TO_SKIP | MANUAL CONFIG
        if typename(attr) in INSTR_MODULES_TO_SKIP:
            return "Skipping attribute as it is one of INSTR_MODULES_TO_SKIP"

        # 5. Skip if the attribute is in modules_to_skip_prefix | MANUAL CONFIG
        for modules_to_skip_prefix in INSTR_MODULES_TO_SKIP:
            if typename(attr).startswith(modules_to_skip_prefix):
                return "Skipping attribute as it is in INSTR_MODULES_TO_SKIP"

        # 6. Skip if the attribute does not belong to the target root module
        if not typename(attr).startswith(self.root_module):
            return "Skipping attribute as it does not belong to the root module"

        return None

    def should_disable_dump(self, attr) -> bool:
        """Check if the dump should be disabled for the attribute.
        If use_full_instr is True, then the dump will not be disabled.
        If funcs_of_inv_interest is provided, then the dump will be disabled for all functions except the ones in funcs_of_inv_interest.
        If the attribute is in WRAP_WITHOUT_DUMP, then the dump will be disabled. Otherwise, the dump will not be disabled.
        """

        if self.use_full_instr:
            return False

        if self.funcs_of_inv_interest is not None:
            if typename(attr) in self.funcs_of_inv_interest:
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
                    _ = getattr(pymodule, attr_name)
                    attr = pymodule.__dict__.get(
                        attr_name
                    )  # this attr should be loaded now
                except Exception as e:
                    get_instrumentation_logger_for_process().debug(
                        f"Depth: {depth}, lazy loading failed for attribute: {attr_name}, Module: {target_name}, Type: {typename(attr)}: {e}"
                    )

            if reason := self._should_skip_instr_attr(attr_name, pymodule):
                get_instrumentation_logger_for_process().debug(
                    f"Depth: {depth}, Skipping attribute: {attr_name}, Reason: {reason}, Module: {target_name}, Type: {typename(attr)}"
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
            ):
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

                wrapped = wrapper(
                    attr,
                    is_bound_method=is_API_bound_method(attr),
                    scan_proxy_in_args=self.scan_proxy_in_args,
                    disable_dump=self.should_disable_dump(attr),
                    dump_stack_trace=self.API_dump_stack_trace,
                    cond_dump=self.cond_dump,
                )
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


class StatelessVarObserver:
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

        for param in self._get_state_copy():
            attributes = param["attributes"]
            if dump_tensor_hash:
                from mldaikon.proxy_wrapper.hash import tensor_hash

                for attr_name, attr in attributes.items():
                    if isinstance(attr, torch.Tensor):
                        attributes[f"{attr_name}_hash"] = tensor_hash(attr)

            dump_trace_VAR(
                {
                    "process_id": os.getpid(),
                    "thread_id": threading.current_thread().ident,
                    "meta_vars": get_meta_vars(),
                    "type": TraceLineType.STATE_CHANGE,
                    "var_type": param["type"],
                    "var_name": param["name"],
                    "attributes": attributes,
                    "time": timestamp,
                }
            )

    def _get_state_copy(self):
        def is_safe_getattr(obj, attr):
            try:
                getattr(obj, attr)
                return True
            except Exception as e:
                get_instrumentation_logger_for_process().warn(
                    f"Failed to get attribute {attr} of parameter {name}, skipping it. Error: {e}"
                )
                return False

        state_copy = []
        for name, param in self.var.named_parameters():
            # if name in self.param_versions:
            #     if param._version == self.param_versions[name]:
            #         # the parameter has not changed, so skip it
            #         print(f"Skipping {name} as it has not changed in step {self.step}")
            #         continue
            # TODO: use a flag to enable/disable this optimization
            self.param_versions[name] = param._version
            state_copy.append(
                {
                    "name": name,
                    "type": typename(param),
                    "attributes": {},
                }
            )
            # only get the attributes that are actual values
            for attr_name in dir(param):
                if attr_name.startswith("__") or not is_safe_getattr(param, attr_name):
                    continue
                attr = getattr(param, attr_name)

                if callable(attr):
                    continue

                if isinstance(attr, torch.Tensor) or attr is None:
                    if attr_name in ["data", "grad"]:
                        if attr is not None:
                            attr = attr.view(-1).tolist()
                        else:
                            attr = (
                                []
                            )  # for polars binding, having too many nones would cause error
                    else:
                        continue
                # try to serialize the attribute, if it fails, then skip it
                try:
                    json.dumps(attr)
                except Exception as e:
                    get_instrumentation_logger_for_process().warn(
                        f"Failed to serialize attribute {attr_name} of parameter {name}, skipping it. Error: {e}"
                    )
                    continue

                state_copy[-1]["attributes"][attr_name] = attr

        return state_copy

    def observe(self):
        """The function is called to observe the state of the model. Each call to this function will
        1. Get the current state of the model
        2. Log the state
        """

        timestamp = datetime.datetime.now().timestamp()

        for param in self._get_state_copy():
            dump_trace_VAR(
                {
                    "process_id": os.getpid(),
                    "thread_id": threading.current_thread().ident,
                    "meta_vars": get_meta_vars(),
                    "type": TraceLineType.STATE_CHANGE,
                    # "var": self.var.__class__.__name__,
                    "var_type": param["type"],  # FIXME: hardcoding the type for now
                    "var_name": param["name"],
                    "attributes": param["attributes"],
                    "time": timestamp,
                }
            )
