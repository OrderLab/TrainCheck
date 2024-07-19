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
from typing import TYPE_CHECKING, Callable, Optional

import torch
import torch.utils

if TYPE_CHECKING:
    from mldaikon.proxy_wrapper.proxy import Proxy  # noqa: F401

from mldaikon.config.config import INSTR_MODULES_TO_SKIP, WRAP_WITHOUT_DUMP
from mldaikon.instrumentor.replace_functions import funcs_to_be_replaced
from mldaikon.proxy_wrapper.proxy_basics import is_proxied, unproxy_arg
from mldaikon.proxy_wrapper.proxy_config import (
    disable_proxy_class,
    enable_C_level_observer,
)
from mldaikon.utils import typename

EXP_START_TIME = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

meta_vars: dict[str, object] = {}

import queue
trace_API_loggers: dict[int, (queue.Queue, threading.Thread, logging.Logger)] = {}
trace_VAR_loggers: dict[int, (queue.Queue, threading.Thread, logging.Logger)] = {}
instrumentation_loggers: dict[int, logging.Logger] = {}

def get_dicts():
    return trace_API_loggers, trace_VAR_loggers

def log_worker(queue, log_filename, logger):
    buffer = []
    level=logging.INFO
    buffer_size = 10000
    while True:
        log_entry = queue.get()
        if log_entry is None:
            if buffer:
                # with open(log_filename, 'a') as f:
                #     f.write('\n'.join(buffer) + '\n')
                logger.log(level, '\n'.join(buffer) + '\n')
            queue.task_done()
            break
        buffer.append(log_entry)
        if len(buffer) >= buffer_size:
            # with open(log_filename, 'a') as f:
            #     f.write('\n'.join(buffer) + '\n')
            logger.log(level, '\n'.join(buffer) + '\n')
            buffer.clear()
        queue.task_done()

disable_proxy_class = disable_proxy_class

_instancemethod_t = type(torch._C._distributed_c10d.ProcessGroup.broadcast)

METRIC_INSTRUMENTED_FUNC_LIST: dict[str, list[str]] = {"dump": [], "no_dump": []}


class TraceLineType:
    FUNC_CALL_PRE = "function_call (pre)"
    FUNC_CALL_POST = "function_call (post)"
    FUNC_CALL_POST_EXCEPTION = "function_call (post) (exception)"
    STATE_CHANGE = "state_change"


def get_trace_API_logger_for_process():
    pid = os.getpid()
    script_name = os.getenv("MAIN_SCRIPT_NAME")
    assert (
        script_name is not None
    ), "MAIN_SCRIPT_NAME is not set, examine the instrumented code to see if os.environ['MAIN_SCRIPT_NAME'] is set in the main function"

    if pid in trace_API_loggers:
        return trace_API_loggers[pid][0]
    
    log_queue = queue.Queue()
    log_filename = f"{script_name}_mldaikon_trace_API_{EXP_START_TIME}_{pid}.log"
    logger = logging.getLogger(f"trace_API_{pid}")
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_filename)
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(file_handler)
    log_thread = threading.Thread(target=log_worker, args=(log_queue, log_filename, logger))
    log_thread.start()

    trace_API_loggers[pid] = (log_queue, log_thread, logger)
    return log_queue


def get_trace_VAR_logger_for_process():
    pid = os.getpid()
    script_name = os.getenv("MAIN_SCRIPT_NAME")
    assert (
        script_name is not None
    ), "MAIN_SCRIPT_NAME is not set, examine the instrumented code to see if os.environ['MAIN_SCRIPT_NAME'] is set in the main function"

    if pid in trace_VAR_loggers:
        return trace_VAR_loggers[pid][0]

    log_queue = queue.Queue()
    log_filename = f"{script_name}_mldaikon_trace_VAR_{EXP_START_TIME}_{pid}.log"
    logger = logging.getLogger(f"trace_VAR_{pid}")
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_filename)
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(file_handler)
    log_thread = threading.Thread(target=log_worker, args=(log_queue, log_filename, logger))
    log_thread.start()

    trace_VAR_loggers[pid] = (log_queue, log_thread, logger)
    return log_queue


def dump_trace_API(trace: dict, level=logging.INFO):
    """add a timestamp (unix) to the trace and dump it to the trace log file"""
    log_queue = get_trace_API_logger_for_process()
    trace["time"] = datetime.datetime.now().timestamp()
    log_queue.put(json.dumps(trace))


def dump_trace_VAR(trace: dict, level=logging.INFO):
    """add a timestamp (unix) to the trace and dump it to the trace log file"""
    log_queue = get_trace_VAR_logger_for_process()
    if "time" not in trace:
        trace["time"] = datetime.datetime.now().timestamp()
    log_queue.put(json.dumps(trace))

def get_instrumentation_logger_for_process():
    pid = os.getpid()
    script_name = os.getenv("MAIN_SCRIPT_NAME")
    assert (
        script_name is not None
    ), "MAIN_SCRIPT_NAME is not set, examine the instrumented code to see if os.environ['MAIN_SCRIPT_NAME'] is set in the main function"

    if pid in instrumentation_loggers:
        return instrumentation_loggers[pid]

    logger = logging.getLogger(f"instrumentation_{pid}")
    logger.setLevel(logging.INFO)
    log_file = f"{script_name}_mldaikon_instrumentation_{EXP_START_TIME}_{pid}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(file_handler)
    instrumentation_loggers[pid] = logger
    return logger


def is_c_level_function(original_function):
    return not hasattr(original_function, "__code__")


def global_wrapper(
    original_function, is_bound_method, scan_proxy_in_args, *args, **kwargs
):
    import uuid

    func_call_id = uuid.uuid4().hex

    logger = logging.getLogger(__name__)

    # Get the current thread object
    current_thread = threading.current_thread()
    # Get the thread ID
    thread_id = current_thread.ident
    process_id = os.getpid()

    func_name = typename(original_function)

    pre_record = {
        "func_call_id": func_call_id,
        "thread_id": thread_id,
        "process_id": process_id,
        "meta_vars": meta_vars,
        "type": TraceLineType.FUNC_CALL_PRE,
        "function": func_name,
        "is_bound_method": is_bound_method,
        "obj_id": None if not is_bound_method else id(args[0]),
        "proxy_obj_names": [
            ["", ""]
        ],  # HACK: this is a hack to make polars schema inference work (it samples the first 100 rows to infer the schema)
    }

    C_level_call = is_c_level_function(original_function)

    if scan_proxy_in_args:
        proxy_in_args = []

        def find_proxy_in_args(args):
            for i, arg in enumerate(args):
                if is_proxied(
                    arg
                ):  # Ziming: get rid of directly import Proxy from external module, use proxy_basics instead
                    print(
                        f"Found proxy {arg.__dict__['var_name']} in function {func_name}"
                    )
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

    dump_trace_API(pre_record)
    if enable_C_level_observer and C_level_call:
        from mldaikon.proxy_wrapper.proxy_observer import add_observer_to_func

        original_function = add_observer_to_func(original_function, unproxy=True)
    elif C_level_call:
        args = [unproxy_arg(arg) for arg in args]
        kwargs = {k: unproxy_arg(v) for k, v in kwargs.items()}
    try:
        result = original_function(*args, **kwargs)
    except Exception as e:
        dump_trace_API(
            {
                "func_call_id": func_call_id,
                "thread_id": thread_id,
                "process_id": process_id,
                "meta_vars": meta_vars,
                "type": TraceLineType.FUNC_CALL_POST_EXCEPTION,
                "function": func_name,
                "args": [f"{arg}" for arg in args],
                "kwargs": [f"{k}={v}" for k, v in kwargs.items()],
                "exception": str(e),
                "exception_type": f"{type(e)}",
                "traceback": traceback.format_exc(),
                "is_bound_method": is_bound_method,
                "obj_id": None if not is_bound_method else id(args[0]),
            },
            logging.ERROR,
        )
        logger.error(f"Error in {func_name}: {type(e)} {e}")
        raise e

    post_record = (
        pre_record.copy()
    )  # copy the pre_record (though we don't actually need to copy anything)
    post_record["type"] = TraceLineType.FUNC_CALL_POST
    dump_trace_API(post_record, logging.INFO)

    return result


def core_wrapper(original_function, *args, **kwargs):
    """same as global_wrapper but without the logging, will have lower overhead than global_wrapper
    We use this wrapper on the functions that are not helpful for invariant inference,  but still needs to be instrumented to handle proxy classes
    """
    C_level_call = is_c_level_function(original_function)
    if C_level_call:

        def unproxy_arg(arg):
            if hasattr(arg, "is_ml_daikon_proxied_obj"):
                return unproxy_arg(arg._obj)
            elif type(arg) in [list]:
                return [unproxy_arg(element) for element in arg]
            elif type(arg) in [tuple]:
                return tuple(unproxy_arg(element) for element in arg)
            else:
                return arg

        args = [unproxy_arg(arg) for arg in args]
        kwargs = {k: unproxy_arg(v) for k, v in kwargs.items()}
    return original_function(*args, **kwargs)


def wrapper(original_function, is_bound_method, scan_proxy_in_args, disable_dump=False):
    if not disable_dump:
        METRIC_INSTRUMENTED_FUNC_LIST["dump"].append(typename(original_function))

        @functools.wraps(original_function)
        def wrapped(*args, **kwargs):
            return global_wrapper(
                original_function, is_bound_method, scan_proxy_in_args, *args, **kwargs
            )

    else:
        METRIC_INSTRUMENTED_FUNC_LIST["no_dump"].append(typename(original_function))

        @functools.wraps(original_function)
        def wrapped(*args, **kwargs):
            return core_wrapper(original_function, *args, **kwargs)

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
        allow_disable_dump: bool,
        funcs_of_inv_interest: Optional[list[str]] = None,
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
            allow_disable_dump (bool):
                Whether to allow disabling the dump of the trace on certain functions. Regardless of this flag, the function will still be instrumented.
                Refer to WRAP_WITHOUT_DUMP in config.py for the list of functions/modules that will have the dump disabled.
            funcs_of_inv_interest (Optional[List[Callable]]):
                An optional list of functions that are of interest for invariant inference.
                If provided, all functions not in this list will be instrumented with dump disabled,
                and the functions in this list will be instrumented with dump enabled. NOTE: If this list is provided, allow_disable_dump must be set to True. WRAP_WITHOUT_DUMP will be ignored.

        Returns:
            None
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
        self.allow_disable_dump = allow_disable_dump
        self.funcs_of_inv_interest = funcs_of_inv_interest

        if self.funcs_of_inv_interest is not None and not self.allow_disable_dump:
            get_instrumentation_logger_for_process().fatal(
                "Invariants are provided but allow_disable_dump is False. Selective instrumentation cannot be done. Please set allow_disable_dump to True or remove the invariants"
            )
            raise ValueError(
                "Invariants are provided but allow_disable_dump is False. Selective instrumentation cannot be done. Please set allow_disable_dump to True or remove the invariants"
            )

        if self.funcs_of_inv_interest is not None:
            get_instrumentation_logger_for_process().info(
                f"Functions of interest for invariant inference: {self.funcs_of_inv_interest}"
            )

    def instrument(self) -> int:
        if not self.instrumenting:
            return 0

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
            assert set(METRIC_INSTRUMENTED_FUNC_LIST["dump"]) == set(
                self.funcs_of_inv_interest
            ), "METRIC_INSTRUMENTED_FUNC_LIST['dump'] != funcs_of_inv_interest"

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
            and attr_name not in ["__init__", "__call__"]
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
        If allow_disable_dump is False, then the dump will not be disabled.
        If funcs_of_inv_interest is provided, then the dump will be disabled for all functions except the ones in funcs_of_inv_interest.
        If the attribute is in WRAP_WITHOUT_DUMP, then the dump will be disabled. Otherwise, the dump will not be disabled.
        """

        if not self.allow_disable_dump:
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
        self.step = (
            0  # HACK: this is a hack to get the step number as we observe every step
        )
        meta_vars.update({"step": self.step})
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
                    "meta_vars": meta_vars,
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
        self.step += 1
        meta_vars.update({"step": self.step})

        timestamp = datetime.datetime.now().timestamp()

        for param in self._get_state_copy():
            dump_trace_VAR(
                {
                    "process_id": os.getpid(),
                    "thread_id": threading.current_thread().ident,
                    "meta_vars": meta_vars,
                    "type": TraceLineType.STATE_CHANGE,
                    # "var": self.var.__class__.__name__,
                    "var_type": param["type"],  # FIXME: hardcoding the type for now
                    "var_name": param["name"],
                    "attributes": param["attributes"],
                    "time": timestamp,
                }
            )

