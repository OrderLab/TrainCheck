import datetime
import functools
import inspect
import json
import logging
import os
import random
import threading
import traceback
import types

import torch
import torch.utils

# from mldaikon.proxy_wrapper.proxy import Proxy
from mldaikon.proxy_wrapper.config import disable_proxy_class
from mldaikon.utils import typename

EXP_START_TIME = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

meta_vars: dict[str, object] = {}
# TODO: refactor the skipped_modules logic. Use an attribute to mark if the module is wrapped or skipped or not.

trace_API_loggers: dict[int, logging.Logger] = {}
trace_VAR_loggers: dict[int, logging.Logger] = {}
instrumentation_loggers: dict[int, logging.Logger] = {}


disable_proxy_class = disable_proxy_class


def get_trace_API_logger_for_process():
    pid = os.getpid()
    script_name = os.getenv("MAIN_SCRIPT_NAME")
    assert (
        script_name is not None
    ), "MAIN_SCRIPT_NAME is not set, examine the instrumented code to see if os.environ['MAIN_SCRIPT_NAME'] is set in the main function"

    if pid in trace_API_loggers:
        return trace_API_loggers[pid]

    logger = logging.getLogger(f"trace_API_{pid}")
    logger.setLevel(logging.INFO)
    log_file = f"{script_name}_mldaikon_trace_API_{EXP_START_TIME}_{pid}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(file_handler)
    trace_API_loggers[pid] = logger
    return logger


def get_trace_VAR_logger_for_process():
    pid = os.getpid()
    script_name = os.getenv("MAIN_SCRIPT_NAME")
    assert (
        script_name is not None
    ), "MAIN_SCRIPT_NAME is not set, examine the instrumented code to see if os.environ['MAIN_SCRIPT_NAME'] is set in the main function"

    if pid in trace_VAR_loggers:
        return trace_VAR_loggers[pid]

    logger = logging.getLogger(f"trace_VAR_{pid}")
    logger.setLevel(logging.INFO)
    log_file = f"{script_name}_mldaikon_trace_VAR_{EXP_START_TIME}_{pid}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(file_handler)
    trace_VAR_loggers[pid] = logger
    return logger


def dump_trace_API(trace: dict, level=logging.INFO):
    """add a timestamp (unix) to the trace and dump it to the trace log file"""
    logger = get_trace_API_logger_for_process()
    if "time" not in trace:
        trace["time"] = datetime.datetime.now().timestamp()
    logger.log(level, json.dumps(trace))


def dump_trace_VAR(trace: dict, level=logging.INFO):
    """add a timestamp (unix) to the trace and dump it to the trace log file"""
    logger = get_trace_VAR_logger_for_process()
    if "time" not in trace:
        trace["time"] = datetime.datetime.now().timestamp()
    logger.log(level, json.dumps(trace))


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


def global_wrapper(original_function, *args, **kwargs):
    func_call_id = random.randint(0, 1000)

    # Get the current thread object
    current_thread = threading.current_thread()
    # Get the thread ID
    thread_id = current_thread.ident
    process_id = os.getpid()

    func_name = original_function.__name__
    if hasattr(original_function, "__module__"):
        module_name = original_function.__module__
    else:
        module_name = "unknown"

    func_name = f"{module_name}.{func_name}"

    dump_trace_API(
        {
            "func_call_id": func_call_id,
            "thread_id": thread_id,
            "process_id": process_id,
            "meta_vars": meta_vars,
            "type": "function_call (pre)",
            "function": func_name,
        }
    )
    try:
        ### Safe but inefficient: use inspect.getsource to check if the function is a C level function
        # C_level_call = False
        # try:
        #     # check if the original function is a C level function
        #     inspect.getsource(original_function)
        # except Exception as e:
        #     C_level_call = True
        C_level_call = is_c_level_function(original_function)
        # Not Safe for wrapped functions: check if the original function is a builtin_function_or_method
        # if isinstance(original_function, types.BuiltinFunctionType):
        if C_level_call:
            # print(f"Wrapping {original_function}")
            def unproxy_arg(arg):

                if hasattr(arg,"is_proxied_obj"):
                    return unproxy_arg(arg._obj)
                elif type(arg) in [list]:
                    return [unproxy_arg(element) for element in arg]
                elif type(arg) in [tuple]:
                    return tuple(unproxy_arg(element) for element in arg)
                else:
                    return arg

            args = [unproxy_arg(arg) for arg in args]
            # args = unproxy_arg(args[0])
            kwargs = {k: unproxy_arg(v) for k, v in kwargs.items()}

        # def unwrap_proxies(obj):
        #     if isinstance(obj, Proxy):
        #         return unwrap_proxies(obj._obj)
        #     elif isinstance(obj, list):
        #         for i in range(len(obj)):
        #             obj[i] = unwrap_proxies(obj[i])
        #         return obj
        # Ziming: comment out the dict unwrapping here, it would interfere
        # with the _try_get_data functionality in dataloader
        # elif isinstance(obj, dict):
        #     for key in obj:
        #         obj[key] = unwrap_proxies(obj[key], level+1)
        #     return obj
        # elif isinstance(obj, tuple):
        #     obj = tuple(unwrap_proxies(item) for item in obj)
        #     return obj
        # elif isinstance(obj, types.ModuleType):
        #     return obj
        # else:
        #     return obj

        # if not disable_proxy_class:
        #     args = [unwrap_proxies(arg) for arg in args]
        #     kwargs = {k: unwrap_proxies(v) for k, v in kwargs.items()}
        result = original_function(*args, **kwargs)
    except Exception as e:
        dump_trace_API(
            {
                "func_call_id": func_call_id,
                "thread_id": thread_id,
                "process_id": process_id,
                "meta_vars": meta_vars,
                "type": "function_call (post) (exception)",
                "function": func_name,
                "args": [f"{arg}" for arg in args],
                "kwargs": [f"{k}={v}" for k, v in kwargs.items()],
                "exception": str(e),
                "traceback": traceback.format_exc(),
            },
            logging.ERROR,
        )
        print(f"Error in {func_name}: {e}")
        raise e
    dump_trace_API(
        {
            "func_call_id": func_call_id,
            "thread_id": thread_id,
            "process_id": process_id,
            "meta_vars": meta_vars,
            "type": "function_call (post)",
            "function": func_name,
        },
        logging.INFO,
    )
    return result


def wrapper(original_function):
    @functools.wraps(original_function)
    def wrapped(*args, **kwargs):
        return global_wrapper(original_function, *args, **kwargs)

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


# def init_wrapper(original_init):
#     @functools.wraps(original_init)
#     def wrapped_init(self, *args, **kwargs):
#         print(f"wrapped_init for {self.__class__.__name__}")
#         if isinstance(self, torch._ops._OpNamespace):
#             result = original_init(self, *args) if args else None
#         else:
#             try:
#                 result = original_init(self, *args, **kwargs)
#             except Exception as e:
#                 get_instrumentation_logger_for_process().error(f"Error in __init__ of {self.__class__.__name__}: {e}")
#                 print(f"Error in __init__ of {self.__class__.__name__}: {e}")
#                 return None

#         serialized_args = [safe_serialize(arg) for arg in args]
#         serialized_kwargs = {k: safe_serialize(v) for k, v in kwargs.items()}
#         print(
#             f"Initialized {self.__class__.__name__} with args: {serialized_args} and kwargs: {serialized_kwargs}"
#         )
#         logger_trace.info(
#             json.dumps(
#                 {
#                     "thread_id": threading.current_thread().ident,
#                     "process_id": os.getpid(),
#                     "type": "class_init",
#                     "class": self.__class__.__name__,
#                     "args": serialized_args,
#                     "kwargs": serialized_kwargs,
#                 }
#             )
#         )

#         self = Proxy(self, log_level=logging.INFO, logdir="proxy_logs.log")

#         return result

#     return wrapped_init

# Ziming: temporarily disable the new_wrapper because it is relatively unstable

# def new_wrapper(original_new_func):
#     if getattr(original_new_func, "_is_wrapped", False):
#         get_instrumentation_logger_for_process().warning(
#             f"__new__ of {original_new_func.__name__} is already wrapped"
#         )
#         print(f"__new__ of {original_new_func.__name__} is already wrapped")
#         return original_new_func

#     @functools.wraps(original_new_func)
#     def wrapped_new(cls, *args, **kwargs):
#         import random

#         # generate a random id for the function call
#         func_id = str(random.randint(0, 10))

#         print(f"idx: {func_id} wrapped_new for {cls.__name__}")
#         if isinstance(cls, torch._ops._OpNamespace):
#             print(f"idx: {func_id} CALLing original_new_func")
#             result = original_new_func(cls)
#             print(f"idx: {func_id} EXITing original_new_func")
#         else:
#             try:
#                 print(f"idx: {func_id} CALLing original_new_func")
#                 result = original_new_func(cls)
#                 print(f"idx: {func_id} EXITing original_new_func")
#             except Exception as e:
#                 print(f"idx: {func_id} Error in __new__ of {cls.__name__}: {e}")
#                 get_instrumentation_logger_for_process().error(
#                     f"idx: {func_id} Error in __new__ of {cls.__name__}: {e}"
#                 )
#                 return None
#         try:
#             print(
#                 f"idx: {func_id} Initializing {cls.__name__} with Args: {args}, Kwargs: {kwargs}"
#             )
#             result.__init__(*args, **kwargs)
#         except Exception as e:
#             print(f"idx: {func_id} Error in __init__ of {cls.__name__}: {e}")
#             get_instrumentation_logger_for_process().error(
#                 f"idx: {func_id} Error in __init__ of {cls.__name__}: {e}"
#             )
#             return None

#         if cls.__name__ in INCLUDED_WRAP_LIST:
#             print(
#                 f"idx: {func_id} Initalized {cls.__name__} , now creating the proxy class"
#             )
#             result = Proxy(
#                 result, log_level=logging.INFO, logdir=proxy_log_dir
#             )

#         return result

#     # Mark this function as wrapped
#     wrapped_new._is_wrapped = True

#     return wrapped_new


instrumented_modules = set()
skipped_modules: set[types.ModuleType | type | types.FunctionType] = set()
skipped_functions = set()

# there are certain modules that we don't want to instrument (for example, download(), tqdm, etc.)
modules_to_skip = [
    "torch.fx",
    # "torch.jit",
    # "torch._jit",
    # "torch._C",
    "torch._sources",  # FIXME: cannot handle this module, instrumenting it will lead to exceptions: TypeError: module, class, method, function, traceback, frame, or code object was expected, got builtin_function_or_method
]


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
    ):
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

        # TODO: check if self.target or self.root_module is in the modules_to_skip list

        # remove the target from the skipped_modules set
        if target in skipped_modules and self.instrumenting:
            assert not callable(target), f"Skipping callable {target} is not supported"
            skipped_modules.remove(target)

    def check_if_to_skip(self, attr: type):
        if typename(attr) in modules_to_skip:
            return True

        for modules_to_skip_prefix in modules_to_skip:
            if typename(attr).startswith(modules_to_skip_prefix):
                return True

        # attr should also be skipped if the attr does belong to the target
        if not typename(attr).startswith(typename(self.target)):
            return True

        return False

    def instrument(self):
        if self.instrumenting:
            self.instrumented_count = self._instrument_module(self.target)
            return self.instrumented_count
        return 0

    def _instrument_module(self, pymodule: types.ModuleType | type, depth=0):
        target_name = pymodule.__name__

        if pymodule in instrumented_modules or pymodule in skipped_modules:
            get_instrumentation_logger_for_process().info(
                f"Depth: {depth}, Skipping module: {target_name}"
            )
            return 0

        get_instrumentation_logger_for_process().info(
            f"Depth: {depth}, Instrumenting module: {target_name}"
        )
        instrumented_modules.add(pymodule)

        count_wrapped = 0
        for attr_name in dir(pymodule):
            if not hasattr(pymodule, attr_name):
                # handle __abstractmethods__ attribute
                get_instrumentation_logger_for_process().info(
                    f"Depth: {depth}, Skipping attribute as it does not exist: {attr_name}"
                )
                continue

            attr = pymodule.__dict__.get(
                attr_name, None
            )  # getattr(pymodule, attr_name)

            if attr is None:
                get_instrumentation_logger_for_process().info(
                    f"Depth: {depth}, Skipping attribute as it is None: {attr_name}"
                )
                """
                TODO: From my observation, this is happening for the attributes that are implemented in C extension, which usually include math ops and other low level operations for tensors
                , such as tensor.add_.
                We should support these operations as well. Reason is in PyTorch-FORUM84911.
                """
                continue

            # TODO: fix the bug "TypeError: module, class, method, function, traceback, frame, or code object was expected, got builtin_function_or_method"
            if "getfile" in attr_name:
                get_instrumentation_logger_for_process().info(
                    f"Depth: {depth}, Skipping attribute as it is getfile: {attr_name}"
                )
                continue

            # skip magic methods
            if attr_name.startswith("__"):
                get_instrumentation_logger_for_process().info(
                    f"Depth: {depth}, Skipping magic functions: {attr_name}"
                )
                # if callable(attr): # TODO: understand why callable leads to issues
                if isinstance(attr, types.FunctionType):
                    skipped_functions.add(attr)
                elif isinstance(attr, types.ModuleType):
                    skipped_modules.add(attr)
                continue

            """Current Issue with private attributes:

            Background: 
            
                There are actually no *private* attributes in Python. 
                The single underscore prefix is a *convention* that is used to indicate 
                that the attribute should not be accessed directly. 
            
                The double underscore function names such as `__init__` are actually 
                'magic' functions in Python. 
                These functions are super useful and are used to control the behavior of the class. 
                To know more, refer to this link: https://rszalski.github.io/magicmethods/    
                A lot of these magic functions, if instrumented, can be helpful. For example,
                arguments to __init__ can be printed to understand how the class is initialized.
                Also, incepting `__setattr__` can keep track of variable (e.g. weights and gradients) 
                changes in the class.

            Issue:
                The problem is that if we are to instrument all the magic functions, the `__repr__` 
                is leading to some troubles. 

                ```
                Before calling __repr__
                Exception in __repr__: maximum recursion depth exceeded
                Exception in extra_repr: maximum recursion depth exceeded while getting the repr of an object
                Before calling __repr__
                Before calling extra_repr
                ```
                This is because `__repr__` is called by `extra_repr` and `extra_repr` is called by `__repr__`.


            Plan for now:
                This feature is being delayed for now. The original motivation for tracking the magic functions is to 
                capture invalid configs when initialting the classes. For example, if a optimizer is intialized with
                no learnable parameters, it is a sign of a bug. 
                However, we plan to delay this feature for now. The reasons are two fold:
                1. In most ML pipelines, the class initialization code are only run once. This means if we want to 
                    infer invariants from these initializations, we need to combine trace from multiple pipelines. I think
                    it is better to just focus on the main training loop for now as these are executed multiple times and 
                    we can infer things from just one pipeline.
                2. The second reason is these invalid initializations usually have symptoms during the training loop.
                    If the optimizer is not passed with learnable parameters, we will observe that no updates are being performed
                    to the model weights. This is a clear symptom that can be captured during the training loop.

                So for now, we will skip the magic functions.
            """

            # if callable(attr):
            """
              File "/home/yuxuan/gitrepos/ml-daikon/src/instrumentor/tracer.py", line 243, in _instrument_module
                if attr in skipped_functions:
            TypeError: unhashable type: 'instancemethod'
              File "/home/yuxuan/miniconda3/envs/PyTorch-FORUM84911/lib/python3.10/site-packages/torch/library.py", line 109, in impl
                elif isinstance(op_name, OpOverload):
            TypeError: isinstance() arg 2 must be a type, a tuple of types, or a union
            """

            if self.check_if_to_skip(attr):
                get_instrumentation_logger_for_process().info(
                    f"Depth: {depth}, Skipping due to modules_to_skip: {typename(attr)}"
                )
                continue

            if isinstance(attr, types.FunctionType) or isinstance(
                attr, types.BuiltinFunctionType
            ):
                # if isinstance(attr
                try:
                    if attr in skipped_functions:
                        get_instrumentation_logger_for_process().info(
                            f"Depth: {depth}, Skipping function: {typename(attr)}"
                        )
                        continue

                except Exception as e:
                    get_instrumentation_logger_for_process().fatal(
                        f"Depth: {depth}, Error while checking if function {typename(attr)} is in skipped_functions: {e}"
                    )
                    continue
                get_instrumentation_logger_for_process().info(
                    f"Instrumenting function: {typename(attr)}"
                )
                wrapped = wrapper(attr)
                try:
                    setattr(pymodule, attr_name, wrapped)
                except Exception as e:
                    # handling immutable types and attrs that have no setters
                    get_instrumentation_logger_for_process().info(
                        f"Depth: {depth}, Skipping function {typename(attr)} due to error: {e}"
                    )
                    continue
                count_wrapped += 1
            elif isinstance(attr, types.ModuleType):
                if attr in skipped_modules:
                    get_instrumentation_logger_for_process().info(
                        f"Depth: {depth}, Skipping module: {typename(attr)}"
                    )
                    continue
                if not typename(attr).startswith(
                    self.root_module
                ):  # TODO: refine the logic of how to rule out irrelevant modules
                    get_instrumentation_logger_for_process().info(
                        f"Depth: {depth}, Skipping module due to irrelevant name:{typename(attr)}"
                    )
                    skipped_modules.add(attr)
                    continue

                get_instrumentation_logger_for_process().info(
                    f"Depth: {depth}, Recursing into module: {typename(attr)}"
                )
                count_wrapped += self._instrument_module(attr, depth + 1)

            elif inspect.isclass(attr):
                get_instrumentation_logger_for_process().info(
                    f"Depth: {depth}, Recursing into class: {typename(attr)}"
                )
                if not attr.__module__.startswith(self.root_module):
                    get_instrumentation_logger_for_process().info(
                        f"Depth: {depth}, Skipping class {typename(attr)} due to irrelevant module: {attr.__module__}"
                    )
                    continue
                count_wrapped += self._instrument_module(attr, depth + 1)

        get_instrumentation_logger_for_process().info(
            f"Depth: {depth}, Wrapped {count_wrapped} functions in module {target_name}"
        )
        return count_wrapped


class StatefulVarObserver:
    """
    Tracker for the state of a variable. This variable itself cannot be reassigned, i.e. var.attr = new_value is allowed but not var = new_var.

    Currently only suports torch models.

    The difference of this class with StatelessVarObserver is that this class keeps track of the previous state of the variable.
    During each observation, the current state is compared with the previous state and the differences are dumped.
    """

    def __init__(self, var):
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

        timestamp = datetime.datetime.now().timestamp()
        self.current_state = self._get_state_copy()

        for param in self.current_state:
            dump_trace_VAR(
                {
                    "process_id": os.getpid(),
                    "thread_id": threading.current_thread().ident,
                    "meta_vars": meta_vars,
                    "type": "state_change",
                    "var_type": param["type"],
                    "var_name": param["name"],
                    "change": {
                        "value": {
                            "old": param[
                                "param"
                            ],  # HACK: this is a hack for polars to get consistent schemas
                            "new": param[
                                "param"
                            ],  # HACK: this is a hack for polars to get consistent schemas
                        },
                        "attributes": {
                            "old": param[
                                "attributes"
                            ],  # HACK: this is a hack for polars to get consistent schemas
                            "new": param[
                                "attributes"
                            ],  # HACK: this is a hack for polars to get consistent schemas
                        },
                    },
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
            param_list = param.clone().detach().tolist()

            # # HACK: if the param_list is 2 dimensional, then add a dummy dimension to make it 2D
            if not isinstance(param_list[0], list):
                param_list = [param_list]

            state_copy.append(
                {
                    "name": name,
                    "type": typename(param),
                    "attributes": {
                        "param_value": param_list,
                    },
                }
            )
            # only get the attributes that are actual values
            for attr_name in dir(param):
                if attr_name.startswith("__") or not is_safe_getattr(param, attr_name):
                    continue
                attr = getattr(param, attr_name)

                if callable(attr):
                    continue

                if isinstance(attr, torch.Tensor):
                    # skipping the tensor values as we should have already captured them
                    # also, the fields in tensor such as `H` and `T` are just views of the same tensor`
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
        2. Compare it with the previous state
        3. Log the differences
            The differences are computed by comparing the below values:
                - The value of the tensor.
                - The attributes of the tensor, such as requires_grad, device, tensor_model_parallel, etc.
        """
        self.step += 1
        meta_vars.update({"step": self.step})

        timestamp = datetime.datetime.now().timestamp()

        state_copy = self._get_state_copy()
        for old_param, new_param in zip(self.current_state, state_copy):
            # three types of changes: value, attributes, and both
            msg_dict = {
                "process_id": os.getpid(),
                "thread_id": threading.current_thread().ident,
                "meta_vars": meta_vars,
                "type": "state_change",
                # "var": self.var.__class__.__name__,
                "var_type": old_param["type"],  # FIXME: hardcoding the type for now
                "var_name": old_param["name"],
                "time": timestamp,
            }
            if old_param["param"] != new_param["param"]:
                if "change" not in msg_dict:
                    msg_dict["change"] = {}
                msg_dict["change"]["value"] = {
                    "old": old_param["param"],
                    "new": new_param["param"],
                }
            if old_param["attributes"] != new_param["attributes"]:
                if "change" not in msg_dict:
                    msg_dict["change"] = {}
                msg_dict["change"]["attributes"] = {
                    "old": old_param["attributes"],
                    "new": new_param["attributes"],
                }
            if "change" in msg_dict:
                dump_trace_VAR(msg_dict, logging.INFO)

        self.current_state = state_copy


class StatelessVarObserver(StatefulVarObserver):
    """
    Tracker for the state of a variable. This variable itself cannot be reassigned, i.e. var.attr = new_value is allowed but not var = new_var.

    Currently only suports torch models.

    The difference of this class with StatefulVarObserver is that this class does not keep track of the previous state of the variable.
    Only the current state is dumped during each observation, regardless of whether the state has changed or not.
    """

    def __init__(self, var):
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

        timestamp = datetime.datetime.now().timestamp()

        for param in self._get_state_copy():
            dump_trace_VAR(
                {
                    "process_id": os.getpid(),
                    "thread_id": threading.current_thread().ident,
                    "meta_vars": meta_vars,
                    "type": "state_change",
                    "var_type": param["type"],
                    "var_name": param["name"],
                    "attributes": param["attributes"],
                    "time": timestamp,
                }
            )

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
                    "type": "state_change",
                    # "var": self.var.__class__.__name__,
                    "var_type": param["type"],  # FIXME: hardcoding the type for now
                    "var_name": param["name"],
                    "attributes": param["attributes"],
                    "time": timestamp,
                }
            )


if __name__ == "__main__":
    Instrumentor(torch).instrument()
