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

import src.proxy_wrapper.proxy as ProxyWrapper
from src.config.config import INCLUDED_WRAP_LIST, proxy_log_dir

logger_instrumentation = logging.getLogger("instrumentation")
logger_trace = logging.getLogger("trace")

meta_vars: dict[str, object] = {}


def typename(o):
    if isinstance(o, torch.Tensor):
        return o.type()
    module = ""
    class_name = ""
    if (
        hasattr(o, "__module__")
        and o.__module__ != "builtins"
        and o.__module__ != "__builtin__"
        and o.__module__ is not None
    ):
        module = o.__module__ + "."
    if hasattr(o, "__qualname__"):
        class_name = o.__qualname__
    elif hasattr(o, "__name__"):
        class_name = o.__name__
    else:
        class_name = o.__class__.__name__
    return module + class_name


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

    # logger_trace.info({'type': 'function_call (pre)', 'function': original_function.__name__, 'args': args, 'kwargs': kwargs})
    logger_trace.info(
        json.dumps(
            {
                "func_call_id": func_call_id,
                "thread_id": thread_id,
                "process_id": process_id,
                "meta_vars": meta_vars,
                "type": "function_call (pre)",
                "function": func_name,
            }
        )
    )
    try:
        result = original_function(*args, **kwargs)
    except Exception as e:
        logger_trace.error(
            json.dumps(  # keep the error logs as json
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
                }
            )
        )
        print(f"Error in {func_name}: {e}")
        raise e
    # logger_trace.info({'type': 'function_call (post)', 'function': original_function.__name__, 'result': result})
    logger_trace.info(
        json.dumps(
            {
                "func_call_id": func_call_id,
                "thread_id": thread_id,
                "process_id": process_id,
                "meta_vars": meta_vars,
                "type": "function_call (post)",
                "function": func_name,
            }
        )
    )
    return result


def wrapper(original_function):
    @functools.wraps(original_function)
    def wrapped(*args, **kwargs):
        return global_wrapper(original_function, *args, **kwargs)

    return wrapped


EXCLUDED_CLASSES = (torch.utils.data._utils.worker.WorkerInfo,)


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


def init_wrapper(original_init):
    @functools.wraps(original_init)
    def wrapped_init(self, *args, **kwargs):
        print(f"wrapped_init for {self.__class__.__name__}")
        if isinstance(self, torch._ops._OpNamespace):
            result = original_init(self, *args) if args else None
        else:
            try:
                result = original_init(self, *args, **kwargs)
            except Exception as e:
                logging.error(f"Error in __init__ of {self.__class__.__name__}: {e}")
                print(f"Error in __init__ of {self.__class__.__name__}: {e}")
                return None

        serialized_args = [safe_serialize(arg) for arg in args]
        serialized_kwargs = {k: safe_serialize(v) for k, v in kwargs.items()}
        print(
            f"Initialized {self.__class__.__name__} with args: {serialized_args} and kwargs: {serialized_kwargs}"
        )
        logger_trace.info(
            json.dumps(
                {
                    "thread_id": threading.current_thread().ident,
                    "process_id": os.getpid(),
                    "type": "class_init",
                    "class": self.__class__.__name__,
                    "args": serialized_args,
                    "kwargs": serialized_kwargs,
                }
            )
        )

        self = ProxyWrapper.Proxy(self, log_level=logging.INFO, logdir="proxy_logs.log")

        return result

    return wrapped_init


def new_wrapper(original_new_func):
    if getattr(original_new_func, "_is_wrapped", False):
        logging.warning(f"__new__ of {original_new_func.__name__} is already wrapped")
        print(f"__new__ of {original_new_func.__name__} is already wrapped")
        return original_new_func

    @functools.wraps(original_new_func)
    def wrapped_new(cls, *args, **kwargs):
        import random

        # generate a random id for the function call
        func_id = str(random.randint(0, 10))

        print(f"idx: {func_id} wrapped_new for {cls.__name__}")
        if isinstance(cls, torch._ops._OpNamespace):
            print(f"idx: {func_id} CALLing original_new_func")
            result = original_new_func(cls)
            print(f"idx: {func_id} EXITing original_new_func")
        else:
            try:
                print(f"idx: {func_id} CALLing original_new_func")
                result = original_new_func(cls)
                print(f"idx: {func_id} EXITing original_new_func")
            except Exception as e:
                print(f"idx: {func_id} Error in __new__ of {cls.__name__}: {e}")
                logging.error(f"idx: {func_id} Error in __new__ of {cls.__name__}: {e}")
                return None
        try:
            print(
                f"idx: {func_id} Initializing {cls.__name__} with Args: {args}, Kwargs: {kwargs}"
            )
            result.__init__(*args, **kwargs)
        except Exception as e:
            print(f"idx: {func_id} Error in __init__ of {cls.__name__}: {e}")
            logging.error(f"idx: {func_id} Error in __init__ of {cls.__name__}: {e}")
            return None

        if cls.__name__ in INCLUDED_WRAP_LIST:
            print(
                f"idx: {func_id} Initalized {cls.__name__} , now creating the proxy class"
            )
            result = ProxyWrapper.Proxy(
                result, log_level=logging.INFO, logdir=proxy_log_dir
            )

        return result

    # Mark this function as wrapped
    wrapped_new._is_wrapped = True

    return wrapped_new


instrumented_modules = set()
skipped_modules: set[types.ModuleType | type | types.FunctionType] = set()
skipped_functions = set()

# there are certain modules that we don't want to instrument (for example, download(), tqdm, etc.)
modules_to_skip = [
    "torch.fx",
    "torch.jit",
    "torch._jit",
    "torch._C",
]


class instrumentor:
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
        if isinstance(target, types.ModuleType):
            self.root_module = target.__name__.split(".")[0]
        elif inspect.isclass(target):
            self.root_module = target.__module__.split(".")[0]
        elif callable(target):
            raise ValueError(
                """Unsupported target type. This instrumentor does not support function, 
                due to inability to swap the original function with the wrapper function 
                in the namespace. However, you can use the wrapper function directly by 
                setting 
                    `func = wrapper(func)`
                """
            )
        else:
            raise ValueError(
                "Unsupported target type. This instrumentor only supports module, class."
            )
        self.instrumented_count = 0
        self.target = target

        # remove the target from the skipped_modules set
        if target in skipped_modules:
            skipped_modules.remove(target)

    def instrument(self):
        self.instrumented_count = self._instrument_module(self.target)
        return self.instrumented_count

    def _instrument_module(self, pymodule: types.ModuleType | type, depth=0):
        target_name = pymodule.__name__

        if pymodule in instrumented_modules or pymodule in skipped_modules:
            logger_instrumentation.info(
                f"Depth: {depth}, Skipping module: {target_name}"
            )
            return 0

        logger_instrumentation.info(
            f"Depth: {depth}, Instrumenting module: {target_name}"
        )
        instrumented_modules.add(pymodule)

        count_wrapped = 0
        for attr_name in dir(pymodule):
            if not hasattr(pymodule, attr_name):
                # handle __abstractmethods__ attribute
                logger_instrumentation.info(
                    f"Depth: {depth}, Skipping attribute as it does not exist: {attr_name}"
                )
                continue

            attr = pymodule.__dict__.get(
                attr_name, None
            )  # getattr(pymodule, attr_name)

            if attr is None:
                logger_instrumentation.info(
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
                logger_instrumentation.info(
                    f"Depth: {depth}, Skipping attribute as it is getfile: {attr_name}"
                )
                continue

            # skip private attributes
            if attr_name.startswith("__"):
                logger_instrumentation.info(
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

            if isinstance(attr, types.FunctionType) or isinstance(
                attr, types.BuiltinFunctionType
            ):
                # if isinstance(attr
                try:
                    if attr in skipped_functions:
                        logger_instrumentation.info(
                            f"Depth: {depth}, Skipping function: {attr_name}"
                        )
                        continue
                except Exception as e:
                    logger_instrumentation.fatal(
                        f"Depth: {depth}, Error while checking if function {attr_name} is in skipped_functions: {e}"
                    )
                    continue
                logger_instrumentation.info(f"Instrumenting function: {attr_name}")
                wrapped = wrapper(attr)
                try:
                    setattr(pymodule, attr_name, wrapped)
                except Exception as e:
                    # handling immutable types and attrs that have no setters
                    logger_instrumentation.info(
                        f"Depth: {depth}, Skipping function {attr_name} due to error: {e}"
                    )
                    continue
                count_wrapped += 1
            elif isinstance(attr, types.ModuleType):
                if attr.__name__ in modules_to_skip:
                    logger_instrumentation.info(
                        f"Depth: {depth}, Skipping module due to modules_to_skip: {attr_name}"
                    )
                    continue

                if attr in skipped_modules:
                    logger_instrumentation.info(
                        f"Depth: {depth}, Skipping module: {attr_name}"
                    )
                    continue
                if not attr.__name__.startswith(
                    self.root_module
                ):  # TODO: refine the logic of how to rule out irrelevant modules
                    logger_instrumentation.info(
                        f"Depth: {depth}, Skipping module due to irrelevant name:{attr_name}"
                    )
                    skipped_modules.add(attr)
                    continue

                logger_instrumentation.info(
                    f"Depth: {depth}, Recursing into module: {attr_name}"
                )
                count_wrapped += self._instrument_module(attr, depth + 1)

            elif inspect.isclass(attr):
                logger_instrumentation.info(
                    f"Depth: {depth}, Recursing into class: {attr_name}"
                )
                if not attr.__module__.startswith(self.root_module):
                    logger_instrumentation.info(
                        f"Depth: {depth}, Skipping class {attr_name} due to irrelevant module: {attr.__module__}"
                    )
                    continue
                count_wrapped += self._instrument_module(attr, depth + 1)

        logger_instrumentation.info(
            f"Depth: {depth}, Wrapped {count_wrapped} functions in module {target_name}"
        )
        return count_wrapped


class StateVarObserver:
    """
    Currently only suports torch models
    TODO: Generalize this to general python objects
    """

    def __init__(self, var):
        # Get the current thread object
        if isinstance(var, list):
            assert (
                len(var) == 1
            ), "Currently only supports single variable, please use multiple observers for multiple variables."
            var = var[0]
        assert isinstance(var, torch.nn.Module), "Currently only supports torch models."
        self.var = var
        self.current_state = self._get_state_copy()
        # dump the initial state
        logger_trace.info(
            # also not compressing here (we want to maintain tensor information)
            json.dumps(
                {
                    "process_id": os.getpid(),
                    "thread_id": threading.current_thread().ident,
                    "meta_vars": meta_vars,
                    "type": "state_dump",
                    "var": self.var.__class__.__name__,
                    "var_type": typename(self.var),
                    "state": self.current_state,
                }
            )
        )

    def _get_state_copy(self):
        return {
            name: param.clone().detach()
            for name, param in self.model.named_parameters()
        }

    def has_changed(self):
        # Check if there is any change in the model parameters
        for name in self.current_state:
            if not torch.equal(self.current_state[name], self.model.state_dict()[name]):
                logger_trace.info(f"State variable {name} has changed")
                logger_trace.info(
                    json.dumps(
                        {
                            "thread_id": threading.current_thread().ident,
                            "process_id": os.getpid(),
                            "type": "state_variable_change",
                            "variable": name,
                        }
                    )
                )
                self.current_state = self._get_state_copy()

        def is_safe_getattr(obj, attr):
            try:
                getattr(obj, attr)
                return True
            except Exception as e:
                print(
                    f"Failed to get attribute {attr} of parameter {name}, skipping it. Error: {e}"
                )
                return False

        state_copy = []
        for name, param in self.var.named_parameters():
            state_copy.append(
                {
                    "name": name,
                    "type": typename(param),
                    "param": param.clone().detach().tolist(),
                    "properties": {},
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
                    continue
                # try to serialize the attribute, if it fails, then skip it
                try:
                    json.dumps(attr)  # skipping compression
                except Exception as e:
                    logger_instrumentation.warn(
                        f"Failed to serialize attribute {attr_name} of parameter {name}, skipping it. Error: {e}"
                    )
                    continue

                state_copy[-1]["properties"][attr_name] = attr

        return state_copy

    def observe(self):
        """The function is called to observe the state of the model. Each call to this function will
        1. Get the current state of the model
        2. Compare it with the previous state
        3. Log the differences
            The differences are computed by comparing the below values:
                - The value of the tensor.
                - The properties of the tensor, such as requires_grad, device, tensor_model_parallel, etc.
        """
        state_copy = self._get_state_copy()
        for old_param, new_param in zip(self.current_state, state_copy):
            # three types of changes: value, properties, and both
            msg_dict = {
                "process_id": os.getpid(),
                "thread_id": threading.current_thread().ident,
                "meta_vars": meta_vars,
                "type": "state_change",
                # "var": self.var.__class__.__name__,
                "var_type": old_param["type"],  # FIXME: hardcoding the type for now
                "var_name": old_param["name"],
            }
            if old_param["param"] != new_param["param"]:
                if "change" not in msg_dict:
                    msg_dict["change"] = {}
                msg_dict["change"]["value"] = {
                    "old": old_param["param"],
                    "new": new_param["param"],
                }
            if old_param["properties"] != new_param["properties"]:
                if "change" not in msg_dict:
                    msg_dict["change"] = {}
                msg_dict["change"]["properties"] = {
                    "old": old_param["properties"],
                    "new": new_param["properties"],
                }
            if "change" in msg_dict:
                logger_trace.info(json.dumps(msg_dict))

        self.current_state = state_copy


if __name__ == "__main__":
    instrumentor(torch).instrument()
