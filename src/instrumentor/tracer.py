import functools
import inspect
import json
import logging
import os
import threading
import types
import uuid

import torch

logger_instrumentation = logging.getLogger("instrumentation")
logger_trace = logging.getLogger("trace")

meta_vars: dict[str, object] = {}


def global_wrapper(original_function, *args, **kwargs):
    func_id = str(uuid.uuid4())
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
                "uuid": func_id,
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
            json.dumps(
                {
                    "uuid": func_id,
                    "thread_id": thread_id,
                    "process_id": process_id,
                    "meta_vars": meta_vars,
                    "type": "function_call (post) (exception)",
                    "function": func_name,
                    "args": [f"{arg}" for arg in args],
                    "kwargs": [f"{k}={v}" for k, v in kwargs.items()],
                    "exception": str(e),
                }
            )
        )
        raise e
    # logger_trace.info({'type': 'function_call (post)', 'function': original_function.__name__, 'result': result})
    logger_trace.info(
        json.dumps(
            {
                "uuid": func_id,
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


instrumented_modules = set()
skipped_modules: set[types.ModuleType | type | types.FunctionType] = set()
skipped_functions = set()

# there are certain modules that we don't want to instrument (for example, download(), tqdm, etc.)
modules_to_skip = ["torch.fx"]


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

            if isinstance(attr, types.FunctionType):
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
            json.dumps(
                {
                    "process_id": os.getpid(),
                    "thread_id": threading.current_thread().ident,
                    "meta_vars": meta_vars,
                    "type": "state_dump",
                    "var": self.var.__class__.__name__,
                    "var_type": "torch.nn.Module",  # FIXME: hardcoding the type for now
                    "state": self.current_state,
                }
            )
        )

    def _get_state_copy(self):
        def is_safe_getattr(obj, attr):
            try:
                getattr(obj, attr)
                return True
            except Exception as e:
                logger_trace.warn(
                    f"Failed to get attribute {attr} of parameter {name}, skipping it. Error: {e}"
                )
                return False

        state_copy = []
        for name, param in self.var.named_parameters():
            state_copy.append(
                {
                    "name": name,
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
                    json.dumps(attr)
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
                "var": self.var.__class__.__name__,
                "var_type": "torch.nn.Module",  # FIXME: hardcoding the type for now
                "name": old_param["name"],
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
