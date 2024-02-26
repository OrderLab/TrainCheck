import types
import inspect
import functools
import torch
import logging
import os
import sys
import json
import uuid
import threading


logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)
logger_trace = logging.getLogger("trace")


def global_wrapper(original_function, *args, **kwargs):
    func_id = str(uuid.uuid4())
    # Get the current thread object
    current_thread = threading.current_thread()
    # Get the thread ID
    thread_id = current_thread.ident
    process_id = os.getpid()

    # logger_trace.info({'type': 'function_call (pre)', 'function': original_function.__name__, 'args': args, 'kwargs': kwargs})
    logger_trace.info(
        json.dumps(
            {
                "uuid": func_id,
                "thread_id": thread_id,
                "process_id": process_id,
                "type": "function_call (pre)",
                "function": original_function.__name__,
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
                    "type": "function_call (post) (exception)",
                    "function": original_function.__name__,
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
                "type": "function_call (post)",
                "function": original_function.__name__,
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
skipped_modules = set()
skipped_functions = set()

# there are certain modules that we don't want to instrument (for example, download(), tqdm, etc.)
modules_to_skip = ["torch.fx"]


def instrument(pymodule: types.ModuleType, depth=0):
    if pymodule in instrumented_modules or pymodule in skipped_modules:
        logger.info(f"Depth: {depth}, Skipping module: {pymodule.__name__}")
        return 0

    logger.info(f"Depth: {depth}, Instrumenting module: {pymodule.__name__}")
    instrumented_modules.add(pymodule)

    count_wrapped = 0
    for attr_name in dir(pymodule):
        if not hasattr(pymodule, attr_name):
            # handle __abstractmethods__ attribute
            logger.info(
                f"Depth: {depth}, Skipping attribute as it does not exist: {attr_name}"
            )
            continue

        attr = pymodule.__dict__.get(attr_name, None)  # getattr(pymodule, attr_name)

        if attr is None:
            logger.info(
                f"Depth: {depth}, Skipping attribute as it is None: {attr_name}"
            )
            """
            TODO: From my observation, this is happening for the attributes that are implemented in C extension, which usually include math ops and other low level operations for tensors
            , such as tensor.add_.
            We should support these operations as well. Reason is in PyTorch-FORUM84911.
            """
            continue

        # skip private attributes
        if attr_name.startswith("__"):
            logger.info(f"Depth: {depth}, Skipping magic functions: {attr_name}")
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

        if isinstance(attr, types.FunctionType) or isinstance(
            attr, types.BuiltinFunctionType
        ):
            if attr in skipped_functions:
                logger.info(f"Depth: {depth}, Skipping function: {attr_name}")
                continue

            logger.info(f"Instrumenting function: {attr_name}")
            wrapped = wrapper(attr)
            try:
                setattr(pymodule, attr_name, wrapped)
            except Exception as e:
                # handling immutable types and attrs that have no setters
                logger.info(
                    f"Depth: {depth}, Skipping function {attr_name} due to error: {e}"
                )
                continue
            count_wrapped += 1
        elif isinstance(attr, types.ModuleType):
            if attr.__name__ in modules_to_skip:
                logger.info(
                    f"Depth: {depth}, Skipping module due to modules_to_skip: {attr_name}"
                )
                continue

            if attr in skipped_modules:
                logger.info(f"Depth: {depth}, Skipping module: {attr_name}")
                continue
            if not attr.__name__.startswith("torch"):
                logger.info(
                    f"Depth: {depth}, Skipping module due to irrelevant name:{attr_name}"
                )
                skipped_modules.add(attr)
                continue

            logger.info(f"Depth: {depth}, Recursing into module: {attr_name}")
            count_wrapped += instrument(attr, depth + 1)

        elif inspect.isclass(attr):
            logger.info(f"Depth: {depth}, Recursing into class: {attr_name}")
            if not attr.__module__.startswith("torch"):
                logger.info(
                    f"Depth: {depth}, Skipping class {attr_name} due to irrelevant module: {attr.__module__}"
                )
                continue
            count_wrapped += instrument(attr, depth + 1)

    logger.info(
        f"Depth: {depth}, Wrapped {count_wrapped} functions in module {pymodule.__name__}"
    )
    return count_wrapped


class StateVarObserver:
    """
    Currently only suports torch models
    """

    def __init__(self, model):
        # Get the current thread object
        self.model = model
        self.current_state = self._get_state_copy()

    def _get_state_copy(self):
        # Return a copy of the current state of the model
        return {
            name: param.clone().detach()
            for name, param in self.model.named_parameters()
        }

    def has_changed(self):
        # Check if there is any change in the model parameters
        for name in self.current_state:
            if not torch.equal(self.current_state[name], self.model.state_dict()[name]):
                # logger_trace.info(f"State variable {name} has changed")
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
                return True
        self.current_state = self._get_state_copy()
        return False
