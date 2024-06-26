import functools
import tokenize as tokenize

import torch.optim.adam as adam
import torch.optim.sgd as sgd

try:
    from deepspeed.runtime.bf16_optimizer import BF16_Optimizer
except ImportError:
    pass
from torch._C._distributed_c10d import ProcessGroup

from mldaikon.proxy_wrapper.proxy_basics import is_proxied, unproxy_arg

#################################################
###         Proxied Torch functions


def unproxy_func(func):
    original_func = func

    @functools.wraps(original_func)
    def wrapper(*args, **kwargs):
        args = [unproxy_arg(arg) for arg in args]
        kwargs = {k: unproxy_arg(v) for k, v in kwargs.items()}
        return original_func(*args, **kwargs)

    return wrapper


ProcessGroup.broadcast = unproxy_func(ProcessGroup.__dict__.get("broadcast"))
ProcessGroup.allreduce = unproxy_func(ProcessGroup.__dict__.get("allreduce"))
ProcessGroup.allgather = unproxy_func(ProcessGroup.__dict__.get("allgather"))
tokenize._tokenize = unproxy_func(tokenize.__dict__.get("_tokenize"))  # type: ignore
if "BF16_Optimizer" in globals():
    BF16_Optimizer._flatten_dense_tensors_aligned = unproxy_func(
        BF16_Optimizer.__dict__.get("_flatten_dense_tensors_aligned")
    )
    BF16_Optimizer._update_storage_to_flattened_tensor = unproxy_func(
        BF16_Optimizer.__dict__.get("_update_storage_to_flattened_tensor")
    )

#################################################


def observe_proxy_var(var, phase):
    if hasattr(var, "is_ml_daikon_proxied_obj"):
        var.dump_trace(phase)
    else:
        NotImplementedError(f"observe method not implemented for {var}")


def add_observer_to_func(func):
    original_func = func

    @functools.wraps(original_func)
    def wrapper(*args, **kwargs):
        observe_var = []
        for arg in args:
            # if the arg is list or tuple, check if it contains proxied object
            if type(arg) in [list, tuple]:
                for element in arg:
                    if is_proxied(element):
                        observe_var.append(element)
            if is_proxied(arg):
                observe_var.append(arg)
        # pre observe
        for var in observe_var:
            observe_proxy_var(var, "pre_observe")
        result = original_func(*args, **kwargs)
        # post observe
        for var in observe_var:
            observe_proxy_var(var, "post_observe")
        return result

    return wrapper


#################################################

adam.adam = add_observer_to_func(adam.__dict__.get("adam"))
sgd.sgd = add_observer_to_func(sgd.__dict__.get("sgd"))
