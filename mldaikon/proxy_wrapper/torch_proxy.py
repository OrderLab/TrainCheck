import functools
import tokenize as tokenize
from typing import List, Tuple

import torch
import torch.distributed
import torch.optim.adam as adam
import torch.optim.optimizer as torch_optimizer
import torch.optim.sgd as sgd

try:
    from deepspeed.runtime.bf16_optimizer import BF16_Optimizer
except ImportError:
    pass
from torch._C._distributed_c10d import ProcessGroup
from torch.optim.optimizer import (
    _foreach_supported_types,
    _get_foreach_kernels_supported_devices,
    _get_fused_kernels_supported_devices,
)

from mldaikon.proxy_wrapper.proxy_basics import is_proxied, unproxy_arg

#################################################
###         Proxied Torch functions

original_default_to_fused_or_foreach = torch_optimizer._default_to_fused_or_foreach


def _default_to_fused_or_foreach(
    params: List[torch.Tensor], differentiable: bool, use_fused: bool = False
) -> Tuple[bool, bool]:
    print("_default_to_fused_or_foreach_function wrapped")
    if torch.jit.is_scripting() or differentiable:
        return False, False

    fused_supported_devices = _get_fused_kernels_supported_devices()
    foreach_supported_devices = _get_foreach_kernels_supported_devices()
    fused = use_fused and all(
        p is None
        or (
            type(p) in _foreach_supported_types
            or hasattr(p, "is_ml_daikon_proxied_obj")
            and p.device.type in fused_supported_devices
            and torch.is_floating_point(p)
        )
        for p in params
    )
    foreach = not fused and all(
        p is None
        or (
            type(p) in _foreach_supported_types
            or hasattr(p, "is_ml_daikon_proxied_obj")
            and p.device.type in foreach_supported_devices
        )
        for p in params
    )
    return fused, foreach


torch_optimizer._default_to_fused_or_foreach = _default_to_fused_or_foreach


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
