import functools
from typing import List, Tuple
import torch
import logging
import torch.distributed
import torch.optim.optimizer as torch_optimizer
from torch.optim.optimizer import (
    _get_fused_kernels_supported_devices,
    _get_foreach_kernels_supported_devices,
    _foreach_supported_types,
)
from mldaikon.proxy_wrapper.proxy_basics import unproxy_arg, is_proxied
from torch._C._distributed_c10d import ReduceOp, ProcessGroup
from deepspeed.runtime.bf16_optimizer import BF16_Optimizer

#################################################
###         Proxied Torch functions

original_default_to_fused_or_foreach = torch_optimizer._default_to_fused_or_foreach


def _default_to_fused_or_foreach(
    params: List[torch.Tensor], differentiable: bool, use_fused: bool = False
) -> Tuple[bool, bool]:
    if torch.jit.is_scripting() or differentiable:
        return False, False

    fused_supported_devices = _get_fused_kernels_supported_devices()
    foreach_supported_devices = _get_foreach_kernels_supported_devices()
    fused = use_fused and all(
        p is None
        or (
            type(p) in _foreach_supported_types
            or hasattr(p, "is_proxied_obj")
            and p.device.type in fused_supported_devices
            and torch.is_floating_point(p)
        )
        for p in params
    )
    foreach = not fused and all(
        p is None
        or (
            type(p) in _foreach_supported_types
            or hasattr(p, "is_proxied_obj")
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


ProcessGroup.broadcast = unproxy_func(ProcessGroup.broadcast)
ProcessGroup.allreduce = unproxy_func(ProcessGroup.allreduce)
ProcessGroup.allgather = unproxy_func(ProcessGroup.allgather)
BF16_Optimizer._flatten_dense_tensors_aligned = unproxy_func(
    BF16_Optimizer._flatten_dense_tensors_aligned
)
BF16_Optimizer._update_storage_to_flattened_tensor = unproxy_func(
    BF16_Optimizer._update_storage_to_flattened_tensor
)


#################################################

