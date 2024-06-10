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


original_processgroup_broadcast = ProcessGroup.broadcast

# unproxy the arguments before calling the original function
@functools.wraps(original_processgroup_broadcast)
def processgroup_broadcast(
    *args, **kwargs
) -> None:
    args = [unproxy_arg(arg) for arg in args]
    kwargs = {k: unproxy_arg(v) for k, v in kwargs.items()}
    return original_processgroup_broadcast(*args, **kwargs)

ProcessGroup.broadcast = processgroup_broadcast

original_bf16_optimizer_flatten_dense_tensors_aligned =
BF16_Optimizer._flatten_dense_tensors_aligned

def _flatten_dense_tensors_aligned(self, tensor_list, alignment):
    # Your code here
    pass

BF16Optimizer._flatten_dense_tensors_aligned = _flatten_dense_tensors_aligned