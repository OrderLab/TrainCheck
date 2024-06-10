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
from torch._C._distributed_c10d import ReduceOp

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
