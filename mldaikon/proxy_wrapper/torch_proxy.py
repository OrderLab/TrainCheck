from typing import List, Tuple
import torch
import logging
import torch.distributed
from torch.optim.optimizer import _get_fused_kernels_supported_devices, _get_foreach_kernels_supported_devices, _foreach_supported_types
from torch._C._distributed_c10d import ReduceOp
from .proxy import Proxy

#################################################
###         Proxied Torch functions

original_default_to_fused_or_foreach = torch.optim.optimizer._default_to_fused_or_foreach

def _default_to_fused_or_foreach(params: List[torch.Tensor],
                                 differentiable: bool,
                                 use_fused: bool = False) -> Tuple[bool, bool]:
    if torch.jit.is_scripting() or differentiable:
        return False, False

    fused_supported_devices = _get_fused_kernels_supported_devices()
    foreach_supported_devices = _get_foreach_kernels_supported_devices()
    fused = use_fused and all(
        p is None or (type(p) in _foreach_supported_types or hasattr(p, 'is_proxied_obj') 
                      and
                      p.device.type in fused_supported_devices and
                      torch.is_floating_point(p)) for p in params
    )
    foreach = not fused and all(
        p is None or (type(p) in _foreach_supported_types  or hasattr(p, 'is_proxied_obj') 
                      and
                      p.device.type in foreach_supported_devices) for p in params
    )
    return fused, foreach

torch.optim.optimizer._default_to_fused_or_foreach = _default_to_fused_or_foreach

# Save the original broadcast function
original_broadcast = torch.distributed.broadcast


def broadcast(tensor, src, group=None, async_op=False):
    # Perform the original broadcast operation
    if type(tensor) is Proxy:
        tensor = tensor._obj
    if type(src) is Proxy:
        src = src._obj
    if type(group) is Proxy:
        group = group._obj
    original_broadcast(tensor, src, group, async_op)

    # Wrap the first argument in a Proxy object
    # tensor = Proxy(tensor, logdir='proxy_log.log', log_level=logging.INFO)


# Override the broadcast function
torch.distributed.broadcast = broadcast

original_all_reduce = torch.distributed.all_reduce


def all_reduce(tensor, op=ReduceOp.SUM, group=None, async_op=False):
    if type(tensor) is Proxy:
        tensor = tensor._obj
    if type(op) is Proxy:
        op = op._obj
    if type(group) is Proxy:
        group = group._obj
    # Perform the original all_reduce operation
    original_all_reduce(tensor, op, group, async_op)

    # Wrap the first argument in a Proxy object
    # tensor = Proxy(tensor, logdir='proxy_log.log', log_level=logging.INFO)


# Override the all_reduce function
torch.distributed.all_reduce = all_reduce
