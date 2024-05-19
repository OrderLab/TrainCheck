import torch
import logging
import torch.distributed
from torch._C._distributed_c10d import ReduceOp
from .proxy import Proxy
#################################################
###         Proxied Torch functions

# Save the original broadcast function
original_broadcast = torch.distributed.broadcast

def broadcast(tensor, src, group=None, async_op=False):
    # Perform the original broadcast operation
    original_broadcast(tensor, src, group, async_op)

    # Wrap the first argument in a Proxy object
    tensor = Proxy(tensor, logdir='proxy_log.log', log_level=logging.INFO)

# Override the broadcast function
torch.distributed.broadcast = broadcast

original_all_reduce = torch.distributed.all_reduce

def all_reduce(tensor, op=ReduceOp.SUM, group=None, async_op=False):
    # Perform the original all_reduce operation
    original_all_reduce(tensor, op, group, async_op)

    # Wrap the first argument in a Proxy object
    tensor = Proxy(tensor, logdir='proxy_log.log', log_level=logging.INFO)

# Override the all_reduce function
torch.distributed.all_reduce = all_reduce
