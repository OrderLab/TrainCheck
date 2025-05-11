import tokenize as tokenize

try:
    from deepspeed.runtime.bf16_optimizer import BF16_Optimizer
except ImportError:
    pass
from torch._C._distributed_c10d import ProcessGroup

from traincheck.proxy_wrapper.proxy_basics import unproxy_func

#################################################
###         Proxied Torch functions


setattr(ProcessGroup, "broadcast", unproxy_func(ProcessGroup.__dict__.get("broadcast")))
setattr(ProcessGroup, "allreduce", unproxy_func(ProcessGroup.__dict__.get("allreduce")))
setattr(ProcessGroup, "allgather", unproxy_func(ProcessGroup.__dict__.get("allgather")))

setattr(tokenize, "_tokenize", unproxy_func(tokenize.__dict__.get("_tokenize")))
if "BF16_Optimizer" in globals():
    BF16_Optimizer._flatten_dense_tensors_aligned = unproxy_func(
        BF16_Optimizer.__dict__.get("_flatten_dense_tensors_aligned")
    )
    BF16_Optimizer._update_storage_to_flattened_tensor = unproxy_func(
        BF16_Optimizer.__dict__.get("_update_storage_to_flattened_tensor")
    )
