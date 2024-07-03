import torch.optim.optimizer as optimizer_

from mldaikon.proxy_wrapper.proxy_basics import adapt_func_for_proxy
from mldaikon.utils import typename

_default_to_fused_or_foreach = adapt_func_for_proxy(
    optimizer_.__dict__.get("_default_to_fused_or_foreach")
)
setattr(optimizer_, "_default_to_fused_or_foreach", _default_to_fused_or_foreach)


funcs_to_be_replaced = {
    typename(_default_to_fused_or_foreach): _default_to_fused_or_foreach,
}
