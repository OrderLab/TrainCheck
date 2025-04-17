import logging

import torch.optim.optimizer as optimizer_

from traincheck.proxy_wrapper.proxy_basics import adapt_func_for_proxy
from traincheck.utils import typename


def is_funcs_to_be_unproxied(original_func):
    if not hasattr(original_func, "__module__") or not hasattr(
        original_func, "__name__"
    ):
        return False
    original_func_fullname = original_func.__module__ + "." + original_func.__name__
    return any(
        [
            func_to_be_unproxied in original_func_fullname
            for func_to_be_unproxied in funcs_to_be_unproxied
        ]
    )


funcs_to_be_unproxied = [
    # torch.save,
    "torch.serialization.save",
    # torch.load,
    "torch.serialization.load",
]

funcs_to_be_replaced = {}

original__default_to_fused_or_foreach = optimizer_.__dict__.get(
    "_default_to_fused_or_foreach"
)
if original__default_to_fused_or_foreach is None:
    logger = logging.getLogger(__name__)
    logger.warning(
        "The function _default_to_fused_or_foreach is not found in the module torch.optim.optimizer"
    )
else:
    _default_to_fused_or_foreach = adapt_func_for_proxy(
        optimizer_.__dict__.get("_default_to_fused_or_foreach")
    )
    setattr(optimizer_, "_default_to_fused_or_foreach", _default_to_fused_or_foreach)
    funcs_to_be_replaced[typename(_default_to_fused_or_foreach)] = (
        _default_to_fused_or_foreach
    )
