import torch.optim.optimizer as optimizer_

from mldaikon.proxy_wrapper.proxy_basics import transform_function
from mldaikon.utils import typename

# @functools.wraps(optimizer_._default_to_fused_or_foreach)
# def _default_to_fused_or_foreach(
#     params: list[torch.Tensor], differentiable: bool, use_fused: bool = False
# ) -> tuple[bool, bool]:
#     print("_default_to_fused_or_foreach_function wrapped")
#     if torch.jit.is_scripting() or differentiable:
#         return False, False
#     fused_supported_devices = optimizer_._get_fused_kernels_supported_devices()
#     foreach_supported_devices = optimizer_._get_foreach_kernels_supported_devices()
#     fused = use_fused and all(
#         p is None
#         or (
#             type(p) in optimizer_._foreach_supported_types
#             or hasattr(p, "is_ml_daikon_proxied_obj")
#             and p.device.type in fused_supported_devices
#             and torch.is_floating_point(p)
#         )
#         for p in params
#     )
#     foreach = not fused and all(
#         p is None
#         or (
#             type(p) in optimizer_._foreach_supported_types
#             or hasattr(p, "is_ml_daikon_proxied_obj")
#             and p.device.type in foreach_supported_devices
#         )
#         for p in params
#     )
#     return fused, foreach
_default_to_fused_or_foreach = transform_function(
    optimizer_.__dict__.get("_default_to_fused_or_foreach")
)
setattr(optimizer_, "_default_to_fused_or_foreach", _default_to_fused_or_foreach)


funcs_to_be_replaced = {
    typename(_default_to_fused_or_foreach): _default_to_fused_or_foreach,
}
