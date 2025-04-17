import functools


def is_proxied(obj):
    try:
        if obj is not None and "is_traincheck_proxied_obj" in obj.__dict__:
            return True
    except Exception:
        return False
    return False


def unproxy_arg(arg, inspect_torch_module=False):

    if is_proxied(arg):
        return unproxy_arg(arg._obj, inspect_torch_module)
    elif type(arg) in [list]:
        return [unproxy_arg(element, inspect_torch_module) for element in arg]
    elif type(arg) in [tuple]:
        return tuple(unproxy_arg(element, inspect_torch_module) for element in arg)
    # if it is a torch module, unproxy all its named children
    elif inspect_torch_module:
        import torch

        if isinstance(arg, torch.nn.Module):
            for name, module in arg.named_children():
                arg._modules[name] = unproxy_arg(module, inspect_torch_module)
            # handle named_parameters
            for name, param in arg.named_parameters():
                arg._parameters[name] = unproxy_arg(param, inspect_torch_module)
            return arg

        return arg
    else:
        return arg


# Proxy class to wrap the torch._VF module
class VFProxy:
    def __init__(self, vf_module):
        self._vf_module = vf_module

    def __getattr__(self, name):
        attr = getattr(self._vf_module, name)

        if callable(attr):
            return self.unproxy_func(attr)
        else:
            return attr

    def unproxy_func(self, func, inspect_torch_module=False):
        original_func = func

        @functools.wraps(original_func)
        def wrapper(*args, **kwargs):
            args = [unproxy_arg(arg, inspect_torch_module) for arg in args]
            kwargs = {k: unproxy_arg(v) for k, v in kwargs.items()}
            return original_func(*args, **kwargs)

        return wrapper
