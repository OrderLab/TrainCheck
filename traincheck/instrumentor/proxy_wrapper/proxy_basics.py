import ast
import functools
import inspect
from collections.abc import Callable

import astor

import traincheck.config.config as config
from traincheck.instrumentor.proxy_wrapper.list_writeback import (
    call_with_list_writebacks,
    list_writeback_positions,
    prepare_list_writebacks,
)
from traincheck.utils import typename


def is_compile_internal_module(obj):
    mod = getattr(type(obj), "__module__", "") or ""
    if any(mod.startswith(p) for p in config.COMPILE_INTERNAL_MODULE):
        return True
    name = type(obj).__name__
    if mod.startswith("torch._dynamo") and name != "OptimizedModule":
        return True
    return False


def is_fake_tensor(x) -> bool:
    if not config.USE_TORCH_COMPILE:
        return False
    try:
        from torch._subclasses.fake_tensor import FakeTensor
        from torch.fx import Proxy as FxProxy

        if isinstance(x, FakeTensor):
            return True
        if isinstance(x, FxProxy):
            return True
    except Exception:
        pass

    try:
        if is_compile_internal_module(x):
            return True
    except Exception:
        return True

    try:
        if x.device.type == "meta":
            return True
    except Exception:
        return True

    return False


def is_proxied(obj):
    try:
        if is_fake_tensor(obj):
            return False
        if obj is not None and "is_traincheck_proxied_obj" in obj.__dict__:
            return True
    except Exception:
        return False
    return False


def is_proxyparameter(obj):
    try:
        if is_fake_tensor(obj):
            return False
        if obj is not None and "is_traincheck_proxyparameter" in obj.__dict__:
            return True
    except Exception:
        return False
    return False


def unproxy_arg(arg, inspect_torch_module=False):

    if is_proxied(arg):
        return unproxy_arg(arg._obj, inspect_torch_module)
    elif type(arg) in [list]:
        # Always a new list: the caller's list keeps its proxies (an optimizer's
        # param list must, since its state is keyed by them). APIs that return
        # results by replacing slots of a list argument are registered in
        # list_writeback, and unproxy_func writes those slots back (issue #176).
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


def unproxy_args_kwargs(args, kwargs, inspect_torch_module=False):
    args = [unproxy_arg(arg, inspect_torch_module) for arg in args]
    kwargs = {k: unproxy_arg(v) for k, v in kwargs.items()}
    return args, kwargs


def unproxy_func(
    func: Callable[..., object],
    inspect_torch_module: bool = False,
    api_name: str | None = None,
) -> Callable[..., object]:
    """Wrap ``func`` so that it is called with unproxied arguments.

    ``api_name`` is the name ``func`` is registered under in
    ``LIST_WRITEBACK_ARGUMENTS``; ``None`` derives it with ``typename``. Slots
    that ``func`` replaces in a registered list argument are written back to
    the caller's list, also when ``func`` raises. Registered parameters are
    resolved here, so a registry entry that does not match ``func`` raises
    ``ListWritebackError`` when wrapping rather than at a call.
    """
    original_func = func
    lookup_name = typename(func) if api_name is None else api_name
    positions = list_writeback_positions(func, lookup_name)

    @functools.wraps(original_func)
    def wrapper(*args: object, **kwargs: object) -> object:
        callee_args, callee_kwargs = unproxy_args_kwargs(
            args, kwargs, inspect_torch_module
        )
        if positions is None:
            return original_func(*callee_args, **callee_kwargs)
        pending = prepare_list_writebacks(
            lookup_name, positions, args, kwargs, callee_args, callee_kwargs
        )
        return call_with_list_writebacks(
            original_func, pending, callee_args, callee_kwargs
        )

    return wrapper


def type_handle_traincheck_proxy(x):
    if hasattr(x, "is_traincheck_proxied_obj"):
        return type(x._obj)
    return type(x)


class TypeToIsInstanceTransformer(ast.NodeTransformer):
    # add from mldaiokn.proxy_wrapper.proxy_basics import type_handle_traincheck_proxy after function definition
    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        # Inject code right after the def statement
        inject_code = """
from traincheck.instrumentor.proxy_wrapper.proxy_basics import type_handle_traincheck_proxy
"""
        inject_node = ast.parse(inject_code).body
        node.body = inject_node + node.body
        return node

    def visit_Call(self, node):
        self.generic_visit(node)

        # Check if the call is type(xxx)
        if (
            isinstance(node.func, ast.Name)
            and node.func.id == "type"
            and len(node.args) == 1
        ):

            # Replace type(xxx) with type_handle_traincheck_proxy(xxx)
            new_node = ast.Call(
                func=ast.Name(id="type_handle_traincheck_proxy", ctx=ast.Load()),
                args=node.args,
                keywords=[],
            )

            return ast.copy_location(new_node, node)
        return node


def adapt_func_for_proxy(func):
    """Adapt a function to work with proxied objects.
    - Replace type() calls with type_handle_traincheck_proxy() so that type(ProxyObj) returns type(ProxyObj._obj) instead of Proxy
    """

    source = inspect.getsource(func)
    tree = ast.parse(source)
    transformer = TypeToIsInstanceTransformer()
    new_tree = transformer.visit(tree)
    ast.fix_missing_locations(new_tree)
    new_code = astor.to_source(new_tree)

    # Define a new dictionary to execute the transformed code
    new_locals = {}
    exec(new_code, func.__globals__, new_locals)

    # Return the transformed function object
    return new_locals[func.__name__]
