import ast
import functools
import inspect

import astor


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


def unproxy_func(func, inspect_torch_module=False):
    original_func = func

    @functools.wraps(original_func)
    def wrapper(*args, **kwargs):
        args = [unproxy_arg(arg, inspect_torch_module) for arg in args]
        kwargs = {k: unproxy_arg(v) for k, v in kwargs.items()}
        return original_func(*args, **kwargs)

    return wrapper


def unproxy_args_kwargs(args, kwargs, inspect_torch_module=False):
    args = [unproxy_arg(arg, inspect_torch_module) for arg in args]
    kwargs = {k: unproxy_arg(v) for k, v in kwargs.items()}
    return args, kwargs


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
from traincheck.proxy_wrapper.proxy_basics import type_handle_traincheck_proxy
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
