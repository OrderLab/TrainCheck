import functools

from mldaikon.proxy_wrapper.proxy_basics import is_proxied, unproxy_func


def observe_proxy_var(var, phase):
    if hasattr(var, "is_ml_daikon_proxied_obj"):
        var.dump_trace(phase)
    else:
        NotImplementedError(f"observe method not implemented for {var}")


def add_observer_to_func(func, unproxy=False):
    original_func = func

    @functools.wraps(original_func)
    def wrapper(*args, **kwargs):
        observe_var = []
        for arg in args:
            # if the arg is list or tuple, check if it contains proxied object
            if type(arg) in [list, tuple]:
                for element in arg:
                    if is_proxied(element):
                        observe_var.append(element)
            if is_proxied(arg):
                observe_var.append(arg)
        # pre observe
        for var in observe_var:
            observe_proxy_var(var, "pre_observe")
        if unproxy:
            result = unproxy_func(original_func)(*args, **kwargs)
        else:
            result = original_func(*args, **kwargs)
        # post observe
        for var in observe_var:
            observe_proxy_var(var, "post_observe")
        return result

    return wrapper
