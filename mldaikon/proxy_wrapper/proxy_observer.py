import functools
import typing

from traincheck.config.config import should_disable_proxy_dumping
from traincheck.utils import typename

if typing.TYPE_CHECKING:
    from traincheck.proxy_wrapper.proxy import Proxy
from .proxy_basics import is_proxied, unproxy_func


def observe_proxy_var(
    var: "Proxy",
    phase,
    observe_api_name: str,
):
    # update the proxy object's timestamp
    var.update_timestamp()

    if phase == "post_observe":
        var.register_object()

    if should_disable_proxy_dumping():
        # do nothing but return, obj state dumps should be triggered separately
        return None

    var.dump_trace(phase=phase, dump_loc=observe_api_name)


def add_observer_to_func(original_function, unproxy=False):
    original_function_name = typename(original_function)

    @functools.wraps(original_function)
    def wrapper(*args, **kwargs):
        proxied_vars = []
        for arg in args:
            # if the arg is list or tuple, check if it contains proxied object
            if type(arg) in [list, tuple]:
                for element in arg:
                    if is_proxied(element):
                        proxied_vars.append(element)
            if is_proxied(arg):
                proxied_vars.append(arg)

        # pre observe
        for i, var in enumerate(proxied_vars):
            observe_proxy_var(
                var,
                "pre_observe",
                original_function_name,
            )

        processed_function = original_function
        if unproxy:
            processed_function = unproxy_func(original_function)

        result = processed_function(*args, **kwargs)

        # post observe
        for i, var in enumerate(proxied_vars):
            observe_proxy_var(
                var,
                "post_observe",
                original_function_name,
            )
        return result

    return wrapper
