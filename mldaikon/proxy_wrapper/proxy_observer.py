import functools

from mldaikon.proxy_wrapper.config import only_dump_when_change
from mldaikon.proxy_wrapper.proxy_basics import is_proxied, unproxy_func


def observe_proxy_var(
    var, phase, only_dump_when_change=True, pre_observed_var=None, trace_info=None
):
    if hasattr(var, "is_ml_daikon_proxied_obj"):
        if only_dump_when_change:
            if phase == "pre_observe":
                trace_info = var.dump_trace(phase, only_dump_when_change)
                assert trace_info is not None, "trace_info should not be None"
                return trace_info

            elif phase == "post_observe":
                assert (
                    pre_observed_var is not None
                ), "pre_observed_var should not be None"
                assert trace_info is not None, "trace_info should not be None"
                var.dump_trace(
                    phase, only_dump_when_change, pre_observed_var, trace_info
                )
        var.dump_trace(phase)
        return None
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
        trace_info = len(observe_var) * [None]
        for i, var in enumerate(observe_var):
            if only_dump_when_change:
                import copy

                try:
                    pre_observed_var = copy.deepcopy(var)
                except Exception:
                    pre_observed_var = var

            else:
                pre_observed_var = var
            trace_info[i] = observe_proxy_var(
                pre_observed_var, "pre_observe", only_dump_when_change
            )
        if unproxy:
            result = unproxy_func(original_func)(*args, **kwargs)
        else:
            result = original_func(*args, **kwargs)
        # post observe
        for i, var in enumerate(observe_var):
            observe_proxy_var(
                var,
                "post_observe",
                only_dump_when_change,
                pre_observed_var,
                trace_info[i],
            )
        return result

    return wrapper
