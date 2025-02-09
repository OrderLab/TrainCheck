import functools

from mldaikon.config.config import should_disable_proxy_dumping
from mldaikon.utils import typename

from .proxy_basics import is_proxied, unproxy_func
from .proxy_config import auto_observer_config


def observe_proxy_var(
    var,
    phase,
    observe_loc: str,
    only_dump_when_change=True,
    pre_observed_var=None,
    trace_info=None,
):

    if is_proxied(var):
        # register/update the var to be observed in post_observe
        if phase == "post_observe":
            var.register_object()

        if should_disable_proxy_dumping():
            # do nothing but return, obj state dumps should be triggered separately
            return None

        if only_dump_when_change:
            if phase == "pre_observe":
                trace_info = var.dump_trace(
                    phase, only_dump_when_change, dump_loc=observe_loc
                )
                assert trace_info is not None, "trace_info should not be None"
                return trace_info

            elif phase == "post_observe":
                assert (
                    pre_observed_var is not None
                ), "pre_observed_var should not be None"
                assert trace_info is not None, "trace_info should not be None"
                var.dump_trace(
                    phase,
                    only_dump_when_change,
                    pre_observed_var,
                    trace_info,
                    dump_loc=observe_loc,
                )
        else:
            var.dump_trace(phase)
        return None
    else:
        NotImplementedError(f"observe method not implemented for {var}")


def add_observer_to_func(original_function, unproxy=False):
    only_dump_when_change = auto_observer_config["only_dump_when_change"]
    original_function_name = typename(original_function)

    @functools.wraps(original_function)
    def wrapper(*args, **kwargs):
        observe_var = []
        proxied_var = []
        for arg in args:
            # if the arg is list or tuple, check if it contains proxied object
            if type(arg) in [list, tuple]:
                for element in arg:
                    if is_proxied(element):
                        proxied_var.append(element)
                        observe_var.append(element)
            if is_proxied(arg):
                proxied_var.append(arg)
                observe_var.append(arg)

        # pre observe
        trace_info = len(observe_var) * [None]
        for i, var in enumerate(observe_var):
            if only_dump_when_change:
                import copy

                try:
                    pre_observed_var = copy.deepcopy(var)
                except Exception as e:
                    import traceback

                    raise Exception(
                        f"Error in deepcopy for {var} of type {type(var._obj)},"
                        f"please implement deepcopy for the class,"
                        f"or set only_dump_when_change=False"
                        f"Error: {e}"
                        f"Stack Trace: {traceback.format_exc()}"
                    )
            else:
                pre_observed_var = var
            trace_info[i] = observe_proxy_var(
                pre_observed_var,
                "pre_observe",
                original_function_name,
                only_dump_when_change,
            )

        processed_function = original_function
        if unproxy:
            processed_function = unproxy_func(original_function)

        result = processed_function(*args, **kwargs)

        # post observe
        for i, var in enumerate(observe_var):
            observe_proxy_var(
                var,
                "post_observe",
                original_function_name,
                only_dump_when_change,
                pre_observed_var,
                trace_info[i],
            )
        return result

    return wrapper
