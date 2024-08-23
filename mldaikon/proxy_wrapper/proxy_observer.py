import functools

from mldaikon.instrumentor.tracer import should_dump_trace
from mldaikon.proxy_wrapper.proxy_basics import is_proxied, unproxy_func
from mldaikon.proxy_wrapper.proxy_config import auto_observer_config
from mldaikon.utils import typename


def observe_proxy_var(
    var, phase, only_dump_when_change=True, pre_observed_var=None, trace_info=None
):
    if is_proxied(var):
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
        else:
            var.dump_trace(phase)
        return None
    else:
        NotImplementedError(f"observe method not implemented for {var}")


def add_observer_to_func(original_function, cond_dump, unproxy=False):
    only_dump_when_change = auto_observer_config["only_dump_when_change"]

    @functools.wraps(original_function)
    def wrapper(*args, **kwargs):
        observe_var = []
        for arg in args:
            # if the arg is list or tuple, check if it contains proxied object
            if type(arg) in [list, tuple]:
                for element in arg:
                    if is_proxied(element):
                        if should_dump_trace(
                            cond_dump,
                            None,
                            f"VAR: {typename(element)}: {element.__dict__['var_name']}",
                            None,
                            None,
                        ):
                            observe_var.append(element)
                        else:
                            # TODO: @ziming-zh what's a good way to dump a log here? Does logger = logging.getLogger(__name__) work?
                            pass
            if is_proxied(arg):
                if should_dump_trace(
                    cond_dump,
                    None,
                    f"VAR: {typename(arg)}: {arg.__dict__['var_name']}",
                    None,
                    None,
                ):
                    observe_var.append(arg)
                else:
                    # TODO: @ziming-zh what's a good way to dump a log here? Does logger = logging.getLogger(__name__) work?
                    pass

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
                pre_observed_var, "pre_observe", only_dump_when_change
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
                only_dump_when_change,
                pre_observed_var,
                trace_info[i],
            )
        return result

    return wrapper
