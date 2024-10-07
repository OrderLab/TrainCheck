import re

from mldaikon.invariant.base_cls import FailedHypothesis, Invariant, Relation
from mldaikon.trace.trace import Trace


def filter_functions_with_tensors(all_func_call_events) -> list[str]:
    """
    Filter out the functions that don't have tensors as args or return values.

    Question: some functions return the expected autocast type and thus the return type is dtype instead of tensor, ideally we also want
    to capture those.

    Note: It is assumed that all func call events related to a function will have same input output schema
    (i.e. if tensor showed up in one func call event, it will show up in all func call events of that function)
    """
    tensor_pattern = r"torch\..*Tensor"
    funcs_with_tensors: list[str] = []
    for func_name, func_call_ids_and_events in all_func_call_events.items():
        func_has_tensor = False
        for func_call_event in func_call_ids_and_events.values():
            for arg in func_call_event.args:
                assert len(arg) == 1
                arg_type = list(arg.keys())[0]
                if re.match(tensor_pattern, arg_type):
                    func_has_tensor = True
                    break

            for kwarg_type in func_call_event.kwargs:
                if re.match(tensor_pattern, kwarg_type):
                    func_has_tensor = True
                    break

            for return_value in func_call_event.return_values:
                if re.match(tensor_pattern, return_value):
                    func_has_tensor = True
                    break
            if func_has_tensor:
                break
        if func_has_tensor:
            funcs_with_tensors.append(func_name)

    return funcs_with_tensors


class ConsistentTransientVarsRelation(Relation):
    """Infer common properties of transient variables that are consistent across function calls.

    For example, if you have a function that is called multiple times, and the function args and return values
    """

    @staticmethod
    def infer(trace: Trace) -> tuple[list[Invariant], list[FailedHypothesis]]:

        all_func_names = trace.get_func_names()
        all_func_call_ids = {
            func_name: trace.get_func_call_ids(func_name)
            for func_name in all_func_names
        }
        all_func_call_events = {
            func_name: {
                func_call_id: trace.query_func_call_event(func_call_id)
                for func_call_id in func_call_ids
            }
            for func_name, func_call_ids in all_func_call_ids.items()
        }

        funcs_with_tensors = filter_functions_with_tensors(all_func_call_events)
        print(funcs_with_tensors)

        return [], []

        # now let's reason about the input and output properties of these function calls' args and return values

        # let's make the assumption that we are only interested in the functions that have tensors as args or return values

        # functions that do not have tensors as input but have them as output --> factory functions
        # functions that have tensors as both input and output --> mathematical operations

        # we need an abstraction over the trace to get a specific function call's args and return values
        # for now let's get them from the raw json trace

        # get all the function calls.

        # find properties inside.

        # instead of doing this, will it just make more sense to differentiate between stages and only infer in one single step if
        # we are in training or testing stage?

        # Get all the function args and return values, and try to find the properties that
        # are consistent across the function calls.

        # 1. group by properties if some of them show great statistical consistency

        # Question: are u only targeting function args and return values? What about consistency relationships between transient
        # variables and the long-term variables (e.g. model, optimizers?)

        # Insight: internal, transient variables are kinda separate from long-term variables like the model and optimizer.
        # I think it will be fine to treat them separately.

        # the simplest case: only matmul is called multiple times and you have them both inside and outside the autocast regions

        # need additional properties about these functions in precondition inference
