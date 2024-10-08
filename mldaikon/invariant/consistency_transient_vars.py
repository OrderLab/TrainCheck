import re

from mldaikon.invariant.base_cls import (
    APIParam,
    CheckerResult,
    Example,
    ExampleList,
    FailedHypothesis,
    Hypothesis,
    Invariant,
    Relation,
    VarTypeParam,
)

from mldaikon.invariant.precondition import find_precondition
from mldaikon.trace.trace import Trace
from mldaikon.trace.types import (
    FuncCallEvent,
    FuncCallExceptionEvent,
    IncompleteFuncCallEvent,
)

TENSOR_PATTERN = r"torch\..*Tensor"


def filter_functions_with_tensors(all_func_call_events) -> list[str]:
    """
    Filter out the functions that don't have tensors as args or return values.

    Question: some functions return the expected autocast type and thus the return type is dtype instead of tensor, ideally we also want
    to capture those.

    Note: It is assumed that all func call events related to a function will have same input output schema
    (i.e. if tensor showed up in one func call event, it will show up in all func call events of that function)
    """

    funcs_with_tensors: list[str] = []
    for func_name, func_call_ids_and_events in all_func_call_events.items():
        func_has_tensor = False
        for func_call_event in func_call_ids_and_events.values():
            for arg in func_call_event.args:
                assert len(arg) == 1
                arg_type = list(arg.keys())[0]
                if re.match(TENSOR_PATTERN, arg_type):
                    func_has_tensor = True
                    break

            for kwarg_type in func_call_event.kwargs:
                if re.match(TENSOR_PATTERN, kwarg_type):
                    func_has_tensor = True
                    break

            for return_value in func_call_event.return_values:
                if re.match(TENSOR_PATTERN, return_value):
                    func_has_tensor = True
                    break
            if func_has_tensor:
                break
        if func_has_tensor:
            funcs_with_tensors.append(func_name)

    return funcs_with_tensors


def get_returned_tensors(
    func_call_event: FuncCallEvent | FuncCallExceptionEvent | IncompleteFuncCallEvent,
) -> list[dict]:
    """
    Get all the tensors that are returned by the function calls.
    """
    assert not isinstance(
        func_call_event, (FuncCallExceptionEvent, IncompleteFuncCallEvent)
    ), "Exceptions or incomplete function calls don't have return values."

    returned_tensors = []
    for return_type, attributes in func_call_event.return_values.items():
        if re.match(TENSOR_PATTERN, return_type):
            returned_tensors.append(attributes)
    assert (
        len(returned_tensors) > 0
    ), "No tensors found in the return values of the function calls."
    return returned_tensors


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

        relevant_func_call_events = {
            func_name: func_call_ids_and_events
            for func_name, func_call_ids_and_events in all_func_call_events.items()
            if func_name in funcs_with_tensors
        }

        # now, group the function calls by the properties of the input and output tensors

        # now that we have the functions we want to work with, how do we infer the properties of the transient variables?

        # hypothesize over properties being a specific value,

        # consistent ones seem to work well with the invariant hypothesis

        # we are not here to replicate input and shaping constraints, but there might be some interesting properties that we can infer.

        # infer 1: beijie: input / output relationship
        # also, we migth be able to infer the matmul related issues
        # can we replicate the pytea/NeuRI code here?
        pass

        # infer 2: input having specific properties
        # prop-ML-related: norm, max, min, mean, std,
        # prop-Control-related: shape, dtype, etc.
        pass

        # infer 3: output having specific properties
        # prop-ML-related: norm, max, min, mean, std,
        pass
        # how do we infer the properties of the transient variables?
        all_hypotheses = {}
        for func_name in relevant_func_call_events:
            # infer per function
            all_returned_tensors = []
            for func_call_event in relevant_func_call_events[func_name].values():
                # infer per function call
                returned_tensors = get_returned_tensors(func_call_event)
                all_returned_tensors.append((func_call_event, returned_tensors))

            # generate the number of times each property showed up
            properties_occur_num: dict[str, dict[object, int]] = {}
            properties_corresponding_func_call: dict[
                str,
                dict[
                    object,
                    list[
                        FuncCallEvent | FuncCallExceptionEvent | IncompleteFuncCallEvent
                    ],
                ],
            ] = {}
            for func_call_event, returned_tensors in all_returned_tensors:
                for returned_tensor in returned_tensors:
                    for prop, prop_val in returned_tensor.items():
                        if prop not in properties_occur_num:
                            properties_occur_num[prop] = {}
                            properties_corresponding_func_call[prop] = {}
                        if prop_val not in properties_occur_num[prop]:
                            properties_occur_num[prop][prop_val] = 0
                            properties_corresponding_func_call[prop][prop_val] = []
                        properties_occur_num[prop][prop_val] += 1
                        properties_corresponding_func_call[prop][prop_val].append(
                            func_call_event
                        )

            hypotheses_for_func: list[Hypothesis] = []
            # generate a hypothesis for each property
            for prop, prop_values in properties_occur_num.items():
                for prop_val, prop_val_count in prop_values.items():
                    # hypothesis priority can be given based on the number of times the property showed up
                    hypothesis = Hypothesis(
                        invariant=Invariant(
                            relation=ConsistentTransientVarsRelation,
                            params=[
                                APIParam(api_full_name=func_name),
                                VarTypeParam(
                                    var_type="torch.Tensor",
                                    attr_name=prop,
                                    const_value=prop_val,
                                ),
                            ],
                            precondition=None,
                            text_description=f"{prop} of the tensors returned by the function {func_name} is consistently {prop_val}.",
                        ),
                        positive_examples=ExampleList({"pre_event"}),
                        negative_examples=ExampleList({"pre_event"}),
                    )

                    # let's add positive and negative examples
                    for func_call_event in properties_corresponding_func_call[prop][
                        prop_val
                    ]:
                        example = Example({"pre_event": [func_call_event.pre_record]})
                        hypothesis.positive_examples.add_example(example)

                    for prop_val_other, prop_val_count_other in prop_values.items():
                        if prop_val_other == prop_val:
                            continue
                        for func_call_event in properties_corresponding_func_call[prop][
                            prop_val_other
                        ]:
                            example = Example(
                                {"pre_event": [func_call_event.pre_record]}
                            )
                            hypothesis.negative_examples.add_example(example)

                    hypotheses_for_func.append(hypothesis)

            all_hypotheses[func_name] = hypotheses_for_func
        # positive example is the function calls corresponding to the hypothesis
        # negative example is the function calls that do not correspond to the hypothesis (i.e. func calls that returned different values)

        # now that we have the hypotheses for each function, we can return them
        # we can also return the failed hypotheses if any
        # we can also return the properties that are consistent across the function calls

        # infer precondition for these hypotheses
        print(all_hypotheses)

        for func_name, hypotheses in all_hypotheses.items():
            for hypothesis in hypotheses:
                precondition = find_precondition(hypothesis)
                print(precondition)

        print("done")

        # precondition inference: function name can be a precondition

        # can we let relation tell the precondition inference algorithm about what is already assumed?
        # then we solve the step issue.

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

    @staticmethod
    def evaluate(value_group: list) -> bool:
        raise NotImplementedError

    @staticmethod
    def static_check_all(
        trace: Trace, inv: Invariant, check_relation_first: bool
    ) -> CheckerResult:
        raise NotImplementedError
