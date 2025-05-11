import logging
import re
from typing import Hashable

import pandas as pd
from tqdm import tqdm

from traincheck.invariant.base_cls import (
    APIParam,
    Arguments,
    CheckerResult,
    Example,
    ExampleList,
    FailedHypothesis,
    Hypothesis,
    InputOutputParam,
    Invariant,
    Relation,
    VarTypeParam,
    make_hashable,
)
from traincheck.invariant.precondition import find_precondition
from traincheck.trace.trace import Trace
from traincheck.trace.types import (
    FuncCallEvent,
    FuncCallExceptionEvent,
    IncompleteFuncCallEvent,
)
from traincheck.utils import safe_isnan

TENSOR_PATTERN = r"torch\..*Tensor"
PARAMETER_KEYWORD = "Parameter"
ATTR_SKIP = "_ML_DAIKON_data_ID"

# _CACHE_PATH = "func_with_tensors.pkl"


def safe_equality(obj1: object, obj2: object) -> bool:
    """
    Check if two objects are equal, handling NaN values.
    """
    if safe_isnan(obj1) and safe_isnan(obj2):
        return True

    if safe_isnan(obj1) or safe_isnan(obj2):
        return False

    return obj1 == obj2


# for each value observed, form a dict of {value: path to access the value}
def _get_tensor_value_paths(tensors: list[dict]) -> dict:
    logger = logging.getLogger(__name__)

    values: dict[Hashable, list[list[str | int]]] = {}
    for idx, tensor in enumerate(tensors):
        for prop, prop_val in tensor.items():
            if isinstance(prop_val, (list, tuple)):
                # mainly for shape and ndims related fields
                for val_idx, val in enumerate(prop_val):
                    if not isinstance(val, Hashable):
                        # we can't hash unhashable types
                        logger.warning(
                            f"Unhashable prop {prop}: {str(prop_val)} found in the input tensor, skipping the value."
                        )
                        continue
                    if val not in values:
                        values[val] = []

                    values[val].append([idx, prop, val_idx])

            else:
                if not isinstance(prop_val, Hashable):
                    # we can't hash unhashable types
                    logger.warning(
                        f"Unhashable prop {prop}: {str(prop_val)} found in the input tensor, skipping the value."
                    )
                    continue
                if prop_val not in values:
                    values[prop_val] = []

                values[prop_val].append([idx, prop])
    return values


def filter_functions_with_tensors(
    all_func_call_events, output_has_tensors: bool, input_has_tensors: bool
) -> list[str]:
    """
    Filter out the functions that don't have tensors as args or return values.

    Question: some functions return the expected autocast type and thus the return type is dtype instead of tensor, ideally we also want
    to capture those.

    Note: It is assumed that all func call events related to a function will have same input output schema
    (i.e. if tensor showed up in one func call event, it will show up in all func call events of that function)
    """
    # if os.path.exists(_CACHE_PATH):
    #     with open(_CACHE_PATH, "rb") as f:
    #         return pickle.load(f)

    funcs_with_tensors: list[str] = []
    for func_name, func_call_ids_and_events in all_func_call_events.items():
        func_satisfy_requirement = False
        func_has_input_tensor = False
        func_has_output_tensor = False
        for func_call_event in func_call_ids_and_events.values():
            for arg in func_call_event.args.values():  # TODO: handle new format
                assert len(arg) == 1
                arg_type = list(arg.keys())[0]
                if re.match(TENSOR_PATTERN, arg_type) or PARAMETER_KEYWORD in arg_type:
                    func_has_input_tensor = True
                    break

            for kwarg_type in func_call_event.kwargs:
                if (
                    re.match(TENSOR_PATTERN, kwarg_type)
                    or PARAMETER_KEYWORD in kwarg_type
                ):
                    func_has_input_tensor = True
                    break

            if isinstance(func_call_event, (FuncCallEvent)):
                return_values = func_call_event.return_values
                if isinstance(return_values, dict):
                    return_values = [return_values]
                if safe_isnan(return_values):
                    return_values = []
                for return_value in return_values:
                    type_value = list(return_value.keys())[0]
                    if (
                        re.match(TENSOR_PATTERN, type_value)
                        or PARAMETER_KEYWORD in type_value
                    ):
                        func_has_output_tensor = True
                        break
            if func_has_input_tensor and input_has_tensors and not output_has_tensors:
                func_satisfy_requirement = True
                break
            if func_has_output_tensor and output_has_tensors and not input_has_tensors:
                func_satisfy_requirement = True
                break
            if (
                func_has_input_tensor
                and func_has_output_tensor
                and output_has_tensors
                and input_has_tensors
            ):
                func_satisfy_requirement = True
                break
        if func_satisfy_requirement:
            funcs_with_tensors.append(func_name)

    # with open(_CACHE_PATH, "wb") as f:
    #     pickle.dump(funcs_with_tensors, f)
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
    # dict[str, dict[str, object]] | list[dict[str, dict[str, object]]]
    returned_tensors = []
    return_values = func_call_event.return_values
    if isinstance(return_values, dict):
        return_values = [return_values]
    if safe_isnan(return_values):
        return []
    for return_value in return_values:
        type_value = list(return_value.keys())[0]
        attributes = return_value[type_value]
        if re.match(TENSOR_PATTERN, type_value) or PARAMETER_KEYWORD in type_value:
            # let's pop the ATTR_SKIP attribute
            if ATTR_SKIP in attributes:
                attributes.pop(ATTR_SKIP)
            returned_tensors.append(attributes)
    return returned_tensors


def get_input_tensors(
    func_call_event: FuncCallEvent | FuncCallExceptionEvent | IncompleteFuncCallEvent,
) -> list[dict]:
    """
    Get all the input tensors that are passed to the function calls.
    """
    input_tensors = []
    for arg in func_call_event.args.values():
        assert len(arg) == 1
        arg_type = list(arg.keys())[0]
        if re.match(TENSOR_PATTERN, arg_type) or PARAMETER_KEYWORD in arg_type:
            input_tensors.append(arg[arg_type])
    # ["type":value]
    for kwarg in func_call_event.kwargs.values():
        assert len(kwarg) == 1
        kwarg_type = list(kwarg.keys())[0]
        if re.match(TENSOR_PATTERN, kwarg_type) or PARAMETER_KEYWORD in kwarg_type:
            input_tensors.append(kwarg[kwarg_type])
    # {name:type:value}
    return input_tensors


def get_input_thresholds(
    func_call_event: FuncCallEvent | FuncCallExceptionEvent | IncompleteFuncCallEvent,
) -> tuple[list[dict], list[dict]]:
    """
    Get all the input thresholds that are passed to the function calls.

    Output: [{arg_name: threshold_value}]  TODO: theoretically, we should be able to parse arbitrary threshold values in nested structures, here we assume
    that the threshold values are primitive types passed as direct arguments to the function calls, but this might not be true for GenerationConfig
    """
    min_thresholds = []
    max_thresholds = []

    arguments = Arguments(
        func_call_event.args,
        func_call_event.kwargs,
        func_call_event.func_name,
        consider_default_values=True,
    )

    for arg_name, arg_type_and_value in arguments.arguments.items():
        arg_value = list(arg_type_and_value.values())[0]
        if not isinstance(arg_value, (int, float)):
            continue
        if "min" in arg_name:
            min_thresholds.append({arg_name: arg_value})
        if "max" in arg_name:
            max_thresholds.append({arg_name: arg_value})

    return min_thresholds, max_thresholds


def get_events_of_funcs_with_tensors(
    all_func_names, trace, output_has_tensors=True, input_has_tensors=True
):
    # HACK: remove all torch.overrides
    all_func_names = [
        func_name for func_name in all_func_names if "torch.override" not in func_name
    ]
    # remove all functions with "._" in them
    all_func_names = [
        func_name for func_name in all_func_names if "._" not in func_name
    ]
    # remove all functions with "._is_" in them
    all_func_names = [
        func_name for func_name in all_func_names if ".is_" not in func_name
    ]

    # if os.path.exists(_CACHE_PATH):
    #     with open(_CACHE_PATH, "rb") as f:
    #         all_func_names = pickle.load(f)

    all_func_call_ids = {
        func_name: trace.get_func_call_ids(func_name) for func_name in all_func_names
    }

    # sampling 1000 if more than 1000
    import random

    all_func_call_ids = {
        func_name: (
            random.sample(func_call_ids, 1000)
            if len(func_call_ids) > 1000
            else func_call_ids
        )
        for func_name, func_call_ids in all_func_call_ids.items()
    }

    all_func_call_events = {
        func_name: {
            func_call_id: trace.query_func_call_event(func_call_id)
            for func_call_id in tqdm(func_call_ids, desc=f"Querying {func_name} events")
        }
        for func_name, func_call_ids in all_func_call_ids.items()  # PROBABLY THERE'S SOMETHING WE CAN DO VIA STATIC ANALYSIS
    }

    funcs_with_tensors = filter_functions_with_tensors(
        all_func_call_events, output_has_tensors, input_has_tensors
    )

    relevant_func_call_events = {
        func_name: func_call_ids_and_events
        for func_name, func_call_ids_and_events in all_func_call_events.items()
        if func_name in funcs_with_tensors
    }

    return relevant_func_call_events


class ConsistentOutputRelation(Relation):
    """Infer common properties of transient variables that are consistent across function calls.

    For example, if you have a function that is called multiple times, and the function args and return values
    """

    @staticmethod
    def generate_hypothesis(trace) -> list[Hypothesis]:
        logger = logging.getLogger(__name__)

        all_func_names = trace.get_func_names()
        relevant_func_call_events = get_events_of_funcs_with_tensors(
            all_func_names, trace, output_has_tensors=True, input_has_tensors=False
        )

        all_hypotheses = {}
        for func_name in tqdm(relevant_func_call_events, desc="Inferring hypotheses"):
            # infer per function
            all_returned_tensors: list[tuple[FuncCallEvent, list[dict]]] = []
            skip_func = False
            for func_call_event in relevant_func_call_events[func_name].values():
                # infer per function call
                returned_tensors = get_returned_tensors(func_call_event)
                if len(returned_tensors) == 0:
                    logger.warning(
                        f"No tensors found in the return values of the function call {func_call_event}, previously number of tensors found: {[len(t[1]) for t in all_returned_tensors]}"
                    )
                    skip_func = True
                    break
                assert isinstance(func_call_event, FuncCallEvent)
                all_returned_tensors.append((func_call_event, returned_tensors))
            if skip_func:
                continue

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
                        if not isinstance(prop_val, Hashable):
                            # make it hashable
                            prop_val = make_hashable(prop_val)
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

                    for prop in properties_occur_num:
                        if prop not in returned_tensor:
                            if pd.NA not in properties_occur_num[prop]:
                                properties_occur_num[prop][pd.NA] = 0
                                properties_corresponding_func_call[prop][pd.NA] = []
                            properties_occur_num[prop][pd.NA] += 1
                            properties_corresponding_func_call[prop][pd.NA].append(
                                func_call_event
                            )

            hypotheses_for_func: list[Hypothesis] = []
            # generate a hypothesis for each property
            for prop, prop_values in properties_occur_num.items():
                for prop_val, _ in prop_values.items():
                    if not isinstance(prop_val, Hashable):
                        # make it hashable
                        prop_val = make_hashable(prop_val)
                    # hypothesis priority can be given based on the number of times the property showed up
                    hypothesis = Hypothesis(
                        invariant=Invariant(
                            relation=ConsistentOutputRelation,
                            params=[
                                APIParam(api_full_name=func_name),
                                VarTypeParam(
                                    var_type="torch.Tensor",
                                    attr_name=prop,
                                    const_value=make_hashable(prop_val),
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
                        try:
                            if safe_equality(prop_val, prop_val_other):
                                continue
                        except TypeError:
                            print(
                                f"TypeError: {prop_val} {safe_isnan(prop_val)} {type(prop_val)} and {prop_val_other} {safe_isnan(prop_val_other)} are not comparable, skipping this property."
                            )
                            raise
                        for func_call_event in properties_corresponding_func_call[prop][
                            prop_val_other
                        ]:
                            example = Example(
                                {"pre_event": [func_call_event.pre_record]}
                            )
                            hypothesis.negative_examples.add_example(example)

                    hypothesis.invariant.num_positive_examples = len(
                        hypothesis.positive_examples
                    )
                    hypothesis.invariant.num_negative_examples = len(
                        hypothesis.negative_examples
                    )

                    hypotheses_for_func.append(hypothesis)

            all_hypotheses[func_name] = hypotheses_for_func
        # return all_hypotheses
        return sum(all_hypotheses.values(), [])

    @staticmethod
    def collect_examples(trace, hypothesis):
        inv = hypothesis.invariant
        # get all the function calls
        assert len(inv.params) == 2
        assert isinstance(inv.params[0], APIParam)
        assert isinstance(inv.params[1], VarTypeParam)

        func_name = inv.params[0].api_full_name
        # get all the function calls for the function
        func_call_ids = trace.get_func_call_ids(func_name)

        # down sample to 1000
        import random

        if len(func_call_ids) > 1000:
            func_call_ids = random.sample(func_call_ids, 1000)

        for func_call_id in tqdm(
            func_call_ids, desc=f"Adding examples for {inv.text_description}"
        ):
            func_call_event = trace.query_func_call_event(func_call_id)
            if isinstance(
                func_call_event, (FuncCallExceptionEvent, IncompleteFuncCallEvent)
            ):
                continue

            returned_tensors = get_returned_tensors(func_call_event)
            if len(returned_tensors) == 0:
                # add negative example
                example = Example({"pre_event": [func_call_event.pre_record]})
                hypothesis.negative_examples.add_example(example)
                continue

            # TODO: this might be wrong due to make hashable used in infer, proceed with caution
            for returned_tensor in returned_tensors:
                prop = inv.params[1].attr_name
                prop_val = inv.params[1].const_value
                if (
                    prop not in returned_tensor
                    or make_hashable(returned_tensor[prop]) != prop_val
                ):
                    # add negative example
                    example = Example({"pre_event": [func_call_event.pre_record]})
                    hypothesis.negative_examples.add_example(example)
                else:
                    # add positive example
                    example = Example({"pre_event": [func_call_event.pre_record]})
                    hypothesis.positive_examples.add_example(example)

        hypothesis.invariant.num_positive_examples = len(hypothesis.positive_examples)
        hypothesis.invariant.num_negative_examples = len(hypothesis.negative_examples)

    @staticmethod
    def infer(trace: Trace) -> tuple[list[Invariant], list[FailedHypothesis]]:

        all_hypotheses = ConsistentOutputRelation.generate_hypothesis(trace)

        invariants = []
        failed_hypotheses = []
        for hypothesis in all_hypotheses:
            precondition = find_precondition(hypothesis, [trace])
            print(precondition)
            if precondition is not None:
                hypothesis.invariant.precondition = precondition
                invariants.append(hypothesis.invariant)
            else:
                print(f"Could not find precondition for {hypothesis}")
                failed_hypotheses.append(
                    FailedHypothesis(hypothesis, "Precondition not found")
                )

        print("done")

        return invariants, failed_hypotheses

    @staticmethod
    def evaluate(value_group: list) -> bool:
        raise NotImplementedError

    @staticmethod
    def static_check_all(
        trace: Trace, inv: Invariant, check_relation_first: bool
    ) -> CheckerResult:
        # let's make all the invs specific to API output properties now

        assert inv.precondition is not None, "The precondition should not be None."

        # get all the function calls
        assert len(inv.params) == 2
        assert isinstance(inv.params[0], APIParam)
        assert isinstance(inv.params[1], VarTypeParam)

        func_name = inv.params[0].api_full_name
        # get all the function calls for the function
        func_call_ids = trace.get_func_call_ids(func_name)

        triggered = False
        # for each function call, check if the property holds
        for func_call_id in tqdm(
            func_call_ids, desc=f"Checking invariant {inv.text_description}"
        ):
            func_call_event = trace.query_func_call_event(func_call_id)
            if isinstance(
                func_call_event, (FuncCallExceptionEvent, IncompleteFuncCallEvent)
            ):
                continue

            # check for precondition here
            if not inv.precondition.verify(
                [func_call_event.pre_record], "pre_event", trace
            ):
                continue

            triggered = True

            returned_tensors = get_returned_tensors(func_call_event)
            if len(returned_tensors) == 0:
                return CheckerResult(
                    trace=[func_call_event.pre_record],
                    invariant=inv,
                    check_passed=False,
                    triggered=True,
                )

            # TODO: this might be wrong due to make hashable used in infer, proceed with caution
            for returned_tensor in returned_tensors:
                prop = inv.params[1].attr_name
                prop_val = inv.params[1].const_value
                if (
                    prop not in returned_tensor
                    or make_hashable(returned_tensor[prop]) != prop_val
                ):
                    return CheckerResult(
                        trace=[func_call_event.pre_record],
                        invariant=inv,
                        check_passed=False,
                        triggered=True,
                    )

        return CheckerResult(
            trace=None,
            invariant=inv,
            check_passed=True,
            triggered=triggered,
        )
        # raise NotImplementedError

    @staticmethod
    def get_precondition_infer_keys_to_skip(hypothesis: Hypothesis) -> list[str]:
        return []


class ConsistentInputOutputRelation(Relation):
    """Infer common properties that should be enforced across the input and output of a function call.

    This relation is mainly implemented to support constraints supported/inferred by PyTea (ICSE'22) and NeuRI (FSE'23)
    """

    @staticmethod
    def generate_hypothesis(trace: Trace) -> list[Hypothesis]:
        logger = logging.getLogger(__name__)

        all_func_names = trace.get_func_names()
        relevant_func_call_events = get_events_of_funcs_with_tensors(
            all_func_names, trace, output_has_tensors=True, input_has_tensors=True
        )

        # for these func_call_events, we obtain the properties that are consistent across the input and output tensors
        # we can then generate hypotheses for these properties

        all_hypotheses: dict[
            str, dict[tuple[InputOutputParam, InputOutputParam], Hypothesis]
        ] = {}

        for func_name in tqdm(
            relevant_func_call_events,
            desc="Infer hypotheses for consistent input output relation on functions",
        ):
            logger.info(f"Infer hypotheses for {func_name}")
            # TBD: test matmul input / output shape constraints though we did not instrument matmul
            api_param = APIParam(api_full_name=func_name)
            for func_event in tqdm(
                relevant_func_call_events[func_name].values(),
                desc=f"Infer hypotheses for {func_name}",
            ):
                # try to form hypothesis for each function call

                # get the input and output tensors of the function call
                input_tensors = get_input_tensors(func_event)
                # potentially we need to attach the tensors to their signature in the function definition, for now let's assume that the tensors in the same order as they are defined and use index to access them
                output_tensors = get_returned_tensors(func_event)

                input_values_paths = _get_tensor_value_paths(input_tensors)
                output_values_paths = _get_tensor_value_paths(output_tensors)

                # find the common values between the input and output tensors and form hypotheses for them
                input_values = set(input_values_paths.keys())
                output_values = set(output_values_paths.keys())

                common_values = input_values.intersection(output_values)

                for common_value in common_values:
                    input_paths = input_values_paths[common_value]
                    output_paths = output_values_paths[common_value]
                    combinations_of_paths = [
                        (input_path, output_path)
                        for input_path in input_paths
                        for output_path in output_paths
                    ]
                    if isinstance(common_value, bool):
                        # hack: for flags, we keep it simple: only values with the same prop name can be considered consistent
                        # e.g. "True" consistency across input and output tensors will only be about "requires_grad" v.s. "requires_grad", not "requires_grad" v.s. "is_cuda"
                        combinations_of_paths = [
                            (input_path, output_path)
                            for input_path, output_path in combinations_of_paths
                            if input_path == output_path
                        ]

                    # add hypothesis for each common value, combine the paths to access the value in the input and output tensors
                    for input_path, output_path in combinations_of_paths:
                        input_param = InputOutputParam(
                            name="input_tensors",
                            index=input_path[0],
                            type="torch.Tensor",
                            additional_path=tuple(input_path[1:]),
                            api_name=func_name,
                            is_input=True,
                        )

                        output_param = InputOutputParam(
                            name="output_tensors",
                            index=output_path[0],
                            type="torch.Tensor",
                            additional_path=tuple(output_path[1:]),
                            api_name=func_name,
                            is_input=False,
                        )

                        if func_name not in all_hypotheses:
                            all_hypotheses[func_name] = {}

                        if (input_param, output_param) in all_hypotheses[func_name]:
                            continue

                        hypothesis = Hypothesis(
                            invariant=Invariant(
                                relation=ConsistentInputOutputRelation,
                                params=[input_param, api_param, output_param],
                                precondition=None,
                                text_description=f"The value {common_value} is consistent across the input {input_path} and output {output_path} tensors of the function {func_name}.",
                            ),
                            positive_examples=ExampleList(
                                {"pre_event"}
                            ),  # q: do we need input attributes as precondition here? probably not
                            negative_examples=ExampleList(
                                {"pre_event"}
                            ),  # q: do we need input attributes as precondition here? probably not
                        )

                        all_hypotheses[func_name][
                            (input_param, output_param)
                        ] = hypothesis

            # if no hypothesis is formed for the function, we skip it
            if func_name not in all_hypotheses:
                continue

            # now, scan for positive and negative examples
            for func_event in relevant_func_call_events[func_name].values():
                input_tensors = get_input_tensors(func_event)
                output_tensors = get_returned_tensors(func_event)

                # for each hypothesis, check if it holds for this function call
                for (input_param, output_param), hypothesis in all_hypotheses[
                    func_name
                ].items():
                    # FIXME: sometimes there's no matching value in the input and output tensors, we need to handle this case, for now we skip it
                    try:
                        input_value = input_param.get_value_from_list_of_tensors(
                            input_tensors
                        )
                        output_value = output_param.get_value_from_list_of_tensors(
                            output_tensors
                        )
                    except (IndexError, KeyError):
                        logger.warning(
                            f"Could not find the matching value in the input and output tensors for the hypothesis {hypothesis}, skipping this function call."
                        )
                        continue
                    if input_value == output_value:
                        example = Example({"pre_event": [func_event.pre_record]})
                        hypothesis.positive_examples.add_example(example)
                    else:
                        example = Example({"pre_event": [func_event.pre_record]})
                        hypothesis.negative_examples.add_example(example)

            # now, update the number of positive and negative examples
            for hypothesis in all_hypotheses[func_name].values():
                hypothesis.invariant.num_positive_examples = len(
                    hypothesis.positive_examples
                )
                hypothesis.invariant.num_negative_examples = len(
                    hypothesis.negative_examples
                )

        hypos_to_return: list[Hypothesis] = []
        for hypotheses in all_hypotheses.values():
            hypos_to_return.extend(hypotheses.values())

        return hypos_to_return

    @staticmethod
    def collect_examples(trace, hypothesis):
        inv = hypothesis.invariant

        assert len(inv.params) == 3

        input_param, api_param, output_param = inv.params

        assert isinstance(input_param, InputOutputParam)
        assert isinstance(api_param, APIParam)
        assert isinstance(output_param, InputOutputParam)
        assert inv.params[0].is_input
        assert not inv.params[2].is_input

        logger = logging.getLogger(__name__)

        # get all the function calls
        func_name = api_param.api_full_name
        func_call_ids = trace.get_func_call_ids(func_name)

        import random

        if len(func_call_ids) > 1000:
            func_call_ids = random.sample(func_call_ids, 1000)

        for func_call_id in tqdm(
            func_call_ids, desc=f"Checking invariant {inv.text_description}"
        ):
            func_call_event = trace.query_func_call_event(func_call_id)
            if isinstance(
                func_call_event, (FuncCallExceptionEvent, IncompleteFuncCallEvent)
            ):
                continue

            input_tensors = get_input_tensors(func_call_event)
            output_tensors = get_returned_tensors(func_call_event)
            try:
                input_value = input_param.get_value_from_list_of_tensors(input_tensors)
                output_value = output_param.get_value_from_list_of_tensors(
                    output_tensors
                )
            except (IndexError, KeyError):
                logger.warning(
                    f"Could not find the value to be used in input or output tensors for the hypothesis {inv}, skipping this function call."
                )
                continue

            if input_value != output_value:
                # add negative example
                example = Example({"pre_event": [func_call_event.pre_record]})
                hypothesis.negative_examples.add_example(example)
            else:
                # add positive example
                example = Example({"pre_event": [func_call_event.pre_record]})
                hypothesis.positive_examples.add_example(example)

    @staticmethod
    def infer(trace: Trace) -> tuple[list[Invariant], list[FailedHypothesis]]:
        all_hypotheses = ConsistentInputOutputRelation.generate_hypothesis(trace)

        # now that we have the hypotheses for each function, we can start precondition inference
        invariants = []
        failed_hypotheses = []
        for hypothesis in all_hypotheses:
            precondition = find_precondition(hypothesis, [trace])
            if precondition is not None:
                hypothesis.invariant.precondition = precondition
                invariants.append(hypothesis.invariant)
            else:
                failed_hypotheses.append(
                    FailedHypothesis(hypothesis, "Precondition not found")
                )

        return invariants, failed_hypotheses

    @staticmethod
    def evaluate(value_group: list) -> bool:
        raise NotImplementedError

    @staticmethod
    def static_check_all(trace, inv, check_relation_first):

        assert inv.precondition is not None, "The precondition should not be None."
        assert len(inv.params) == 3

        input_param, api_param, output_param = inv.params

        assert isinstance(input_param, InputOutputParam)
        assert isinstance(api_param, APIParam)
        assert isinstance(output_param, InputOutputParam)
        assert inv.params[0].is_input
        assert not inv.params[2].is_input

        logger = logging.getLogger(__name__)

        # get all the function calls
        func_name = api_param.api_full_name
        func_call_ids = trace.get_func_call_ids(func_name)
        triggered = False

        for func_call_id in tqdm(
            func_call_ids, desc=f"Checking invariant {inv.text_description}"
        ):
            func_call_event = trace.query_func_call_event(func_call_id)
            if isinstance(
                func_call_event, (FuncCallExceptionEvent, IncompleteFuncCallEvent)
            ):
                continue

            # check for precondition here
            if not inv.precondition.verify(
                [func_call_event.pre_record], "pre_event", trace
            ):
                continue

            input_tensors = get_input_tensors(func_call_event)
            output_tensors = get_returned_tensors(func_call_event)
            try:
                input_value = input_param.get_value_from_list_of_tensors(input_tensors)
                output_value = output_param.get_value_from_list_of_tensors(
                    output_tensors
                )
            except (IndexError, KeyError):
                logger.warning(
                    f"Could not find the value to be checked in input or output tensors for the hypothesis {inv}, skipping this function call."
                )
                continue

            triggered = True
            if input_value != output_value:
                return CheckerResult(
                    trace=[func_call_event.pre_record],
                    invariant=inv,
                    check_passed=False,
                    triggered=True,
                )

        return CheckerResult(
            trace=None,
            invariant=inv,
            check_passed=True,
            triggered=triggered,
        )

    @staticmethod
    def get_precondition_infer_keys_to_skip(hypothesis: Hypothesis) -> list[str]:
        return []


class ThresholdRelation(Relation):
    """Infer common properties that should be enforced across the input and output of a function call.

    This relation is mainly implemented to support constraints supported/inferred by PyTea (ICSE'22) and NeuRI (FSE'23)
    """

    @staticmethod
    def generate_hypothesis(trace: Trace) -> list[Hypothesis]:
        # get the function calls that have tensors or nn.Modules as both input and output
        logger = logging.getLogger(__name__)
        all_func_names = trace.get_func_names()
        relevant_func_call_events = get_events_of_funcs_with_tensors(
            all_func_names, trace, output_has_tensors=True, input_has_tensors=False
        )
        max_hypotheses: dict[
            str, dict[tuple[InputOutputParam, InputOutputParam], Hypothesis]
        ] = {}
        min_hypotheses: dict[
            str, dict[tuple[InputOutputParam, InputOutputParam], Hypothesis]
        ] = {}

        for func_name in tqdm(
            relevant_func_call_events,
            desc="Infer hypotheses for threshold relation on functions",
        ):
            logger.info(f"Infer hypotheses for {func_name}")
            api_param = APIParam(api_full_name=func_name)
            for func_event in tqdm(
                relevant_func_call_events[func_name].values(),
                desc=f"Infer hypotheses for {func_name}",
            ):
                min_thresholds, max_thresholds = get_input_thresholds(func_event)
                output_tensors = get_returned_tensors(func_event)

                output_values_paths = _get_tensor_value_paths(
                    output_tensors
                )  # {value:path}
                for (
                    output_value,
                    output_paths,
                ) in output_values_paths.items():  # {value:path}
                    if not isinstance(output_value, (int, float)):
                        continue
                    for output_path in output_paths:
                        assert not isinstance(
                            output_path[0], (list, tuple)
                        ), f"Index should be a single value, got {output_path}"
                        output_param = InputOutputParam(
                            name="output_tensors",
                            index=output_path[0],
                            type="torch.Tensor",
                            additional_path=tuple(output_path[1:]),
                            api_name=func_name,
                            is_input=False,
                        )

                        # try form min threshold hypotheses
                        for min_threshold in min_thresholds:
                            threshold_name, threshold_value = list(
                                min_threshold.items()
                            )[0]
                            input_param = InputOutputParam(
                                name=threshold_name,
                                index=None,
                                type=str(type(threshold_value)),
                                additional_path=None,
                                api_name=func_name,
                                is_input=True,
                            )
                            if (
                                func_name in min_hypotheses
                                and (input_param, output_param)
                                in min_hypotheses[func_name]
                            ):
                                continue

                            if (
                                output_value >= threshold_value
                                and (output_value - threshold_value) / threshold_value
                                <= 0.2
                            ):

                                if func_name not in min_hypotheses:
                                    min_hypotheses[func_name] = {}

                                logger.info(
                                    f"Forming hypothesis for {func_name}: output param {output_param} is bounded below by min threshold {input_param}"
                                )
                                min_hypotheses[func_name][
                                    (input_param, output_param)
                                ] = Hypothesis(
                                    invariant=Invariant(
                                        relation=ThresholdRelation,
                                        params=[
                                            output_param,
                                            api_param,
                                            input_param,
                                        ],  # the first param should be larger or equal to the second param
                                        precondition=None,
                                        text_description=f"Output tensor's value at {output_param.additional_path} is consistently larger than or equal to the min input threshold {input_param.name} for the function {func_name}.",
                                    ),
                                    positive_examples=ExampleList(
                                        {"pre_event"}
                                    ),  # q: do we need input attributes as precondition here? probably not
                                    negative_examples=ExampleList(
                                        {"pre_event"}
                                    ),  # q: do we need input attributes as precondition here? probably not
                                )

                        # try form max threshold hypotheses
                        for max_threshold in max_thresholds:
                            threshold_name, threshold_value = list(
                                max_threshold.items()
                            )[0]
                            input_param = InputOutputParam(
                                name=threshold_name,
                                index=None,
                                type=str(type(threshold_value)),
                                additional_path=None,
                                api_name=func_name,
                                is_input=True,
                            )
                            if (
                                func_name in max_hypotheses
                                and (input_param, output_param)
                                in max_hypotheses[func_name]
                            ):
                                continue

                            if (
                                output_value <= threshold_value
                                and (threshold_value - output_value) / threshold_value
                                <= 0.2
                            ):

                                if func_name not in max_hypotheses:
                                    max_hypotheses[func_name] = {}

                                logger.info(
                                    f"Forming hypothesis for {func_name}: output param {output_param} is bounded above by max threshold {input_param}"
                                )
                                max_hypotheses[func_name][
                                    (input_param, output_param)
                                ] = Hypothesis(
                                    invariant=Invariant(
                                        relation=ThresholdRelation,
                                        params=[
                                            input_param,
                                            api_param,
                                            output_param,
                                        ],  # the first param should be larger or equal to the second param
                                        precondition=None,
                                        text_description=f"Output tensor's value at {output_param.additional_path} is consistently less than or equal to the max input threshold {input_param.name} for the function {func_name}.",
                                    ),
                                    positive_examples=ExampleList(
                                        {"pre_event"}
                                    ),  # q: do we need input attributes as precondition here? probably not
                                    negative_examples=ExampleList(
                                        {"pre_event"}
                                    ),  # q: do we need input attributes as precondition here? probably not
                                )

            # now, scan for positive and negative examples
            for func_event in relevant_func_call_events[func_name].values():
                min_thresholds, max_thresholds = get_input_thresholds(func_event)
                output_tensors = get_returned_tensors(func_event)
                argument = Arguments(
                    func_event.args,
                    func_event.kwargs,
                    func_event.func_name,
                    consider_default_values=True,
                )
                output_tensors = get_returned_tensors(func_event)
                if func_name in min_hypotheses:
                    for (input_param, output_param), hypothesis in min_hypotheses[
                        func_name
                    ].items():
                        min_threshold = input_param.get_value_from_arguments(argument)
                        output_value = output_param.get_value_from_list_of_tensors(
                            output_tensors
                        )

                        example = Example({"pre_event": [func_event.pre_record]})
                        if output_value >= min_threshold:
                            hypothesis.positive_examples.add_example(example)
                        else:
                            hypothesis.negative_examples.add_example(example)

                if func_name in max_hypotheses:
                    for (input_param, output_param), hypothesis in max_hypotheses[
                        func_name
                    ].items():
                        print(argument.arguments)
                        max_threshold = input_param.get_value_from_arguments(argument)
                        output_value = output_param.get_value_from_list_of_tensors(
                            output_tensors
                        )

                        example = Example({"pre_event": [func_event.pre_record]})
                        if output_value <= max_threshold:
                            hypothesis.positive_examples.add_example(example)
                        else:
                            hypothesis.negative_examples.add_example(example)

        hypos_to_return: list[Hypothesis] = []
        for hypotheses in min_hypotheses.values():
            hypos_to_return.extend(hypotheses.values())
        for hypotheses in max_hypotheses.values():
            hypos_to_return.extend(hypotheses.values())
        return hypos_to_return

    @staticmethod
    def collect_examples(trace, hypothesis):
        inv = hypothesis.invariant

        assert len(inv.params) == 3
        max_param, api_param, min_param = inv.params
        assert isinstance(max_param, InputOutputParam)
        assert isinstance(api_param, APIParam)
        assert isinstance(min_param, InputOutputParam)

        if max_param.is_input:
            assert not min_param.is_input
            is_threshold_min = False
            input_param = max_param
            output_param = min_param
        else:
            assert min_param.is_input
            is_threshold_min = True
            input_param = min_param
            output_param = max_param

        func_name = api_param.api_full_name
        # get all function calls for the function
        func_call_ids = trace.get_func_call_ids(func_name)

        import random

        if len(func_call_ids) > 1000:
            func_call_ids = random.sample(func_call_ids, 1000)

        for func_call_id in tqdm(
            func_call_ids, desc=f"Checking invariant {inv.text_description}"
        ):
            func_call_event = trace.query_func_call_event(func_call_id)
            if isinstance(
                func_call_event, (FuncCallExceptionEvent, IncompleteFuncCallEvent)
            ):
                continue

            threshold_value = input_param.get_value_from_arguments(
                Arguments(
                    func_call_event.args,
                    func_call_event.kwargs,
                    func_call_event.func_name,
                    consider_default_values=True,
                )
            )
            output_value = output_param.get_value_from_list_of_tensors(
                get_returned_tensors(func_call_event)
            )

            example = Example({"pre_event": [func_call_event.pre_record]})
            if is_threshold_min:
                if output_value >= threshold_value:
                    # add positive example
                    hypothesis.positive_examples.add_example(example)
                else:
                    # add negative example
                    hypothesis.negative_examples.add_example(example)

            else:
                if output_value <= threshold_value:
                    # add positive example
                    hypothesis.positive_examples.add_example(example)
                else:
                    # add negative example
                    hypothesis.negative_examples.add_example(example)

    @staticmethod
    def infer(trace: Trace) -> tuple[list[Invariant], list[FailedHypothesis]]:
        # now that we have the hypotheses for each function, we can start precondition inference
        all_hypotheses = ThresholdRelation.generate_hypothesis(trace)

        invariants = []
        failed_hypotheses = []
        for hypothesis in all_hypotheses:
            precondition = find_precondition(hypothesis, [trace])
            if precondition is not None:
                hypothesis.invariant.precondition = precondition
                invariants.append(hypothesis.invariant)
            else:
                failed_hypotheses.append(
                    FailedHypothesis(hypothesis, "Precondition not found")
                )

        return invariants, failed_hypotheses

    @staticmethod
    def evaluate(value_group: list) -> bool:
        raise NotImplementedError

    @staticmethod
    def static_check_all(trace, inv, check_relation_first):
        # get the first param and the second param, the first param should be larger or equal to the second param
        # the first param should be larger or equal to the second param
        assert inv.precondition is not None, "The precondition should not be None."
        assert len(inv.params) == 3
        max_param, api_param, min_param = inv.params
        assert isinstance(max_param, InputOutputParam)
        assert isinstance(api_param, APIParam)
        assert isinstance(min_param, InputOutputParam)

        if max_param.is_input:
            assert not min_param.is_input
            is_threshold_min = False
            input_param = max_param
            output_param = min_param
        else:
            assert min_param.is_input
            is_threshold_min = True
            input_param = min_param
            output_param = max_param

        func_name = api_param.api_full_name
        # get all function calls for the function
        func_call_ids = trace.get_func_call_ids(func_name)

        if len(func_call_ids) == 0:
            return CheckerResult(
                trace=None, invariant=inv, check_passed=True, triggered=False
            )

        triggered = False
        for func_call_id in tqdm(
            func_call_ids, desc=f"Checking invariant {inv.text_description}"
        ):
            func_call_event = trace.query_func_call_event(func_call_id)
            if isinstance(
                func_call_event, (FuncCallExceptionEvent, IncompleteFuncCallEvent)
            ):
                continue

            # check for precondition here
            if not inv.precondition.verify(
                [func_call_event.pre_record], "pre_event", trace
            ):
                continue

            triggered = True

            threshold_value = input_param.get_value_from_arguments(
                Arguments(
                    func_call_event.args,
                    func_call_event.kwargs,
                    func_call_event.func_name,
                    consider_default_values=True,
                )
            )
            output_value = output_param.get_value_from_list_of_tensors(
                get_returned_tensors(func_call_event)
            )

            if is_threshold_min:
                if output_value >= threshold_value:
                    continue
            else:
                if output_value <= threshold_value:
                    continue

            return CheckerResult(
                trace=[func_call_event.pre_record],
                invariant=inv,
                check_passed=False,
                triggered=True,
            )

        return CheckerResult(
            trace=None,
            invariant=inv,
            check_passed=True,
            triggered=triggered,
        )

    @staticmethod
    def get_precondition_infer_keys_to_skip(hypothesis: Hypothesis) -> list[str]:
        return []
