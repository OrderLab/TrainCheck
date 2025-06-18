import logging
import random
import time
from typing import Type

import numpy as np
from tqdm import tqdm

from traincheck.config.config import ANALYSIS_SKIP_FUNC_NAMES
from traincheck.instrumentor.tracer import TraceLineType
from traincheck.invariant.base_cls import (
    APIParam,
    Arguments,
    CheckerResult,
    Example,
    ExampleList,
    FailedHypothesis,
    Hypothesis,
    Invariant,
    Param,
    Relation,
    VarNameParam,
    VarTypeParam,
    calc_likelihood,
    construct_api_param,
    construct_var_param_from_var_change,
    is_signature_empty,
)
from traincheck.invariant.precondition import find_precondition
from traincheck.invariant.symbolic_value import generalize_values
from traincheck.trace.trace import Trace
from traincheck.trace.types import (
    ALL_EVENT_TYPES,
    FuncCallEvent,
    FuncCallExceptionEvent,
    IncompleteFuncCallEvent,
    VarChangeEvent,
)
from traincheck.utils import typename

PARENT_GROUP_NAME = "parent_func_call_pre"
VAR_GROUP_NAME = "var_events"

""" Possible Optimizations for Inference Speed of the APIContainRelation:
    - Parallelization
    - Checking lower level function calls first and use the results to infer the higher level function calls (i.e. module.to should reuse the results from module._apply)
    - Heuristics to handle recursive function calls (module._apply can have ~4000 lines of trace which contains many recursive calls). The get_func_call_events utility should provide an option to skip the recursive calls.
"""


def can_func_be_bound_method(
    trace: Trace,
    func_name: str,
    var_type: str | None = None,
    attr_name: str | None = None,
) -> bool:
    """Checks if during each invocaton of the function, there are variables that are not changed
    but are causally related to the function. If such variables exist, it means that the function
    's negative examples can be found by looking at the variables that are not changed. Otherwise,
    the negative examples will be the function call itself.
    """

    func_call_ids = trace.get_func_call_ids(func_name)

    for func_call_id in func_call_ids:
        if not trace.get_var_ids_unchanged_but_causally_related(
            func_call_id, var_type, attr_name
        ):
            return False
    return True


cache_events_scanner: dict[
    str, list[FuncCallEvent | FuncCallExceptionEvent | VarChangeEvent]
] = {}


def _group_events_by_type(events: list):
    """Group the events by their type, for event types not present in the list, an empty list is created"""
    grouped_events: dict[
        Type[FuncCallEvent]
        | Type[FuncCallExceptionEvent]
        | Type[IncompleteFuncCallEvent]
        | Type[VarChangeEvent],
        list[
            FuncCallEvent
            | FuncCallExceptionEvent
            | IncompleteFuncCallEvent
            | VarChangeEvent
        ],
    ] = {event_type: [] for event_type in ALL_EVENT_TYPES}
    for event in events:
        grouped_events[type(event)].append(event)
    return grouped_events


def events_scanner(
    trace: Trace, func_call_id: str
) -> list[FuncCallEvent | FuncCallExceptionEvent | VarChangeEvent]:
    """Scan the trace, and return the first set of events that happened within the pre and post
    events of the parent_func_name.

    Args:
        trace: Trace
            - the trace to be scanned
        func_call_id: str
            - the function call id of the parent function, which should correspond to two events (entry and exit)
    """
    logger = logging.getLogger(__name__)

    # implement cache for the events
    if func_call_id in cache_events_scanner:
        logger.debug(f"Using cached events for the function call: {func_call_id}")
        return cache_events_scanner[func_call_id]
    entry_time = time.time()
    events = trace.query_high_level_events_within_func_call(
        func_call_id=func_call_id,
    )
    cache_events_scanner[func_call_id] = events
    exit_time = time.time()
    logger.debug(
        f"Scanned the trace for events, return {len(events)} events, took {exit_time - entry_time} seconds"
    )
    return events


def prune_func_call(
    total_func_calls,
    parent_event_type,
    list_num_events_scanned: list[int],
    parent_event_types_seen: list,
) -> bool:
    """Pruning logic for determining whether a function's contained events should be processed and queried

    Args:
        total_func_calls: int
            - the total number of function calls
        list_num_events_scanned: list[int]
            - the list of number of events scanned for the function calls so far

    Returns:
        bool
            - whether the function call should be pruned or not

    The pruning logic is as follows:
    - If the number of function calls total present in the trace is less than 1000, no pruning is done
    - Pruning is probabilistic, with initial probability of (1 - 1000/total_func_calls)
    - If `list_num_events_scanned` indicates that the last 10 function calls have the same number of events scanned, the pruning probability is increased by 100 times for the next function call
    - We use random.random() to determine whether the function call should be pruned or not
    """

    logger = logging.getLogger(__name__)

    # if the event type has not even been seen before, don't prune
    if (
        len(parent_event_types_seen) == 0
        or parent_event_type != parent_event_types_seen[-1]
    ):
        return False

    MAX_FUNC_CALLS = 1000
    not_pruning_prob = min(MAX_FUNC_CALLS / total_func_calls, 1)

    if (
        len(list_num_events_scanned) > 10
        and len(set(list_num_events_scanned[-10:])) == 1
    ):
        # look at the last 10 number of events scanned, if they are all the same, skip the function call with a probability
        not_pruning_prob /= 100
        logger.debug(
            f"Same number of events observed in the last 10 attempts: {list_num_events_scanned[0]}, increasing the pruning probability to: {1 - not_pruning_prob}"
        )

    is_skipping = random.random() > not_pruning_prob
    if is_skipping:
        logger.debug(
            f"Skipping the function call due to sampling with a pruning probability of: {1 - not_pruning_prob}"
        )
    return is_skipping


def _merge_hypotheses(hypotheses: list[Hypothesis]) -> list[Hypothesis]:
    assert len(hypotheses) > 0, "Expected at least one hypotheses to be merged"

    logger = logging.getLogger(__name__)
    pos_exp_group_names = hypotheses[0].positive_examples.get_group_names()
    neg_exp_group_names = hypotheses[0].negative_examples.get_group_names()
    logger.debug(
        f"In _merge_hypothesis Positive example group names: {pos_exp_group_names}, Negative example group names: {neg_exp_group_names}"
    )

    used_dynamic_analysis = pos_exp_group_names == neg_exp_group_names

    # try to merge the hypotheses if individual hypotheses have too low likelihood
    all_likelihoods = [hypo.calc_likelihood() for hypo in hypotheses]

    # calculate the number of positive and negative examples after merge
    all_positive_examples = set()
    all_positive_parent_examples = set()
    all_negative_examples = set()

    for idx in range(len(hypotheses)):

        pre_records = hypotheses[idx].positive_examples.get_group_from_examples(
            PARENT_GROUP_NAME
        )
        exps_with_only_parent = [
            Example({PARENT_GROUP_NAME: pre_record}) for pre_record in pre_records
        ]
        all_positive_parent_examples.update(exps_with_only_parent)
        all_positive_examples.update(hypotheses[idx].positive_examples.examples)
        all_negative_examples.update(hypotheses[idx].negative_examples.examples)

    # let's remove those negative examples that are present in the positive examples of other hypotheses
    if used_dynamic_analysis:
        all_negative_examples = all_negative_examples.difference(all_positive_examples)
    else:
        all_negative_examples = all_negative_examples.difference(
            all_positive_parent_examples
        )

    # calculate the likelihood of the merged hypotheses now
    merged_likelihood = calc_likelihood(
        len(all_positive_examples), len(all_negative_examples)
    )

    if merged_likelihood / np.mean(all_likelihoods) < 1.3:
        # no merging if the likelihood is not significantly higher
        return hypotheses

    # construct the param for the merged hypotheses
    merged_child_param = hypotheses[0].invariant.params[1].with_no_customization()
    # get the values to be generalized
    all_customized_fields = [
        child_param.get_customized_fields()
        for child_param in [hypo.invariant.params[1] for hypo in hypotheses]
    ]
    all_customizable_fields = set(merged_child_param.get_customizable_field_names())
    for field in all_customizable_fields:
        if all(
            field not in customized_fields
            for customized_fields in all_customized_fields
        ):
            continue

        assert all(
            field in customized_fields for customized_fields in all_customized_fields
        ), "Expected all hypotheses to have the same customizable fields"
        values = [
            customized_fields[field] for customized_fields in all_customized_fields
        ]
        generalized_value = generalize_values(values)
        setattr(merged_child_param, field, generalized_value)

    # construct the merged hypotheses
    merged_hypothesis = Hypothesis(
        invariant=Invariant(
            relation=hypotheses[0].invariant.relation,
            params=[
                hypotheses[0].invariant.params[0],
                merged_child_param,
            ],
            text_description="TBD merged",
            num_positive_examples=len(all_positive_examples),
            num_negative_examples=len(all_positive_examples),
            precondition=None,  # to be inferred later
        ),
        positive_examples=ExampleList.from_iterable_of_examples(
            all_positive_examples, pos_exp_group_names
        ),
        negative_examples=ExampleList.from_iterable_of_examples(
            all_negative_examples, neg_exp_group_names
        ),  # this is not correct as in the case of no negative examples, we should still have the negative examples group names initialized
    )

    return [merged_hypothesis]


def _get_parent_type(
    trace: Trace, parent_func_call_id: str
) -> Type[FuncCallEvent | FuncCallExceptionEvent | IncompleteFuncCallEvent]:
    parent_post_record = trace.get_post_func_call_record(parent_func_call_id)
    parent_event_type: Type[
        FuncCallEvent | FuncCallExceptionEvent | IncompleteFuncCallEvent
    ]
    if parent_post_record:
        if parent_post_record["type"] == TraceLineType.FUNC_CALL_POST:
            parent_event_type = FuncCallEvent
        elif parent_post_record["type"] == TraceLineType.FUNC_CALL_POST_EXCEPTION:
            parent_event_type = FuncCallExceptionEvent
        else:
            assert False, f"Unknown event type: {parent_post_record['type']}"
    else:
        parent_event_type = IncompleteFuncCallEvent

    return parent_event_type


class APIContainRelation(Relation):
    """Relation that checks if the API contain relation holds.
    In the API contain relation, an parent API call will always contain the child API call.
    """

    """ After this PR, APIContainRelation should be able to tell events that are semantically different apart. 
    For now, for example for the APICall related events, we don't distinguish whether the API failed or not.
    Also for VarChangeEvent, we don't consider the values before/after the change.

    In this PR we are trying to count for these delicate differences, and generalize only the common parts.
    Thus, we should be able to infer the additional invariant that `zero_grad` always lead to a `VarChangeEvent` from "anything" to "None".
    TODOs:
    - [ ] Refine the intersection logic, instead of relying on intersects provided by set class and Event class's __eq__ method, we should
    define a function that does this compare
        - [ ] Compare every attribute of the event, and return True if all attributes are the same
        - [ ] Try to generalize the event when seeing the same type event with different attributes 
    - [ ] Make the Dynamic Analysis part less ad-hoc as of its current form in the code
    """

    @staticmethod
    def generate_hypothesis(trace) -> list[Hypothesis]:
        # let's play it dumb here,
        _, _, hypotheses = APIContainRelation.infer(
            trace, return_successful_hypotheses=True
        )  # type: ignore

        # set all precond to None
        for hypo in hypotheses:
            hypo.invariant.precondition = None

        return hypotheses

    @staticmethod
    def collect_examples(trace, hypothesis):
        """A wrapper for the real collect_examples method. Adding stage-based inference support"""
        if hypothesis.valid_only_for_stages:
            for (
                stage,
                stage_trace,
            ) in (
                trace.get_traces_for_stage().items()
            ):  # This raise an error if in mutli-trace inference some inputs are annotated and some are not, we can also just skip the unannotated traces
                if stage not in hypothesis.valid_stages:
                    continue
                APIContainRelation._collect_examples(stage_trace, hypothesis)
        else:
            APIContainRelation._collect_examples(trace, hypothesis)

    @staticmethod
    def _collect_examples(trace, hypothesis):
        # TODO: only collect examples for a specific stage if the hypothesis is stage-based.
        logger = logging.getLogger(__name__)

        inv = hypothesis.invariant
        assert (
            len(inv.params) == 2
        ), "Expected 2 parameters for APIContainRelation, one for the parent function name, and one for the child event name"
        parent_param, child_param = inv.params[0], inv.params[1]
        assert isinstance(
            parent_param, APIParam
        ), "Expected the first parameter to be an APIParam"
        assert isinstance(
            child_param, (APIParam, VarTypeParam, VarNameParam)
        ), "Expected the second parameter to be an APIParam or VarTypeParam (VarNameParam not supported yet)"

        if (
            not isinstance(child_param, APIParam)
            and not trace.is_var_instrumented_proxy()
        ):
            logger.warning(
                "The current trace is not collected with the proxy-based variable instrumentation, skip example collection"
            )
            return

        parent_func_name = parent_param.api_full_name

        parent_func_call_ids = trace.get_func_call_ids(
            parent_func_name
        )  # should be sorted by time to reflect timeliness

        check_for_unchanged_vars = False
        if not isinstance(child_param, APIParam):
            if VAR_GROUP_NAME in hypothesis.negative_examples.get_group_names():
                check_for_unchanged_vars = True

        nums_contained_events = []
        kind_of_parent_events = []
        for i, parent_func_call_id in tqdm(
            enumerate(parent_func_call_ids), desc="Collecting examples"
        ):
            parent_event_type = _get_parent_type(trace, parent_func_call_id)
            if not (i < 10 or i > len(parent_func_call_ids) - 10) and prune_func_call(
                len(parent_func_call_ids),
                parent_event_type,
                nums_contained_events,
                kind_of_parent_events,
            ):
                continue

            contained_events = events_scanner(trace, parent_func_call_id)
            nums_contained_events.append(len(contained_events))
            kind_of_parent_events.append(parent_event_type)

            grouped_events = _group_events_by_type(contained_events)
            if isinstance(child_param, APIParam):
                contained_events = (
                    grouped_events.get(FuncCallEvent, [])
                    + grouped_events.get(FuncCallExceptionEvent, [])
                    + grouped_events.get(IncompleteFuncCallEvent, [])
                )
                for event in contained_events:
                    if child_param.check_event_match(event):
                        # add a positive example
                        parent_pre_record = trace.get_pre_func_call_record(
                            parent_func_call_id
                        )
                        example = Example()
                        example.add_group(PARENT_GROUP_NAME, [parent_pre_record])
                        hypothesis.positive_examples.add_example(example)
                        break
                else:
                    # add a negative example
                    parent_pre_record = trace.get_pre_func_call_record(
                        parent_func_call_id
                    )
                    example = Example()
                    example.add_group(PARENT_GROUP_NAME, [parent_pre_record])
                    hypothesis.negative_examples.add_example(example)
            else:
                contained_events = grouped_events.get(VarChangeEvent, [])
                found = False
                for event in contained_events:
                    if child_param.check_event_match(event):
                        # add a positive example
                        parent_pre_record = trace.get_pre_func_call_record(
                            parent_func_call_id
                        )
                        example = Example()
                        example.add_group(PARENT_GROUP_NAME, [parent_pre_record])
                        hypothesis.positive_examples.add_example(example)
                        found = True
                        if check_for_unchanged_vars:
                            example.add_group(VAR_GROUP_NAME, event.get_traces())
                        else:
                            # only one example is enough if we don't have var state in the example
                            break
                if check_for_unchanged_vars:
                    assert (
                        found
                    ), "Expected the positive example to be found in the case of dynamic analysis"
                    unchanged_var_ids = (
                        trace.get_var_ids_unchanged_but_causally_related(
                            parent_func_call_id,
                            child_param.var_type,
                            child_param.attr_name,
                        )
                    )
                    # add these unchanged vars as negative examples
                    for var_id in unchanged_var_ids:
                        example = Example()
                        example.add_group(PARENT_GROUP_NAME, [parent_pre_record])
                        example.add_group(
                            VAR_GROUP_NAME,
                            trace.get_var_raw_event_before_time(
                                var_id, parent_pre_record["time"]
                            ),
                        )
                        hypothesis.negative_examples.add_example(example)

                if not found:
                    # add a negative example
                    parent_pre_record = trace.get_pre_func_call_record(
                        parent_func_call_id
                    )
                    example = Example()
                    example.add_group(PARENT_GROUP_NAME, [parent_pre_record])
                    hypothesis.negative_examples.add_example(example)

    @staticmethod
    def infer(
        trace: Trace, return_successful_hypotheses: bool = False
    ) -> tuple[list[Invariant], list[FailedHypothesis]]:
        """Infer Invariants without Preconditions"""
        # enable stage-based inference for API Contain Relation
        logger = logging.getLogger(__name__)

        if not trace.is_stage_annotated():
            invs, failed_hypos, succ_hypos = APIContainRelation._infer(trace)
            if return_successful_hypotheses:
                return invs, failed_hypos, succ_hypos  # type: ignore
            return invs, failed_hypos

        invariants_by_stage: dict[Invariant, list[str]] = {}
        invariants_to_hypos: dict[Invariant, list[Hypothesis]] = {}
        failed_hypothesis_by_stage: dict[FailedHypothesis, list[str]] = {}
        for stage, stage_trace in trace.get_traces_for_stage().items():
            logger.info(
                "Stage annotation detected in trace, enabling stage-based inference"
            )
            invariants, failed_hypotheses, successful_hypotheses = (
                APIContainRelation._infer(stage_trace)
            )
            for invariant, hypo in zip(invariants, successful_hypotheses):
                if invariant not in invariants_by_stage:
                    invariants_by_stage[invariant] = []
                if invariant not in invariants_to_hypos:
                    invariants_to_hypos[invariant] = []
                invariants_by_stage[invariant].append(stage)
                invariants_to_hypos[invariant].append(hypo)

            for failed_hypothesis in failed_hypotheses:
                if failed_hypothesis not in failed_hypothesis_by_stage:
                    failed_hypothesis_by_stage[failed_hypothesis] = []
                failed_hypothesis_by_stage[failed_hypothesis].append(stage)

        # merge the inferred invariants, for the unmerged invariants, add the stage information to the precondition
        merged_invariants = []
        merged_successful_hypotheses: list[Hypothesis] = []
        all_stages = trace.get_all_stages()
        for invariant in invariants_by_stage:
            supported_stages = set(invariants_by_stage[invariant])
            hypothesis = invariants_to_hypos[invariant][0]
            for other_hypothesis in invariants_to_hypos[invariant][1:]:
                hypothesis.positive_examples.examples.extend(
                    other_hypothesis.positive_examples.examples
                )
                hypothesis.negative_examples.examples.extend(
                    other_hypothesis.negative_examples.examples
                )
            hypothesis.add_stage_info(supported_stages, all_stages)
            merged_invariants.append(hypothesis.get_invariant(all_stages))
            merged_successful_hypotheses.append(hypothesis)

        # HACK: add the stage info to the failed hypotheses through the text description as well NEED TO CHANGE IF WE SUPPORT INVARIANT REFINEMENT
        for failed_hypothesis in failed_hypothesis_by_stage:
            not_supported_stages = failed_hypothesis_by_stage[failed_hypothesis]
            if failed_hypothesis.hypothesis.invariant.text_description is None:
                failed_hypothesis.hypothesis.invariant.text_description = ""
            failed_hypothesis.hypothesis.invariant.text_description += (
                f" (FAILED IN in stages: {not_supported_stages})"
            )

        if return_successful_hypotheses:
            return merged_invariants, list(failed_hypothesis_by_stage.keys()), merged_successful_hypotheses  # type: ignore
        return merged_invariants, list(failed_hypothesis_by_stage.keys())

    @staticmethod
    def _infer(
        trace: Trace,
    ) -> tuple[list[Invariant], list[FailedHypothesis], list[Hypothesis]]:
        """Infer Invariants with Preconditions"""

        logger = logging.getLogger(__name__)

        # split the trace into groups based on (process_id and thread_id)
        hypotheses: dict[
            APIParam, dict[APIParam | VarTypeParam | VarNameParam, Hypothesis]
        ] = {}
        hypothesis_should_use_causal_vars_for_negative_examples: dict[
            str, dict[str, dict[str | tuple[str, ...], bool]]
        ] = {}
        func_names = trace.get_func_names()

        func_names = [
            func_name
            for func_name in func_names
            if not any(skip in func_name for skip in ANALYSIS_SKIP_FUNC_NAMES)
        ]

        func_names = [
            func_name for func_name in func_names if not is_signature_empty(func_name)
        ]

        if len(func_names) == 0:
            logger.warning(
                "No function calls found in the trace, skipping the analysis"
            )
            return [], [], []

        for parent in tqdm(
            func_names, desc="Scanning through function calls to generate hypotheses"
        ):
            is_parent_a_bound_method = trace.get_func_is_bound_method(parent)
            logger.debug(f"Starting the analysis for the parent function: {parent}")
            parent_func_call_ids = trace.get_func_call_ids(parent)
            logger.debug(
                f"Found {len(parent_func_call_ids)} invocations for the function: {parent}"
            )

            # down sampling the function calls
            if len(parent_func_call_ids) > 1000:
                # down sample the function calls, keep the first 10, last 10, and randomly sample 100 from the rest
                parent_func_call_ids = (
                    parent_func_call_ids[:10]
                    + random.sample(parent_func_call_ids[10:-10], 100)  # type: ignore
                    + parent_func_call_ids[-10:]
                )

            all_contained_events: dict[
                str, list[FuncCallEvent | FuncCallExceptionEvent | VarChangeEvent]
            ] = {}

            nums_contained_events: list[int] = []
            kind_of_parent_events: list[
                Type[FuncCallEvent | FuncCallExceptionEvent | IncompleteFuncCallEvent]
            ] = []
            for i, parent_func_call_id in enumerate(parent_func_call_ids):
                parent_event_type = _get_parent_type(trace, parent_func_call_id)

                if not (
                    i < 10 or i > len(parent_func_call_ids) - 10
                ) and prune_func_call(
                    len(parent_func_call_ids),
                    parent_event_type,
                    nums_contained_events,
                    kind_of_parent_events,
                ):
                    continue
                contained_events = events_scanner(
                    trace=trace, func_call_id=parent_func_call_id
                )
                nums_contained_events.append(len(contained_events))
                all_contained_events[parent_func_call_id] = contained_events
                kind_of_parent_events.append(parent_event_type)

            # MARK: HYPOTHESIS CREATION
            """Create hypotheses for each specific event (a event is defined by its __dict__)"""
            parent_param = APIParam(parent)
            hypos_for_dynamic_analysis: list[tuple[Param, Param]] = []
            passing_events_for_API_contain_API: dict[
                APIParam,
                dict[
                    APIParam,
                    list[
                        FuncCallEvent | FuncCallExceptionEvent | IncompleteFuncCallEvent
                    ],
                ],
            ] = {}

            for (
                parent_func_call_id,
                local_contained_events,
            ) in all_contained_events.items():
                parent_event = trace.query_func_call_event(parent_func_call_id)
                parent_param = construct_api_param(
                    parent_event
                ).with_no_customization()  # don't use args/kwargs/exceptions for distinguishing the parent functions
                if parent_param not in hypotheses:
                    hypotheses[parent_param] = {}
                    hypothesis_should_use_causal_vars_for_negative_examples[parent] = {}

                events_grouped_by_type = _group_events_by_type(local_contained_events)
                for event in events_grouped_by_type[VarChangeEvent]:
                    # dynamic analysis can be applied to VarChangeEvent
                    # child_param = VarNameParam(event.var_id.var_type, event.attr_name)
                    child_param: APIParam | VarTypeParam | VarNameParam = (
                        construct_var_param_from_var_change(event)
                    )

                    if child_param in hypotheses[parent_param]:
                        continue
                    # if dynamic analysis is available for this child_param, we can use it to find negative examples
                    should_use_causal_vars_for_negative_examples = False
                    if isinstance(child_param, VarTypeParam):
                        # dynamic analysis only applicable to Var type based analysis
                        should_use_causal_vars_for_negative_examples = (
                            is_parent_a_bound_method
                            and can_func_be_bound_method(
                                trace, parent, event.var_id.var_type, event.attr_name
                            )
                        )
                        if should_use_causal_vars_for_negative_examples:
                            hypos_for_dynamic_analysis.append(
                                (parent_param, child_param)
                            )
                    hypotheses[parent_param][child_param] = Hypothesis(
                        Invariant(
                            relation=APIContainRelation,
                            params=[parent_param, child_param],
                            precondition=None,
                            text_description=f"{parent} contains {child_param} of type {typename(event)}",
                        ),
                        positive_examples=ExampleList(
                            {PARENT_GROUP_NAME, VAR_GROUP_NAME}
                            if should_use_causal_vars_for_negative_examples
                            else {PARENT_GROUP_NAME}
                        ),
                        negative_examples=ExampleList(
                            {PARENT_GROUP_NAME, VAR_GROUP_NAME}
                            if should_use_causal_vars_for_negative_examples
                            else {PARENT_GROUP_NAME}
                        ),
                    )

                events_grouped_by_type.pop(VarChangeEvent)
                for event_type in events_grouped_by_type:
                    for event in events_grouped_by_type[event_type]:
                        child_param = construct_api_param(event).with_no_customization()
                        # append the event to the passing_events_for_API_contain_API
                        if parent_param not in passing_events_for_API_contain_API:
                            passing_events_for_API_contain_API[parent_param] = {}
                        if (
                            child_param
                            not in passing_events_for_API_contain_API[parent_param]
                        ):
                            passing_events_for_API_contain_API[parent_param][
                                child_param
                            ] = []
                        passing_events_for_API_contain_API[parent_param][
                            child_param
                        ].append(event)

                        if child_param not in hypotheses[parent_param]:
                            hypotheses[parent_param][child_param] = Hypothesis(
                                Invariant(
                                    relation=APIContainRelation,
                                    params=[parent_param, child_param],
                                    precondition=None,
                                    text_description=f"{parent} contains {child_param} of type {typename(event)}",
                                ),
                                positive_examples=ExampleList({PARENT_GROUP_NAME}),
                                negative_examples=ExampleList({PARENT_GROUP_NAME}),
                            )

            # MARK: EVENT MERGING TO CREATE FINE-GRAINED CHILD PARAMS
            for parent_param in passing_events_for_API_contain_API:
                for child_param in passing_events_for_API_contain_API[parent_param]:
                    events = passing_events_for_API_contain_API[parent_param][
                        child_param
                    ]

                    def _merge_child_API_events(
                        events: list[
                            FuncCallEvent
                            | FuncCallExceptionEvent
                            | IncompleteFuncCallEvent
                        ],
                    ) -> APIParam:
                        """Given a list of API events belonging to the same parent function, merge them to create a fine-grained child_param
                        i.e. consistency of arguments, etc.
                        """
                        arguments: Arguments | None = None
                        func_name = events[0].func_name
                        for event in events:
                            if arguments is None:
                                args = event.args
                                kwargs = event.kwargs
                                arguments = Arguments(args, kwargs, func_name)
                            else:
                                args = event.args
                                kwargs = event.kwargs
                                event_arguments = Arguments(args, kwargs, func_name)
                                arguments = arguments.merge_with(event_arguments)

                            if arguments.is_empty():
                                return APIParam(func_name)

                        assert (
                            isinstance(arguments, Arguments)
                            and not arguments.is_empty()
                        )
                        return APIParam(
                            func_name,
                            arguments=arguments,
                        )

                    merged_child_param = _merge_child_API_events(events)
                    hypotheses[parent_param][child_param] = Hypothesis(
                        Invariant(
                            relation=APIContainRelation,
                            params=[parent_param, merged_child_param],
                            precondition=None,
                            text_description=f"{parent} contains {merged_child_param}",
                        ),
                        positive_examples=ExampleList({PARENT_GROUP_NAME}),
                        negative_examples=ExampleList({PARENT_GROUP_NAME}),
                    )

            # MARK: PRECONDITION INFERENCE PREPARATION
            # scan the child_func_names for positive and negative examples
            for (
                parent_func_call_id,
                local_contained_events,
            ) in all_contained_events.items():
                parent_event = trace.query_func_call_event(parent_func_call_id)
                parent_param = construct_api_param(parent_event).with_no_customization()
                parent_hypos = hypotheses[
                    parent_param
                ].copy()  # keep record of all hypotheses related to the parent function

                # adding positive examples
                events_grouped_by_type = _group_events_by_type(local_contained_events)
                for event in events_grouped_by_type[VarChangeEvent]:
                    child_param = construct_var_param_from_var_change(event)
                    assert (
                        child_param in hypotheses[parent_param]
                    ), f"Internal error: child_param {child_param} not found in the hypotheses during the example collection phase"
                    parent_hypos.pop(
                        child_param, None
                    )  # same child event can occur multiple times in a particular parent event, due to the above assert it is save to use None to ignore the KeyError
                    example = Example()
                    example.add_group(PARENT_GROUP_NAME, [parent_event.pre_record])

                    if (parent_param, child_param) in hypos_for_dynamic_analysis:
                        example.add_group(VAR_GROUP_NAME, event.get_traces())
                    hypotheses[parent_param][child_param].positive_examples.add_example(
                        example
                    )

                events_grouped_by_type.pop(VarChangeEvent)
                for event_type in events_grouped_by_type:
                    for event in events_grouped_by_type[event_type]:
                        child_param = construct_api_param(event).with_no_customization()
                        assert (
                            child_param in hypotheses[parent_param]
                        ), f"Internal error: child_param {child_param} not found in the hypotheses during the example collection phase"

                        parent_hypos.pop(
                            child_param, None
                        )  # same child event can occur multiple times in a particular parent event, due to the above assert it is save to use None to ignore the KeyError
                        hypotheses[parent_param][
                            child_param
                        ].positive_examples.add_example(
                            Example({PARENT_GROUP_NAME: [parent_event.pre_record]})
                        )

                # adding negative examples for hypotheses that are not modified above
                for child_param in parent_hypos:
                    if (parent_param, child_param) not in hypos_for_dynamic_analysis:
                        hypotheses[parent_param][
                            child_param
                        ].negative_examples.add_example(
                            Example({PARENT_GROUP_NAME: [parent_event.pre_record]})
                        )
                    else:
                        # use dynamic analysis to find negative examples
                        assert isinstance(
                            child_param, (VarTypeParam)
                        )  # dynamic analysis only applicable to Var type based analysis

                        unchanged_var_ids = (
                            trace.get_var_ids_unchanged_but_causally_related(
                                parent_func_call_id,
                                child_param.var_type,
                                child_param.attr_name,
                            )
                        )
                        for var_id in unchanged_var_ids:
                            example = Example()
                            example.add_group(
                                PARENT_GROUP_NAME, [parent_event.pre_record]
                            )
                            example.add_group(
                                VAR_GROUP_NAME,
                                trace.get_var_raw_event_before_time(
                                    var_id, parent_event.pre_record["time"]
                                ),
                            )
                            hypotheses[parent_param][
                                child_param
                            ].negative_examples.add_example(example)

        # extra step: merge invariants for Var Change Related Invariants
        # QUESTION: how do we do the same merging thing for APIParam?
        for parent_param in hypotheses:
            # group the child_hypotheses by core_fields
            all_mergeable_hypotheses: dict[
                VarNameParam | VarTypeParam, list[Hypothesis]
            ] = {}
            for child_param in hypotheses[parent_param]:
                if isinstance(child_param, APIParam):
                    continue
                core_child_param = child_param.with_no_customization()
                if core_child_param not in all_mergeable_hypotheses:
                    all_mergeable_hypotheses[core_child_param] = []
                all_mergeable_hypotheses[core_child_param].append(
                    hypotheses[parent_param][child_param]
                )

            # hypotheses[parent_param] = {} # this is wrong, we should only remove the hypotheses that are passed to the merge_hypotheses function
            # remove all the hypotheses that have the second param as VarNameParam or VarTypeParam
            to_be_merged_child_params = []
            for child_param in hypotheses[parent_param]:
                if isinstance(child_param, (VarNameParam | VarTypeParam)):
                    to_be_merged_child_params.append(child_param)
            for child_param in to_be_merged_child_params:
                hypotheses[parent_param].pop(child_param)

            # for each key in all_mergeable_hypotheses, invoke the hypotheses merging process.
            for hypotheses_to_be_merged in all_mergeable_hypotheses.values():
                merged_successful_hypotheses = _merge_hypotheses(
                    hypotheses_to_be_merged
                )
                # delete original hypotheses in the original `hypotheses` structure
                for hypo in merged_successful_hypotheses:
                    new_child_param = hypo.invariant.params[1]
                    assert isinstance(new_child_param, (VarNameParam | VarTypeParam))
                    hypotheses[parent_param][
                        new_child_param
                    ] = hypo  # index 1 here is to get the var_name or var_type param, a hack. Correctness should be guaranteed by the isinstance above

        pbar = tqdm(
            total=len(
                [
                    child_param
                    for parent_param in hypotheses
                    for child_param in hypotheses[parent_param]
                ]
            ),
            desc="Infering preconditions",
        )
        all_invariants: list[Invariant] = []
        failed_hypotheses = []
        successful_hypotheses = []
        for parent_param in hypotheses:
            for child_param in hypotheses[parent_param]:
                hypo = hypotheses[parent_param][child_param]
                pbar.update(1)
                logger.debug(
                    f"Starting the inference of precondition for the hypotheses: {hypo.invariant.text_description}"
                )
                found_precondition = find_precondition(hypo, [trace])
                if (
                    found_precondition is not None
                ):  # TODO: abstract this precondition inference part to a function
                    hypo.invariant.precondition = found_precondition
                    hypo.invariant.num_positive_examples = len(hypo.positive_examples)
                    hypo.invariant.num_negative_examples = len(hypo.negative_examples)
                    all_invariants.append(hypo.invariant)
                    successful_hypotheses.append(hypo)
                else:
                    logger.debug(f"Precondition not found for the hypotheses: {hypo}")
                    failed_hypotheses.append(
                        FailedHypothesis(hypo, "Precondition not found")
                    )

        return all_invariants, failed_hypotheses, successful_hypotheses

    @staticmethod
    def evaluate(value_group: list) -> bool:
        """Evaluate the relation based on the given value group.

        Args:
            value_group: list
                - [0]: str - the function name to be contained
                - [1]: set - a set of function names (child API calls)

        TODO:
            - extend [0] to not only function calls but also other types of events, such as data updates, etc.
        """

        assert (
            len(value_group) == 2
        ), "Expected 2 arguments for APIContainRelation, #1 the function name to be contained, #2 a set of function names"
        assert isinstance(
            value_group[0], str
        ), "Expected the first argument to be a string"
        assert isinstance(
            value_group[1], set
        ), "Expected the second argument to be a set"

        expected_child_func = value_group[0]
        seen_child_funcs = value_group[1]

        return expected_child_func in seen_child_funcs

    @staticmethod
    def static_check_all(
        trace: Trace, inv: Invariant, check_relation_first: bool
    ) -> CheckerResult:
        """Check the invariant on the trace

        NOTE: this function takes one invariant at a time, and checks if the invariant holds on the trace. However, if multiple invariants targets the same parent function,
        we should be batching the checks for the same parent function to avoid redundant computation (mainly from `events_scanner`). ### TODO ITEM ###
        """

        assert (
            len(inv.params) == 2
        ), f"Expected 2 parameters for APIContainRelation, one for the parent function name, and one for the child event name: {inv.params[0].to_dict()}"

        parent_param, child_param = inv.params[0], inv.params[1]
        assert isinstance(
            parent_param, APIParam
        ), "Expected the first parameter to be an APIParam"
        assert isinstance(
            child_param, (APIParam, VarTypeParam, VarNameParam)
        ), "Expected the second parameter to be an APIParam or VarTypeParam (VarNameParam not supported yet)"

        # TODO: support VarNameParam in the future

        logger = logging.getLogger(__name__)

        parent_func_name = parent_param.api_full_name
        preconditions = inv.precondition
        inv_triggered = (
            False  # should be set to True if precondition is met at least once
        )

        assert (
            preconditions is not None
        ), "Expected the precondition to be set for the invariant"

        parent_func_call_ids = trace.get_func_call_ids(
            parent_func_name
        )  # should be sorted by time to reflect timeliness

        skip_var_unchanged_check = (
            VAR_GROUP_NAME not in preconditions.get_group_names()
            or len(preconditions.get_group(VAR_GROUP_NAME)) == 0
            or preconditions.is_group_unconditional(VAR_GROUP_NAME)
        )

        if not skip_var_unchanged_check:
            assert isinstance(
                child_param, VarTypeParam
            ), "Expected the child parameter to be a VarTypeParam"
            if not can_func_be_bound_method(
                trace, parent_func_name, child_param.var_type, child_param.attr_name
            ):
                logger.warning(
                    """The invariant includes a precondition for the variables that are changed/unchanged, to enforce this precondition, you should be running the trace collector with the --scan_proxy_in_args flag to collect the trace.
Defaulting to skip the var preconditon check for now.
                    """
                )
                skip_var_unchanged_check = True

        # the main checking loop: the online checker function will be the body of this loop, which will be called repeatedly
        nums_contained_events: list[int] = []
        kind_of_parent_events: list[
            Type[FuncCallEvent | FuncCallExceptionEvent | IncompleteFuncCallEvent]
        ] = []
        for parent_func_call_id in tqdm(
            parent_func_call_ids, desc=f"Checking invariants for {inv.text_description}"
        ):
            parent_event_type = _get_parent_type(trace, parent_func_call_id)
            kind_of_parent_events.append(parent_event_type)
            # check for parent precondition
            parent_pre_record = trace.get_pre_func_call_record(parent_func_call_id)

            var_unchanged_check_passed = True
            found_expected_child_event = False

            if not check_relation_first:
                # precondition check
                if not preconditions.verify(
                    [parent_pre_record], PARENT_GROUP_NAME, trace
                ):
                    logger.debug(
                        f"Precondition not met for the parent function: {parent_func_name} at {parent_func_call_id}: {parent_pre_record['time']} at {trace.get_time_precentage(parent_pre_record['time'])}, skipping the contained events check"
                    )
                    continue

                # precondition passed
                inv_triggered = True

                # invariant check
                events = events_scanner(trace=trace, func_call_id=parent_func_call_id)
                nums_contained_events.append(len(events))
                for event in events:
                    if child_param.check_event_match(event):
                        found_expected_child_event = True
                        break
            else:
                # invariant check
                events = events_scanner(trace=trace, func_call_id=parent_func_call_id)
                nums_contained_events.append(len(events))
                for event in events:
                    if child_param.check_event_match(event):
                        found_expected_child_event = True
                        break

                # precondition check
                if not preconditions.verify(
                    [parent_pre_record], PARENT_GROUP_NAME, trace
                ):
                    logger.debug(
                        f"Precondition not met for the parent function: {parent_func_name} at {parent_func_call_id}: {parent_pre_record['time']} at {trace.get_time_precentage(parent_pre_record['time'])}, skipping the contained events check"
                    )
                    continue

                # precondition passed
                inv_triggered = True

            if not skip_var_unchanged_check:
                assert isinstance(
                    child_param, VarTypeParam
                ), "Expected the child parameter to be a VarTypeParam"
                # get the unchanged vars that are causally related to the parent function
                unchanged_var_ids = trace.get_var_ids_unchanged_but_causally_related(
                    parent_func_call_id, child_param.var_type, child_param.attr_name
                )
                assert (
                    len(unchanged_var_ids) > 0
                ), f"Internal error: can_func_be_bound_method returned True but no unchanged vars found for the parent function: {parent_func_name} at {parent_func_call_id}: {parent_pre_record['time']} at {trace.get_time_precentage(parent_pre_record['time'])}"
                # get the var change events for the unchanged vars
                unchanged_var_states = [
                    trace.get_var_raw_event_before_time(
                        var_id, parent_pre_record["time"]
                    )
                    for var_id in unchanged_var_ids
                ]
                for unchanged_var_state in unchanged_var_states:
                    # verify that no precondition is met for the unchanged vars
                    # MARK: precondition 2
                    if not preconditions.verify(
                        unchanged_var_state, VAR_GROUP_NAME, trace
                    ):
                        logger.error(
                            f"INV CHECK ERROR: Precondition met for the unchanged vars for the parent function: {parent_func_name} at {parent_func_call_id}: {parent_pre_record['time']} at {trace.get_time_precentage(parent_pre_record['time'])}"
                        )
                        var_unchanged_check_passed = False
                        break

            if (skip_var_unchanged_check and not found_expected_child_event) or (
                not skip_var_unchanged_check and not var_unchanged_check_passed
            ):
                logger.error(
                    f"INV CHECK ERROR: Expected child event not found in the contained events for the parent function: {parent_func_name} at {parent_func_call_id}: {parent_pre_record['time']} at {trace.get_time_precentage(parent_pre_record['time'])}"
                )

                # TODO: improve reported error message + include the trace for variables that didn't change in the causal relation case
                assert (
                    inv_triggered
                ), "Expected the invariant to be triggered, check internal logic correctness"
                result = CheckerResult(
                    trace=[parent_pre_record],
                    invariant=inv,
                    check_passed=False,
                    triggered=inv_triggered,
                )
                return result

        return CheckerResult(
            trace=None,
            invariant=inv,
            check_passed=True,
            triggered=inv_triggered,
        )

    @staticmethod
    def get_precondition_infer_keys_to_skip(hypothesis: Hypothesis) -> list[str]:
        return []
