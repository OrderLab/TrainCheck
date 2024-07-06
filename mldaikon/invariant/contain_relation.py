import logging

import polars as pl
from tqdm import tqdm

from mldaikon.instrumentor.tracer import TraceLineType
from mldaikon.invariant.base_cls import (
    APIParam,
    Example,
    ExampleList,
    Hypothesis,
    Invariant,
    Param,
    Relation,
    VarTypeParam,
)
from mldaikon.invariant.precondition import find_precondition
from mldaikon.trace.trace import Trace
from mldaikon.trace.types import FuncCallEvent, FuncCallExceptionEvent, VarChangeEvent
from mldaikon.utils import typename

PARENT_GROUP_NAME = "parent_func_call_pre"
VAR_GROUP_NAME = "var_events"


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

    func_call_ids = (
        trace.events.filter(
            (pl.col("type") == TraceLineType.FUNC_CALL_PRE)
            & (pl.col("function") == func_name)
        )
        .select("func_call_id")
        .to_series()
    )
    for func_call_id in func_call_ids:
        if not trace.get_var_ids_unchanged_but_causally_related(
            func_call_id, var_type, attr_name
        ):
            return False
    return True


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
    pre_call_record = trace.get_pre_func_call_record(func_call_id)
    post_call_record = trace.get_post_func_call_record(func_call_id)

    process_id = pre_call_record["process_id"]
    thread_id = pre_call_record["thread_id"]

    time_range = (
        pre_call_record["time"],
        post_call_record["time"],
    )

    events = trace.query_high_level_events_within_time(
        time_range=time_range, process_id=process_id, thread_id=thread_id
    )
    return events


class APIContainRelation(Relation):
    """Relation that checks if the API contain relation holds.
    In the API contain relation, an parent API call will always contain the child API call.
    """

    @staticmethod
    def infer(trace: Trace) -> list[Invariant]:
        """Infer Invariants with Preconditions"""

        logger = logging.getLogger(__name__)

        # split the trace into groups based on (process_id and thread_id)
        hypothesis: dict[str, dict[str, dict[str | tuple[str, ...], Hypothesis]]] = {}
        hypothesis_should_use_causal_vars_for_negative_examples: dict[
            str, dict[str, dict[str | tuple[str, ...], bool]]
        ] = {}
        func_names = trace.get_func_names()

        func_names = [
            func_name for func_name in func_names if "adam.step" in func_name.lower()
        ]

        if len(func_names) == 0:
            logger.warning(
                "No function calls found in the trace, skipping the analysis"
            )
            return []

        for parent in tqdm(
            func_names, desc="Scanning through function calls to generate hypotheses"
        ):
            is_parent_a_bound_method = trace.get_func_is_bound_method(parent)
            logger.debug(f"Starting the analysis for the parent function: {parent}")
            # get all parent pre event indexes
            parent_func_call_ids = (
                trace.events.filter(
                    (pl.col("type") == TraceLineType.FUNC_CALL_PRE)
                    & (pl.col("function") == parent)
                )
                .select("func_call_id")
                .to_series()
            )
            logger.debug(
                f"Found {len(parent_func_call_ids)} invocations for the function: {parent}"
            )
            all_contained_events: list[
                list[FuncCallEvent | FuncCallExceptionEvent | VarChangeEvent]
            ] = []
            for parent_func_call_id in parent_func_call_ids:
                # get all contained events (can be any child function calls, var changes, etc.)
                contained_events = events_scanner(
                    trace=trace, func_call_id=parent_func_call_id
                )
                all_contained_events.append(contained_events)

            """Create hypothesis for each "kind" of contained events
                For FuncCall events, the "kind" is defined by the function name
                For VarChange events, the "kind" is defined by the variable type
            """
            hypothesis[parent] = {}
            hypothesis_should_use_causal_vars_for_negative_examples[parent] = {}
            for local_contained_events in all_contained_events:
                for event in local_contained_events:
                    high_level_event_type = typename(event)
                    target: str | tuple[str, ...] = (
                        event.func_name
                        if isinstance(event, (FuncCallEvent, FuncCallExceptionEvent))
                        else (event.var_id.var_type, event.attr_name)
                    )

                    if high_level_event_type not in hypothesis[parent]:
                        hypothesis[parent][high_level_event_type] = {}
                        hypothesis_should_use_causal_vars_for_negative_examples[parent][
                            high_level_event_type
                        ] = {}

                    if target not in hypothesis[parent][high_level_event_type]:
                        should_use_causal_vars_for_negative_examples = (
                            can_func_be_bound_method(
                                trace, parent, event.var_id.var_type, event.attr_name
                            )
                            if is_parent_a_bound_method
                            and isinstance(event, VarChangeEvent)
                            else False
                        )  # can_func_be_bound_method is super costly so we only call it when necessary
                        hypothesis_should_use_causal_vars_for_negative_examples[parent][
                            high_level_event_type
                        ][target] = should_use_causal_vars_for_negative_examples

                        group_names = (
                            {PARENT_GROUP_NAME, VAR_GROUP_NAME}
                            if isinstance(event, VarChangeEvent)
                            else {PARENT_GROUP_NAME}
                        )

                        params: list[Param] = [APIParam(parent)]
                        if isinstance(event, VarChangeEvent):
                            params.append(
                                VarTypeParam(event.var_id.var_type, event.attr_name)
                            )

                            # params.append(VarNameParam(event.var_id.var_type, event.attr_name)) # TODO: add a switch to enable using the attribute name as a parameter
                        elif isinstance(event, FuncCallEvent):
                            params.append(APIParam(event.func_name))

                        hypothesis[parent][high_level_event_type][target] = Hypothesis(
                            Invariant(
                                relation=APIContainRelation,
                                params=params,
                                precondition=None,
                                text_description=f"{parent} contains {target} of type {typename(event)}",
                                # text_description=f"{parent} (is_bound_method: {is_parent_a_bound_method}, should_use_causal_vars_for_negative_examples: {should_use_causal_vars_for_negative_examples}) contains {target} of type {typename(event)}",
                            ),
                            positive_examples=ExampleList(group_names),
                            negative_examples=ExampleList(
                                group_names
                                if should_use_causal_vars_for_negative_examples
                                else {PARENT_GROUP_NAME}
                            ),
                        )
                        """If the parent function is a bound method, we can leverage the dynamic analysis to find variables that are not changed but are causally related to the parent function.
                            These variables can be used as negative examples for the hypothesis.
                            If the parent function is not a bound method, we won't be able to find such variables that are not changed.
                            Another way to figure this out is to establish the causal relationship from the function input
                        """

            # scan the child_func_names for positive and negative examples
            for parent_func_call_id, local_contained_events in zip(
                parent_func_call_ids, all_contained_events
            ):
                touched: dict[str, set] = {}
                pre_record = trace.get_pre_func_call_record(parent_func_call_id)
                for event in local_contained_events:
                    high_level_event_type = typename(event)

                    target = (
                        event.func_name
                        if isinstance(event, (FuncCallEvent, FuncCallExceptionEvent))
                        else (event.var_id.var_type, event.attr_name)
                    )

                    assert target in hypothesis[parent][high_level_event_type]
                    example = Example()
                    example.add_group(PARENT_GROUP_NAME, [pre_record])
                    if isinstance(event, VarChangeEvent):
                        example.add_group(VAR_GROUP_NAME, event.get_traces())
                    hypothesis[parent][high_level_event_type][
                        target
                    ].positive_examples.add_example(example)

                    if high_level_event_type not in touched:
                        touched[high_level_event_type] = set()
                    touched[high_level_event_type].add(target)

                for high_level_event_type in hypothesis[parent]:
                    for target in hypothesis[parent][high_level_event_type]:
                        if hypothesis_should_use_causal_vars_for_negative_examples[
                            parent
                        ][high_level_event_type][target]:
                            assert high_level_event_type == typename(VarChangeEvent)
                            unchanged_var_ids = (
                                trace.get_var_ids_unchanged_but_causally_related(
                                    parent_func_call_id,
                                    target[0],
                                    target[1],
                                )
                            )
                            for var_id in unchanged_var_ids:
                                example = Example()
                                example.add_group(PARENT_GROUP_NAME, [pre_record])
                                example.add_group(
                                    VAR_GROUP_NAME,
                                    trace.get_var_raw_event_before_time(
                                        var_id, pre_record["time"]
                                    ),
                                )
                                hypothesis[parent][high_level_event_type][
                                    target
                                ].negative_examples.add_example(example)
                            continue

                        # if we haven't seen the target in the current trace, add this API invocation as a negative example
                        if (
                            high_level_event_type not in touched
                            or target not in touched[high_level_event_type]
                        ):
                            example = Example()
                            example.add_group(PARENT_GROUP_NAME, [pre_record])
                            hypothesis[parent][high_level_event_type][
                                target
                            ].negative_examples.add_example(example)

        pbar = tqdm(
            total=len(
                [
                    hypothesis[parent][event_type][target]
                    for parent in hypothesis
                    for event_type in hypothesis[parent]
                    for target in hypothesis[parent][event_type]
                ]
            ),
            desc="Infering preconditions",
        )
        all_invariants: list[Invariant] = []
        all_hypotheses = []
        for parent in hypothesis:
            for high_level_event_type in hypothesis[parent]:
                for target in hypothesis[parent][high_level_event_type]:
                    pbar.update(1)
                    h = hypothesis[parent][high_level_event_type][target]
                    logger.debug(
                        f"Starting the inference of precondition for the hypothesis: {h.invariant.text_description}"
                    )
                    found_precondition = find_precondition(h)
                    if (
                        found_precondition is not None
                    ):  # TODO: abstract this precondition inference part to a function
                        h.invariant.precondition = found_precondition
                        all_invariants.append(h.invariant)
                        all_hypotheses.append((h, f"{high_level_event_type}"))
                    else:
                        logger.debug(f"Precondition not found for the hypothesis: {h}")

        # sort the hypotheses for debugging purposes
        all_hypotheses.sort(key=lambda h: len(h[0].positive_examples), reverse=True)
        return all_invariants

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
    def static_check_all(trace: Trace, inv: Invariant) -> bool:
        """Check the invariant on the trace

        NOTE: this function takes one invariant at a time, and checks if the invariant holds on the trace. However, if multiple invariants targets the same parent function,
        we should be batching the checks for the same parent function to avoid redundant computation (mainly from `events_scanner`). ### TODO ITEM ###
        """

        assert (
            len(inv.params) == 2
        ), "Expected 2 parameters for APIContainRelation, one for the parent function name, and one for the child event name"

        parent_param, child_param = inv.params[0], inv.params[1]
        assert isinstance(
            parent_param, APIParam
        ), "Expected the first parameter to be an APIParam"
        assert isinstance(
            child_param, (APIParam, VarTypeParam)
        ), "Expected the second parameter to be an APIParam or VarTypeParam (VarNameParam not supported yet)"

        logger = logging.getLogger(__name__)

        parent_func_name = parent_param.api_full_name
        preconditions = inv.precondition
        assert preconditions is not None, "Expected the precondition to be set"

        parent_func_call_ids = (
            trace.events.filter((pl.col("function") == parent_func_name))
            .select("func_call_id")
            .unique()
            .to_series()
        )
        # should be sorted by time to reflect timeliness

        parent_preconditions = preconditions.get_group(PARENT_GROUP_NAME)
        skip_parent_check = (
            len(parent_preconditions) == 0
        )  # this is no precondition for the parent function (not even unconditional ones)

        skip_var_unchanged_check = (
            VAR_GROUP_NAME not in preconditions.get_group_names()
            or len(preconditions.get_group(VAR_GROUP_NAME)) == 0
            or preconditions.is_group_unconditional(VAR_GROUP_NAME)
        )
        logger.debug(
            f"skip_parent_check {skip_parent_check}, skip_var_unchanged_check: {skip_var_unchanged_check}"
        )
        if not skip_var_unchanged_check:
            assert isinstance(
                child_param, VarTypeParam
            ), "Expected the child parameter to be a VarTypeParam"
            var_preconditions = preconditions.get_group(VAR_GROUP_NAME)
            if not can_func_be_bound_method(
                trace, parent_func_name, child_param.var_type, child_param.attr_name
            ):
                logger.warning(
                    """The invariant includes a precondition for the variables that are changed/unchanged, to enforce this precondition, you should be running the trace collector with the --scan_proxy_in_args flag to collect the trace.
Defaulting to skip the var preconditon check for now.
                    """
                )
                skip_var_unchanged_check = True

        for parent_func_call_id in parent_func_call_ids:
            # check for parent precondition
            parent_pre_record = trace.get_pre_func_call_record(parent_func_call_id)
            if not skip_parent_check:
                if not any(
                    precond.verify([parent_pre_record])
                    for precond in parent_preconditions
                ):
                    # if no precondition is met, continue without checking the child events
                    logger.debug(
                        f"Precondition not met for the parent function {parent_func_name} at {parent_pre_record}, skipping the check for the child events on inv: {inv.text_description}"
                    )
                    continue

            var_unchanged_check_passed = True
            found_expected_child_event = False
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
                    if any(
                        precond.verify(unchanged_var_state)
                        for precond in var_preconditions
                    ):
                        logger.error(
                            f"INV CHECK ERROR: Precondition met for the unchanged vars for the parent function: {parent_func_name} at {parent_func_call_id}: {parent_pre_record['time']} at {trace.get_time_precentage(parent_pre_record['time'])}"
                        )
                        var_unchanged_check_passed = False
                        break

                # var_change_events = [event for event in trace.query_var_changes_within_func_call(parent_func_call_id) if event.var_id.var_type == child_param.var_type and event.attr_name == child_param.attr_name]
                # TODO: enable precondition refinement by checking whether there are any var change events that don't satisfy the precondition
            else:
                # no preconditions for the child events in the current implementation

                # get all contained events (can be any child function calls, var changes, etc.)
                contained_events = events_scanner(
                    trace=trace,
                    func_call_id=parent_func_call_id,
                )

                # check if the expected child event is in the contained events
                if isinstance(child_param, (APIParam, VarTypeParam)):
                    for event in contained_events:
                        if isinstance(child_param, APIParam) and isinstance(
                            event, FuncCallEvent
                        ):
                            if event.func_name == child_param.api_full_name:
                                # found expected child event, break the loop as the invariant holds and we can check next parent function invocation
                                logger.debug(
                                    f"Found the expected child event: {child_param.api_full_name} in the contained events for the parent function: {parent_func_name} at {parent_func_call_id}: {parent_pre_record['time']} at {trace.get_time_precentage(parent_pre_record['time'])}"
                                )
                                found_expected_child_event = True
                                # no precondition refinement for APIParam child events yet as the infer algorithm doesn't infer preconditions for child API events
                                break

                        if isinstance(child_param, VarTypeParam) and isinstance(
                            event, VarChangeEvent
                        ):
                            if (
                                event.var_id.var_type == child_param.var_type
                                and event.attr_name == child_param.attr_name
                            ):
                                # found expected child event, break the loop as the invariant holds and we can check next parent function invocation
                                logger.debug(
                                    f"Found the expected child event: {child_param.var_type} in the contained events for the parent function: {parent_func_name} at {parent_func_call_id}: {parent_pre_record['time']} at {trace.get_time_precentage(parent_pre_record['time'])}"
                                )
                                found_expected_child_event = True
                                # no precondition refinement for VarTypeParam child events yet as the infer algorithm doesn't infer preconditions for child Var events
                                break
                else:
                    raise ValueError(
                        f"Unsupported parameter type for child_param: {child_param}"
                    )

            if (skip_var_unchanged_check and not found_expected_child_event) or (
                not skip_var_unchanged_check and not var_unchanged_check_passed
            ):
                logger.error(
                    f"INV CHECK ERROR: Expected child event not found in the contained events for the parent function: {parent_func_name} at {parent_func_call_id}: {parent_pre_record['time']} at {trace.get_time_precentage(parent_pre_record['time'])}"
                )
                return False

        return True
