import logging

import polars as pl
from tqdm import tqdm

from mldaikon.instrumentor.tracer import TraceLineType
from mldaikon.invariant.base_cls import (
    Example,
    ExampleList,
    Hypothesis,
    Invariant,
    Relation,
)
from mldaikon.invariant.precondition import find_precondition
from mldaikon.trace.trace import Trace
from mldaikon.trace.types import FuncCallEvent, FuncCallExceptionEvent, VarChangeEvent
from mldaikon.utils import typename


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
    trace: Trace, func_pre_call_idx: int, parent_func_name: str
) -> list[FuncCallEvent | FuncCallExceptionEvent | VarChangeEvent]:
    """Scan the trace, and return the first set of events that happened within the pre and post
    events of the parent_func_name.

    Args:
        trace: Trace
            - the trace to be scanned
        parent_func_name: str
            - the parent function name

    Note that if the first record of the trace is not the pre-event of the parent_func_name, or
    the pre-event has no post-event, the function will return None. Otherwise, it will return the
    set of events that happened within the first post-event of the parent_func_name on the same
    thread and process.
    """
    pre_call_record = trace.events.row(index=func_pre_call_idx, named=True)
    assert (
        trace.events.row(index=func_pre_call_idx, named=True)["function"]
        == parent_func_name
    ), "The func_pre_call_idx should be the pre-event of the parent function"

    func_post_call_idx = trace.get_func_post_call_idx(func_pre_call_idx)
    post_call_record = trace.events.row(index=func_post_call_idx, named=True)

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

    def __init__(self):
        pass

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
            parent_pre_call_indices = trace.events.select(
                pl.arg_where(
                    (pl.col("type") == TraceLineType.FUNC_CALL_PRE)
                    & (pl.col("function") == parent)
                )
            ).to_series()
            logger.debug(
                f"Found {len(parent_pre_call_indices)} invocations for the function: {parent}"
            )
            all_contained_events: list[
                list[FuncCallEvent | FuncCallExceptionEvent | VarChangeEvent]
            ] = []
            for idx in parent_pre_call_indices:
                # get all contained events (can be any child function calls, var changes, etc.)
                contained_events = events_scanner(
                    trace=trace, func_pre_call_idx=idx, parent_func_name=parent
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

                    param_selectors = [
                        target,  # TODO: refactor the relation evaluate logic to be specific-trace-agnotic
                        lambda func_pre_call_idx: events_scanner(
                            trace=trace,
                            func_pre_call_idx=func_pre_call_idx,
                            parent_func_name=parent,
                        ),
                    ]
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
                            {"parent_func_call_pre", "var_events"}
                            if isinstance(event, VarChangeEvent)
                            else {"parent_func_call_pre"}
                        )
                        hypothesis[parent][high_level_event_type][target] = Hypothesis(
                            Invariant(
                                relation=APIContainRelation(),
                                param_selectors=param_selectors,
                                precondition=None,
                                text_description=f"{parent} (is_bound_method: {is_parent_a_bound_method}, should_use_causal_vars_for_negative_examples: {should_use_causal_vars_for_negative_examples}) contains {target} of type {typename(event)}",
                            ),
                            positive_examples=ExampleList(group_names),
                            negative_examples=ExampleList(
                                group_names
                                if should_use_causal_vars_for_negative_examples
                                else {"parent_func_call_pre"}
                            ),
                        )
                        """If the parent function is a bound method, we can leverage the dynamic analysis to find variables that are not changed but are causally related to the parent function.
                            These variables can be used as negative examples for the hypothesis.
                            If the parent function is not a bound method, we won't be able to find such variables that are not changed.
                            Another way to figure this out is to establish the causal relationship from the function input
                        """

            # scan the child_func_names for positive and negative examples
            for idx, local_contained_events in zip(
                parent_pre_call_indices, all_contained_events
            ):
                touched: dict[str, set] = {}
                pre_record = trace.events.row(index=idx, named=True)
                for event in local_contained_events:
                    high_level_event_type = typename(event)

                    target = (
                        event.func_name
                        if isinstance(event, (FuncCallEvent, FuncCallExceptionEvent))
                        else (event.var_id.var_type, event.attr_name)
                    )

                    assert target in hypothesis[parent][high_level_event_type]
                    example = Example()
                    example.add_group("parent_func_call_pre", [pre_record])
                    if isinstance(event, VarChangeEvent):
                        example.add_group("var_events", event.get_traces())
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
                            pre_record = trace.events.row(index=idx, named=True)
                            unchanged_var_ids = (
                                trace.get_var_ids_unchanged_but_causally_related(
                                    pre_record["func_call_id"],
                                    target[0],
                                    target[1],
                                )
                            )
                            for var_id in unchanged_var_ids:
                                example = Example()
                                example.add_group("parent_func_call_pre", [pre_record])
                                example.add_group(
                                    "var_events",
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
                            example.add_group("parent_func_call_pre", [pre_record])
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
                    h.invariant.precondition = find_precondition(h)
                    if (
                        h.invariant.precondition is not None
                    ):  # TODO: abstract this precondition inference part to a function
                        all_invariants.append(h.invariant)
                        all_hypotheses.append((h, f"{high_level_event_type}"))
                    else:
                        logger.debug(f"Precondition not found for the hypothesis: {h}")

        # sort the hypotheses for debugging purposes
        all_hypotheses.sort(key=lambda h: len(h[0].positive_examples), reverse=True)
        for h, desc in all_hypotheses:
            print(desc, h._print_debug())

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
