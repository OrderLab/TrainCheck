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
        hypothesis: dict[str, dict[str, dict[str, Hypothesis]]] = {}
        func_names = trace.get_func_names()
        if len(func_names) == 0:
            logger.warning(
                "No function calls found in the trace, skipping the analysis"
            )
            return []

        for parent in tqdm(
            func_names, desc="Scanning through function calls to generate hypotheses"
        ):
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
            all_contained_events = []
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
            for local_contained_events in all_contained_events:
                for event in local_contained_events:
                    target = (
                        event.func_name
                        if isinstance(event, (FuncCallEvent, FuncCallExceptionEvent))
                        else f"{event.var_id.var_type}.{event.attr_name}"
                    )
                    param_selectors = [
                        target,  # TODO: refactor the relation evaluate logic to be specific-trace-agnotic
                        lambda func_pre_call_idx: events_scanner(
                            trace=trace,
                            func_pre_call_idx=func_pre_call_idx,
                            parent_func_name=parent,
                        ),
                    ]
                    if typename(event) not in hypothesis[parent]:
                        hypothesis[parent][typename(event)] = {}

                    if target not in hypothesis[parent][typename(event)]:
                        group_names = (
                            {"parent_func_call_pre", "child_events"}
                            if isinstance(event, VarChangeEvent)
                            else {"parent_func_call_pre"}
                        )
                        hypothesis[parent][typename(event)][target] = Hypothesis(
                            Invariant(
                                relation=APIContainRelation(),
                                param_selectors=param_selectors,
                                precondition=None,
                                text_description=f"{parent} contains {target} of type {typename(event)}",
                            ),
                            positive_examples=ExampleList(group_names),
                            negative_examples=ExampleList({"parent_func_call_pre"}),
                        )

            # scan the child_func_names for positive and negative examples
            for idx, local_contained_events in zip(
                parent_pre_call_indices, all_contained_events
            ):
                touched: dict[str, set] = {}
                pre_record = trace.events.row(index=idx, named=True)
                for high_level_event_type in hypothesis[parent]:
                    for event in local_contained_events:
                        target = (
                            event.func_name
                            if isinstance(
                                event, (FuncCallEvent, FuncCallExceptionEvent)
                            )
                            else f"{event.var_id.var_type}.{event.attr_name}"
                        )
                        if target in hypothesis[parent][high_level_event_type]:
                            example = Example()
                            example.add_group("parent_func_call_pre", [pre_record])
                            if isinstance(event, VarChangeEvent):
                                example.add_group("child_events", event.get_traces())
                            hypothesis[parent][high_level_event_type][
                                target
                            ].positive_examples.add_example(example)

                        if high_level_event_type not in touched:
                            touched[high_level_event_type] = set()
                        touched[high_level_event_type].add(target)

                for high_level_event_type in hypothesis[parent]:
                    for target in hypothesis[parent][high_level_event_type]:
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

        logging.debug("Starting the inference of preconditions for the hypothesis")

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
