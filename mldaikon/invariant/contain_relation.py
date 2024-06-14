import logging

import polars as pl

from mldaikon.instrumentor.tracer import TraceLineType
from mldaikon.invariant.base_cls import Hypothesis, Invariant, Relation
from mldaikon.invariant.precondition import find_precondition
from mldaikon.trace.trace import Trace, VarChange


def events_scanner(
    trace: Trace, func_pre_call_idx: int, parent_func_name: str
) -> set[str] | None:
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

    process_id = pre_call_record["process_id"]
    thread_id = pre_call_record["thread_id"]

    # get the events that happened within the pre and post events of the parent_func_name
    func_names = (
        trace.events.slice(
            func_pre_call_idx + 1, length=func_post_call_idx - func_pre_call_idx - 1
        )
        .filter(
            (pl.col("process_id") == process_id) & (pl.col("thread_id") == thread_id)
        )
        .select("function")
        .to_series()
        .unique()
        .to_list()
    )

    """
    TODO: currently, we only return the set of function names, but we should return the set of events instead as data changes are also important
        Fix this after we formalize the event types & variable changes event format 
    """
    # logger.debug(f"Found {len(func_names)} events between the pre and post events of the parent_func_name: {parent_func_name}")
    return set(func_names)


def var_change_scanner(
    trace: Trace, func_pre_call_idx: int, parent_func_name: str
) -> set[VarChange] | None:
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

    var_changes = trace.query_var_changes_within_time_and_process(
        (pre_call_record["time"], post_call_record["time"]), process_id
    )

    return set(var_changes)


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
        hypothesis_api: dict[str, dict[str, Hypothesis]] = {}
        hypothesis_var: dict[str, dict[str, Hypothesis]] = {}
        func_names = trace.get_func_names()
        if len(func_names) == 0:
            logger.warning(
                "No function calls found in the trace, skipping the analysis"
            )
            return []

        # sort the function names for to put adam.step at the front
        func_names.sort(key=lambda x: "adam.step" in x, reverse=True)

        for parent in func_names:
            logger.debug(f"Starting the analysis for the parent function: {parent}")
            # get all parent pre event indexes
            parent_pre_idx = trace.events.select(
                pl.arg_where(
                    (pl.col("type") == TraceLineType.FUNC_CALL_PRE)
                    & (pl.col("function") == parent)
                )
            ).to_series()
            logger.debug(
                f"Found {len(parent_pre_idx)} invocations for the function: {parent}"
            )
            all_child_func_names: list[set[str]] = []
            all_var_changes = []
            for idx in parent_pre_idx:
                # get all child post events
                child_func_names = events_scanner(
                    trace=trace, func_pre_call_idx=idx, parent_func_name=parent
                )
                if child_func_names is None:
                    raise ValueError(
                        "The events_scanner should return a set of function names during inference."
                    )
                all_child_func_names.append(child_func_names)

                # get all variable changes
                var_changes = var_change_scanner(
                    trace=trace, func_pre_call_idx=idx, parent_func_name=parent
                )
                if var_changes is None:
                    raise ValueError(
                        "The var_change_scanner should return a set of variable changes during inference."
                    )
                all_var_changes.append(var_changes)

            # get the unique child_func_names

            unique_seen_child_func_names = set(
                item for sublist in all_child_func_names for item in sublist
            )
            # create hypothesis_api for each child_func_name
            hypothesis_api[parent] = {}
            hypothesis_var[parent] = {}
            logger.debug(
                f"Creating {len(unique_seen_child_func_names)} hypotheses for the parent function: {parent}"
            )
            for child_func_name in unique_seen_child_func_names:
                param_selectors = [
                    child_func_name,
                    lambda func_pre_call_idx: events_scanner(
                        trace=trace,
                        func_pre_call_idx=func_pre_call_idx,
                        parent_func_name=parent,
                    ),
                ]

                hypothesis_api[parent][child_func_name] = Hypothesis(
                    Invariant(
                        relation=APIContainRelation(),
                        param_selectors=param_selectors,
                        precondition=None,
                    ),
                    positive_examples=[],
                    negative_examples=[],
                )

            # scan the child_func_names for positive and negative examples
            for idx, child_func_names in zip(parent_pre_idx, all_child_func_names):
                for expected_child_func in unique_seen_child_func_names:
                    parent_pre_event = trace.events.row(
                        index=idx, named=True
                    )  # NOTE: for this specific analysis, the examples we use is just the pre-event, if in other cases we need to use other parts of the trace, we may collect them in the hypothesis_api construction phase

                    # assumption: the precondition can only resides in the parent pre-event
                    if expected_child_func in child_func_names:
                        hypothesis_api[parent][
                            expected_child_func
                        ].positive_examples.append([parent_pre_event])
                    else:
                        hypothesis_api[parent][
                            expected_child_func
                        ].negative_examples.append([parent_pre_event])

        ## precondition inference
        for p, child_hypotheses in hypothesis_api.items():
            for k, h in child_hypotheses.items():
                h.invariant.precondition = find_precondition(h)

        all_invariants: list[Invariant] = []
        all_hypotheses = []
        for p, child_hypotheses in hypothesis_api.items():
            for k, h in child_hypotheses.items():
                all_invariants.append(h.invariant)
                all_hypotheses.append((h, f"{p} contains {k}"))

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
