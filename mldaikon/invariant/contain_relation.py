import logging

import polars as pl

from mldaikon.invariant.base_cls import Hypothesis, Invariant, Relation
from mldaikon.ml_daikon_trace import Trace


def events_scanner(trace_df: pl.DataFrame, parent_func_name: str) -> set[str] | None:
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
    logger = logging.getLogger(__name__)

    # get the first record of the trace
    first_record = trace_df.row(index=0, named=True)
    if (
        first_record["function"] != parent_func_name
        or first_record["type"] != "function_call (pre)"
    ):
        logger.error(
            f"The first record of the trace is not the pre-event of the parent_func_name: {parent_func_name}"
        )
        return None

    # adding a index column to the trace_df as polars does not have a built-in function to get the index of a row

    # get the process_id and thread_id of the first record
    process_id = first_record["process_id"]
    thread_id = first_record["thread_id"]
    func_call_id = first_record["func_call_id"]

    # get the post-event of the parent_func_name, according to func_call_id
    post_idx = trace_df.select(
        pl.arg_where(
            (
                pl.col("type").is_in(
                    ["function_call (post)", "function_call (post) (exception)"]
                )
            )
            & (pl.col("function") == parent_func_name)
            & (pl.col("func_call_id") == func_call_id)
        )
    ).to_series()

    num_records = len(post_idx)
    if num_records == 0:
        logger.error(
            f"No post-event found for the parent_func_name: {parent_func_name}"
        )
        return None

    # This assert no longer holds as we do not use UUIDs for func_call_id
    # assert (
    #     num_records == 1
    # ), "There should be only one post-event for the parent_func_name"
    post_idx = post_idx[0]  # the first post-event
    post_event = trace_df.row(index=post_idx, named=True)
    assert (
        post_event["process_id"] == process_id and post_event["thread_id"] == thread_id
    ), "The post-event of the parent_func_name should be on the same thread and process"

    # get the events that happened within the pre and post events of the parent_func_name
    func_names = (
        trace_df.slice(0, post_idx)
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
        hypothesis: dict[str, dict[str, Hypothesis]] = {}
        func_names = (
            trace.events.select("function").drop_nulls().unique().to_series().to_list()
        )

        for parent in func_names:
            logger.debug(f"Starting the analysis for the parent function: {parent}")
            # get all parent pre event indexes
            parent_pre_idx = trace.events.select(
                pl.arg_where(
                    (pl.col("type") == "function_call (pre)")
                    & (pl.col("function") == parent)
                )
            ).to_series()
            logger.debug(
                f"Found {len(parent_pre_idx)} invocations for the function: {parent}"
            )
            all_child_func_names: list[set[str]] = []
            for idx in parent_pre_idx:
                # get all child post events
                child_func_names = events_scanner(
                    trace_df=trace.events.slice(idx, None), parent_func_name=parent
                )
                if child_func_names is None:
                    raise ValueError(
                        "The events_scanner should return a set of function names during inference."
                    )
                all_child_func_names.append(child_func_names)

            unique_seen_child_func_names = set(
                [item for sublist in all_child_func_names for item in sublist]
            )
            # create hypothesis for each child_func_name
            hypothesis[parent] = {}
            logger.debug(
                f"Creating {len(unique_seen_child_func_names)} hypotheses for the parent function: {parent}"
            )
            for child_func_name in unique_seen_child_func_names:
                param_selectors = [
                    child_func_name,
                    lambda trace_df: events_scanner(
                        trace_df=trace_df, parent_func_name=parent
                    ),
                ]

                hypothesis[parent][child_func_name] = Hypothesis(
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
                    )  # NOTE: for this specific analysis, the examples we use is just the pre-event, if in other cases we need to use other parts of the trace, we may collect them in the hypothesis construction phase

                    # assumption: the precondition can only resides in the parent pre-event
                    if expected_child_func in child_func_names:
                        hypothesis[parent][
                            expected_child_func
                        ].positive_examples.append(Trace([parent_pre_event]))
                    else:
                        hypothesis[parent][
                            expected_child_func
                        ].negative_examples.append(Trace([parent_pre_event]))

        all_invariants: list[Invariant] = []
        all_hypotheses = []
        for p, child_hypotheses in hypothesis.items():
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
