import polars as pl
import logging
import tqdm

from src_new.invariant.base_cls import Relation, Invariant, Hypothesis
from src_new.trace import Trace

def events_scanner(trace_df: pl.DataFrame, parent_func_name: str) -> set[str]|None:
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
    if first_record["function"] != parent_func_name or first_record["type"] != "function_call (pre)":
        logger.error(f"The first record of the trace is not the pre-event of the parent_func_name: {parent_func_name}")
        return None
    
    # adding a index column to the trace_df as polars does not have a built-in function to get the index of a row

    # get the process_id and thread_id of the first record
    process_id = first_record["process_id"]
    thread_id = first_record["thread_id"]
    uuid = first_record["uuid"]

    # get the post-event of the parent_func_name, according to uuid
    post_idx = trace_df.select(
                    pl.arg_where(pl.col("uuid")==first_record["uuid"])
                    ).to_series()

    num_records = len(post_idx) - 1
    if num_records == 0:
        logger.error(f"No post-event found for the parent_func_name: {parent_func_name}")
        return None
    assert num_records == 1, "There should be only one post-event for the parent_func_name"

    post_idx = post_idx[1]
    post_event = trace_df.row(index=post_idx, named=True)
    assert post_event["process_id"] == process_id and post_event["thread_id"] == thread_id, "The post-event of the parent_func_name should be on the same thread and process"

    # get the events that happened within the pre and post events of the parent_func_name
    func_names = trace_df.slice(0, post_idx).filter((pl.col("process_id") == process_id) 
                                        & (pl.col("thread_id") == thread_id) 
                                        ).select("function").to_series().unique().to_list()
    
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
        func_names = trace.events["function"].drop_nulls().drop_nans().unique().to_list()

        # first pass, create hypothesis
        logger.debug(f"Found {len(func_names)} unique function names in the trace., creating {len(func_names)**2} hypotheses.")
        for func_name in func_names:
            for child_func_name in func_names:
                if func_name == child_func_name:
                    continue

                param_selectors = [
                    child_func_name,
                    lambda trace_df: events_scanner(trace_df=trace_df, parent_func_name=func_name)
                ]
                
                if func_name not in hypothesis:
                    hypothesis[func_name] = {}

                hypothesis[func_name][child_func_name] = Hypothesis(
                    Invariant(
                    relation=APIContainRelation(),
                    params=param_selectors,
                    precondition=None
                    ),
                    positive_examples=[],
                    negative_examples=[]
                )
        logger.debug(f"Created {len(func_names)**2} hypotheses.")
        # second pass, evaluate the hypothesis by finding positive and negative examples
        """Naive implementation would be to iterate over all hypotheses and for each hypothesis,
        we run a pass over the trace to find negative and positive examples.
        """
        logger.debug(f"Starting the second pass to evaluate hypotheses.")
        for parent in hypothesis:
            logger.debug(f"Starting the second pass for the parent function: {parent}, verifying {len(hypothesis[parent])} hypotheses.")
            # get all parent pre event indexes
            parent_pre_idx = trace.events.select(
                pl.arg_where((pl.col("type")=='function_call (pre)')
                             & (pl.col("function")==parent))
                             ).to_series()
            logger.debug(f"Found {len(parent_pre_idx)} pre-events for the parent function: {parent}")
            for idx in parent_pre_idx:
                # get all child post events
                child_func_names = events_scanner(trace_df=trace.events.slice(idx, None), parent_func_name=parent)

                # prepare positive and negative examples
                parent_pre_event = trace.events.row(index=idx, named=True)
                parent_post_idx = trace.events.select(
                    pl.arg_where(pl.col("uuid")==parent_pre_event["uuid"])
                    ).to_series()[1]
                events = trace.events.slice(idx, length=parent_post_idx-idx).filter(
                    pl.col("thread_id")==parent_pre_event["thread_id"],
                    pl.col("process_id")==parent_pre_event["process_id"]
                )

            for child in hypothesis[parent]:
                if child in child_func_names:
                    hypothesis[parent][child].positive_examples.append(Trace(events))
                else:
                    hypothesis[parent][child].negative_examples.append(Trace(events))

        # third pass, evaluate the hypothesis
        return hypothesis

    def evaluate(self, value_group: list) -> bool:
        """Evaluate the relation based on the given value group.
        
        Args:
            value_group: list
                - [0]: str - the function name to be contained
                - [1]: set - a set of function names (child API calls)

        TODO: 
            - extend [0] to not only function calls but also other types of events, such as data updates, etc.
        """
        

        assert len(value_group) == 2, "Expected 2 arguments for APIContainRelation, #1 the function name to be contained, #2 a set of function names"
        assert isinstance(value_group[0], str), "Expected the first argument to be a string"
        assert isinstance(value_group[1], set), "Expected the second argument to be a set"

        expected_child_func = value_group[0]
        seen_child_funcs = value_group[1]

        return expected_child_func in seen_child_funcs
