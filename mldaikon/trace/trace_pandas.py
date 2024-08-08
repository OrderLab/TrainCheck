import logging
import re

import polars as pl
import modin.pandas as pd
from tqdm import tqdm
from modin.pandas import json_normalize

from mldaikon.config import config
from mldaikon.instrumentor.tracer import TraceLineType
from mldaikon.trace.types import (
    AttrState,
    FuncCallEvent,
    FuncCallExceptionEvent,
    Liveness,
    VarChangeEvent,
    VarInstId,
)

logger = logging.getLogger(__name__)

# TODO: formalize the trace schema for efficient polars processing
def test_dump(df):
    df = df.sort_values(by='func_call_id')
    df.to_json('pandas_trace.json', orient='records', lines=True)

def get_attr_name(col_name: str) -> str:
    if config.VAR_ATTR_PREFIX not in col_name:
        raise ValueError(f"{col_name} does not contain the tracker_var_field_prefix.")
    return col_name[len(config.VAR_ATTR_PREFIX) :]


class Trace:
    def __init__(self, events, truncate_incomplete_func_calls=True):
        self.events = events
        self.var_ids = None
        self.var_insts = None
        self.var_changes = None

        if isinstance(events, list) and all(
            [isinstance(e, pd.DataFrame) for e in events]
        ):
            self.events = pd.concat(events, ignore_index=True)
        elif isinstance(events, list) and all([isinstance(e, dict) for e in events]):
            self.events = pd.DataFrame(events)
            self.events = json_normalize(self.events.to_dict(orient='records'), sep=".")

        assert isinstance(
            self.events, pd.DataFrame
        ), "events should be a DataFrame, list of DataFrames, or a list of dictionaries."

        assert (
            "time" in self.events.columns
        ), "time column not found in the events, cannot infer invariants as the analysis does a lot of queries based on time."

        logger.warning(
            "Infer engine won't sort the events by time anymore as sorting is costly for large traces. Please make sure every separate file is sorted by time, and traces belong to the same process should be in the same file. For variable traces, we will still be sorting them by time so no need to worry about that."
        )
        # test_dump(self.events)
        if truncate_incomplete_func_calls:
            logger.info("Truncating incomplete trailing function calls from the trace.")
            self._rm_incomplete_trailing_func_calls()
            logger.info(
                "Done truncating incomplete trailing function calls from the trace."
            )



    def _rm_incomplete_trailing_func_calls(self):
        """Remove incomplete trailing function calls from the trace. For why incomplete function calls exist, refer to https://github.com/OrderLab/ml-daikon/issues/31

        This function would group the function calls by `func_call_id` which is unique for each function call. Thus, each `func_call_id` should
        exactly correspond to two trace records (one pre-call and one post-call/exception). If there is only one record for a `func_call_id`,
        the function is "incomplete" and should be handled with care.

        For each incomplete function call, there will be three cases:
        1. The function call is the outermost function call of the process: In this case, we will treat it as a complete function call and ignore it.
        2. The function call is not the outermost function call of the process,
           2.1 If the function call is on a sub-thread and close enough to the outermost function call post event (config.INCOMPLETE_FUNC_CALL_SECONDS_TO_OUTERMOST_POST), we will remove it.
           2.2 If the function call is on the main-thread or on a sub-thread but not close enough to the outermost function call post event, we will raise an error.

        Raises:
            ValueError: If an incomplete function call is not close enough to the outermost function call post event.
            AssertionError: If the incomplete function call is not on a different thread than the outermost function call.
        """
        logger = logging.getLogger(__name__)

        if "function" not in self.events.columns:
            logger.warning(
                "function column not found in the events, no function related invariants will be extracted."
            )
            return
        
        func_call_groups = self.events.groupby("func_call_id").size().reset_index(name='count')

        incomplete_func_call_ids = func_call_groups[func_call_groups["count"] == 1]["func_call_id"]

        # print(incomplete_func_call_ids)
        incomplete_func_call_records = self.events[self.events["func_call_id"].isin(incomplete_func_call_ids)]
        # test_dump(incomplete_func_call_records)
        for _, row in incomplete_func_call_records.iterrows():  
            assert (
                row["type"] == TraceLineType.FUNC_CALL_PRE
            ), "Incomplete function call is not a pre-call event."
            logger.warning(f"Incomplete function call detected: {row}")
            process_id = row["process_id"]
            thread_id = row["thread_id"]

            filtered_events_pre = self.events[
                (self.events["type"] == TraceLineType.FUNC_CALL_PRE) &
                (self.events["process_id"] == process_id) &
                (self.events["time"] == self.get_start_time(process_id))
            ]

            if not filtered_events_pre.empty:
                outermost_func_call_pre = filtered_events_pre.iloc[0]
            else:
                outermost_func_call_pre = None

            filtered_events_post = self.events[
                (self.events["type"] == TraceLineType.FUNC_CALL_POST) &
                (self.events["func_call_id"] == outermost_func_call_pre["func_call_id"])
            ]

            outermost_incomplete = False
            if not filtered_events_post.empty:
                outermost_func_call_post = filtered_events_post.iloc[0]
            else:
                outermost_incomplete = True
                outermost_func_call_post = None


            if row["func_call_id"] == outermost_func_call_pre["func_call_id"]:
                logger.warning(
                    f"The outermost function call is incomplete: {outermost_func_call_pre['function']} with id {outermost_func_call_pre['func_call_id']}. Will treat it as a complete function call."
                )
                # print(f"The outermost function call is incomplete: {outermost_func_call_pre['function']} with id {outermost_func_call_pre['func_call_id']}. Will treat it as a complete function call.")
                continue

            if not outermost_incomplete:
                assert (
                    thread_id != outermost_func_call_pre["thread_id"]
                ), f"Incomplete function call (func_call_id: {row['func_call_id']}) (name: {row["function"]}) is not on a different thread than outermost function (func_call_id: {outermost_func_call_pre['func_call_id']}) (name: {outermost_func_call_pre["function"]}) on process {process_id}. Please Investigate."

                if (
                    row["time"]
                    > outermost_func_call_post["time"]
                    - config.INCOMPLETE_FUNC_CALL_SECONDS_TO_OUTERMOST_POST
                ):
                    logger.warning(f"Removing incomplete function call: {row}")
                    self.events = self.events[self.events["func_call_id"] != row["func_call_id"]]
                else:
                    raise ValueError(
                        f"Incomplete function call is not close enough to the outermost function call post event: {row}"
                    )
                
            else:
                assert row["time"] == self.get_end_time(
                    process_id, thread_id
                ), f"Incomplete function call is not the last event for the process {process_id} and thread {thread_id}."

                logger.warning(
                    f"Removing incomplete function call: {row} as the outermost function call is also incomplete."
                )

                self.events = self.events[self.events["func_call_id"] != row["func_call_id"]]
                logger.warning(f"Removing incomplete function call: {row}")

        

        # test_dump(self.events)

    def get_start_time(self, process_id=None, thread_id=None) -> int:
        """Get the start time of the trace. If process_id or thread_id is provided, the start time of the specific process or thread will be returned."""
        if process_id is not None and thread_id is not None:
            return self.events[
                (self.events["process_id"] == process_id)
                & (self.events["thread_id"] == thread_id)
            ]["time"].min()

        if process_id is not None:
            return self.events[self.events["process_id"] == process_id]["time"].min()

        if thread_id is not None:
            return self.events[self.events["thread_id"] == thread_id]["time"].min()

        return self.events["time"].min()

    def get_end_time(self, process_id=None, thread_id=None) -> int:
        """Get the start time of the trace. If process_id or thread_id is provided, the start time of the specific process or thread will be returned."""

        if process_id is not None and thread_id is not None:
            return self.events[
                (self.events["process_id"] == process_id)
                & (self.events["thread_id"] == thread_id)
            ]["time"].max()

        if process_id is not None:
            return self.events[self.events["process_id"] == process_id]["time"].max()

        if thread_id is not None:
            return self.events[self.events["thread_id"] == thread_id]["time"].max()

        return self.events["time"].max()
            
            
    def get_func_names(self) -> list[str]:
        """Find all function names from the trace."""
        if "function" not in self.events.columns:
            logger.warning(
                "function column not found in the events, no function related invariants will be extracted."
            )
            return []
        return self.events["function"].dropna().unique().tolist()
    

    def get_func_is_bound_method(self, func_name: str) -> bool:
        """Check if a function is bound to a class (i.e. method of a object).

        Args:
            func_name (str): The name of the function.

        Returns:
            bool: True if the function is bound to a class, False otherwise.

        Raises:
            AssertionError: If the boundness information is not found for the function.

        A function is bound to a class if *all* the function calls of the function are made on an object.
        """
        if "function" not in self.events.columns:
            logger.warning(
                "function column not found in the events, no function related invariants will be extracted."
            )
            return False

        is_bound_method = (
            self.events[self.events["function"] == func_name]["is_bound_method"]
            .dropna()
            .tolist()
        )

        assert (
            None not in is_bound_method
        ), f"Boundness information not found for {func_name}"
        assert all(
            is_bound_method[0] == is_bound_method[i]
            for i in range(1, len(is_bound_method))
        ), f"Boundness information is not consistent for {func_name}"
        return is_bound_method[0]


def read_trace_file(
    file_path: str | list[str], truncate_incomplete_func_calls=True
) -> Trace:
    """Reads the trace file and returns the trace instance."""
    if isinstance(file_path, list):
        dfs = []
        for f in file_path:
            try:
                dfs.append(pd.read_json(f, lines=True))
                logger.info(f"Done reading {f}")
            except Exception as e:
                logger.error(f"Failed to read {f} due to {e}. aborting")
                raise e
        logger.info("Concatenating the DataFrames")
        events = pd.concat(dfs, ignore_index=True)
        logger.info("Done concatenating the DataFrames")
    else:
        events = pd.read_json(file_path, lines=True)
        logger.info(f"Done reading {file_path}")
    
    
    events = json_normalize(events.to_dict(orient='records'), sep=".")
    # TODO: normalize abandon the empty dicts, check whether it matters

    # events.to_json('pandas_events.json', orient='records', lines=True)

    return Trace(
        events,
        truncate_incomplete_func_calls=truncate_incomplete_func_calls,
    )

