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

        if truncate_incomplete_func_calls:
            logger.info("Truncating incomplete trailing function calls from the trace.")
            self._rm_incomplete_trailing_func_calls()
            logger.info(
                "Done truncating incomplete trailing function calls from the trace."
            )

    def _rm_incomplete_trailing_func_calls(self):

        pass


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

    return Trace(
        events,
        truncate_incomplete_func_calls=truncate_incomplete_func_calls,
    )

