import logging
import re
from typing import NamedTuple

import polars as pl
from tqdm import tqdm

from mldaikon.config import config
from mldaikon.instrumentor.tracer import TraceLineType

logger = logging.getLogger(__name__)

# TODO: formalize the trace schema for efficient polars processing


class VarInstId(NamedTuple):
    process_id: int
    var_name: str
    var_type: str


class Liveness:
    def __init__(self, start_time: int | None, end_time: int | None):
        self.start_time = start_time
        self.end_time = end_time


class AttrState:
    def __init__(self, value: type, liveness: Liveness, traces: list[dict]):
        self.value: type = value
        self.liveness: Liveness = liveness
        self.traces = traces


class VarChange:
    def __init__(
        self,
        var_id: VarInstId,
        attr_name: str,
        change_time: int,
        old_state: AttrState,
        new_state: AttrState,
    ):
        self.var_id = var_id
        self.attr_name = attr_name
        self.change_time = change_time
        self.old_state = old_state
        self.new_state = new_state


def _unnest_all(schema, separator):
    def _unnest(schema, path=[]):
        for name, dtype in schema.items():
            base_type = dtype.base_type()

            if base_type == pl.Struct:
                yield from _unnest(dtype.to_schema(), path + [name])
            else:
                yield path + [name], dtype

    for (col, *fields), dtype in _unnest(schema):
        expr = pl.col(col)

        for field in fields:
            expr = expr.struct[field]

        not_empty_fields = [f for f in fields if f.strip() != ""]
        if len(not_empty_fields) > 0:
            name = separator.join([col] + not_empty_fields)
        else:
            name = col

        yield expr.alias(name)


def unnest_all(df: pl.DataFrame, separator=".") -> pl.DataFrame:
    return df.select(_unnest_all(df.schema, separator))


def get_attr_name(col_name: str) -> str:
    if config.VAR_ATTR_PREFIX not in col_name:
        raise ValueError(f"{col_name} does not contain the tracker_var_field_prefix.")
    return col_name[len(config.VAR_ATTR_PREFIX) :]


class Trace:
    def __init__(self, events):
        self.events = events
        self.var_ids = None
        self.var_insts = None
        self.var_changes = None

        if isinstance(events, list) and all(
            [isinstance(e, pl.DataFrame) for e in events]
        ):
            self.events = pl.concat(events, how="diagonal_relaxed")
        elif isinstance(events, list) and all([isinstance(e, dict) for e in events]):
            self.events = pl.DataFrame(events)
            self.events = unnest_all(self.events)

        assert isinstance(
            self.events, pl.DataFrame
        ), "events should be a DataFrame, list of DataFrames, or a list of dictionaries."

        try:
            self.events = self.events.sort("time", descending=False)
        except pl.PolarsError:
            raise ValueError(
                "Failed to sort the events by time. Check if the time column is present in the events."
            )

    def filter(self, predicate):
        # TODO: need to think about how to implement this, as pre-conditions for bugs like DS-1801 needs to take multiple events into account
        raise NotImplementedError("filter method is not implemented yet.")

    def get_start_time(self) -> int:
        return self.events["time"].min()

    def get_end_time(self) -> int:
        return self.events["time"].max()

    def get_func_names(self) -> list[str]:
        """Find all function names from the trace."""
        if "function" not in self.events.columns:
            logger.warning(
                "function column not found in the events, no function related invariants will be extracted."
            )
            return []
        return (
            self.events.select("function").drop_nulls().unique().to_series().to_list()
        )

    def get_var_ids(self) -> list[VarInstId]:
        """Find all variables (uniquely identified by name, type and process id) from the trace."""
        # Identification of Variables --> (variable_name, process_id)
        if self.var_ids is not None:
            return self.var_ids

        if "var_name" not in self.events.columns:
            logger.warning(
                "var_name column not found in the events, no variable related invariants will be extracted."
            )
            return []

        variables = (
            self.events.select("var_name", "var_type", "process_id")
            .drop_nulls()
            .unique()
        )

        self.var_ids = []
        for var_id in variables.rows(named=True):
            self.var_ids.append(
                VarInstId(
                    var_id["process_id"],
                    var_id["var_name"],
                    var_id["var_type"],
                )
            )
        return self.var_ids

    def get_var_insts(self) -> dict[VarInstId, dict[str, list[AttrState]]]:
        if self.var_insts is not None:
            return self.var_insts

        var_ids = self.get_var_ids()
        if len(var_ids) == 0:
            logger.warning("No variables found in the trace.")
            return {}
        var_insts = {}
        for var_id in tqdm(var_ids, desc="Indexing Variable Instances"):
            var_inst_states = self.events.filter(
                pl.col("process_id") == var_id.process_id,
                pl.col("var_name") == var_id.var_name,
                pl.col("var_type") == var_id.var_type,
            )

            state_changes = var_inst_states.filter(
                pl.col("type") == TraceLineType.STATE_CHANGE
            )

            # init attribute values for this variable
            attr_values = {}
            for state_change in state_changes.rows(named=True):
                for col in state_change:
                    if col.startswith(config.VAR_ATTR_PREFIX):
                        attr_name = get_attr_name(col)
                        # pruning out the attributes that might be properties
                        if any(
                            [
                                re.match(pattern, attr_name) is not None
                                for pattern in config.PROP_ATTR_PATTERNS
                            ]
                        ) or any(
                            [
                                self.events[col].dtype == _type
                                for _type in config.PROP_ATTR_TYPES
                            ]
                        ):
                            continue

                        if attr_name not in attr_values:
                            attr_values[attr_name] = [
                                AttrState(
                                    state_change[col],
                                    Liveness(state_change["time"], None),
                                    [state_change],
                                )
                            ]
                        else:
                            if attr_values[attr_name][-1].value != state_change[col]:
                                attr_values[attr_name][-1].liveness.end_time = (
                                    state_change["time"]
                                )
                                attr_values[attr_name].append(
                                    AttrState(
                                        state_change[col],
                                        Liveness(state_change["time"], None),
                                        [state_change],
                                    )
                                )
                            else:
                                attr_values[attr_name][-1].traces.append(state_change)

            # set end time for the last state change
            for attr_name in attr_values:
                if attr_values[attr_name][-1].liveness.end_time is None:
                    attr_values[attr_name][-1].liveness.end_time = self.get_end_time()
            var_insts[var_id] = attr_values
        for var_id in var_insts:
            for attr in var_insts[var_id]:
                for value in var_insts[var_id][attr]:
                    if len(value.traces) == 0:
                        print(f"Warning: No traces found for {var_id} {attr}")
        self.var_insts = var_insts
        return self.var_insts

    def get_var_changes(self) -> list[VarChange]:
        if self.var_changes is not None:
            return self.var_changes

        var_insts = self.get_var_insts()

        self.var_changes = []
        for var_id in var_insts:
            for attr in var_insts[var_id]:
                for i in range(1, len(var_insts[var_id][attr])):

                    change_time = var_insts[var_id][attr][i].liveness.start_time
                    old_state = var_insts[var_id][attr][i - 1]
                    new_state = var_insts[var_id][attr][i]
                    assert (
                        change_time is not None
                    ), f"Start time not found for {var_id} {attr} {var_insts[var_id][attr][i].value}"
                    self.var_changes.append(
                        VarChange(
                            var_id=var_id,
                            attr_name=attr,
                            change_time=change_time,
                            old_state=old_state,
                            new_state=new_state,
                        )
                    )

        return self.var_changes

    def query_var_changes_within_time(
        self, time_range: tuple[int, int]
    ) -> list[VarChange]:

        var_changes = self.get_var_changes()
        return [
            var_change
            for var_change in var_changes
            if time_range[0] <= var_change.change_time <= time_range[1]
        ]

    def query_var_changes_within_time_and_process(
        self, time_range: tuple[int, int], process_id: int
    ) -> list[VarChange]:
        var_changes = self.get_var_changes()
        return [
            var_change
            for var_change in var_changes
            if time_range[0] <= var_change.change_time <= time_range[1]
            and var_change.var_id.process_id == process_id
        ]

    def scan_to_groups(self, param_selectors: list):
        """Extract from trace, groups of events

        args:
            param_selectors: list
                A list of functions that take an event as input and return a value retrieved from the event.
                If a value is not found, the function should return None. The length of the list should be equal to
                the number of variables in the relation.
        """

        # raise NotImplementedError("group method is not implemented for pandas yet.")

        groups: list[list[object]] = []
        group: list[object] = []
        param_selector_itr = iter(param_selectors)
        curr_param_selector = next(param_selector_itr, None)
        assert curr_param_selector is not None, "param_selectors should not be empty"

        for e in self.events:
            if curr_param_selector is None:
                groups.append(group)
                group = []
                param_selector_itr = iter(param_selectors)
                curr_param_selector = next(param_selector_itr, None)
            elif not callable(curr_param_selector):
                # curr_param_selector is a value
                group.append(curr_param_selector)
            elif curr_param_selector(e):
                group.append(curr_param_selector(e))
                curr_param_selector = next(param_selector_itr, None)

        # dealing with the left-over group
        if len(group) == len(param_selectors):
            groups.append(group)
        elif len(group) > 0:
            logger.debug(
                f"Left-over group: {group} not added to groups as it does not have enough elements."
            )

        return groups

    def get_func_post_call_idx(self, func_pre_call_idx: int) -> int:
        """Get the post call index of a function given its pre-call index."""
        # get the first record of the trace at the offset
        pre_call_record = self.events.row(index=func_pre_call_idx, named=True)

        assert (
            pre_call_record["type"] == TraceLineType.FUNC_CALL_PRE
        ), "The record at the given index is not a function call (pre) event."

        function = pre_call_record["function"]
        process_id = pre_call_record["process_id"]
        thread_id = pre_call_record["thread_id"]
        func_call_id = pre_call_record["func_call_id"]

        # get the idx of the first post-event
        func_post_call_indices = self.events.select(
            pl.arg_where(
                (
                    pl.col("type").is_in(
                        [
                            TraceLineType.FUNC_CALL_POST,
                            TraceLineType.FUNC_CALL_POST_EXCEPTION,
                        ]
                    )
                )
                & (pl.col("function") == function)
                & (pl.col("func_call_id") == func_call_id)
                & (pl.col("process_id") == process_id)
                & (pl.col("thread_id") == thread_id)
            )
        ).to_series()

        # find the first post-idx > offset as there might be multiple calls with the same func_call_id
        func_post_call_idx = [
            idx for idx in func_post_call_indices if idx > func_pre_call_idx
        ][0]
        return func_post_call_idx


def read_trace_file(file_path: str | list[str]) -> Trace:
    """Reads the trace file and returns the trace instance."""
    if isinstance(file_path, list):
        events = pl.concat(
            [pl.read_ndjson(f) for f in file_path], how="diagonal_relaxed"
        )
    else:
        events = pl.read_ndjson(file_path)
    return Trace(unnest_all(events))
