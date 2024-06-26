import logging
import re

import polars as pl
from tqdm import tqdm

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
    def __init__(self, events, truncate_incomplete_func_calls=True):
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

        if truncate_incomplete_func_calls:
            self._rm_incomplete_trailing_func_calls()

    def _rm_incomplete_trailing_func_calls(self):
        """Remove incomplete trailing function calls from the trace. https://github.com/OrderLab/ml-daikon/issues/31"""
        logger = logging.getLogger(__name__)

        if "function" not in self.events.columns:
            logger.warning(
                "function column not found in the events, no function related invariants will be extracted."
            )
            return
        # group function calls by func_call_id ## NOTE: BREAKING BEHAVIOR IF FUNC_CALL_ID IS NOT UNIQUE
        func_call_groups = self.events.groupby("func_call_id").count()
        # find the func_call_ids that have only one record
        incomplete_func_call_ids = func_call_groups.filter(pl.col("count") == 1)[
            "func_call_id"
        ]

        # retrieve the traces of the incomplete function calls
        func_call_records = self.events.filter(
            pl.col("func_call_id").is_in(incomplete_func_call_ids)
        )
        for record in func_call_records.rows(named=True):
            assert (
                record["type"] == TraceLineType.FUNC_CALL_PRE
            ), "Incomplete function call is not a pre-call event."
            logger.warning(f"Incomplete function call detected: {record}")
            process_id = record["process_id"]
            thread_id = record["thread_id"]

            # assert the incomplete function call happens after or before  the outermost function call on that process
            outermost_func_call_pre = self.events.filter(
                pl.col("type") == TraceLineType.FUNC_CALL_PRE,
                pl.col("process_id") == process_id,
            ).row(
                0, named=True
            )  # events is pre-sorted by time, so no need to sort again

            assert (
                thread_id != outermost_func_call_pre["thread_id"]
            ), "Incomplete function call is not on a different thread. Please Investigate."

            outermost_func_call_post = self.events.filter(
                pl.col("func_call_id") == outermost_func_call_pre["func_call_id"],
            ).row(1, named=True)

            if (
                record["time"]
                > outermost_func_call_post["time"]
                - config.INCOMPLETE_FUNC_CALL_SECONDS_TO_OUTERMOST_POST
            ):
                logger.warning(f"Removing incomplete function call: {record}")
                self.events = self.events.filter(
                    pl.col("func_call_id") != record["func_call_id"]
                )
            else:
                raise ValueError(
                    f"Incomplete function call is not close enough to the outermost function call post event: {record}"
                )

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

    def get_func_is_bound_method(self, func_name: str) -> bool:
        """Check if a function is bound to a class."""
        if "function" not in self.events.columns:
            logger.warning(
                "function column not found in the events, no function related invariants will be extracted."
            )
            return False

        is_bound_method = (
            self.events.filter(pl.col("function") == func_name)
            .select("is_bound_method")
            .to_series()
            .to_list()
        )
        assert (
            None not in is_bound_method
        ), f"Boundness information not found for {func_name}"
        assert all(
            is_bound_method[0] == is_bound_method[i]
            for i in range(1, len(is_bound_method))
        ), f"Boundness information is not consistent for {func_name}"
        return is_bound_method[0]

    def get_causally_related_vars(self, func_call_id) -> list[VarInstId]:
        """Find all variables that are causally related to a function call."""

        # get the pre-call event of the function call
        func_call_pre_event = self.events.filter(
            pl.col("type") == TraceLineType.FUNC_CALL_PRE,
            pl.col("func_call_id") == func_call_id,
        ).row(0, named=True)

        func_call_time = func_call_pre_event["time"]

        # get the process id of the function call
        assert func_call_pre_event[
            "is_bound_method"
        ], f"Causal relation extraction is only supported for bound methods, got {func_call_pre_event['function']} which is not"
        obj_id = func_call_pre_event["obj_id"]

        # find all function calls that are related to this object prior to the function call
        related_func_call_pre_events = self.events.filter(
            pl.col("type") == TraceLineType.FUNC_CALL_PRE,
            pl.col("obj_id") == obj_id,
            pl.col("time") < func_call_pre_event["time"],
        )

        process_id = func_call_pre_event["process_id"]
        thread_id = func_call_pre_event["thread_id"]
        for related_func_call_pre_event in related_func_call_pre_events.rows(
            named=True
        ):
            # having the assertion failure does not mean that we run into a bug, but it is something that we should investigate
            assert (
                related_func_call_pre_event["process_id"] == process_id
            ), "Related function call is on a different process."
            assert (
                related_func_call_pre_event["thread_id"] == thread_id
            ), "Related function call is on a different thread."

        # find all variables that are related to the function calls
        related_func_call_ids = related_func_call_pre_events["func_call_id"].to_list()

        # take a look at each var's last trace before the function call time to determine if it is causally related
        related_vars = []
        var_ids = self.get_var_ids()
        for var_id in var_ids:
            trace_before_func_call = self.events.filter(
                pl.col("process_id") == var_id.process_id,
                pl.col("var_name") == var_id.var_name,
                pl.col("var_type") == var_id.var_type,
                pl.col("time") < func_call_time,
            )
            if len(trace_before_func_call) == 0:
                continue
            last_trace = trace_before_func_call.row(-1, named=True)
            if (
                len(
                    [
                        related_func_call_id
                        for related_func_call_id in related_func_call_ids
                        if related_func_call_id in last_trace["causal_func_call_ids"]
                    ]
                )
                > 0
            ):
                # the variable is causally related to the function call
                related_vars.append(var_id)
        return related_vars

    def get_vars_not_changed_but_causally_related(
        self,
        func_call_id: str,
        var_type: str | None = None,
        attr_name: str | None = None,
    ) -> list[VarInstId]:
        related_vars = self.get_causally_related_vars(func_call_id)
        changed_vars = self.query_var_changes_within_func_call(func_call_id)

        related_vars_not_changed = []
        if var_type is not None:
            related_vars = [
                var_id for var_id in related_vars if var_id.var_type == var_type
            ]
            changed_vars = [
                var_change
                for var_change in changed_vars
                if var_change.var_id.var_type == var_type
            ]
        if attr_name is not None:
            changed_vars = [
                var_change
                for var_change in changed_vars
                if var_change.attr_name == attr_name
            ]

        for var_id in related_vars:
            if any([var_change.var_id == var_id for var_change in changed_vars]):
                continue
            related_vars_not_changed.append(var_id)
        return related_vars_not_changed

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

    def get_var_raw_event_before_time(self, var_id: VarInstId, time: int) -> list[dict]:
        """Get all raw events of a variable."""
        return self.events.filter(
            pl.col("process_id") == var_id.process_id,
            pl.col("var_name") == var_id.var_name,
            pl.col("var_type") == var_id.var_type,
            pl.col("time") < time,
        ).rows(named=True)

    def get_var_changes(self) -> list[VarChangeEvent]:
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
                        VarChangeEvent(
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
    ) -> list[VarChangeEvent]:

        var_changes = self.get_var_changes()
        return [
            var_change
            for var_change in var_changes
            if time_range[0] <= var_change.change_time <= time_range[1]
        ]

    def query_var_changes_within_time_and_process(
        self, time_range: tuple[int, int], process_id: int
    ) -> list[VarChangeEvent]:
        var_changes = self.get_var_changes()
        return [
            var_change
            for var_change in var_changes
            if time_range[0] <= var_change.change_time <= time_range[1]
            and var_change.var_id.process_id == process_id
        ]

    def query_var_changes_within_func_call(
        self, func_call_id: str
    ) -> list[VarChangeEvent]:
        func_call_pre_event = self.events.filter(
            pl.col("type") == TraceLineType.FUNC_CALL_PRE,
            pl.col("func_call_id") == func_call_id,
        ).row(0, named=True)

        func_call_post_event = self.events.filter(
            pl.col("type") == TraceLineType.FUNC_CALL_POST,
            pl.col("func_call_id") == func_call_id,
        ).row(0, named=True)

        return self.query_var_changes_within_time(
            (func_call_pre_event["time"], func_call_post_event["time"])
        )

    def scan_to_groups(self, param_selectors: list):
        """Extract from trace, groups of events

        args:
            param_selectors: list
                A list of functions that take an event as input and return a value retrieved from the event.
                If a value is not found, the function should return None. The length of the list should be equal to
                the number of variables in the relation.
        """

        raise NotImplementedError("group method is not implemented yet.")

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

    def query_func_call_events_within_time(
        self, time_range: tuple[int, int], process_id: int, thread_id: int
    ) -> list[FuncCallEvent | FuncCallExceptionEvent]:
        """Extract all function call events from the trace."""
        func_call_events: list[FuncCallEvent | FuncCallExceptionEvent] = []
        func_pre_call_event_indices = (
            self.events.select(
                pl.arg_where(
                    (pl.col("type") == TraceLineType.FUNC_CALL_PRE)
                    & (
                        pl.col("time").is_between(
                            time_range[0], time_range[1], closed="none"
                        )
                    )
                    & (pl.col("process_id") == process_id)
                    & (pl.col("thread_id") == thread_id)
                )
            )
            .to_series()
            .to_list()
        )

        for func_pre_call_idx in func_pre_call_event_indices:
            func_post_call_idx = self.get_func_post_call_idx(func_pre_call_idx)
            pre_record = self.events.row(index=func_pre_call_idx, named=True)
            post_record = self.events.row(index=func_post_call_idx, named=True)
            assert (
                post_record["time"] < time_range[1]
            ), f"Post call event found after the time range {time_range}"
            func_call_events.append(
                FuncCallEvent(pre_record["function"], pre_record, post_record)
                if post_record["type"] == TraceLineType.FUNC_CALL_POST
                else FuncCallExceptionEvent(
                    pre_record["function"], pre_record, post_record
                )
            )
        return func_call_events

    def query_high_level_events_within_time(
        self, time_range: tuple[int, int], process_id: int, thread_id: int
    ) -> list[FuncCallEvent | FuncCallExceptionEvent | VarChangeEvent]:

        high_level_func_call_events = self.query_func_call_events_within_time(
            time_range, process_id, thread_id
        )
        high_level_var_change_events = self.query_var_changes_within_time_and_process(
            time_range, process_id
        )

        return high_level_func_call_events + high_level_var_change_events


def read_trace_file(
    file_path: str | list[str], truncate_incomplete_func_calls=True
) -> Trace:
    """Reads the trace file and returns the trace instance."""
    if isinstance(file_path, list):
        events = pl.concat(
            [pl.read_ndjson(f) for f in file_path], how="diagonal_relaxed"
        )
    else:
        events = pl.read_ndjson(file_path)
    return Trace(
        unnest_all(events),
        truncate_incomplete_func_calls=truncate_incomplete_func_calls,
    )
