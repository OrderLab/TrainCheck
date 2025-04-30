import logging

import polars as pl

from traincheck.config import config
from traincheck.instrumentor.tracer import TraceLineType
from traincheck.trace.types import (
    AttrState,
    ContextManagerState,
    FuncCallEvent,
    FuncCallExceptionEvent,
    IncompleteFuncCallEvent,
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
    logger.info("Unnesting all columns in the DataFrame.")
    unnested_df = df.select(_unnest_all(df.schema, separator))
    logger.info("Done unnesting all columns in the DataFrame.")
    return unnested_df


def get_attr_name(col_name: str) -> str:
    if config.VAR_ATTR_PREFIX not in col_name:
        raise ValueError(f"{col_name} does not contain the tracker_var_field_prefix.")
    return col_name[len(config.VAR_ATTR_PREFIX) :]


class Trace:
    def __init__(self, events, truncate_incomplete_func_calls=True):
        """Initializes the trace instance.

        Args:
            events (pl.DataFrame | list[pl.DataFrame] | list[dict]): The events (underlying object containing all the records) of the trace. It can be
                - a single DataFrame
                - a list of DataFrames (will be concatnated into one ), or
                - a list of dictionaries (will be converted into a DataFrame)
            truncate_incomplete_func_calls (bool, optional): Whether to truncate incomplete trailing function calls from the trace. Defaults to True.
                look at the doc of `_rm_incomplete_trailing_func_calls` for more information.

        What this function does:
            - Concatenates the DataFrames if the events is a list of DataFrames.
            - Converts the list of dictionaries into a DataFrame if the events is a list of dictionaries.
            - Truncates incomplete trailing function calls from the trace if `truncate_incomplete_func_calls` is True.
            - Checks if the time column is present in the events DataFrame.
        """
        self.events = events
        self.truncate_incomplete_func_calls = truncate_incomplete_func_calls
        self.var_changes: list[VarChangeEvent] | None = None
        self.cache_id_2_func_call_event: dict[
            str, FuncCallEvent | FuncCallExceptionEvent | IncompleteFuncCallEvent
        ] = {}
        raise NotImplementedError("This class should not be instantiated directly.")

    def _rm_incomplete_trailing_func_calls(self):
        """Remove incomplete trailing function calls from the trace. For why incomplete function calls exist, refer to https://github.com/OrderLab/traincheck/issues/31

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
        raise NotImplementedError(
            "This function should be implemented in the child class."
        )

    def get_all_stages(self) -> set[str]:
        """Get all stages in the trace."""
        raise NotImplementedError(
            "This function should be implemented in the child class."
        )

    def get_traces_for_stage(self) -> dict[str, "Trace"]:
        """Get all stages and their corresponding trace object in the trace."""
        raise NotImplementedError(
            "This function should be implemented in the child class."
        )

    def is_func_called(self, func_name: str, stage: None | str) -> bool:
        """Check if a function is called in the trace."""
        raise NotImplementedError(
            "This function should be implemented in the child class."
        )

    def _index_context_manager_meta_vars(self):
        """Identify context manager entry and exit events, and add them to the meta_vars."""
        raise NotImplementedError(
            "This function should be implemented in the child class."
        )

    def is_stage_annotated(self) -> bool:
        """Check if the trace has meta_vars.stage annotations."""
        raise NotImplementedError(
            "This function should be implemented in the child class."
        )

    def is_var_instrumented_proxy(self) -> bool:
        """Check if the trace was collected with proxy model instrumentation."""
        raise NotImplementedError(
            "This function should be implemented in the child class."
        )

    def query_active_context_managers(
        self, time: float, process_id: int, thread_id: int
    ) -> list[ContextManagerState]:
        """Given a timestamp, query all active context managers at that time.

        Args:
            time (float): The timestamp to query the active context managers.
            process_id (int): The process id to query the active context managers.
            thread_id (int): The thread id to query the active context managers.

        Returns:
            list[dict]: A list of active context managers at the given time.
            Each dict will be like:
            {
                "context_manager_name": "context_manager_name",
                "args": {"arg_name": "arg_value", ...},
                "kwargs": {"kwarg_name": "kwarg_value", ...},
                "start_time": start_time,
                "end_time": end_time,
            }
        """
        raise NotImplementedError(
            "This function should be implemented in the child class."
        )

    def get_meta_vars(self, time, process_id, thread_id) -> dict | None:
        """Get the meta variables at a specific time, process and thread."""
        raise NotImplementedError(
            "This function should be implemented in the child class."
        )

    def get_start_time(self, process_id=None, thread_id=None) -> float:
        """Get the start time of the trace. If process_id or thread_id is provided, the start time of the specific process or thread will be returned."""
        raise NotImplementedError(
            "This function should be implemented in the child class."
        )

    def get_end_time(self, process_id=None, thread_id=None) -> float:
        """Get the start time of the trace. If process_id or thread_id is provided, the start time of the specific process or thread will be returned."""
        raise NotImplementedError(
            "This function should be implemented in the child class."
        )

    def get_process_ids(self) -> list[int]:
        """Find all process ids from the trace."""
        raise NotImplementedError(
            "This function should be implemented in the child class."
        )

    def get_thread_ids(self) -> list[int]:
        """Find all thread ids from the trace."""
        raise NotImplementedError(
            "This function should be implemented in the child class."
        )

    def get_func_names(self) -> list[str]:
        """Find all function names from the trace."""
        raise NotImplementedError(
            "This function should be implemented in the child class."
        )

    def get_max_num_consecutive_call_func(self, func_name: str) -> int:
        """Find the maximum number of contiguous calls to a function in the trace."""
        raise NotImplementedError(
            "This function should be implemented in the child class."
        )

    def get_func_call_ids(self, func_name: str = "") -> list[str]:
        """Find all function call ids from the trace.
        Optionally, filter by function name.
        """
        raise NotImplementedError(
            "This function should be implemented in the child class."
        )

    def get_column_dtype(self, column_name: str) -> type:
        """Get the data type of a column in the trace.
        When implementing this in schemaless dataframes, just use the first non-null value in the column to infer the type, and print a warning saying that the type might not be accurate.
        """
        raise NotImplementedError(
            "This function should be implemented in the child class."
        )

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
        raise NotImplementedError(
            "This function should be implemented in the child class."
        )

    def get_causally_related_vars(self, func_call_id) -> set[VarInstId]:
        """Find all variables that are causally related to a function call.
        By casually related, we mean that the variables have been accessed or modified by the object (with another method) that the function call is made on.
        """
        raise NotImplementedError(
            "This function should be implemented in the child class."
        )

    def get_var_ids_unchanged_but_causally_related(
        self,
        func_call_id: str,
        var_type: str | None = None,
        attr_name: str | None = None,
    ) -> list[VarInstId]:
        """Find all variables that are causally related to a function call but not changed within the function call.

        Casually related vars: Variables are accessed or modified by the object that the function call is bound to.
        """
        related_vars = self.get_causally_related_vars(func_call_id)
        changed_vars = self.query_var_changes_within_func_call(func_call_id)

        related_vars_not_changed = []
        if var_type is not None:
            related_vars = {
                var_id for var_id in related_vars if var_id.var_type == var_type
            }
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
        raise NotImplementedError(
            "This function should be implemented in the child class."
        )

    def get_var_insts(self) -> dict[VarInstId, dict[str, list[AttrState]]]:
        """Index and get all variable instances from the trace.

        Returns:
            dict[VarInstId, dict[str, list[AttrState]]]: A dictionary mapping variable instances to their attributes and their states.
            {
                VarInstId(process_id, var_name, var_type): {
                    attr_name: [AttrState(value, liveness, traces), ...],
                    ...
                },
            }

        Consecutive traces reporting the same value will be merged into one `AttrState` object. So consecutive AttrState objects will not have the same value.
        """

        raise NotImplementedError(
            "This function should be implemented in the child class."
        )

    def get_var_raw_event_before_time(self, var_id: VarInstId, time: int) -> list[dict]:
        """Get all original trace records of a variable before time."""
        raise NotImplementedError(
            "This function should be implemented in the child class."
        )

    def get_var_changes(self) -> list[VarChangeEvent]:
        """Get all variable changes events from the trace.

        Essentially, this function will comprise consecutive states of the same variable attribute as a single change event.

        Returns:
            list[VarChangeEvent]: A list of all variable change events.
        """

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
        """Extract all variable change events from the trace, within a specific time range."""
        var_changes = self.get_var_changes()
        return [
            var_change
            for var_change in var_changes
            if time_range[0] <= var_change.change_time <= time_range[1]
        ]

    def query_var_changes_within_time_and_process(
        self, time_range: tuple[int | float, int | float], process_id: int
    ) -> list[VarChangeEvent]:
        """Extract all variable change events from the trace, within a specific time range and process."""
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
        """Extract all variable change events from the trace, within the duration of a specific function call."""
        pre_record = self.get_pre_func_call_record(func_call_id)
        post_record = self.get_post_func_call_record(func_call_id)

        start_time = pre_record["time"]

        if post_record is None:
            end_time = (
                self.get_end_time(pre_record["process_id"], pre_record["thread_id"])
                + 0.001
            )
        else:
            end_time = post_record["time"]

        return self.query_var_changes_within_time_and_process(
            (start_time, end_time), process_id=pre_record["process_id"]
        )

    def get_pre_func_call_record(self, func_call_id: str) -> dict:
        """Get the pre call record of a function given its func_call_id."""
        raise NotImplementedError(
            "This function should be implemented in the child class."
        )

    def get_post_func_call_record(self, func_call_id: str) -> dict | None:
        """Get the post call record of a function given its func_call_id.
        Returns None if the post call event is not found and the pre-call event is the outermost function call. (see the doc of `_rm_incomplete_trailing_func_calls`)
        """
        raise NotImplementedError(
            "This function should be implemented in the child class."
        )

    def query_func_call_event(
        self, func_call_id: str
    ) -> FuncCallEvent | FuncCallExceptionEvent | IncompleteFuncCallEvent:
        """Extract a function call event from the trace, given its func_call_id."""
        if func_call_id in self.cache_id_2_func_call_event:
            return self.cache_id_2_func_call_event[func_call_id]

        pre_record = self.get_pre_func_call_record(func_call_id)
        post_record = self.get_post_func_call_record(func_call_id)

        event: FuncCallEvent | FuncCallExceptionEvent | IncompleteFuncCallEvent
        if post_record is None:
            # query the end time of the trace on the specific process and thread
            potential_end_time = self.get_end_time(
                pre_record["process_id"], pre_record["thread_id"]
            )
            event = IncompleteFuncCallEvent(
                pre_record["function"], pre_record, potential_end_time
            )

        elif post_record["type"] == TraceLineType.FUNC_CALL_POST:
            event = FuncCallEvent(pre_record["function"], pre_record, post_record)

        elif post_record["type"] == TraceLineType.FUNC_CALL_POST_EXCEPTION:
            event = FuncCallExceptionEvent(
                pre_record["function"], pre_record, post_record
            )
        else:
            raise ValueError(f"Unknown function call event type: {post_record['type']}")

        self.cache_id_2_func_call_event[func_call_id] = event
        return event

    def query_func_call_events_within_time(
        self,
        time_range: tuple[int | float, int | float],
        process_id: int,
        thread_id: int,
    ) -> list[FuncCallEvent | FuncCallExceptionEvent]:
        """Extract all function call events from the trace, within a specific time range, process and thread."""
        raise NotImplementedError(
            "This function should be implemented in the child class."
        )

    def query_high_level_events_within_func_call(
        self, func_call_id: str
    ) -> list[FuncCallEvent | FuncCallExceptionEvent | VarChangeEvent]:
        """Extract all high-level events (function calls and variable changes) within a specific function call."""
        pre_record = self.get_pre_func_call_record(func_call_id)
        post_record = self.get_post_func_call_record(func_call_id)
        if post_record is None:
            logger.warning(
                f"Post call event not found for {func_call_id} ({pre_record['function']})"
            )
            # let's get the end time of the trace on the specific process and thread
            end_time = (
                self.get_end_time(pre_record["process_id"], pre_record["thread_id"])
                + 0.001
            )  # adding a small value to make sure the end time is after the last event on the process and thread
            time_range = (pre_record["time"], end_time)
        else:
            time_range = (pre_record["time"], post_record["time"])
        process_id = pre_record["process_id"]
        thread_id = pre_record["thread_id"]

        high_level_func_call_events = self.query_func_call_events_within_time(
            time_range, process_id, thread_id
        )
        high_level_var_change_events = self.query_var_changes_within_func_call(
            func_call_id
        )

        return high_level_func_call_events + high_level_var_change_events

    def get_time_precentage(self, time: int) -> float:
        return (time - self.get_start_time()) / (
            self.get_end_time() - self.get_start_time()
        )
