import logging
import re

import polars as pl
from tqdm import tqdm

from mldaikon.config import config
from mldaikon.instrumentor.tracer import TraceLineType
from mldaikon.trace.trace import Trace
from mldaikon.trace.types import (
    AttrState,
    FuncCallEvent,
    FuncCallExceptionEvent,
    IncompleteFuncCallEvent,
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
    logger.info("Unnesting all columns in the DataFrame.")
    unnested_df = df.select(_unnest_all(df.schema, separator))
    logger.info("Done unnesting all columns in the DataFrame.")
    return unnested_df


def get_attr_name(col_name: str) -> str:
    if config.VAR_ATTR_PREFIX not in col_name:
        raise ValueError(f"{col_name} does not contain the tracker_var_field_prefix.")
    return col_name[len(config.VAR_ATTR_PREFIX) :]


class TracePolars(Trace):
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

        assert (
            "time" in self.events.columns
        ), "time column not found in the events, cannot infer invariants as the analysis does a lot of queries based on time."

        # TODO: we need to handle the schema issue in polars quickly as it will force us to split the traces by schema, which will cause traces from one process to be split into multiple files and thus we will have to sort the entire trace by time, which is costly
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
        # group function calls by func_call_id ## NOTE: BREAKING BEHAVIOR IF FUNC_CALL_ID IS NOT UNIQUE
        func_call_groups = self.events.group_by("func_call_id").count()
        # find the func_call_ids that have only one record
        incomplete_func_call_ids = func_call_groups.filter(pl.col("count") == 1)[
            "func_call_id"
        ]

        # retrieve the traces of the incomplete function calls
        incomplete_func_call_records = self.events.filter(
            pl.col("func_call_id").is_in(incomplete_func_call_ids)
        )
        for record in incomplete_func_call_records.rows(
            named=True
        ):  # order of events doesn't matter as we are querying by func_call_id
            assert (
                record["type"] == TraceLineType.FUNC_CALL_PRE
            ), "Incomplete function call is not a pre-call event."
            logger.warning(f"Incomplete function call detected: {record}")
            process_id = record["process_id"]
            thread_id = record["thread_id"]

            # assert the incomplete function call happens after or before  the outermost function call on that process
            outermost_func_call_pre_record = self.events.filter(
                pl.col("type") == TraceLineType.FUNC_CALL_PRE,
                pl.col("process_id") == process_id,
                pl.col("time") == self.get_start_time(process_id),
            ).row(0, named=True)

            # see if the outermost function is also incomplete
            outermost_func_call_post = self.events.filter(
                pl.col("type") == TraceLineType.FUNC_CALL_POST,
                pl.col("func_call_id")
                == outermost_func_call_pre_record["func_call_id"],
            )

            if outermost_func_call_post.height == 0:
                outermost_incomplete = True
            else:
                outermost_incomplete = False
                outermost_func_call_post_record = outermost_func_call_post.row(
                    0, named=True
                )

            # if the incomplete function is the outermost function, we should be handling it differently
            if record["func_call_id"] == outermost_func_call_pre_record["func_call_id"]:
                logger.warning(
                    f"The outermost function call is incomplete: {outermost_func_call_pre_record['function']} with id {outermost_func_call_pre_record['func_call_id']}. Will treat it as a complete function call."
                )
                # nothing is done here at the pre-processing stage, incomplete outermost call is handled in other query functions
                continue

            if not outermost_incomplete:
                # if outermost function is complete, everything inside on the same thread should also be complete
                assert (
                    thread_id != outermost_func_call_pre_record["thread_id"]
                ), f"Incomplete function call (func_call_id: {record['func_call_id']}) is not on a different thread than outermost function (func_call_id: {outermost_func_call_pre_record['func_call_id']}) on process {process_id}. Please Investigate."

                if (
                    record["time"]
                    > outermost_func_call_post_record["time"]
                    - config.INCOMPLETE_FUNC_CALL_SECONDS_TO_OUTERMOST_POST
                ):
                    logger.warning(f"Removing incomplete function call: {record}")
                    prev_var_height = self.events.filter(
                        pl.col("type") == "state_change"
                    ).height
                    self.events = self.events.filter(
                        pl.col("func_call_id").ne_missing(record["func_call_id"])
                    )
                    assert (
                        self.events.filter(pl.col("type") == "state_change").height
                        == prev_var_height
                    ), f"Incomplete function call removal incorrect, removed var traces incorrectly. func_call_id: {record['func_call_id']}, prev_height: {prev_var_height}, new_height: {self.events.filter(pl.col('type') == 'state_change').height}"
                else:
                    raise ValueError(
                        f"Incomplete function call is not close enough to the outermost function call post event: {record}"
                    )

            else:
                # the outermost function is also incomplete, just delete the current incomplete function call and everything after it on the same thread and process FIXME: this can make the end time estimation (i.e. the end time) of the outermost function call even more inaccurate
                logger.warning(
                    f"Removing incomplete function call and everything after it on the same thread and process: {record}"
                )

                prev_var_height = self.events.filter(
                    pl.col("type") == "state_change"
                ).height
                self.events = self.events.filter(
                    (pl.col("process_id").ne_missing(process_id))
                    | (pl.col("thread_id").ne_missing(thread_id))
                    | (pl.col("time") < record["time"])
                )

                assert (
                    self.events.filter(pl.col("type") == "state_change").height
                    == prev_var_height
                ), "Incomplete function call removal incorrect, removed var traces incorrectly."
                # assert the record's func_call_id is no longer in the events
                assert (
                    self.events.filter(
                        pl.col("func_call_id") == record["func_call_id"]
                    ).height
                    == 0
                ), f"Incomplete function call is not removed: {record}"

    def get_start_time(self, process_id=None, thread_id=None) -> float:
        """Get the start time of the trace. If process_id or thread_id is provided, the start time of the specific process or thread will be returned."""
        if process_id is not None and thread_id is not None:
            return self.events.filter(
                pl.col("process_id") == process_id, pl.col("thread_id") == thread_id
            )["time"].min()

        if process_id is not None:
            return self.events.filter(pl.col("process_id") == process_id)["time"].min()

        if thread_id is not None:
            return self.events.filter(pl.col("thread_id") == thread_id)["time"].min()

        return self.events["time"].min()

    def get_end_time(self, process_id=None, thread_id=None) -> float:
        """Get the start time of the trace. If process_id or thread_id is provided, the start time of the specific process or thread will be returned."""

        if process_id is not None and thread_id is not None:
            return self.events.filter(
                pl.col("process_id") == process_id, pl.col("thread_id") == thread_id
            )["time"].max()

        if process_id is not None:
            return self.events.filter(pl.col("process_id") == process_id)["time"].max()

        if thread_id is not None:
            return self.events.filter(pl.col("thread_id") == thread_id)["time"].max()

        return self.events["time"].max()

    def get_process_ids(self) -> list[int]:
        """Find all process ids from the trace."""
        return (
            self.events.select("process_id").drop_nulls().unique().to_series().to_list()
        )

    def get_thread_ids(self) -> list[int]:
        """Find all thread ids from the trace."""
        return (
            self.events.select("thread_id").drop_nulls().unique().to_series().to_list()
        )

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

    def get_func_call_ids(self, func_name: str = "") -> list[str]:
        """Find all function call ids from the trace."""
        if "func_call_id" not in self.events.columns:
            logger.warning(
                "func_call_id column not found in the events, no function related invariants will be extracted."
            )
            return []

        if func_name:
            return (
                self.events.filter(pl.col("function") == func_name)
                .select("func_call_id")
                .drop_nulls()
                .unique()
                .to_series()
                .to_list()
            )

        return (
            self.events.select("func_call_id")
            .drop_nulls()
            .unique()
            .to_series()
            .to_list()
        )

    def get_column_dtype(self, column_name: str) -> type:
        """Get the data type of a column in the trace.
        When implementing this in schemaless dataframes, just use the first non-null value in the column to infer the type, and print a warning saying that the type might not be accurate.
        """
        return self.events[column_name].dtype

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

    def get_causally_related_vars(self, func_call_id) -> set[VarInstId]:
        """Find all variables that are causally related to a function call.
        By casually related, we mean that the variables have been accessed or modified by the object (with another method) that the function call is made on.
        """

        # get the pre-call event of the function call
        func_call_pre_event = self.events.filter(
            pl.col("type") == TraceLineType.FUNC_CALL_PRE,
            pl.col("func_call_id") == func_call_id,
        ).row(
            0, named=True
        )  # order of events doesn't matter as the result here is a single row

        # get the process id of the function call
        assert func_call_pre_event[
            "is_bound_method"
        ], f"Causal relation extraction is only supported for bound methods, got {func_call_pre_event['function']} which is not"
        obj_id = func_call_pre_event[
            "obj_id"
        ]  # the object id (address) of the object that the function is bound to

        # find all function calls that are related to this object prior to the function call
        related_func_call_pre_events = self.events.filter(
            pl.col("type") == TraceLineType.FUNC_CALL_PRE,
            pl.col("obj_id") == obj_id,
            pl.col("time") < func_call_pre_event["time"],
        )

        process_id = func_call_pre_event["process_id"]
        thread_id = func_call_pre_event["thread_id"]

        causally_related_var_ids: set[VarInstId] = set()
        for related_func_call_pre_event in related_func_call_pre_events.rows(
            named=True
        ):  # order of events doesn't matter as we are querying by time when we get the related_func_call_pre_events variable
            # having the assertion failure does not mean that we run into a bug, but it is something that we should investigate
            assert (
                related_func_call_pre_event["process_id"] == process_id
            ), "Related function call is on a different process."
            assert (
                related_func_call_pre_event["thread_id"] == thread_id
            ), "Related function call is on a different thread."
            for var_name, var_type in related_func_call_pre_event["proxy_obj_names"]:
                if var_name == "" and var_type == "":
                    continue
                causally_related_var_ids.add(VarInstId(process_id, var_name, var_type))

        return causally_related_var_ids

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
        for var_id in variables.rows(named=True):  # order of events doesn't matter
            self.var_ids.append(
                VarInstId(
                    var_id["process_id"],
                    var_id["var_name"],
                    var_id["var_type"],
                )
            )
        return self.var_ids

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
            ).sort(
                "time"
            )  # we need to sort the events by time to make sure the order of the events is correct

            state_changes = var_inst_states.filter(
                pl.col("type") == TraceLineType.STATE_CHANGE
            )

            # init attribute values for this variable
            attr_values = {}
            for state_change in state_changes.rows(
                named=True
            ):  # order of events matters here as we assume the rows of the events are ordered by time, we can refactor this to use the time column to sort the rows, but it is not necessary as the states for one single var should be on the same process, which is naturally orderred by time
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
        """Get all original trace records of a variable before time."""
        return self.events.filter(
            pl.col("process_id") == var_id.process_id,
            pl.col("var_name") == var_id.var_name,
            pl.col("var_type") == var_id.var_type,
            pl.col("time") < time,
        ).rows(
            named=True
        )  # order of events doesn't matter as we already query by time

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
        pre_records = self.events.filter(
            pl.col("func_call_id") == func_call_id,
            pl.col("type") == TraceLineType.FUNC_CALL_PRE,
        )

        assert (
            pre_records.height == 1
        ), f"{pre_records.height} pre call events found for {func_call_id}, expected 1"
        pre_record = pre_records.row(
            0, named=True
        )  # order of events doesn't matter as the result here is a single row

        return pre_record

    def get_post_func_call_record(self, func_call_id: str) -> dict | None:
        """Get the post call record of a function given its func_call_id.
        Returns None if the post call event is not found and the pre-call event is the outermost function call. (see the doc of `_rm_incomplete_trailing_func_calls`)
        """
        post_records = self.events.filter(
            pl.col("func_call_id") == func_call_id,
            pl.col("type").is_in(
                [TraceLineType.FUNC_CALL_POST, TraceLineType.FUNC_CALL_POST_EXCEPTION]
            ),
        )

        assert (
            post_records.height <= 1
        ), f"{post_records.height} post call events found for {func_call_id}, expected 1"

        if post_records.height == 0:
            logger.warning(f"No post call event found for {func_call_id}")
            # check if the pre-call event is the outermost function call
            pre_record = self.get_pre_func_call_record(func_call_id)
            outermost_func_call_pre = self.events.filter(
                pl.col("type") == TraceLineType.FUNC_CALL_PRE,
                pl.col("process_id") == pre_record["process_id"],
                pl.col("thread_id") == pre_record["thread_id"],
            ).row(0, named=True)

            if pre_record["func_call_id"] == outermost_func_call_pre["func_call_id"]:
                return None
            else:
                raise ValueError(
                    f"No post call event found for {func_call_id}, but it is not the outermost function call."
                )

        post_record = post_records.row(
            0, named=True
        )  # order of events doesn't matter as the result here is a single row

        return post_record

    def query_func_call_event(
        self, func_call_id: str
    ) -> FuncCallEvent | FuncCallExceptionEvent | IncompleteFuncCallEvent:
        """Extract a function call event from the trace, given its func_call_id."""
        pre_record = self.get_pre_func_call_record(func_call_id)
        post_record = self.get_post_func_call_record(func_call_id)

        if post_record is None:
            # query the end time of the trace on the specific process and thread
            potential_end_time = self.get_end_time(
                pre_record["process_id"], pre_record["thread_id"]
            )
            return IncompleteFuncCallEvent(
                pre_record["function"], pre_record, potential_end_time
            )

        if post_record["type"] == TraceLineType.FUNC_CALL_POST:
            return FuncCallEvent(pre_record["function"], pre_record, post_record)

        if post_record["type"] == TraceLineType.FUNC_CALL_POST_EXCEPTION:
            return FuncCallExceptionEvent(
                pre_record["function"], pre_record, post_record
            )

        raise ValueError(f"Unknown function call event type: {post_record['type']}")

    def query_func_call_events_within_time(
        self,
        time_range: tuple[int | float, int | float],
        process_id: int,
        thread_id: int,
    ) -> list[FuncCallEvent | FuncCallExceptionEvent]:
        """Extract all function call events from the trace, within a specific time range, process and thread."""
        func_call_events: list[FuncCallEvent | FuncCallExceptionEvent] = []
        func_call_records = self.events.filter(
            (
                pl.col("type").is_in(
                    [
                        TraceLineType.FUNC_CALL_PRE,
                        TraceLineType.FUNC_CALL_POST,
                        TraceLineType.FUNC_CALL_POST_EXCEPTION,
                    ]
                )
            )
            & (pl.col("time").is_between(time_range[0], time_range[1], closed="none"))
            & (pl.col("process_id") == process_id)
            & (pl.col("thread_id") == thread_id)
        )

        # group function calls by func_call_id
        func_call_groups = func_call_records.group_by("func_call_id")
        for func_call_id, func_call_records in func_call_groups:
            if func_call_records.height == 1:
                # check if it is the outermost function call
                pre_record = func_call_records.row(0, named=True)
                outermost_func_call_pre = self.events.filter(
                    pl.col("type") == TraceLineType.FUNC_CALL_PRE,
                    pl.col("process_id") == pre_record["process_id"],
                    pl.col("thread_id") == pre_record["thread_id"],
                ).row(0, named=True)

                if (
                    pre_record["func_call_id"]
                    == outermost_func_call_pre["func_call_id"]
                ):
                    func_call_events.append(
                        FuncCallEvent(pre_record["function"], pre_record, None)  # type: ignore
                    )
                    continue
                else:
                    raise ValueError(
                        f"No post call event found for {func_call_id}, but it is not the outermost function call."
                    )

            assert (
                func_call_records.height == 2
            ), f"Function call records is not 2 for {func_call_id}"
            pre_record = func_call_records.row(0, named=True)
            post_record = func_call_records.row(1, named=True)

            assert (
                pre_record["type"] == TraceLineType.FUNC_CALL_PRE
            ), f"First record for {func_call_id} is not pre, got {pre_record['type']}"
            assert post_record["type"] in [
                TraceLineType.FUNC_CALL_POST,
                TraceLineType.FUNC_CALL_POST_EXCEPTION,
            ], f"Second record for {func_call_id} is not post, got {post_record['type']}"

            func_call_events.append(
                FuncCallEvent(pre_record["function"], pre_record, post_record)
                if post_record["type"] == TraceLineType.FUNC_CALL_POST
                else FuncCallExceptionEvent(
                    pre_record["function"], pre_record, post_record
                )
            )
        return func_call_events

    # def query_high_level_events_within_time(
    #     self, time_range: tuple[int, int], process_id: int, thread_id: int
    # ) -> list[FuncCallEvent | FuncCallExceptionEvent | VarChangeEvent]:

    #     high_level_func_call_events = self.query_func_call_events_within_time(
    #         time_range, process_id, thread_id
    #     )
    #     high_level_var_change_events = self.query_var_changes_within_time_and_process(
    #         time_range, process_id
    #     )

    #     return high_level_func_call_events + high_level_var_change_events

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

    def get_filtered_function(self) -> pl.DataFrame:
        """Filter API calls in traces and return filtered events."""
        events = self.events
        events = events.filter(~events["function"].str.contains(r"\.__*__"))
        events = events.filter(~events["function"].str.contains(r"\._"))
        threshold = 6
        events = events.with_columns(
            (events["function"] != events["function"].shift())
            .cum_sum()
            .alias("group_id")
        )
        group_counts = events.group_by("group_id").agg(
            [
                pl.col("function").first().alias("function"),
                pl.count("function").alias("count"),
            ]
        )
        functions_to_remove = group_counts.filter(pl.col("count") > threshold)[
            "function"
        ]
        events = events.filter(~events["function"].is_in(functions_to_remove))
        return events


def read_trace_file(
    file_path: str | list[str], truncate_incomplete_func_calls=True
) -> Trace:
    """Reads the trace file and returns the trace instance.

    Args:
        file_path (str | list[str]): The path to the trace file or a list of paths to the trace files.
        truncate_incomplete_func_calls (bool, optional): Whether to truncate incomplete trailing function calls from the trace. Defaults to True.
            look at the doc of `_rm_incomplete_trailing_func_calls` for more information.

    Returns:
        Trace: The trace instance.

    Note that nested structures will be flattened and all files passed as input will be concatenated into one trace object.
    """
    if isinstance(file_path, list):
        dfs = []
        for f in file_path:
            try:
                dfs.append(pl.read_ndjson(f))
                logger.info(f"Done reading {f}")
            except Exception as e:
                logger.error(f"Failed to read {f} due to {e}. aborting")
                raise e
        logger.info("Concatenating the DataFrames")
        events = pl.concat(dfs, how="diagonal_relaxed")
        logger.info("Done concatenating the DataFrames")
    else:
        events = pl.read_ndjson(file_path)
        logger.info(f"Done reading {file_path}")

    return TracePolars(
        unnest_all(events),
        truncate_incomplete_func_calls=truncate_incomplete_func_calls,
    )
