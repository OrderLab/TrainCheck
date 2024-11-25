import logging
import re
from typing import Any

import pandas as pd
from tqdm import tqdm

from mldaikon.config import config
from mldaikon.instrumentor.tracer import TraceLineType
from mldaikon.instrumentor.types import PTID
from mldaikon.trace.trace import Trace
from mldaikon.trace.types import (
    MD_NONE,
    AttrState,
    ContextManagerState,
    FuncCallEvent,
    FuncCallExceptionEvent,
    Liveness,
    VarChangeEvent,
    VarInstId,
)
from mldaikon.trace.utils import (
    bind_args_kwargs_to_signature,
    flatten_dict,
    load_signature_from_class_method_name,
    read_jsonlines_flattened_with_md_none,
)

logger = logging.getLogger(__name__)

STAGE_KEY = "meta_vars.stage"


# TODO: formalize the trace schema for efficient polars processing
def test_dump(df):
    df = df.sort_values(by="func_call_id")
    df.to_json("pandas_trace.json", orient="records", lines=True)


def get_attr_name(col_name: str) -> str:
    if config.VAR_ATTR_PREFIX not in col_name:
        raise ValueError(f"{col_name} does not contain the tracker_var_field_prefix.")
    return col_name[len(config.VAR_ATTR_PREFIX) :]


def safe_isnan(value: Any) -> bool:
    return isinstance(value, float) and pd.isna(value)


class TracePandas(Trace):
    def __init__(self, events, truncate_incomplete_func_calls=True):
        self.events = events

        ## all caches
        self.var_ids = None
        self.var_insts = None
        self.var_changes = None

        self.all_func_call_ids: dict[str, list[str]] = {}
        self.context_manager_states: dict[PTID, list[ContextManagerState]]

        if isinstance(events, list) and all(
            [isinstance(e, pd.DataFrame) for e in events]
        ):
            self.events = pd.concat(events, ignore_index=True)
        elif isinstance(events, list) and all([isinstance(e, dict) for e in events]):
            events = [flatten_dict(e) for e in events]
            self.events = pd.DataFrame(events)

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

        self.column_dtypes_cached = {}

        # HACK: init might not be present at the beginning of the trace due to presence of import-time logs
        self._fill_missing_stage_init()
        self._index_context_manager_meta_vars()

    def get_traces_for_stage(self) -> dict[str, "TracePandas"]:  # type: ignore
        """Get the traces split by stages."""

        if not self.is_stage_annotated():
            raise ValueError("Trace is not annotated with stages.")

        traces = {}
        for stage in self.events[STAGE_KEY].unique():
            traces[stage] = TracePandas(
                self.events[self.events[STAGE_KEY] == stage],
                truncate_incomplete_func_calls=False,
            )

        return traces

    def get_all_stages(self) -> list[str]:
        """Get all stages in the trace."""
        if not self.is_stage_annotated():
            raise ValueError("Trace is not annotated with stages.")

        return self.events[STAGE_KEY].unique().tolist()

    def is_func_called(self, func_name: str, stage: None | str):
        """Check if a function is called in the trace."""
        if stage is not None:
            return (
                func_name
                in self.events[
                    (self.events[STAGE_KEY] == stage)
                    & (self.events["function"] == func_name)
                ]["function"].unique()
            )
        return func_name in self.events["function"].unique()

    def _fill_missing_stage_init(self):
        if STAGE_KEY not in self.events.columns:
            return

        # fill all stage being NaN with "init"
        self.events.loc[self.events[STAGE_KEY].isna(), STAGE_KEY] = "init"

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

        func_call_groups = (
            self.events.groupby("func_call_id").size().reset_index(name="count")
        )

        incomplete_func_call_ids = func_call_groups[func_call_groups["count"] == 1][
            "func_call_id"
        ]

        # print(incomplete_func_call_ids)
        incomplete_func_call_records = self.events[
            self.events["func_call_id"].isin(incomplete_func_call_ids)
        ]
        # test_dump(incomplete_func_call_records)
        for _, row in tqdm(
            incomplete_func_call_records.iterrows(),
            desc="Removing Incomplete Function Calls",
        ):
            assert (
                row["type"] == TraceLineType.FUNC_CALL_PRE
            ), "Incomplete function call is not a pre-call event."
            logger.warning(f"Incomplete function call detected: {row}")
            process_id = row["process_id"]
            thread_id = row["thread_id"]

            filtered_events_pre = self.events[
                (self.events["type"] == TraceLineType.FUNC_CALL_PRE)
                & (self.events["process_id"] == process_id)
                & (self.events["time"] == self.get_start_time(process_id))
            ]

            if not filtered_events_pre.empty:
                outermost_func_call_pre = filtered_events_pre.iloc[0]
            else:
                outermost_func_call_pre = None

            filtered_events_post = self.events[
                (self.events["type"] == TraceLineType.FUNC_CALL_POST)
                & (
                    self.events["func_call_id"]
                    == outermost_func_call_pre["func_call_id"]
                )
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
                ), f"""Incomplete function call (func_call_id: {row['func_call_id']}) (name: {row["function"]}) is not on a different thread than outermost function (func_call_id: {outermost_func_call_pre['func_call_id']}) (name: {outermost_func_call_pre["function"]}) on process {process_id}. Please Investigate."""

                if (
                    row["time"]
                    > outermost_func_call_post["time"]
                    - config.INCOMPLETE_FUNC_CALL_SECONDS_TO_OUTERMOST_POST
                ):
                    logger.warning(f"Removing incomplete function call: {row}")
                    self.events = self.events[
                        self.events["func_call_id"] != row["func_call_id"]
                    ]
                else:
                    raise ValueError(
                        f"Incomplete function call is not close enough to the outermost function call post event: {row}"
                    )

            else:
                # assert row["time"] == self.get_end_time(
                #     process_id, thread_id
                # ), f"Incomplete function call is not the last event for the process {process_id} and thread {thread_id}."

                # logger.warning(
                #     f"Removing incomplete function call: {row} as the outermost function call is also incomplete."
                # )

                # self.events = self.events[self.events["func_call_id"] != row["func_call_id"]]
                # logger.warning(f"Removing incomplete function call: {row}")
                self.events = self.events[
                    (self.events["process_id"] != process_id)
                    | (self.events["thread_id"] != thread_id)
                    | (self.events["time"] < row["time"])
                ]

        # test_dump(self.events)

    def _index_context_manager_meta_vars(self):
        """Identify context manager entry and exit events, and add them to the meta_vars."""
        # 1. Find all trace records that are related to __enter__ and __exit__ functions
        if "function" not in self.events.columns:
            self.context_manager_states = None
            return

        if (
            hasattr(self, "context_manager_states")
            and self.context_manager_states is not None
        ):
            return self.context_manager_states

        all_context_managers: dict[PTID, list[ContextManagerState]] = {}

        context_manager_names = self.events[
            (self.events["function"].str.contains("__enter__", na=False))
            & (
                ~self.events["function"].str.contains(
                    "torch.autograd.grad_mode", na=False
                )
            )  # HACK: ignore all autograd related context managers for now as it doesn't have an __init__ event
            & (
                ~self.events["function"].str.contains(
                    "torch.autograd.profiler.record_function", na=False
                )
            )
        ]["function"].unique()

        context_manager_init_pre_records = []
        for context_manager_name in context_manager_names:
            context_manager_name = context_manager_name.removesuffix(".__enter__")
            context_manager_init_pre_records.extend(
                self.events[
                    (self.events["function"] == f"{context_manager_name}.__init__")
                    & (self.events["type"] == TraceLineType.FUNC_CALL_PRE)
                ].to_dict(orient="records")
            )

        for context_manager_init_pre_record in context_manager_init_pre_records:
            init_pre_record = context_manager_init_pre_record
            process_id = init_pre_record["process_id"]
            thread_id = init_pre_record["thread_id"]

            context_manager_name = init_pre_record["function"].removesuffix(".__init__")

            # find nearest enter and exit events with the same obj_id
            obj_id = init_pre_record["obj_id"]
            try:
                enter_post_record = (
                    self.events[
                        (self.events["type"] == TraceLineType.FUNC_CALL_POST)
                        & (
                            self.events["function"]
                            == f"{context_manager_name}.__enter__"
                        )
                        & (self.events["obj_id"] == obj_id)
                        & (self.events["time"] > init_pre_record["time"])
                        & (self.events["process_id"] == process_id)
                        & (self.events["thread_id"] == thread_id)
                    ]
                    .iloc[0]
                    .to_dict()
                )

                exit_pre_record = (
                    self.events[
                        (self.events["type"] == TraceLineType.FUNC_CALL_PRE)
                        & (
                            self.events["function"]
                            == f"{context_manager_name}.__exit__"
                        )
                        & (self.events["obj_id"] == obj_id)
                        & (self.events["time"] > enter_post_record["time"])
                        & (self.events["process_id"] == process_id)
                        & (self.events["thread_id"] == thread_id)
                    ]
                    .iloc[0]
                    .to_dict()
                )
            except IndexError:
                # sometimes this happens. an enter might not have a corresponding exit, vice versa. e.g. torch.serialization._opener.__enter__ and torch.serialization._open_zipfile_writer_file.__exit__
                logger.warning(
                    f"Context manager {context_manager_name} not used properly. Skipping."
                )
                continue

            assert (
                enter_post_record["time"] < exit_pre_record["time"]
            ), f"enter not before exit: {enter_post_record.to_dict()} {exit_pre_record.to_dict()}"

            start_time, end_time = (
                enter_post_record["time"],
                exit_pre_record["time"],
            )

            args = init_pre_record["args"]
            kwargs = init_pre_record["kwargs"]
            signature = load_signature_from_class_method_name(
                init_pre_record["function"]
            )

            binded_args_and_kwargs = bind_args_kwargs_to_signature(
                args, kwargs, signature
            )

            ptid = PTID(process_id, thread_id)
            if ptid not in all_context_managers:
                all_context_managers[ptid] = []

            logger.debug(f"Adding context manager: {context_manager_name} {ptid}")
            all_context_managers[ptid].append(
                ContextManagerState(
                    name=init_pre_record["function"].removesuffix(".__init__"),
                    ptid=ptid,
                    liveness=Liveness(start_time, end_time),
                    input=binded_args_and_kwargs,
                )
            )

        logger.info(f"Found {len(all_context_managers)} context managers.")

        self.context_manager_states = all_context_managers

    def is_stage_annotated(self):
        # ideally we want to have a static manifest for the trace produced by the instrumentor according to the args
        # but for now, we will check if the trace has the stage column
        return STAGE_KEY in self.events.columns

    def is_var_instrumented_proxy(self):
        # hack: check if "mode" is a column in the trace to determine if the trace is a variable trace
        return "mode" in self.events.columns

    def query_active_context_managers(
        self, time: float, process_id: int, thread_id: int
    ) -> list[ContextManagerState]:
        """Query all active context managers at a given time."""
        if not hasattr(self, "context_manager_states"):
            self._index_context_manager_meta_vars()

        if self.context_manager_states is None:
            return []

        ptid = PTID(process_id, thread_id)
        if ptid not in self.context_manager_states:
            return []

        active_context_managers = []
        for context_manager_state in self.context_manager_states[ptid]:
            if context_manager_state.liveness.is_alive(time):
                active_context_managers.append(context_manager_state)
        return active_context_managers

    def get_meta_vars(
        self, time: float, process_id: int, thread_id: int
    ) -> dict[str, Any] | None:
        """Get the meta_vars a given time.

        Return value:
            dict[str, ]: A dictionary of meta variables. Each key in the dict indicates the type of meta variable and should start with a prefix in the following set: {"context_manager", "rank", "stage", "step"}

        NOTE: CHANGING THE RETURN FORMAT WILL INTERFERE WITH THE PRECONDITION INFERENCE

        """

        # if the process or thread id does not exist in the trace, return None
        if (
            process_id not in self.get_process_ids()
            or thread_id not in self.get_thread_ids()
        ):
            return None

        meta_vars = {}
        active_context_managers = self.query_active_context_managers(
            time, process_id, thread_id
        )

        # hack: flatten the meta-vars data structure so it works with precondition inferece
        prefix = "context_managers"
        for _, context_manager in enumerate(active_context_managers):
            meta_vars[f"{prefix}.{context_manager.name}"] = context_manager.to_dict()[
                "input"
            ]

        return meta_vars

    def get_start_time(self, process_id=None, thread_id=None) -> float:
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

    def get_end_time(self, process_id=None, thread_id=None) -> float:
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

    def get_process_ids(self) -> list[int]:
        """Find all process ids from the trace."""
        if "process_id" not in self.events.columns:
            logger.warning(
                "process_id column not found in the events, no process related invariants will be extracted."
            )
            return []
        return self.events["process_id"].dropna().unique().tolist()

    def get_thread_ids(self) -> list[int]:
        """Find all thread ids from the trace."""
        if "thread_id" not in self.events.columns:
            logger.warning(
                "thread_id column not found in the events, no thread related invariants will be extracted."
            )
            return []
        return self.events["thread_id"].dropna().unique().tolist()

    def get_func_call_ids(self, func_name: str = "") -> list[str]:
        """Find all function call ids from the trace."""
        if "func_call_id" not in self.events.columns:
            logger.warning(
                "func_call_id column not found in the events, no function related invariants will be extracted."
            )
            return []

        if func_name in self.all_func_call_ids:
            return self.all_func_call_ids[func_name]

        if func_name:
            result = (
                self.events[self.events["function"] == func_name]["func_call_id"]
                .dropna()
                .unique()
                .tolist()
            )
            self.all_func_call_ids[func_name] = result
            return result

        assert False, "Why do you need to call this function without a function name?"
        return self.events["func_call_id"].dropna().unique().tolist()

    def get_column_dtype(self, column_name: str) -> type:
        # pandas dataframes are schemaless so we have to find the first non-null value to infer the type

        # TODO: this is painfully slow, we need to find a faster way to infer the type
        # find a value that's not nan and not MD_NONE
        if column_name in self.column_dtypes_cached:
            return self.column_dtypes_cached[column_name]

        filtered_values = self.events[column_name].dropna()
        filtered_values = filtered_values[filtered_values != MD_NONE()]

        if filtered_values.empty:
            self.column_dtypes_cached[column_name] = MD_NONE
            return MD_NONE

        # Return the type of the first valid value
        self.column_dtypes_cached[column_name] = type(filtered_values.iloc[0])
        return self.column_dtypes_cached[column_name]

    def get_func_names(self) -> list[str]:
        """Find all function names from the trace."""
        if "function" not in self.events.columns:
            logger.warning(
                "function column not found in the events, no function related invariants will be extracted."
            )
            return []
        return self.events["function"].dropna().unique().tolist()

    def get_max_num_consecutive_call_func(self, func_name: str) -> int:
        """Find the maximum number of contiguous calls to a function in the trace.

        TODO: THIS IS NOT SOUND NOR THE CORRECT WAY TO GET NUMBER OF CONSECUTIVE CALLS OF A FUNCTION. SHOULD ONLY BE CALLED BY THE LEAD/COVER RELATION AS A TEMPORARY PRUNING SOLUTION.
        """
        # Create group IDs for consecutive function calls
        if not hasattr(self, "_cache_group_counts"):

            events = self.events.copy()
            events["group_id"] = (
                events["function"] != events["function"].shift()
            ).cumsum()

            # Group by group_id and count the consecutive function calls
            group_counts = (
                events.groupby("group_id")
                .agg(function=("function", "first"), count=("function", "size"))
                .reset_index(drop=True)
            )
            self._cache_group_counts = (
                group_counts  # a temporary cache for efficient querying
            )
        else:
            group_counts = self._cache_group_counts
        # Filter for the given function name and find the maximum count
        max_consecutive = group_counts.loc[
            group_counts["function"] == func_name, "count"
        ].max()

        return max_consecutive if pd.notna(max_consecutive) else 0

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
        # TODO: check if this is correct
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

    def get_causally_related_vars(self, func_call_id) -> set[VarInstId]:
        """Find all variables that are causally related to a function call.
        By casually related, we mean that the variables have been accessed or modified by the object (with another method) that the function call is made on.
        """
        # TODO: check if this is correct

        # get the pre-call event of the function call
        # func_call_pre_event = self.events.filter(
        #     pl.col("type") == TraceLineType.FUNC_CALL_PRE,
        #     pl.col("func_call_id") == func_call_id,
        # ).row(
        #     0, named=True
        # )  # order of events doesn't matter as the result here is a single row

        # func_call_pre_event = self.events[(self.events["type"] == TraceLineType.FUNC_CALL_PRE) & (self.events["func_call_id"] == func_call_id)].iloc[0]
        filtered_events = self.events[
            (self.events["type"] == TraceLineType.FUNC_CALL_PRE)
            & (self.events["func_call_id"] == func_call_id)
        ]

        if not filtered_events.empty:
            func_call_pre_event = filtered_events.iloc[0].to_dict()
        else:
            func_call_pre_event = None

        # get the process id of the function call
        assert func_call_pre_event[
            "is_bound_method"
        ], f"Causal relation extraction is only supported for bound methods, got {func_call_pre_event['function']} which is not"

        obj_id = func_call_pre_event[
            "obj_id"
        ]  # the object id (address) of the object that the function is bound to

        # find all function calls that are related to this object prior to the function call

        related_func_call_pre_events = self.events[
            (self.events["type"] == TraceLineType.FUNC_CALL_PRE)
            & (self.events["obj_id"] == obj_id)
            & (self.events["time"] < func_call_pre_event["time"])
        ]

        process_id = func_call_pre_event["process_id"]
        thread_id = func_call_pre_event["thread_id"]

        causally_related_var_ids: set[VarInstId] = set()
        for (
            _,
            related_func_call_pre_event,
        ) in (
            related_func_call_pre_events.iterrows()
        ):  # order of events doesn't matter as we are querying by time when we get the related_func_call_pre_events variable
            # having the assertion failure does not mean that we run into a bug, but it is something that we should investigate
            assert (
                related_func_call_pre_event["process_id"] == process_id
            ), "Related function call is on a different process."
            assert (
                related_func_call_pre_event["thread_id"] == thread_id
            ), "Related function call is on a different thread."
            if safe_isnan(related_func_call_pre_event["proxy_obj_names"]):
                continue
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
        # no change
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
        # TODO: check if this is correct
        if self.var_ids is not None:
            return self.var_ids

        if "var_name" not in self.events.columns:
            logger.warning(
                "var_name column not found in the events, no variable related invariants will be extracted."
            )
            return []

        variables = (
            self.events[["var_name", "var_type", "process_id"]]
            .dropna()
            .drop_duplicates()
        )

        self.var_ids = []
        for _, var_id in variables.iterrows():  # order of events doesn't matter
            self.var_ids.append(
                VarInstId(
                    var_id["process_id"],
                    var_id["var_name"],
                    var_id["var_type"],
                )
            )
        # TODO: possible faster manipulation, need test, if it is faster, change all iterrows to apply
        # self.var_ids.extend(
        #     variables.apply(
        #         lambda var_id: VarInstId(var_id["process_id"], var_id["var_name"], var_id["var_type"]),
        #         axis=1
        #     )
        # )

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
        # TODO: check if this is correct
        if self.var_insts is not None:
            return self.var_insts

        var_ids = self.get_var_ids()
        if len(var_ids) == 0:
            logger.warning("No variables found in the trace.")
            return {}
        var_insts = {}

        for var_id in tqdm(var_ids, desc="Indexing Variable Instances"):
            var_inst_states = self.events[
                (self.events["process_id"] == var_id.process_id)
                & (self.events["var_name"] == var_id.var_name)
                & (self.events["var_type"] == var_id.var_type)
            ].sort_values(
                "time"
            )  # we need to sort the events by time to make sure the order of the events is correct

            state_changes = var_inst_states[
                var_inst_states["type"] == TraceLineType.STATE_CHANGE
            ]

            # init attribute values for this variable
            attr_values = {}
            for _, state_change in state_changes.iterrows():
                for col in state_change.index:

                    if "_ML_DAIKON" in col:
                        # IDs are only reserved for the use of DistinctArgumentRelation
                        continue

                    if safe_isnan(state_change[col]):
                        # skip NaN values as NaNs indicate that the attribute is not present in the state
                        continue

                    if col.startswith(config.VAR_ATTR_PREFIX):
                        attr_name = get_attr_name(col)

                        if any(
                            [
                                re.match(pattern, attr_name) is not None
                                for pattern in config.PROP_ATTR_PATTERNS
                            ]
                        ):
                            continue

                        if attr_name not in attr_values:
                            attr_values[attr_name] = [
                                AttrState(
                                    state_change[col],
                                    Liveness(state_change["time"], None),
                                    [state_change.to_dict()],
                                )
                            ]
                        else:
                            if attr_values[attr_name][-1].value != state_change[
                                col
                            ] and not (
                                safe_isnan(attr_values[attr_name][-1].value)
                                and safe_isnan(state_change[col])
                            ):
                                attr_values[attr_name][-1].liveness.end_time = (
                                    state_change["time"]
                                )
                                attr_values[attr_name].append(
                                    AttrState(
                                        state_change[col],
                                        Liveness(state_change["time"], None),
                                        [state_change.to_dict()],
                                    )
                                )
                            else:
                                attr_values[attr_name][-1].traces.append(
                                    state_change.to_dict()
                                )

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
        # TODO: check if this is correct
        filtered_events = self.events[
            (self.events["process_id"] == var_id.process_id)
            & (self.events["var_name"] == var_id.var_name)
            & (self.events["var_type"] == var_id.var_type)
            & (self.events["time"] < time)
        ]

        return [row.to_dict() for _, row in filtered_events.iterrows()]
        # order of events doesn't matter as we already query by time

    def get_var_changes(self) -> list[VarChangeEvent]:
        """Get all variable changes events from the trace.

        Essentially, this function will comprise consecutive states of the same variable attribute as a single change event.

        Returns:
            list[VarChangeEvent]: A list of all variable change events.
        """
        # no change
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

                    # for debugging
                    # import pandas as pd

                    # if not isinstance(old_state.value, Iterable) and not isinstance(new_state.value, Iterable) and not "_ML_DAIKON" in attr:
                    #     assert not pd.isna(  # AssertionError: Old state is NaN for VarInstId(process_id=374887, var_name='gc1.kernel', var_type='torch.nn.Parameter') _ML_DAIKON_grad_ID (why are those ids progatated to var states?)
                    #         old_state.value
                    #     ), f"Old state is NaN for {var_id} {attr}"
                    #     assert not pd.isna(
                    #         new_state.value
                    #     ), f"New state is NaN for {var_id} {attr}"
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
        # no change
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
        # no change
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
        # no change
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
        # TODO: check if this is correct
        pre_records = self.events[
            (self.events["func_call_id"] == func_call_id)
            & (self.events["type"] == TraceLineType.FUNC_CALL_PRE)
        ]

        assert (
            len(pre_records) == 1
        ), f"{len(pre_records)} pre call events found for {func_call_id}, expected 1"

        pre_record = pre_records.iloc[0].to_dict()

        return pre_record

    def get_post_func_call_record(self, func_call_id: str) -> dict | None:
        """Get the post call record of a function given its func_call_id.
        Returns None if the post call event is not found and the pre-call event is the outermost function call. (see the doc of `_rm_incomplete_trailing_func_calls`)
        """
        # TODO: check if this is correct
        post_records = self.events[
            (self.events["func_call_id"] == func_call_id)
            & (
                self.events["type"].isin(
                    [
                        TraceLineType.FUNC_CALL_POST,
                        TraceLineType.FUNC_CALL_POST_EXCEPTION,
                    ]
                )
            )
        ]

        assert (
            len(post_records) <= 1
        ), f"{post_records.shape[0]} post call events found for {func_call_id}, expected 1"

        if len(post_records) == 0:
            logger.warning(f"No post call event found for {func_call_id}")
            # check if the pre-call event is the outermost function call
            pre_record = self.get_pre_func_call_record(func_call_id)

            outermost_func_call_pre = (
                self.events[
                    (self.events["type"] == TraceLineType.FUNC_CALL_PRE)
                    & (self.events["process_id"] == pre_record["process_id"])
                    & (self.events["thread_id"] == pre_record["thread_id"])
                ]
                .iloc[0]
                .to_dict()
            )

            if pre_record["func_call_id"] == outermost_func_call_pre["func_call_id"]:
                return None
            else:
                raise ValueError(
                    f"No post call event found for {func_call_id}, but it is not the outermost function call."
                )

        post_record = post_records.iloc[
            0
        ].to_dict()  # order of events doesn't matter as the result here is a single row

        return post_record

    def query_func_call_events_within_time(
        self,
        time_range: tuple[int | float, int | float],
        process_id: int,
        thread_id: int,
    ) -> list[FuncCallEvent | FuncCallExceptionEvent]:
        """Extract all function call events from the trace, within a specific time range, process and thread."""
        # TODO: check if this is correct
        func_call_events: list[FuncCallEvent | FuncCallExceptionEvent] = []

        func_call_records = self.events[
            (
                self.events["type"].isin(
                    [
                        TraceLineType.FUNC_CALL_PRE,
                        TraceLineType.FUNC_CALL_POST,
                        TraceLineType.FUNC_CALL_POST_EXCEPTION,
                    ]
                )
            )
            & (
                self.events["time"].between(
                    time_range[0], time_range[1], inclusive="neither"
                )
            )
            & (self.events["process_id"] == process_id)
            & (self.events["thread_id"] == thread_id)
        ]

        # group function calls by func_call_id
        func_call_groups = func_call_records.groupby("func_call_id")

        for func_call_id, func_call_records in func_call_groups:
            assert (
                len(func_call_records) == 2
            ), f"Function call records is not 2 for {func_call_id}"

            pre_record = func_call_records.iloc[0].to_dict()
            post_record = func_call_records.iloc[1].to_dict()

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
        # no change
        pre_record = self.get_pre_func_call_record(func_call_id)
        post_record = self.get_post_func_call_record(func_call_id)
        if post_record is None:
            logger.warning(f"Post call event not found for {func_call_id}")
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


def read_trace_file_Pandas(
    file_path: str | list[str], truncate_incomplete_func_calls=True
) -> TracePandas:
    """Reads the trace file and returns the trace instance."""
    if isinstance(file_path, list):
        dfs = []
        for f in file_path:
            try:
                dfs.append(pd.DataFrame(read_jsonlines_flattened_with_md_none(f)))
                logger.info(f"Done reading {f}")
            except Exception as e:
                logger.error(f"Failed to read {f} due to {e}. aborting")
                raise e
        logger.info("Concatenating the DataFrames")
        events = pd.concat(dfs, ignore_index=True)
        logger.info("Done concatenating the DataFrames")
    else:
        events = pd.DataFrame(read_jsonlines_flattened_with_md_none(file_path))
        logger.info(f"Done reading {file_path}")

    return TracePandas(
        events,
        truncate_incomplete_func_calls=truncate_incomplete_func_calls,
    )
