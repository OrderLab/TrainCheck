"""Frozen legacy query methods used as a differential oracle.

Source: OrderLab/TrainCheck commit 9eb4f320d3db34d372145261704b805b88e39bda,
traincheck/trace/trace_pandas.py (SHA256
 e1bb62eae653f95c5498dc921d991af013e8b5717398f94cfbc718812bb600ad).
Only four exact original methods are retained. The production constructor and
supporting types are shared so tests isolate these query/index changes. Defining
this class as TracePandas intentionally makes the original stage factory create
legacy instances too. See legacy_trace_pandas.provenance.json for method hashes.
"""

import logging

import pandas as pd

from traincheck.instrumentor.tracer import TraceLineType
from traincheck.instrumentor.types import PTID
from traincheck.trace.trace_pandas import STAGE_KEY
from traincheck.trace.trace_pandas import TracePandas as ProductionTracePandas
from traincheck.trace.types import ContextManagerState, Liveness
from traincheck.trace.utils import (
    BindedFuncInput,
    bind_args_kwargs_to_signature,
    load_signature_from_class_method_name,
)
from traincheck.utils import safe_isnan

logger = logging.getLogger(__name__)


class TracePandas(ProductionTracePandas):
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
            if not safe_isnan(args):
                signature = load_signature_from_class_method_name(
                    init_pre_record["function"]
                )

                binded_args_and_kwargs = bind_args_kwargs_to_signature(
                    args, kwargs, signature
                )
            else:
                # create an empty BindedFuncInput if args is NaN, as it indicates
                # that we did not record the args and kwargs for this function call
                binded_args_and_kwargs = BindedFuncInput({})

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

    def get_traces_for_stage(self) -> dict[str, "TracePandas"]:  # type: ignore
        """Get the traces split by stages."""

        if not self.is_stage_annotated():
            raise ValueError("Trace is not annotated with stages.")

        if self.stage_traces:
            return self.stage_traces

        traces = {}
        for stage in self.events[STAGE_KEY].unique():
            traces[stage] = TracePandas(
                self.events[self.events[STAGE_KEY] == stage],
                truncate_incomplete_func_calls=False,
            )

        self.stage_traces = traces

        return traces
