"""Rank-local TrainCheck setup for the distributed qualification workload.

Install once in a fresh worker process, before DDP/optimizer construction. The
API wrappers persist until process exit; only explicitly delimited update scopes
emit events. Subclass mode samples the registered model parameters at update
boundaries in addition to their ordinary assignment observations. It does not
register optimizer state or implement cross-rank invariant checking.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypedDict

if TYPE_CHECKING:
    from torch import nn

Mode = Literal["plain", "api", "subclass"]


class TraceReport(TypedDict):
    mode: Mode
    instrumented_apis: int
    scan_proxy_in_args: bool
    registered_parameter_names: list[str]
    completed_trace_steps: list[int]
    tensor_hash_enabled: bool
    model_sampling: str
    optimizer_state_registered: bool
    cross_rank_checks_executed: bool
    trace_directory: str
    buffers_drained: bool
    trace_record_counts: None


@dataclass
class TraceSession:
    """Own one worker's tracing gates; suppression restores its prior state."""

    mode: Mode
    output: Path
    instrumented_apis: int
    parameter_names: tuple[str, ...]
    _closed: bool = False
    _in_step: bool = False
    _steps: list[int] = field(default_factory=list)

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("Trace session is already closed")

    @contextmanager
    def suppress(self) -> Iterator[None]:
        """Exclude bookkeeping from API and subclass records, including errors."""
        self._require_open()
        if self.mode == "plain":
            yield
            return
        from traincheck.config import config
        from traincheck.instrumentor.tracer import suppress_tracing

        previous = config.DISABLE_WRAPPER
        config.DISABLE_WRAPPER = True
        try:
            with suppress_tracing():
                yield
        finally:
            config.DISABLE_WRAPPER = previous

    def _sample_parameters(self, boundary: Literal["before", "after"]) -> None:
        if self.mode != "subclass":
            return
        from traincheck.instrumentor.proxy_wrapper.proxy_registry import (
            get_global_registry,
        )
        from traincheck.instrumentor.tracer import suppress_tracing

        # Keep model dumping enabled while excluding sampling's own Torch APIs.
        # An empty per-type value means full metadata in the existing registry.
        with suppress_tracing():
            get_global_registry().dump_sample(
                dump_loc=f"distributed_probe.update.{boundary}",
                dump_config={"torch.nn.Parameter": {}},
            )

    @contextmanager
    def step(self, step: int) -> Iterator[None]:
        """Trace an absolute update number; a fresh resume may begin after one."""
        self._require_open()
        if not isinstance(step, int) or isinstance(step, bool) or step < 1:
            raise ValueError("Trace step must be a positive integer")
        if self._in_step:
            raise RuntimeError("Trace update scopes cannot overlap")
        if self._steps and step != self._steps[-1] + 1:
            raise ValueError("Trace update numbers must be consecutive")
        self._in_step = True
        try:
            if self.mode == "plain":
                yield
            else:
                from traincheck.config import config
                from traincheck.instrumentor.caches import META_VARS
                from traincheck.instrumentor.control import start_step

                META_VARS["step"] = step - 1
                start_step()
                try:
                    self._sample_parameters("before")
                    yield
                    self._sample_parameters("after")
                finally:
                    config.DISABLE_WRAPPER = True
            # A failed update is not reported as a completed trace scope.
            self._steps.append(step)
        finally:
            self._in_step = False

    def close(self) -> TraceReport:
        """Drain main-thread buffers; actual record counts require exited workers.

        The dumper owns its writer threads and closes their files at process exit.
        Queue completion does not itself prove filesystem flushing, nor coverage
        of buffers created by other application threads. The launcher audits all
        trace files only after the worker process exits successfully.
        """
        self._require_open()
        if self._in_step:
            raise RuntimeError("Cannot close tracing inside an update")
        if self.mode != "plain":
            from traincheck.config import config
            from traincheck.instrumentor import dumper

            config.DISABLE_WRAPPER = True
            with self.suppress():
                for name in ("api_trace_buffer", "var_trace_buffer"):
                    # No buffer means this thread emitted no events of that kind.
                    buffer: object = getattr(dumper.THREAD_DATA, name, None)
                    if buffer is not None:
                        if not isinstance(buffer, dumper.TraceBuffer):
                            raise TypeError(f"Unexpected trace buffer: {name}")
                        buffer._flush()
                        buffer.queue.join()
        self._closed = True
        return {
            "mode": self.mode,
            "instrumented_apis": self.instrumented_apis,
            "scan_proxy_in_args": self.mode == "subclass",
            "registered_parameter_names": list(self.parameter_names),
            "completed_trace_steps": list(self._steps),
            "tensor_hash_enabled": self.mode != "plain",
            "model_sampling": (
                "explicit before/after update plus subclass assignment observations"
                if self.mode == "subclass"
                else "disabled"
            ),
            "optimizer_state_registered": False,
            "cross_rank_checks_executed": False,
            "trace_directory": str(self.output),
            "buffers_drained": True,
            "trace_record_counts": None,
        }


def start_tracing(mode: Mode, output: Path, model: nn.Module) -> TraceSession:
    """Install normal full-collection wrappers without source rewriting/env dump."""
    if mode not in ("plain", "api", "subclass"):
        raise ValueError("Unknown tracing mode")
    output.mkdir(parents=True, exist_ok=False)
    if mode == "plain":
        return TraceSession(mode, output, 0, ())

    import torch

    if not isinstance(model, torch.nn.Module):
        raise TypeError("Tracked model must be a torch.nn.Module")
    os.environ["TRAINCHECK_OUTPUT_DIR"] = str(output.resolve())

    from traincheck.config import config
    from traincheck.instrumentor import tracer
    from traincheck.instrumentor.caches import META_VARS
    from traincheck.instrumentor.proxy_wrapper.proxy_config import tensor_dump_format

    if META_VARS["step"] != 0 or tracer.modules_or_cls_id_instrumented:
        raise RuntimeError("Trace adapter requires a fresh worker process")
    config.DISABLE_WRAPPER = True
    config.MODEL_TRACKER_STYLE = "subclass" if mode == "subclass" else None
    config.INSTRUMENTATION_POLICY = {"interval": 1, "warm_up": 0}
    config.INSTR_OPTS = None
    config.INSTR_OPTS_FILE = str(output / "instr_opts.json")
    tensor_dump_format.update(
        dump_tensor_hash=True, dump_tensor_stats=False, dump_tensor_full=False
    )
    count = tracer.Instrumentor(
        torch,
        scan_proxy_in_args=mode == "subclass",
        use_full_instr=False,
        API_dump_stack_trace=False,
        sampling_interval=1,
        warm_up_steps=0,
    ).instrument()
    if count <= 0:
        raise RuntimeError("Instrumentor did not wrap any API")
    parameter_names: tuple[str, ...] = ()
    session = TraceSession(mode, output, count, parameter_names)
    if mode == "subclass":
        from traincheck.instrumentor.proxy_wrapper.subclass import (
            ProxyParameter,
            proxy_parameter,
        )

        with session.suppress():
            proxy_parameter(model, parent_name="model", should_dump_trace=False)
            named = list(model.named_parameters())
            if not named:
                raise ValueError("Model must contain parameters for subclass tracking")
            if any(not isinstance(parameter, ProxyParameter) for _, parameter in named):
                raise RuntimeError("Some model parameters were not subclass-tracked")
            parameter_names = tuple("model." + name for name, _ in named)
        session.parameter_names = parameter_names
    return session
