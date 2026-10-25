"""Shared helpers for process-based parallelism in the infer engine and checker.

Both traincheck-infer and traincheck-check fan CPU-bound, per-invariant work out
across a ProcessPoolExecutor (threads would serialize on the GIL). The pattern is:

  - The parent passes the traces to each worker exactly once, via the pool's
    initializer (``worker_init``). Workers keep them in a module global so tasks
    reuse them instead of re-pickling the (potentially large) traces per task.
  - Workers only *compute* and return picklable results; all mutation of shared
    state stays in the parent process.

Task functions live next to their call sites (infer_engine.py / checker.py) and
access the traces through ``get_worker_traces()``.
"""

from typing import TypedDict

import traincheck.config.config as config
import traincheck.utils as _tc_utils
from traincheck.trace.trace import Trace


class InferenceConfigSnapshot(TypedDict):
    """Required runtime settings transferred between our parent and workers."""

    ENABLE_PRECOND_SAMPLING: bool
    PRECOND_SAMPLING_THRESHOLD: int
    MAX_EXAMPLE_LIST_SIZE: int | None
    EXAMPLE_SAMPLING_SEED: int


_WORKER_TRACES: list[Trace] | None = None


def config_snapshot() -> InferenceConfigSnapshot:
    """Runtime inference config values that must be propagated to workers.

    These include CLI sampling options and programmatic reservoir settings.
    Workers started with 'spawn' begin from a fresh interpreter and would
    otherwise see only the module defaults. Add future runtime-overridable
    inference configuration here.
    """
    return {
        "ENABLE_PRECOND_SAMPLING": config.ENABLE_PRECOND_SAMPLING,
        "PRECOND_SAMPLING_THRESHOLD": config.PRECOND_SAMPLING_THRESHOLD,
        "MAX_EXAMPLE_LIST_SIZE": config.MAX_EXAMPLE_LIST_SIZE,
        "EXAMPLE_SAMPLING_SEED": config.EXAMPLE_SAMPLING_SEED,
    }


def worker_init(traces: list[Trace], snapshot: InferenceConfigSnapshot) -> None:
    """Initializer run once per worker process.

    Stores the traces in a module global (one copy per worker, not per task) and
    restores runtime inference config values (see config_snapshot). Also suppresses
    the per-invariant inner progress bars, otherwise every worker would fight
    over the terminal.
    """
    global _WORKER_TRACES
    _WORKER_TRACES = traces
    for name, value in snapshot.items():
        setattr(config, name, value)
    _tc_utils._suppress_inner_progress = True


def get_worker_traces() -> list[Trace]:
    """Return the traces stored by worker_init; only valid inside a worker."""
    assert _WORKER_TRACES is not None, "Worker traces were not initialized"
    return _WORKER_TRACES
