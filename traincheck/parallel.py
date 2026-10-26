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

import traincheck.config.config as config
import traincheck.utils as _tc_utils

_WORKER_TRACES: list | None = None


def config_snapshot() -> dict:
    """CLI-overridable config values that must be propagated to workers.

    These are set in the entry points' main() at runtime; workers started with
    the 'spawn' method begin from a fresh interpreter and would otherwise see
    only the module defaults (harmless under 'fork'). Add any future
    CLI-overridable config here.
    """
    return {
        "ENABLE_PRECOND_SAMPLING": config.ENABLE_PRECOND_SAMPLING,
        "PRECOND_SAMPLING_THRESHOLD": config.PRECOND_SAMPLING_THRESHOLD,
    }


def worker_init(traces: list, snapshot: dict) -> None:
    """Initializer run once per worker process.

    Stores the traces in a module global (one copy per worker, not per task) and
    restores CLI-overridable config values (see config_snapshot). Also suppresses
    the per-invariant inner progress bars, otherwise every worker would fight
    over the terminal.
    """
    global _WORKER_TRACES
    _WORKER_TRACES = traces
    for name, value in snapshot.items():
        setattr(config, name, value)
    _tc_utils._suppress_inner_progress = True


def get_worker_traces() -> list:
    """Return the traces stored by worker_init; only valid inside a worker."""
    assert _WORKER_TRACES is not None, "Worker traces were not initialized"
    return _WORKER_TRACES
