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

from collections.abc import Mapping
from typing import TypeAlias, TypedDict

import traincheck.config.config as config
import traincheck.utils as _tc_utils
from traincheck.trace.trace import Trace

RandomSeed: TypeAlias = int | float | str | bytes | bytearray | None


class InferenceConfigSnapshot(TypedDict):
    """Required runtime settings transferred between our parent and workers."""

    ENABLE_PRECOND_SAMPLING: bool
    PRECOND_SAMPLING_THRESHOLD: int
    MAX_EXAMPLE_LIST_SIZE: int | None
    EXAMPLE_SAMPLING_SEED: RandomSeed


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


def _validate_snapshot(snapshot: Mapping[str, object]) -> InferenceConfigSnapshot:
    """Reject incomplete/corrupt worker settings before changing process state."""
    required = {
        "ENABLE_PRECOND_SAMPLING",
        "PRECOND_SAMPLING_THRESHOLD",
        "MAX_EXAMPLE_LIST_SIZE",
        "EXAMPLE_SAMPLING_SEED",
    }
    unexpected = set(snapshot) - required
    if unexpected:
        raise ValueError(f"Unexpected inference settings: {sorted(unexpected)}")
    # Direct access makes every field required; missing fields retain KeyError.
    enabled = snapshot["ENABLE_PRECOND_SAMPLING"]
    threshold = snapshot["PRECOND_SAMPLING_THRESHOLD"]
    capacity = snapshot["MAX_EXAMPLE_LIST_SIZE"]
    seed = snapshot["EXAMPLE_SAMPLING_SEED"]
    if type(enabled) is not bool:  # noqa: E721 - booleans are not integer settings
        raise ValueError("ENABLE_PRECOND_SAMPLING must be a bool")
    if type(threshold) is not int:  # noqa: E721 - match the integer CLI contract
        raise ValueError("PRECOND_SAMPLING_THRESHOLD must be an integer")
    if capacity is not None and (
        type(capacity) is not int or capacity < 0  # noqa: E721 - reservoir contract
    ):
        raise ValueError("MAX_EXAMPLE_LIST_SIZE must be a nonnegative integer or None")
    # Preserve the documented random.Random seed types, including None and bytes.
    if seed is not None and not isinstance(seed, (int, float, str, bytes, bytearray)):
        raise ValueError("EXAMPLE_SAMPLING_SEED must be a supported random.Random seed")
    return {
        "ENABLE_PRECOND_SAMPLING": enabled,
        "PRECOND_SAMPLING_THRESHOLD": threshold,
        "MAX_EXAMPLE_LIST_SIZE": capacity,
        "EXAMPLE_SAMPLING_SEED": seed,
    }


def worker_init(traces: list[Trace], snapshot: Mapping[str, object]) -> None:
    """Validate required settings, then initialize this worker atomically.

    A malformed snapshot must not install some settings or replace worker traces.
    The parent supplies config_snapshot(); accepting a Mapping also lets this
    process boundary validate deserialized or otherwise untrusted payloads.
    """
    settings = _validate_snapshot(snapshot)
    global _WORKER_TRACES
    config.ENABLE_PRECOND_SAMPLING = settings["ENABLE_PRECOND_SAMPLING"]
    config.PRECOND_SAMPLING_THRESHOLD = settings["PRECOND_SAMPLING_THRESHOLD"]
    # The legacy config module infers int from defaults; programmatic reservoir
    # options also support None and the other stdlib seed types. Fixed attribute
    # names avoid pretending arbitrary keys are supported configuration fields.
    setattr(config, "MAX_EXAMPLE_LIST_SIZE", settings["MAX_EXAMPLE_LIST_SIZE"])
    setattr(config, "EXAMPLE_SAMPLING_SEED", settings["EXAMPLE_SAMPLING_SEED"])
    _WORKER_TRACES = traces
    _tc_utils._suppress_inner_progress = True


def get_worker_traces() -> list[Trace]:
    """Return the traces stored by worker_init; only valid inside a worker."""
    assert _WORKER_TRACES is not None, "Worker traces were not initialized"
    return _WORKER_TRACES
