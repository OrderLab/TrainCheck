"""Task-local randomness for inference, independent of process scheduling.

The engine supplies stable semantic task keys in both its serial and process
paths. Direct relation callers outside an engine task keep the legacy stdlib
random API, including control via random.seed(). Sampling rates and reservoirs
are unchanged; the chosen examples can differ from the old global RNG stream.
"""

import hashlib
import json
import random
from contextvars import ContextVar

_TASK_RNG: ContextVar[random.Random | None] = ContextVar(
    "inference_task_rng", default=None
)


class sampling_task:
    """Give one logical task a fresh reproducible RNG without touching global state.

    Keys must be JSON values with stable ordering, never Python hash(), object
    addresses, worker IDs, or arrival order. Trace indexes refer to the engine's
    ordered input list. A versioned digest keeps seeds stable across processes
    and PYTHONHASHSEED values. ContextVar restores enclosing scopes on exceptions
    and isolates concurrently executing threads.
    """

    def __init__(self, *identity):
        self.identity = identity

    def __enter__(self):
        payload = json.dumps(
            ["traincheck-inference-sampling-v1", 0, *self.identity],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        rng = random.Random(int.from_bytes(hashlib.sha256(payload).digest(), "big"))
        self.token = _TASK_RNG.set(rng)
        return rng

    def __exit__(self, exc_type, exc_value, traceback):
        # A class context manager also preserves frozen exception objects; the
        # generator-based contextlib helper may assign exception.__traceback__.
        _TASK_RNG.reset(self.token)
        return False


def sampling_rng():
    """Return the task's RNG, or the legacy module for standalone relation calls."""
    rng = _TASK_RNG.get()
    return random if rng is None else rng
