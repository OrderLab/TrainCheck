"""Task-local randomness for inference, independent of process scheduling.

The engine supplies stable semantic task keys in both its serial and process
paths. Direct relation callers outside an engine task keep the legacy stdlib
random API, including control via random.seed(). Sampling rates and reservoirs
are unchanged; the chosen examples can differ from the old global RNG stream.
"""

import hashlib
import json
import random
from collections.abc import Sequence
from contextvars import ContextVar, Token
from types import TracebackType
from typing import Literal, Protocol, TypeAlias, TypeVar, Union

# The JSON encoder accepts tuples as arrays too; keep that existing contract.
SamplingIdentity: TypeAlias = Union[
    None,
    bool,
    int,
    float,
    str,
    list["SamplingIdentity"],
    tuple["SamplingIdentity", ...],
    dict[str, "SamplingIdentity"],
]
_Sample = TypeVar("_Sample")


class SamplingRNG(Protocol):
    """The shared sampling surface of random.Random and the random module."""

    def random(self) -> float: ...

    def sample(self, population: Sequence[_Sample], k: int) -> list[_Sample]: ...


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

    def __init__(self, *identity: SamplingIdentity) -> None:
        self.identity = identity
        self.token: Token[random.Random | None]

    def __enter__(self) -> random.Random:
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

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        # A class context manager also preserves frozen exception objects; the
        # generator-based contextlib helper may assign exception.__traceback__.
        _TASK_RNG.reset(self.token)
        return False


def sampling_rng() -> SamplingRNG:
    """Return the task's RNG, or the legacy module for standalone relation calls."""
    rng = _TASK_RNG.get()
    if rng is None:
        return random
    return rng
