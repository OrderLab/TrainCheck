"""Check normalization of token-summed losses accumulated under default DDP."""

from dataclasses import dataclass
from math import isclose, isfinite

RULE_ID: str = "ddp_accumulated_global_token_mean"
RELATIVE_TOLERANCE: float = 1e-12
ABSOLUTE_TOLERANCE: float = 0.0


@dataclass(frozen=True)
class Observation:
    """One loss application; step is one-based and microbatch is zero-based.

    local_tokens and global_tokens cover the complete optimizer-update window,
    including every accumulation microbatch. applied_scale is the actual Python
    float multiplying this microbatch's token-summed loss before backward.
    """

    rank: int
    world_size: int
    step: int
    microbatch: int
    accumulation: int
    local_tokens: int
    global_tokens: int
    applied_scale: float


@dataclass(frozen=True)
class Decision:
    rule_id: str
    passed: bool
    observed: float
    expected: float
    explanation: str


def _require_integer(name: str, value: object) -> int:
    """Accept integer values without treating booleans as integer metadata."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, excluding bool")
    return value


def _require_finite_float(name: str, value: object) -> float:
    """Require an actual built-in Python float, not a coerced numeric value."""
    value_type: type[object] = type(value)
    if not isinstance(value, float) or value_type is not float:
        raise TypeError(f"{name} must be a Python float")
    if not isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def evaluate(observation: Observation) -> Decision:
    """Check the multiplier needed for a complete-window global token mean.

    Malformed or unsupported observations raise TypeError or ValueError. A
    finite but mathematically incorrect multiplier produces a failed Decision.
    The caller must treat validation exceptions as invalid observations, never
    as passes or as evidence that this rule executed successfully.
    """
    if not isinstance(observation, Observation):
        raise TypeError("observation must be an Observation")

    rank: int = _require_integer("rank", observation.rank)
    world_size: int = _require_integer("world_size", observation.world_size)
    step: int = _require_integer("step", observation.step)
    microbatch: int = _require_integer("microbatch", observation.microbatch)
    accumulation: int = _require_integer("accumulation", observation.accumulation)
    local_tokens: int = _require_integer("local_tokens", observation.local_tokens)
    global_tokens: int = _require_integer("global_tokens", observation.global_tokens)
    applied_scale: float = _require_finite_float(
        "applied_scale", observation.applied_scale
    )

    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if rank < 0 or rank >= world_size:
        raise ValueError("rank must satisfy 0 <= rank < world_size")
    if step < 1:
        raise ValueError("step must be a positive, one-based update index")
    if accumulation <= 0:
        raise ValueError("accumulation must be positive")
    if microbatch < 0 or microbatch >= accumulation:
        raise ValueError("microbatch must satisfy 0 <= microbatch < accumulation")
    if local_tokens <= 0:
        raise ValueError("local_tokens must be positive for this probe")
    if global_tokens <= 0:
        raise ValueError("global_tokens must be positive")
    if world_size == 1 and global_tokens != local_tokens:
        raise ValueError("single-rank local_tokens and global_tokens must agree")
    if global_tokens < local_tokens + world_size - 1:
        raise ValueError(
            "global_tokens must include local_tokens and positive totals for "
            "every other rank"
        )

    try:
        expected: float = world_size / global_tokens
    except OverflowError as error:
        raise ValueError(
            "required multiplier is not representable as a float"
        ) from error
    if not isfinite(expected) or expected <= 0.0:
        raise ValueError(
            "required multiplier must be representable as a finite positive float"
        )

    passed: bool = isclose(
        applied_scale,
        expected,
        rel_tol=RELATIVE_TOLERANCE,
        abs_tol=ABSOLUTE_TOLERANCE,
    )
    status: str = "matches" if passed else "does not match"
    explanation: str = (
        f"Applied multiplier {applied_scale!r} {status} world_size/global_tokens "
        f"= {expected!r}. Default DDP averages gradients across ranks, so this "
        "factor makes each token-summed microbatch contribute to the mean over "
        "all valid tokens in the complete optimizer-update window. No additional "
        "accumulation divisor is needed because global_tokens already covers "
        f"the full window. Comparison uses rel_tol={RELATIVE_TOLERANCE!r}, "
        f"abs_tol={ABSOLUTE_TOLERANCE!r}."
    )
    return Decision(
        rule_id=RULE_ID,
        passed=passed,
        observed=applied_scale,
        expected=expected,
        explanation=explanation,
    )
