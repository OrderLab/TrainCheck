"""Capture the actual loss multiplier with native TrainCheck API records.

This helper adds scalar observations for one experimental predicate. It does not
infer invariants or register a new built-in TrainCheck relation. Window counts
come from the workload; they are not independently recounted here.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

OBSERVATION_FIELDS = (
    "rank",
    "world_size",
    "step",
    "microbatch",
    "accumulation",
    "local_tokens",
    "global_tokens",
    "applied_scale",
)


def apply_loss_scale(
    loss: torch.Tensor,
    *,
    rank: int,
    world_size: int,
    step: int,
    microbatch: int,
    accumulation: int,
    local_tokens: int,
    global_tokens: int,
    applied_scale: float,
) -> torch.Tensor:
    """Apply the supplied multiplier, retaining the original autograd operation."""
    result: object = loss * applied_scale
    if not isinstance(result, torch.Tensor):
        raise TypeError("Loss multiplication must return a Tensor")
    return result


def wrapped_loss_scale() -> Callable[..., object]:
    """Use the existing wrapper and capture only the primitive keyword arguments.

    The positional loss tensor and return tensor are intentionally unobserved;
    this predicate uses the scalar multiplier, not tensor values or metadata.
    No predicate runs in the training process. TraceSession owns tracing gates
    and buffer drainage, as it does for ordinary PyTorch API wrappers.
    """
    from traincheck.instrumentor.tracer import wrapper

    argument_config: dict[str, object] = {key: {} for key in OBSERVATION_FIELDS}
    return wrapper(
        apply_loss_scale,
        is_bound_method=False,
        scan_proxy_in_args=False,
        dump_stack_trace=False,
        dump_args_config=argument_config,
        dump_ret=False,
        handle_proxy=False,
        trigger_var_dump=False,
    )
