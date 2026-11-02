"""Shared, ordered state enumeration for model sampling and snapshots."""

from collections.abc import Iterator
from typing import Literal

from torch import Tensor, nn

TensorKind = Literal["parameter", "buffer"]


def iter_model_tensors(
    model: nn.Module,
    *,
    include_buffers: bool = False,
    remove_duplicate: bool = True,
) -> Iterator[tuple[TensorKind, str, Tensor]]:
    """Yield named parameters, then optionally all buffers, in registration order.

    Defaults retain VarSampler's parameters-only, duplicate-removing traversal.
    Snapshots request every registered name, including tied aliases and
    nonpersistent buffers. None-valued registrations are not materialized state.
    """
    parameters = (
        model.named_parameters()
        if remove_duplicate
        else model.named_parameters(remove_duplicate=False)
    )
    for name, parameter in parameters:
        yield "parameter", name, parameter
    if include_buffers:
        buffers = (
            model.named_buffers()
            if remove_duplicate
            else model.named_buffers(remove_duplicate=False)
        )
        for name, buffer in buffers:
            yield "buffer", name, buffer
