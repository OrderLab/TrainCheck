"""Microbatch training update."""

from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

from .engine import Batch, backward, loss_for_batch


def run_update(
    model: nn.Module, optimizer: torch.optim.Optimizer, batches: list[Batch]
) -> float:
    if not batches:
        raise ValueError("An update requires at least one batch")
    optimizer.zero_grad(set_to_none=True)
    accumulation = len(batches)
    total = torch.zeros((), device=batches[0].tokens.device)
    for index, batch in enumerate(batches):
        context = (
            model.no_sync()
            if isinstance(model, DistributedDataParallel) and index + 1 < accumulation
            else nullcontext()
        )
        with context:
            loss = loss_for_batch(model, batch)
            total += loss.detach()
            backward(loss / accumulation, accumulation)
    optimizer.step()
    return float((total / accumulation).item())
