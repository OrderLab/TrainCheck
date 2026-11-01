"""Full-batch training update."""

from __future__ import annotations

import torch
import torch.nn as nn

from .engine import Batch, backward, loss_for_batch


def run_update(
    model: nn.Module, optimizer: torch.optim.Optimizer, batches: list[Batch]
) -> float:
    if not batches:
        raise ValueError("An update requires at least one batch")
    optimizer.zero_grad(set_to_none=True)
    combined = Batch(
        torch.cat([batch.tokens for batch in batches]),
        torch.cat([batch.labels for batch in batches]),
    )
    loss = loss_for_batch(model, combined)
    backward(loss, 1)
    optimizer.step()
    return float(loss.detach().item())
