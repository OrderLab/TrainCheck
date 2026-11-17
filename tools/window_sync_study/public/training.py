"""Streaming gradient accumulation for the two-rank token training workload."""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.nn import functional as F
from workload import IGNORE_INDEX, Batch


def train_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    microbatches: Iterable[Batch],
    token_budget: int | None,
) -> None:
    if token_budget is not None and token_budget <= 0:
        raise ValueError("Expected a positive token budget")
    model.train()
    total = 0
    optimizer.zero_grad(set_to_none=True)

    def finish(tokens: int) -> None:
        for parameter in model.parameters():
            if parameter.grad is None:
                raise ValueError("A model parameter has no gradient")
            parameter.grad.mul_(dist.get_world_size() / tokens)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    for inputs, targets in microbatches:
        logits: Tensor = model(inputs)
        loss = F.cross_entropy(
            logits.flatten(0, 1),
            targets.flatten(),
            ignore_index=IGNORE_INDEX,
            reduction="sum",
        )
        torch.autograd.backward(loss)
        count = targets.ne(IGNORE_INDEX).sum()
        dist.all_reduce(count)
        total += int(count.item())
        if token_budget is None or total >= token_budget:
            finish(total)
            total = 0
    if total:
        finish(total)
