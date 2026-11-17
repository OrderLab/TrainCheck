"""Qualified implementation used only by the study operator."""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
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

    if not isinstance(model, DistributedDataParallel):
        raise TypeError("Expected DistributedDataParallel")
    parameters = list(model.parameters())

    def finish(tokens: int) -> None:
        parts = []
        for parameter in parameters:
            if parameter.grad is None:
                raise ValueError("A model parameter has no gradient")
            parts.append(parameter.grad.flatten())
        if token_budget is None:
            for parameter in parameters:
                assert parameter.grad is not None
                parameter.grad.mul_(dist.get_world_size() / tokens)
        else:
            packed = torch.cat(parts)
            dist.all_reduce(packed)
            packed.div_(tokens)
            offset = 0
            for parameter in parameters:
                assert parameter.grad is not None
                after = offset + parameter.numel()
                parameter.grad.copy_(packed[offset:after].view_as(parameter))
                offset = after
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    for inputs, targets in microbatches:

        def backward() -> None:
            logits: Tensor = model(inputs)
            loss = F.cross_entropy(
                logits.flatten(0, 1),
                targets.flatten(),
                ignore_index=IGNORE_INDEX,
                reduction="sum",
            )
            torch.autograd.backward(loss)

        if token_budget is None:
            backward()
        else:
            with model.no_sync():
                backward()
        count = targets.ne(IGNORE_INDEX).sum()
        dist.all_reduce(count)
        total += int(count.item())
        if token_budget is None or total >= token_budget:
            finish(total)
            total = 0
    if total:
        finish(total)
