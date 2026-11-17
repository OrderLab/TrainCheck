"""Training update policy; called by the ordinary DDP entry point."""

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
    if token_budget is not None:
        raise NotImplementedError("Token-budget mode is not implemented")

    model.train()
    for inputs, targets in microbatches:
        optimizer.zero_grad(set_to_none=True)
        logits: Tensor = model(inputs)
        losses = F.cross_entropy(
            logits.flatten(0, 1),
            targets.flatten(),
            ignore_index=IGNORE_INDEX,
            reduction="sum",
        )
        target_count = (targets != IGNORE_INDEX).sum()
        dist.all_reduce(target_count)
        (losses * (dist.get_world_size() / target_count)).backward()
        optimizer.step()
