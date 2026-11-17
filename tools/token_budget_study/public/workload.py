"""Small deterministic token-level training workload."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

Batch = tuple[Tensor, Tensor]
IGNORE_INDEX = -100
VOCABULARY = 29
WIDTH = 12
ROWS = 2
COLUMNS = 12


class TokenModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(VOCABULARY, WIDTH)
        self.projection = nn.Linear(WIDTH, VOCABULARY)

    def forward(self, tokens: Tensor) -> Tensor:
        logits: Tensor = self.projection(self.embedding(tokens).tanh())
        return logits


def batches(
    counts: Sequence[int], rank: int, seed: int, device: torch.device
) -> list[Batch]:
    generator = torch.Generator().manual_seed(seed + 8191 * rank)
    result: list[Batch] = []
    for count in counts:
        if not 0 <= count <= ROWS * COLUMNS:
            raise ValueError("Target count must fit the microbatch")
        size = (ROWS, COLUMNS)
        tokens = torch.randint(VOCABULARY, size, generator=generator)
        targets = torch.randint(VOCABULARY, size, generator=generator)
        padding = torch.randperm(ROWS * COLUMNS, generator=generator)[count:]
        targets.view(-1)[padding] = IGNORE_INDEX
        result.append((tokens.to(device), targets.to(device)))
    return result


def make_optimizer(model: nn.Module) -> torch.optim.SGD:
    return torch.optim.SGD(model.parameters(), lr=0.08, momentum=0.7)
