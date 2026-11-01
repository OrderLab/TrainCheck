"""Fixed model, objective, and reproducible data for the continuation task."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

VOCABULARY = 32
SEQUENCE_LENGTH = 8
EXAMPLES_PER_RANK = 8
LAYERS = 4


@dataclass(frozen=True)
class Batch:
    tokens: torch.Tensor
    labels: torch.Tensor


def make_layer() -> nn.TransformerEncoderLayer:
    return nn.TransformerEncoderLayer(
        d_model=16, nhead=2, dim_feedforward=32, dropout=0.1, batch_first=True
    )


class Transformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(VOCABULARY, 16)
        self.encoder = nn.TransformerEncoder(make_layer(), num_layers=LAYERS)
        # TransformerEncoder clones its prototype. Construct the remaining
        # blocks independently while retaining each module's default initializer.
        for index in range(1, LAYERS):
            self.encoder.layers[index] = make_layer()
        self.readout = nn.Linear(16, VOCABULARY)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        value: object = self.readout(self.encoder(self.embedding(tokens)))
        if not isinstance(value, torch.Tensor):
            raise TypeError("Transformer output must be a tensor")
        return value


def make_batch(seed: int, cursor: int, rank: int, device: torch.device) -> Batch:
    if seed < 0 or cursor < 0 or rank < 0:
        raise ValueError("Seed, cursor, and rank must be nonnegative")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 10000 * cursor + 100 * rank)
    tokens = torch.randint(
        VOCABULARY,
        (EXAMPLES_PER_RANK, SEQUENCE_LENGTH),
        generator=generator,
    )
    return Batch(tokens.to(device), ((tokens + 1) % VOCABULARY).to(device))


def microbatches(batch: Batch, accumulation: int) -> list[Batch]:
    if accumulation < 1 or EXAMPLES_PER_RANK % accumulation:
        raise ValueError("Accumulation must be a positive divisor of eight")
    expected = (EXAMPLES_PER_RANK, SEQUENCE_LENGTH)
    if tuple(batch.tokens.shape) != expected or tuple(batch.labels.shape) != expected:
        raise ValueError("Unexpected complete batch shape")
    size = EXAMPLES_PER_RANK // accumulation
    return [
        Batch(batch.tokens[start : start + size], batch.labels[start : start + size])
        for start in range(0, EXAMPLES_PER_RANK, size)
    ]


def loss_for_batch(model: nn.Module, batch: Batch) -> torch.Tensor:
    output: object = model(batch.tokens)
    if not isinstance(output, torch.Tensor):
        raise TypeError("Model output must be a tensor")
    result: object = F.cross_entropy(
        output.reshape(-1, VOCABULARY), batch.labels.reshape(-1), reduction="mean"
    )
    if not isinstance(result, torch.Tensor):
        raise TypeError("Cross entropy must be a tensor")
    return result


def evaluate(model: nn.Module, seed: int, device: torch.device) -> float:
    modes = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        with torch.no_grad():
            losses = [
                loss_for_batch(model, make_batch(seed, cursor, 0, device)).double()
                for cursor in range(8)
            ]
            result = float(torch.stack(losses).mean().item())
        if not torch.isfinite(torch.tensor(result)):
            raise ValueError("Validation loss is not finite")
        return result
    finally:
        for module, training in modes:
            module.training = training
