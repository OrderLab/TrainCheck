"""Run a reproducible token-prediction training experiment."""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

VOCAB_SIZE = 32
SEQUENCE_LENGTH = 8
EXAMPLES_PER_RANK_UPDATE = 8
EVALUATION_BATCHES = 8


@dataclass(frozen=True)
class Batch:
    tokens: torch.Tensor
    labels: torch.Tensor


@dataclass(frozen=True)
class Options:
    pipeline: Literal["reference", "current"]
    output: Path
    steps: int = 64
    accumulation: int = 4
    seed: int = 20261101
    lr: float = 0.15
    device: Literal["cpu", "cuda"] = "cuda"
    eval_seed: int = 777000

    def validate(self) -> None:
        if self.pipeline not in ("reference", "current"):
            raise ValueError("pipeline must be reference or current")
        if self.device not in ("cpu", "cuda"):
            raise ValueError("device must be cpu or cuda")
        for name, value in (
            ("steps", self.steps),
            ("accumulation", self.accumulation),
            ("seed", self.seed),
            ("eval_seed", self.eval_seed),
        ):
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an integer")
        if not 1 <= self.steps <= 128:
            raise ValueError("steps must be between 1 and 128")
        if self.accumulation < 1 or EXAMPLES_PER_RANK_UPDATE % self.accumulation:
            raise ValueError("accumulation must be a positive divisor of 8")
        if not 0 <= self.seed < 2**31 or not 0 <= self.eval_seed < 2**31:
            raise ValueError("seed and eval_seed must be in [0, 2**31)")
        if isinstance(self.lr, bool) or not isinstance(self.lr, (int, float)):
            raise TypeError("lr must be numeric")
        if not math.isfinite(self.lr) or self.lr <= 0:
            raise ValueError("lr must be finite and positive")


class TinyTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, 16)
        layer = nn.TransformerEncoderLayer(
            d_model=16,
            nhead=2,
            dim_feedforward=32,
            dropout=0.0,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.readout = nn.Linear(16, VOCAB_SIZE)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        result: object = self.readout(self.encoder(self.embedding(tokens)))
        if not isinstance(result, torch.Tensor):
            raise TypeError("Model output must be a tensor")
        return result


def make_batch(seed: int, step: int, rank: int, device: torch.device) -> Batch:
    """Regenerate a complete rank batch without consuming the training RNG."""
    if seed < 0 or step < 0 or rank < 0:
        raise ValueError("seed, step, and rank must be nonnegative")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 10000 * step + 100 * rank)
    tokens = torch.randint(
        VOCAB_SIZE,
        (EXAMPLES_PER_RANK_UPDATE, SEQUENCE_LENGTH),
        generator=generator,
    )
    labels = (tokens + 1) % VOCAB_SIZE
    return Batch(tokens.to(device), labels.to(device))


def split_batch(batch: Batch, accumulation: int) -> list[Batch]:
    if accumulation < 1 or EXAMPLES_PER_RANK_UPDATE % accumulation:
        raise ValueError("accumulation must be a positive divisor of 8")
    if batch.tokens.shape != batch.labels.shape or tuple(batch.tokens.shape) != (
        EXAMPLES_PER_RANK_UPDATE,
        SEQUENCE_LENGTH,
    ):
        raise ValueError("Unexpected complete batch shape")
    size = EXAMPLES_PER_RANK_UPDATE // accumulation
    return [
        Batch(batch.tokens[start : start + size], batch.labels[start : start + size])
        for start in range(0, EXAMPLES_PER_RANK_UPDATE, size)
    ]


def loss_for_batch(model: nn.Module, batch: Batch) -> torch.Tensor:
    result: object = model(batch.tokens)
    if not isinstance(result, torch.Tensor):
        raise TypeError("Model output must be a tensor")
    loss: object = F.cross_entropy(
        result.reshape(-1, VOCAB_SIZE), batch.labels.reshape(-1), reduction="mean"
    )
    if not isinstance(loss, torch.Tensor):
        raise TypeError("Cross entropy must return a tensor")
    return loss


def backward(loss: torch.Tensor, accumulation: int) -> None:
    """Accumulate one microbatch's contribution to the current update."""
    if accumulation < 1:
        raise ValueError("accumulation must be positive")
    (loss / accumulation).backward()


def evaluate(model: nn.Module, device: torch.device, seed: int = 777000) -> float:
    """Compute mean held-out token loss and restore every module's mode.

    The fixed evaluation data uses private generators. Evaluation does not
    consume CPU/CUDA RNG streams or update the model/optimizer.
    """
    modes = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        with torch.no_grad():
            total = torch.zeros((), dtype=torch.float64, device=device)
            for index in range(EVALUATION_BATCHES):
                batch = make_batch(seed, index, 0, device)
                total += loss_for_batch(model, batch).to(torch.float64)
            value = float((total / EVALUATION_BATCHES).item())
    finally:
        for module, training in modes:
            module.training = training
    if not math.isfinite(value):
        raise ValueError("Evaluation loss must be finite")
    return value


def parse_options(argv: list[str] | None = None) -> Options:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline", choices=("reference", "current"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261101)
    parser.add_argument("--eval-seed", type=int, default=777000)
    parser.add_argument("--lr", type=float, default=0.15)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args(argv)
    options = Options(
        args.pipeline,
        args.output,
        args.steps,
        args.accumulation,
        args.seed,
        args.lr,
        args.device,
        args.eval_seed,
    )
    try:
        options.validate()
    except (TypeError, ValueError) as error:
        parser.error(str(error))
    return options


def run(options: Options) -> None:
    from .current import run_update as current_update
    from .reference import run_update as reference_update

    options.validate()
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if rank < 0 or world_size < 1 or rank >= world_size or local_rank < 0:
        raise ValueError("Invalid distributed rank metadata")
    torch.set_num_threads(1)
    device = (
        torch.device("cuda", local_rank)
        if options.device == "cuda"
        else torch.device("cpu")
    )
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(options.seed)
    if world_size > 1:
        dist.init_process_group(
            "nccl" if device.type == "cuda" else "gloo",
            timeout=datetime.timedelta(seconds=60),
        )
    try:
        output = options.output / f"rank{rank}"
        output.mkdir(parents=True, exist_ok=False)
        model = TinyTransformer().to(device).train()
        wrapped: nn.Module = model
        if world_size > 1:
            wrapped = DistributedDataParallel(
                model, device_ids=[local_rank] if device.type == "cuda" else None
            )
        optimizer = torch.optim.SGD(model.parameters(), lr=options.lr, momentum=0.9)
        update = reference_update if options.pipeline == "reference" else current_update
        curve: list[dict[str, int | float]] = [
            {"step": 0, "validation_loss": evaluate(model, device, options.eval_seed)}
        ]
        steps: list[dict[str, int | float]] = []
        started = time.perf_counter()
        for step in range(1, options.steps + 1):
            batch = make_batch(options.seed, step, rank, device)
            microbatches = split_batch(batch, options.accumulation)
            local_loss = update(wrapped, optimizer, microbatches)
            global_loss = torch.tensor(local_loss, dtype=torch.float64, device=device)
            if world_size > 1:
                dist.all_reduce(global_loss)
                global_loss /= world_size
            loss = float(global_loss.item())
            if not math.isfinite(loss):
                raise ValueError("Training loss must be finite")
            steps.append({"step": step, "train_token_loss": loss})
            if step % 8 == 0 or step == options.steps:
                curve.append(
                    {
                        "step": step,
                        "validation_loss": evaluate(model, device, options.eval_seed),
                    }
                )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        seconds = time.perf_counter() - started
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "cpu_rng": torch.get_rng_state(),
                "cuda_rng": (
                    torch.cuda.get_rng_state(device) if device.type == "cuda" else None
                ),
                "completed_steps": options.steps,
                "rank": rank,
                "world_size": world_size,
                "seed": options.seed,
                "accumulation": options.accumulation,
            },
            output / "checkpoint.pt",
        )
        metrics: dict[str, object] = {
            "pipeline": options.pipeline,
            "rank": rank,
            "world_size": world_size,
            "device": str(device),
            "torch_version": str(torch.__version__),
            "seed": options.seed,
            "eval_seed": options.eval_seed,
            "lr": options.lr,
            "accumulation": options.accumulation,
            "examples_per_rank_update": EXAMPLES_PER_RANK_UPDATE,
            "tokens_per_global_update": world_size
            * EXAMPLES_PER_RANK_UPDATE
            * SEQUENCE_LENGTH,
            "optimizer_updates": options.steps,
            "parameter_count": sum(
                parameter.numel() for parameter in model.parameters()
            ),
            "training_seconds_including_validation": seconds,
            "steps": steps,
            "validation_curve": curve,
        }
        (output / "metrics.json").write_text(
            json.dumps(metrics, indent=2, allow_nan=False) + "\n"
        )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def main() -> None:
    run(parse_options())


if __name__ == "__main__":
    main()
