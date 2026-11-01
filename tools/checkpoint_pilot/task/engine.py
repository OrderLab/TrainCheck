"""Prefix training and fresh-process checkpoint continuation."""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypedDict

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

from . import current, reference, state
from .model import (
    Batch,
    Transformer,
    evaluate,
    loss_for_batch,
    make_batch,
    microbatches,
)


class TrainPoint(TypedDict):
    step: int
    train_token_loss: float


class ValidationPoint(TypedDict):
    step: int
    validation_loss: float


@dataclass(frozen=True)
class Options:
    stage: Literal["prefix", "resume"]
    backend: Literal["reference", "current"]
    checkpoint: Path
    output: Path
    steps: int = 12
    seed: int = 20261101
    eval_seed: int = 777000
    lr: float = 0.15
    accumulation: int = 4
    device: Literal["cpu", "cuda"] = "cuda"

    def validate(self) -> None:
        if self.stage not in ("prefix", "resume") or self.backend not in (
            "reference",
            "current",
        ):
            raise ValueError("Invalid stage or backend")
        if self.device not in ("cpu", "cuda"):
            raise ValueError("Invalid device")
        for name, value in (
            ("steps", self.steps),
            ("seed", self.seed),
            ("eval_seed", self.eval_seed),
            ("accumulation", self.accumulation),
        ):
            state.integer(value, name)
        if not 1 <= self.steps <= 128:
            raise ValueError("steps must be in [1,128]")
        if self.seed >= 2**31 or self.eval_seed >= 2**31:
            raise ValueError("Seeds must be less than 2**31")
        if self.accumulation not in (1, 2, 4, 8):
            raise ValueError("accumulation must be a positive divisor of eight")
        if (
            isinstance(self.lr, bool)
            or not isinstance(self.lr, (int, float))
            or not math.isfinite(self.lr)
            or self.lr <= 0
        ):
            raise ValueError("lr must be finite and positive")


def run_update(
    model: nn.Module, optimizer: torch.optim.Optimizer, batches: list[Batch]
) -> float:
    if not batches:
        raise ValueError("An update requires microbatches")
    optimizer.zero_grad(set_to_none=True)
    total = torch.zeros((), device=batches[0].tokens.device)
    for index, batch in enumerate(batches):
        synchronization = (
            model.no_sync()
            if isinstance(model, DistributedDataParallel) and index + 1 < len(batches)
            else nullcontext()
        )
        with synchronization:
            loss = loss_for_batch(model, batch)
            total += loss.detach()
            (loss / len(batches)).backward()
    for parameter in model.parameters():
        if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()):
            raise ValueError("Every parameter must have a finite gradient")
    optimizer.step()
    return float((total / len(batches)).item())


def run(options: Options) -> None:
    options.validate()
    # Absence means ordinary single-process execution, matching torchrun's API.
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    state.validate_rank(rank, world_size)
    if local_rank < 0:
        raise ValueError("LOCAL_RANK must be nonnegative")
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
        if options.stage == "prefix":
            if rank == 0:
                options.checkpoint.mkdir(parents=True, exist_ok=False)
            if world_size > 1:
                dist.barrier()
        elif not options.checkpoint.is_dir():
            raise FileNotFoundError(options.checkpoint)
        model = Transformer().to(device).train()
        wrapped: nn.Module = model
        if world_size > 1:
            wrapped = DistributedDataParallel(
                model, device_ids=[local_rank] if device.type == "cuda" else None
            )
        # Initialization is shared; stochastic training has a stream per rank.
        torch.manual_seed(options.seed + 104729 * rank + 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=options.lr, momentum=0.9)
        backend = reference if options.backend == "reference" else current
        progress = state.Progress(0, 0, options.seed)
        if options.stage == "resume":
            progress = backend.restore_checkpoint(
                options.checkpoint, model, optimizer, rank, world_size, device
            )
            if progress.seed != options.seed:
                raise ValueError("Checkpoint training seed differs from requested seed")
            _, _, restored_groups = state.optimizer_snapshot(model, optimizer)
            if restored_groups[0]["lr"] != options.lr:
                raise ValueError("Requested learning rate differs from checkpoint")
            restored = state.artifact(
                model, optimizer, progress, rank, world_size, device
            )
            torch.save(restored, output / "restored_state.pt")
        curve: list[ValidationPoint] = [
            {
                "step": progress.completed_steps,
                "validation_loss": evaluate(model, options.eval_seed, device),
            }
        ]
        points: list[TrainPoint] = []
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        for _ in range(options.steps):
            batch = make_batch(progress.seed, progress.data_cursor + 1, rank, device)
            local_loss = run_update(
                wrapped, optimizer, microbatches(batch, options.accumulation)
            )
            global_loss = torch.tensor(local_loss, dtype=torch.float64, device=device)
            if world_size > 1:
                dist.all_reduce(global_loss)
                global_loss /= world_size
            loss = float(global_loss.item())
            if not math.isfinite(loss):
                raise ValueError("Training loss is not finite")
            progress = state.Progress(
                progress.completed_steps + 1, progress.data_cursor + 1, progress.seed
            )
            points.append({"step": progress.completed_steps, "train_token_loss": loss})
            if progress.completed_steps % 4 == 0 or len(points) == options.steps:
                curve.append(
                    {
                        "step": progress.completed_steps,
                        "validation_loss": evaluate(model, options.eval_seed, device),
                    }
                )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        seconds = time.perf_counter() - started
        terminal = state.artifact(model, optimizer, progress, rank, world_size, device)
        torch.save(terminal, output / "terminal_state.pt")
        if options.stage == "prefix":
            backend.save_checkpoint(
                options.checkpoint, model, optimizer, progress, rank, world_size, device
            )
        if world_size > 1:
            dist.barrier()
        metrics: dict[str, object] = {
            "stage": options.stage,
            "backend": options.backend,
            "rank": rank,
            "world_size": world_size,
            "seed": options.seed,
            "eval_seed": options.eval_seed,
            "lr": options.lr,
            "accumulation": options.accumulation,
            "device": str(device),
            "torch_version": str(torch.__version__),
            "parameter_count": sum(
                parameter.numel() for parameter in model.parameters()
            ),
            "examples_per_rank_update": 8,
            "tokens_per_global_update": world_size * 64,
            "requested_updates": options.steps,
            "completed_steps": progress.completed_steps,
            "training_seconds_including_validation": seconds,
            "steps": points,
            "validation_curve": curve,
        }
        (output / "metrics.json").write_text(
            json.dumps(metrics, indent=2, allow_nan=False) + "\n"
        )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def parse_options(argv: list[str] | None = None) -> Options:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prefix", "resume"))
    parser.add_argument("--backend", choices=("reference", "current"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20261101)
    parser.add_argument("--eval-seed", type=int, default=777000)
    parser.add_argument("--lr", type=float, default=0.15)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args(argv)
    options = Options(
        args.stage,
        args.backend,
        args.checkpoint,
        args.output,
        args.steps,
        args.seed,
        args.eval_seed,
        args.lr,
        args.accumulation,
        args.device,
    )
    try:
        options.validate()
    except (ValueError, TypeError) as error:
        parser.error(str(error))
    return options


def main() -> None:
    run(parse_options())


if __name__ == "__main__":
    main()
