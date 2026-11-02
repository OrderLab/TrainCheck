"""Capture selected model states during a small single-process or DDP run."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from traincheck.observation import snapshot_model, write_snapshot


@dataclass(frozen=True)
class Settings:
    output: Path
    run_id: str
    device: str
    steps: int


def _environment_integer(name: str) -> int:
    value = os.environ[name]
    if not value.isdecimal():
        raise ValueError(f"{name} must be a nonnegative decimal integer")
    return int(value)


def run(settings: Settings) -> None:
    """Train an illustrative model; every explicit snapshot observes all its state."""
    if settings.steps < 1:
        raise ValueError("steps must be positive")
    if settings.device not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    required = {"RANK", "LOCAL_RANK", "WORLD_SIZE"}
    present = required.intersection(os.environ)
    if present and present != required:
        raise ValueError(
            "distributed execution requires RANK, LOCAL_RANK and WORLD_SIZE"
        )
    rank = _environment_integer("RANK") if present else 0
    local_rank = _environment_integer("LOCAL_RANK") if present else 0
    world_size = _environment_integer("WORLD_SIZE") if present else 1
    if world_size < 1 or rank >= world_size:
        raise ValueError("rank must be in [0, WORLD_SIZE)")
    device = (
        torch.device(settings.device, local_rank)
        if settings.device == "cuda"
        else torch.device("cpu")
    )
    if settings.device == "cuda":
        torch.cuda.set_device(device)
    distributed = world_size > 1
    if distributed:
        dist.init_process_group(backend="nccl" if settings.device == "cuda" else "gloo")
    try:
        torch.manual_seed(1234)
        model = nn.Sequential(nn.Linear(4, 8), nn.Tanh(), nn.Linear(8, 2)).to(device)
        model.register_buffer(
            "example_counter",
            torch.zeros((), dtype=torch.int64, device=device),
            persistent=False,
        )
        training_model: nn.Module = model
        if distributed:
            training_model = DistributedDataParallel(
                model, device_ids=[local_rank] if settings.device == "cuda" else None
            )
        optimizer = torch.optim.SGD(training_model.parameters(), lr=0.01)
        settings.output.mkdir(parents=True, exist_ok=True)

        def observe(phase: str, step: int) -> None:
            snapshot = snapshot_model(
                model, run_id=settings.run_id, rank=rank, phase=phase, step=step
            )
            write_snapshot(
                snapshot, settings.output / f"rank{rank}-{phase}-{step}.json"
            )

        observe("ready", 0)
        inputs = torch.arange(12, dtype=torch.float32, device=device).reshape(3, 4) / 12
        targets = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], device=device)
        for step in range(1, settings.steps + 1):
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(training_model(inputs), targets)
            loss.backward()
            optimizer.step()
            observe("update", step)
            print(
                json.dumps(
                    {"rank": rank, "step": step, "loss": float(loss.detach().cpu())}
                ),
                flush=True,
            )
    finally:
        if distributed:
            dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--steps", type=int, default=2)
    args = parser.parse_args()
    output: object = args.output
    run_id: object = args.run_id
    device: object = args.device
    steps: object = args.steps
    assert isinstance(output, Path)
    assert isinstance(run_id, str)
    assert isinstance(device, str)
    assert isinstance(steps, int)
    run(Settings(output, run_id, device, steps))


if __name__ == "__main__":
    main()
