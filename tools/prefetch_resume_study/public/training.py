"""Training loop and version-one distributed checkpoint format."""

from __future__ import annotations

import math
from collections.abc import Iterator
from pathlib import Path
from typing import TypedDict

import torch
import torch.distributed as dist
from torch import Tensor
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
from workload import Config, Digits, InputSampler, make_loader, model


class RankState(TypedDict):
    input: dict[str, int]
    cpu_rng: Tensor
    cuda_rng: Tensor | None


class Trainer:
    def __init__(self, config: Config, device: torch.device) -> None:
        self.config = config
        self.device = device
        self.dataset = Digits(config.examples)
        self.sampler = InputSampler(self.dataset, config)
        self.loader = make_loader(self.dataset, self.sampler, config)
        self.iterator: Iterator[list[Tensor]] | None = None
        torch.manual_seed(config.seed)
        self.base = model().to(device)
        self.module = DistributedDataParallel(
            self.base, device_ids=[device.index] if device.type == "cuda" else None
        )
        self.optimizer = torch.optim.AdamW(self.module.parameters(), lr=0.002)
        self.schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=40
        )
        self.steps = 0
        torch.manual_seed(config.seed + 104729 * (1 + dist.get_rank()))

    def train_step(self) -> tuple[float, tuple[int, ...]]:
        if self.iterator is None:
            self.iterator = iter(self.loader)
        try:
            images, labels, ids = next(self.iterator)
        except StopIteration:
            self.sampler.seek(self.sampler.epoch + 1, 0)
            self.iterator = iter(self.loader)
            images, labels, ids = next(self.iterator)
        self.optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(
            self.module(images.to(self.device, non_blocking=True)),
            labels.to(self.device, non_blocking=True),
        )
        loss.backward()
        self.optimizer.step()
        self.schedule.step()
        self.steps += 1
        value = float(loss.item())
        if not math.isfinite(value):
            raise ValueError("Nonfinite training loss")
        return value, tuple(ids.tolist())

    def save(self, path: Path) -> None:
        local = RankState(
            input=self.sampler.state_dict(),
            cpu_rng=torch.get_rng_state(),
            cuda_rng=(
                torch.cuda.get_rng_state(self.device)
                if self.device.type == "cuda"
                else None
            ),
        )
        all_ranks: list[object] = [None] * dist.get_world_size()
        dist.all_gather_object(all_ranks, local)
        if dist.get_rank() == 0:
            temporary = path.with_suffix(".tmp")
            torch.save(
                {
                    "version": 1,
                    "steps": self.steps,
                    "model": self.base.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "schedule": self.schedule.state_dict(),
                    "ranks": all_ranks,
                },
                temporary,
            )
            temporary.replace(path)
        dist.barrier()

    def load(self, path: Path) -> None:
        if self.iterator is not None:
            raise RuntimeError("Load a checkpoint before training starts")
        state = torch.load(path, map_location="cpu")
        if state["version"] != 1 or len(state["ranks"]) != dist.get_world_size():
            raise ValueError("Invalid checkpoint version or world size")
        self.base.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.schedule.load_state_dict(state["schedule"])
        self.steps = state["steps"]
        local = state["ranks"][dist.get_rank()]
        position = local["input"]
        self.sampler.seek(position["epoch"], position["offset"])
        torch.set_rng_state(local["cpu_rng"])
        if self.device.type == "cuda":
            if not isinstance(local["cuda_rng"], Tensor):
                raise ValueError("Expected CUDA RNG state")
            torch.cuda.set_rng_state(local["cuda_rng"], self.device)

    def close(self) -> None:
        self.iterator = None
        del self.loader
