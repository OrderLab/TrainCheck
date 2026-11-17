"""MNIST classifier, deterministic partitioning and checkpointable input stream."""

from __future__ import annotations

import os
import random
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Sampler


@dataclass(frozen=True)
class Config:
    seed: int = 37
    workers: int = 2
    batch_size: int = 16
    examples: int = 384
    prefetch: int = 2


class Digits(Dataset[tuple[Tensor, int, int]]):
    def __init__(self, count: int) -> None:
        root = Path(os.environ["TRAINCHECK_STUDY_DATA"])
        data = (root / "train-images.bin").read_bytes()
        labels = (root / "train-labels.bin").read_bytes()
        if not 0 < count <= len(labels) or len(data) != 784 * len(labels):
            raise ValueError("Unexpected MNIST fixture size")
        self.images = torch.frombuffer(
            bytearray(data[: count * 784]), dtype=torch.uint8
        ).reshape(count, 784)
        self.labels = labels[:count]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[Tensor, int, int]:
        return (
            self.images[index].float().div(255).sub(0.1307).div(0.3081),
            self.labels[index],
            index,
        )


class InputSampler(Sampler[int]):
    """Cursor for a partition's current read stream."""

    def __init__(self, data: Digits, config: Config) -> None:
        self.partition: DistributedSampler[int] = DistributedSampler(
            data, shuffle=True, seed=config.seed, drop_last=True
        )
        self.epoch = 0
        self.offset = 0

    def seek(self, epoch: int, offset: int) -> None:
        if epoch < 0 or not 0 <= offset <= len(self.partition):
            raise ValueError("Invalid input stream position")
        self.epoch = epoch
        self.offset = offset
        self.partition.set_epoch(epoch)

    def __iter__(self) -> Iterator[int]:
        indices = list(self.partition)
        for position in range(self.offset, len(indices)):
            self.offset = position + 1
            yield indices[position]

    def __len__(self) -> int:
        return len(self.partition) - self.offset

    def state_dict(self) -> dict[str, int]:
        return {"epoch": self.epoch, "offset": self.offset}


class BatchLoader(Protocol):
    @property
    def num_workers(self) -> int: ...

    def __iter__(self) -> Iterator[list[Tensor]]: ...


def make_loader(dataset: Digits, sampler: InputSampler, config: Config) -> BatchLoader:
    generator = torch.Generator().manual_seed(config.seed + 11 + dist.get_rank())
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        generator=generator,
        prefetch_factor=config.prefetch if config.workers else None,
        num_workers=config.workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.workers > 0,
    )
    # Default collation transforms the dataset's tuples into lists of tensors.
    return cast(BatchLoader, loader)


def model() -> nn.Module:
    return nn.Sequential(
        nn.Linear(784, 96),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(96, 10),
    )


def start(seed: int, use_cuda: bool = True) -> torch.device:
    local = int(os.environ["LOCAL_RANK"])
    if use_cuda:
        torch.cuda.set_device(local)
    dist.init_process_group("nccl" if use_cuda else "gloo")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    random.seed(seed)
    torch.manual_seed(seed)
    return torch.device("cuda", local) if use_cuda else torch.device("cpu")
