"""Ordinary torchrun entry point for the token training task."""

from __future__ import annotations

import argparse
import json
import math
import os

import torch
import torch.distributed as dist
from torch import Tensor
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
from training import train_epoch
from workload import IGNORE_INDEX, TokenModel, batches, make_optimizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--token-budget", type=int)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()
    if args.token_budget is not None and args.token_budget <= 0:
        parser.error("--token-budget must be positive")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    cuda = args.device == "cuda"
    device = torch.device("cuda", local_rank) if cuda else torch.device("cpu")
    if cuda:
        torch.cuda.set_device(device)
    dist.init_process_group("nccl" if cuda else "gloo")
    try:
        torch.manual_seed(args.seed)
        base = TokenModel().to(device)
        model = DistributedDataParallel(base, device_ids=[local_rank] if cuda else None)
        optimizer = make_optimizer(model)
        counts = (13, 19, 6, 8, 21) if rank % 2 == 0 else (8, 11, 14, 20, 9)
        local_batches = batches(counts, rank, args.seed + 1, device)
        train_epoch(model, optimizer, iter(local_batches), args.token_budget)
        with torch.no_grad():
            inputs, targets = local_batches[-1]
            logits: Tensor = base(inputs)
            loss = F.cross_entropy(
                logits.flatten(0, 1), targets.flatten(), ignore_index=IGNORE_INDEX
            )
        value = float(loss.item())
        if not math.isfinite(value):
            raise ValueError("Nonfinite final loss")
        print(
            json.dumps(
                {
                    "rank": rank,
                    "microbatches": len(local_batches),
                    "final_local_loss": value,
                }
            ),
            flush=True,
        )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
