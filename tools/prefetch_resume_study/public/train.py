"""Ordinary DDP training and small release smoke checks."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch
import torch.distributed as dist
from training import Trainer
from workload import Config, start


def smoke(config: Config, root: Path, device: torch.device) -> None:
    checkpoint = root / "smoke.pt"
    zero = replace(config, workers=0)
    baseline = Trainer(zero, device)
    for _ in range(3):
        baseline.train_step()
    baseline.save(checkpoint)
    expected = [baseline.train_step()[0] for _ in range(3)]
    baseline.close()
    resumed = Trainer(zero, device)
    resumed.load(checkpoint)
    actual = [resumed.train_step()[0] for _ in range(3)]
    resumed.close()
    if actual != expected:
        raise AssertionError("Zero-worker continuation smoke failed")
    threaded = Trainer(replace(config, workers=2), device)
    losses = [threaded.train_step()[0] for _ in range(5)]
    threaded.close()
    print(
        json.dumps({"rank": dist.get_rank(), "smoke": "pass", "last_loss": losses[-1]}),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--prefetch", type=int, default=2)
    parser.add_argument("--save", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    options = parser.parse_args()
    config = Config(
        seed=options.seed, workers=options.workers, prefetch=options.prefetch
    )
    device = start(config.seed, options.device == "cuda")
    try:
        if options.smoke:
            smoke(config, Path.cwd(), device)
            return
        trainer = Trainer(config, device)
        try:
            if options.resume is not None:
                trainer.load(options.resume)
            for _ in range(options.steps):
                loss, _ = trainer.train_step()
                print(
                    json.dumps(
                        {"rank": dist.get_rank(), "step": trainer.steps, "loss": loss}
                    ),
                    flush=True,
                )
            if options.save is not None:
                trainer.save(options.save)
        finally:
            trainer.close()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
