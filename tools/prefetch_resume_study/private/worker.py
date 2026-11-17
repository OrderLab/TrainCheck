"""Private two-rank replay of restarted and uninterrupted MNIST training."""

from __future__ import annotations

import argparse
import hashlib
import importlib.abc
import importlib.machinery
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from types import ModuleType
from typing import Protocol

import torch
import torch.distributed as dist
from baseline_training import Trainer as Baseline
from torch import Tensor
from training import Trainer
from workload import Config, start


class ControlImports(importlib.abc.MetaPathFinder):
    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None,
        target: ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        if fullname == "traincheck" or fullname.startswith("traincheck."):
            raise ModuleNotFoundError(f"No module named {fullname!r}", name=fullname)
        return None


@dataclass(frozen=True)
class Case:
    name: str
    config: Config
    split: int
    after: int


CASES = (
    Case("zero-worker", Config(seed=11, workers=0), 3, 4),
    Case("two-workers", Config(seed=29), 3, 5),
    Case("deep-prefetch", Config(seed=53, prefetch=4), 2, 4),
    Case("near-epoch-end", Config(seed=43), 11, 5),
    Case("exact-epoch-end", Config(seed=47), 12, 4),
    Case("partial-batch", Config(seed=41, examples=390, prefetch=3), 10, 6),
)


class Loop(Protocol):
    def train_step(self) -> tuple[float, tuple[int, ...]]: ...

    def save(self, path: Path) -> None: ...

    def close(self) -> None: ...


def run_steps(trainer: Loop, count: int) -> list[tuple[float, tuple[int, ...]]]:
    return [trainer.train_step() for _ in range(count)]


def fingerprint(value: object) -> object:
    if isinstance(value, Tensor):
        raw = value.detach().cpu().contiguous().numpy().tobytes()
        return [str(value.dtype), list(value.shape), hashlib.sha256(raw).hexdigest()]
    if isinstance(value, dict):
        return {k: fingerprint(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(fingerprint(v) for v in value)
    if isinstance(value, list):
        return [fingerprint(v) for v in value]
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    raise TypeError(f"Unexpected checkpoint field of type {type(value)}")


def evaluate(
    case: Case, rank: int, device: torch.device, root: Path
) -> dict[str, object]:
    old = root / f"{case.name}-middle.pt"
    expect_out = root / f"{case.name}-expected.pt"
    actual_out = root / f"{case.name}-actual.pt"

    reference = Baseline(replace(case.config, workers=0), device)
    try:
        expected = run_steps(reference, case.split + case.after)
        reference.save(expect_out)
    finally:
        reference.close()

    trainer = Trainer(case.config, device)
    try:
        observed_worker_setting = trainer.loader.num_workers
        prefix = run_steps(trainer, case.split)
        trainer.save(old)
    finally:
        trainer.close()

    resumed = Trainer(case.config, device)
    try:
        resumed.load(old)
        suffix = run_steps(resumed, case.after)
        resumed.save(actual_out)
        resumed_worker_setting = resumed.loader.num_workers
    finally:
        resumed.close()

    observed = prefix + suffix
    different_ids = [
        i + 1
        for i, (a, b) in enumerate(zip(observed, expected, strict=True))
        if a[1] != b[1]
    ]
    different_loss = [
        i + 1
        for i, (a, b) in enumerate(zip(observed, expected, strict=True))
        if a[0] != b[0]
    ]
    exp = torch.load(expect_out, map_location="cpu")
    act = torch.load(actual_out, map_location="cpu")
    state_disagreements = [
        key
        for key in ("version", "steps", "model", "optimizer", "schedule")
        if fingerprint(exp[key]) != fingerprint(act[key])
    ]
    for key in ("cpu_rng", "cuda_rng"):
        if fingerprint(exp["ranks"][rank][key]) != fingerprint(act["ranks"][rank][key]):
            state_disagreements.append(key)

    status = "pass"
    if (
        observed_worker_setting != case.config.workers
        or resumed_worker_setting != case.config.workers
    ):
        status = "resume_contract"
    elif different_ids or different_loss or state_disagreements:
        status = "silent_wrong"
    return {
        "name": case.name,
        "status": status,
        "split_after_updates": case.split,
        "suffix_updates": case.after,
        "workers": [observed_worker_setting, resumed_worker_setting],
        "prefetch": case.config.prefetch,
        "id_mismatch_steps": different_ids,
        "loss_mismatch_steps": different_loss,
        "state_mismatches": state_disagreements,
        "expected_first_resumed_ids": list(expected[case.split][1][:8]),
        "actual_first_resumed_ids": list(suffix[0][1][:8]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipts", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], required=True)
    options = parser.parse_args()
    if not (Path(__file__).parent / "traincheck").is_dir():
        sys.meta_path.insert(0, ControlImports())
    device = start(37, options.device == "cuda")
    rank = dist.get_rank()
    try:
        for index, case in enumerate(CASES):
            path = options.receipts / f"{index}-rank{rank}.json"
            path.with_suffix(".start").write_text(case.name)
            try:
                result = evaluate(case, rank, device, options.receipts)
            except Exception as exc:
                result = {
                    "name": case.name,
                    "status": "runtime_error",
                    "error": repr(exc),
                }
            with path.open("x") as stream:
                json.dump(result, stream)
            dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
