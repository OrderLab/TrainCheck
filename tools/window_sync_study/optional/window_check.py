"""Opt-in TrainCheck adapter for the public two-rank token training task."""

from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
from workload import IGNORE_INDEX, Batch, TokenModel, batches, make_optimizer

from traincheck._window_collectives import BACKWARD, STEP, audit
from traincheck.observation import ModelSnapshot, compare_ranks, snapshot_model

State = dict[str, Tensor]


@dataclass
class StepState:
    gradients: State
    parameters: State
    momentum: State


def capture(model: nn.Module, optimizer: torch.optim.Optimizer) -> StepState:
    weights: State = {}
    gradients: State = {}
    momentum: State = {}
    for name, item in model.named_parameters():
        weights[name] = item.detach().clone()
        if item.grad is not None:
            gradients[name] = item.grad.detach().clone()
        if item in optimizer.state:
            buffer = optimizer.state[item]["momentum_buffer"]
            if not isinstance(buffer, Tensor):
                raise TypeError("SGD momentum must be a tensor")
            momentum[name] = buffer.detach().clone()
    return StepState(gradients, weights, momentum)


def reference(
    model: nn.Module, rank_batches: list[list[Batch]], budget: int | None
) -> tuple[list[int], list[StepState]]:
    optimizer = make_optimizer(model)
    ends: list[int] = []
    states: list[StepState] = []
    pending: list[Batch] = []
    count = 0
    for step, group in enumerate(zip(*rank_batches, strict=True), 1):
        pending.extend(group)
        count += sum(int(target.ne(IGNORE_INDEX).sum().item()) for _, target in group)
        if budget is not None and count < budget and step != len(rank_batches[0]):
            continue
        optimizer.zero_grad(set_to_none=True)
        logits: Tensor = model(torch.cat([x for x, _ in pending]))
        labels = torch.cat([y for _, y in pending])
        F.cross_entropy(
            logits.flatten(0, 1), labels.flatten(), ignore_index=IGNORE_INDEX
        ).backward()
        optimizer.step()
        ends.append(step)
        states.append(capture(model, optimizer))
        pending.clear()
        count = 0
    return ends, states


FIXTURES = (
    ("fixed", (13, 19, 6, 8, 21), (8, 11, 14, 20, 9), None),
    ("exact", (13, 19, 6, 8, 21), (8, 11, 14, 20, 9), 51),
    ("partial_tail", (13, 19, 6, 8, 21), (8, 11, 14, 20, 9), 60),
    ("epoch_below_budget", (0, 7, 21), (18, 10, 0), 200),
)


def check_case(
    name: str,
    left: tuple[int, ...],
    right: tuple[int, ...],
    budget: int | None,
    device: torch.device,
    rank: int,
) -> bool:
    from training import train_epoch

    torch.manual_seed(17)
    base = TokenModel().to(device)
    groups = [batches(counts, r, 18, device) for r, counts in enumerate((left, right))]
    ends, expected = reference(copy.deepcopy(base), groups, budget)
    model = DistributedDataParallel(base, device_ids=[device.index])
    optimizer = make_optimizer(model)
    actual: list[StepState] = []

    def backward_marker(grad: Tensor) -> None:
        with torch.profiler.record_function(BACKWARD):
            pass

    def forward_marker(
        module: nn.Module, args: tuple[Tensor, ...], out: Tensor
    ) -> None:
        if out.requires_grad:
            out.register_hook(backward_marker)  # type: ignore[no-untyped-call]

    def step_marker(
        opt: torch.optim.Optimizer, args: tuple[object, ...], kwargs: dict[str, object]
    ) -> None:
        with torch.profiler.record_function(STEP):
            pass

    def after_step(
        opt: torch.optim.Optimizer, args: tuple[object, ...], kwargs: dict[str, object]
    ) -> None:
        actual.append(capture(base, opt))

    base.register_forward_hook(forward_marker)
    optimizer.register_step_pre_hook(step_marker)
    optimizer.register_step_post_hook(after_step)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True
    ) as prof:
        train_epoch(model, optimizer, iter(groups[rank]), budget)
        torch.cuda.synchronize()
    communication = audit(
        prof.events(), ends, sum(p.numel() for p in base.parameters())
    )
    errors: list[str] = []
    if len(actual) != len(expected):
        errors.append(f"Expected {len(expected)} updates, got {len(actual)}")
    for index, (obs, target) in enumerate(zip(actual, expected), 1):
        for label, seen, wanted in (
            ("gradient", obs.gradients, target.gradients),
            ("parameter", obs.parameters, target.parameters),
            ("momentum", obs.momentum, target.momentum),
        ):
            try:
                torch.testing.assert_close(seen, wanted, rtol=1e-5, atol=1e-6)
            except AssertionError as exc:
                errors.append(f"Update {index} {label}: {str(exc)[:260]}")
    gathered: list[object] = [None] * dist.get_world_size()
    own = snapshot_model(base, run_id=name, rank=rank, phase="after_epoch")
    dist.all_gather_object(gathered, own)
    snapshots: list[ModelSnapshot] = []
    for snapshot in gathered:
        if not isinstance(snapshot, ModelSnapshot):
            raise TypeError("Expected a TrainCheck model snapshot from every rank")
        snapshots.append(snapshot)
    rank_match = compare_ranks(snapshots, expected_ranks=[0, 1]).equal
    passed = not errors and not communication["issues"] and rank_match
    print(
        json.dumps(
            {
                "traincheck_window_check": name,
                "rank": rank,
                "passed": passed,
                "numeric_issues": errors,
                "communication": communication,
                "rank_match": rank_match,
            }
        ),
        flush=True,
    )
    return passed


def main() -> None:
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dist.init_process_group("nccl")
    try:
        rank = dist.get_rank()
        device = torch.device("cuda", local)
        results = [
            check_case(name, lhs, rhs, budget, device, rank)
            for name, lhs, rhs, budget in FIXTURES
        ]
        if not all(results):
            raise SystemExit(1)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
