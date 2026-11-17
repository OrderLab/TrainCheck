"""Withheld two-rank replay. Run only through grade.py in an isolated process."""

from __future__ import annotations

import argparse
import contextlib
import copy
import importlib.abc
import importlib.machinery
import json
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from types import ModuleType

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
from workload import IGNORE_INDEX, Batch, TokenModel, batches, make_optimizer


class ControlImports(importlib.abc.MetaPathFinder):
    """Control replays expose the same optional-package availability as the agent's run."""

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
class Scenario:
    name: str
    counts: tuple[tuple[int, ...], tuple[int, ...]]
    budget: int | None
    seed: int


SCENARIOS = (
    Scenario("fixed-regression", ((18, 3, 17), (2, 21, 5)), None, 19),
    Scenario("exact-uniform", ((9, 9, 9, 9), (9, 9, 9, 9)), 36, 349),
    Scenario("overshoot-and-tail", ((4, 17, 3, 8, 5, 2), (8, 4, 19, 6, 5, 3)), 29, 723),
    Scenario("rank-skew", ((0, 18, 2, 21, 7), (17, 0, 13, 2, 1)), 26, 182),
    Scenario("single-window", ((1, 22, 5), (19, 2, 2)), 99, 291),
    Scenario("one-microbatch-updates", ((18, 4, 17), (6, 18, 3)), 7, 89),
)
ATOL = 1e-6
RTOL = 1e-5

TensorMap = dict[str, Tensor]


@dataclass
class Step:
    gradients: TensorMap
    parameters: TensorMap
    momentum: TensorMap
    end_microbatch: int


def values(model: nn.Module) -> TensorMap:
    return {name: value.detach().clone() for name, value in model.named_parameters()}


def gradients(model: nn.Module) -> TensorMap:
    return {
        name: value.grad.detach().clone()
        for name, value in model.named_parameters()
        if value.grad is not None
    }


def momentum(model: nn.Module, optimizer: torch.optim.Optimizer) -> TensorMap:
    result: TensorMap = {}
    for name, parameter in model.named_parameters():
        if parameter not in optimizer.state:
            continue
        state = optimizer.state[parameter]
        if not isinstance(state, dict) or "momentum_buffer" not in state:
            continue
        buffer = state["momentum_buffer"]
        if not isinstance(buffer, Tensor):
            raise TypeError("SGD momentum buffer must be a tensor")
        result[name] = buffer.detach().clone()
    return result


def reference(
    model: nn.Module, rank_batches: list[list[Batch]], budget: int | None
) -> list[Step]:
    """Train on each concatenated global update; never use candidate gradients or collectives."""
    opt = make_optimizer(model)
    result: list[Step] = []
    pending: list[Batch] = []
    count = 0
    for index, pair in enumerate(zip(*rank_batches, strict=True)):
        pending.extend(pair)
        count += sum(int(targets.ne(IGNORE_INDEX).sum().item()) for _, targets in pair)
        if budget is not None and count < budget and index != len(rank_batches[0]) - 1:
            continue
        inputs = torch.cat([inputs for inputs, _ in pending])
        labels = torch.cat([labels for _, labels in pending])
        opt.zero_grad(set_to_none=True)
        logits: Tensor = model(inputs)
        loss = F.cross_entropy(
            logits.flatten(0, 1), labels.flatten(), ignore_index=IGNORE_INDEX
        )
        torch.autograd.backward(loss)
        grad = gradients(model)
        opt.step()
        result.append(Step(grad, values(model), momentum(model, opt), index + 1))
        pending.clear()
        count = 0
    return result


def compare_map(label: str, expected: TensorMap, actual: TensorMap) -> list[str]:
    issues = []
    if expected.keys() != actual.keys():
        issues.append(
            f"{label} keys: expected {sorted(expected)}, observed {sorted(actual)}"
        )
    for key in sorted(expected.keys() & actual.keys()):
        left, right = expected[key], actual[key]
        if left.shape != right.shape or left.dtype != right.dtype:
            issues.append(f"{label}/{key} shape or dtype mismatch")
        elif not torch.allclose(left, right, rtol=RTOL, atol=ATOL):
            error = (right - left).abs().max().item()
            issues.append(f"{label}/{key} max absolute error {error:.9g}")
    return issues


def evaluate(
    case: Scenario, rank: int, device: torch.device, audit_communication: bool = False
) -> dict[str, object]:
    torch.manual_seed(case.seed)
    base = TokenModel().to(device)
    plain = copy.deepcopy(base)
    rank_batches = [
        batches(c, r, case.seed + 200, device) for r, c in enumerate(case.counts)
    ]
    expected = reference(plain, rank_batches, case.budget)
    ddp = DistributedDataParallel(
        base, device_ids=[device.index] if device.type == "cuda" else None
    )
    opt = make_optimizer(ddp)
    actual: list[Step] = []
    pre: list[TensorMap] = []
    observed_ids: list[int | None] = []
    boundaries: list[list[int | None]] = []

    def before_step(
        optimizer: torch.optim.Optimizer,
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> None:
        if audit_communication:
            from collectives import STEP

            with torch.profiler.record_function(STEP):
                pass
        pre.append(gradients(base))
        boundaries.append(list(observed_ids))
        observed_ids.clear()

    def after_step(
        optimizer: torch.optim.Optimizer,
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> None:
        actual.append(Step(pre.pop(), values(base), momentum(base, optimizer), 0))

    opt.register_step_pre_hook(before_step)
    opt.register_step_post_hook(after_step)
    outstanding = 0
    peak_outstanding = 0
    max_input_elements = 0

    def observe_forward(
        module: nn.Module, inputs: tuple[Tensor, ...], output: Tensor
    ) -> None:
        nonlocal outstanding, peak_outstanding, max_input_elements
        if not output.requires_grad:
            return
        matching = [
            i
            for i, (tokens, _) in enumerate(rank_batches[rank])
            if torch.equal(inputs[0], tokens)
        ]
        observed_ids.append(matching[0] if len(matching) == 1 else None)
        outstanding += 1
        peak_outstanding = max(peak_outstanding, outstanding)
        max_input_elements = max(max_input_elements, inputs[0].numel())

        def before_backward(grad: Tensor) -> None:
            nonlocal outstanding
            if audit_communication:
                from collectives import BACKWARD

                with torch.profiler.record_function(BACKWARD):
                    pass
            outstanding -= 1

        # Tensor.register_hook in PyTorch 2.2 lacks type annotations.
        output.register_hook(before_backward)  # type: ignore[no-untyped-call]

    base.register_forward_hook(observe_forward)
    try:
        from training import train_epoch

        profiler = (
            torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True
            )
            if audit_communication
            else None
        )
        with profiler if profiler is not None else contextlib.nullcontext():
            train_epoch(ddp, opt, iter(rank_batches[rank]), case.budget)
            if profiler is not None:
                torch.cuda.synchronize()
    except NotImplementedError as exc:
        return {"name": case.name, "status": "unsupported", "error": repr(exc)}
    except Exception as exc:
        # The candidate is external code; record the actual failure.
        return {"name": case.name, "status": "runtime_error", "error": repr(exc)}
    numerical: list[str] = []
    if len(expected) != len(actual):
        numerical.append(f"Expected {len(expected)} updates, observed {len(actual)}")
    for i, (exp, obs) in enumerate(zip(expected, actual)):
        numerical.extend(
            compare_map(f"step {i}/gradients", exp.gradients, obs.gradients)
        )
        numerical.extend(
            compare_map(f"step {i}/parameters", exp.parameters, obs.parameters)
        )
        numerical.extend(compare_map(f"step {i}/momentum", exp.momentum, obs.momentum))
    numerical.extend(
        compare_map("final parameters", expected[-1].parameters, values(base))
    )
    numerical.extend(
        compare_map("final momentum", expected[-1].momentum, momentum(base, opt))
    )
    observed = chain(
        values(base).values(),
        *(
            m.values()
            for step in actual
            for m in (step.gradients, step.parameters, step.momentum)
        ),
    )
    finite = all(bool(torch.isfinite(t).all().item()) for t in observed)
    communication: dict[str, object] | None = None
    if profiler is not None:
        from collectives import audit

        communication = audit(
            profiler.events(),
            [step.end_microbatch for step in expected],
            sum(parameter.numel() for parameter in base.parameters()),
        )
    status = "pass"
    if not finite:
        status = "nonfinite"
    elif numerical:
        status = "silent_wrong"
    elif peak_outstanding > 1 or max_input_elements > rank_batches[rank][0][0].numel():
        status = "memory_contract"
    elif communication is not None and communication["issues"]:
        status = "communication_contract"
    return {
        "name": case.name,
        "status": status,
        "forward_microbatch_ids_at_steps": boundaries,
        "unconsumed_forward_ids": observed_ids,
        "peak_outstanding_grad_forwards": peak_outstanding,
        "max_forward_input_elements": max_input_elements,
        "expected_updates": len(expected),
        "observed_updates": len(actual),
        "issues": numerical,
        "communication": communication,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipts", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], required=True)
    parser.add_argument("--audit-communication", action="store_true")
    args = parser.parse_args()
    rank = int(os.environ["RANK"])
    local = int(os.environ["LOCAL_RANK"])
    device = (
        torch.device("cuda", local) if args.device == "cuda" else torch.device("cpu")
    )
    if not (Path(__file__).parent / "traincheck").is_dir():
        sys.meta_path.insert(0, ControlImports())
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    dist.init_process_group(backend="nccl" if device.type == "cuda" else "gloo")
    try:
        for index, case in enumerate(SCENARIOS):
            path = args.receipts / f"{index}-rank{rank}.json"
            path.with_suffix(".start").write_text(case.name)
            result = evaluate(case, rank, device, args.audit_communication)
            with path.open("x") as stream:
                json.dump(result, stream)
            dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
