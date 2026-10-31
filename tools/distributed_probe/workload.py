"""Small DDP contract probe, not an RSI score or performance benchmark.

Synthetic token batches have unequal valid-token counts on each rank. A separate
global-batch calculation supplies the first-update gradient oracle. Checkpoints
are made only at complete update boundaries. Data is regenerated from absolute
step/rank/microbatch indices, so this probe does not exercise a DataLoader cursor.
"""

from __future__ import annotations

import argparse
import copy
import datetime
import hashlib
import json
import os
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

Precision: TypeAlias = Literal["fp32", "bf16"]
Fault: TypeAlias = Literal["none", "local-mean"]
Instrumentation: TypeAlias = Literal["plain", "api", "subclass"]
StateValue: TypeAlias = (
    torch.Tensor
    | None
    | bool
    | int
    | float
    | str
    | list["StateValue"]
    | tuple["StateValue", ...]
    | dict[str | int, "StateValue"]
)

SEED = 20261031
VOCAB_SIZE = 32
SEQUENCE_LENGTH = 8
BATCH_SIZE = 2


@dataclass(frozen=True)
class Options:
    output: Path
    resume: Path | None
    steps: int
    stop_after: int
    precision: Precision
    fault: Fault
    instrumentation: Instrumentation
    device: Literal["cpu", "cuda"]
    accumulation: int


@dataclass(frozen=True)
class Batch:
    tokens: torch.Tensor
    labels: torch.Tensor
    mask: torch.Tensor


class TinyTransformer(nn.Module):
    """Dropout-free model so independent batching has an exact mathematical oracle."""

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
            raise TypeError("Transformer must return a Tensor")
        return result


def make_batch(step: int, rank: int, microbatch: int, device: torch.device) -> Batch:
    if step < 1 or rank < 0 or microbatch < 0:
        raise ValueError(
            "step must be positive; rank and microbatch must be nonnegative"
        )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(SEED + 10000 * step + 100 * rank + microbatch)
    tokens = torch.randint(
        VOCAB_SIZE, (BATCH_SIZE, SEQUENCE_LENGTH), generator=generator
    )
    labels = (tokens + 1) % VOCAB_SIZE
    # Ranks 0 and 1 have disjoint low/high valid-token-count ranges.
    lengths = torch.tensor(
        [2 + 3 * rank + (step + microbatch + row) % 2 for row in range(BATCH_SIZE)]
    ).clamp(max=SEQUENCE_LENGTH)
    mask = torch.arange(SEQUENCE_LENGTH).unsqueeze(0) < lengths.unsqueeze(1)
    return Batch(tokens.to(device), labels.to(device), mask.to(device))


def token_loss(logits: torch.Tensor, batch: Batch) -> torch.Tensor:
    losses = F.cross_entropy(
        logits.reshape(-1, VOCAB_SIZE), batch.labels.reshape(-1), reduction="none"
    ).reshape_as(batch.labels)
    return (losses * batch.mask).sum()


def loss_scale(
    fault: Fault, world_size: int, global_tokens: int, local_tokens: int
) -> float:
    if world_size < 1 or local_tokens < 1 or global_tokens < local_tokens:
        raise ValueError("Invalid world size or valid-token counts")
    if fault == "none":
        return world_size / global_tokens
    if fault == "local-mean":
        return 1.0 / local_tokens
    raise ValueError(f"Unknown fault: {fault}")


def copy_state(value: object, context: str = "state") -> StateValue:
    """Validate and snapshot framework state without dropping values or coercing types."""
    if isinstance(value, torch.Tensor):
        snapshot = value.detach().cpu().clone()
        if type(snapshot) is not torch.Tensor:
            raise TypeError(f"{context} did not detach into a plain Tensor")
        return snapshot
    if value is None or type(value) in (bool, int, float, str):
        # Explicit branches below narrow primitive values for the type checker.
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
    if isinstance(value, list):
        return [copy_state(child, context) for child in value]
    if isinstance(value, tuple):
        return tuple(copy_state(child, context) for child in value)
    if isinstance(value, dict):
        result: dict[str | int, StateValue] = {}
        for key, child in value.items():
            if type(key) not in (str, int) or not isinstance(key, (str, int)):
                raise TypeError(f"{context} contains an unsupported dictionary key")
            result[key] = copy_state(child, f"{context}.{key}")
        return result
    raise TypeError(f"{context} contains unsupported {type(value).__name__}")


def tensor_mapping(value: object, context: str) -> dict[str, torch.Tensor]:
    if not isinstance(value, dict):
        raise TypeError(f"{context} must be a dictionary")
    result: dict[str, torch.Tensor] = {}
    for key, child in value.items():
        if not isinstance(key, str) or not isinstance(child, torch.Tensor):
            raise TypeError(f"{context} must map names to tensors")
        result[key] = child
    return result


def string_mapping(value: object, context: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"{context} must be a dictionary")
    result: dict[str, object] = {}
    for key, child in value.items():
        if not isinstance(key, str):
            raise TypeError(f"{context} must have string keys")
        result[key] = child
    return result


def model_gradients(model: nn.Module) -> dict[str, torch.Tensor]:
    gradients: dict[str, torch.Tensor] = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            raise ValueError(f"Missing gradient for {name}")
        value = copy_state(parameter.grad, name)
        assert isinstance(value, torch.Tensor)
        gradients[name] = value
    return gradients


def tensor_hash(values: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in values.items():
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(
            value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        )
    return digest.hexdigest()


def gradient_reference(
    model: nn.Module,
    world_size: int,
    accumulation: int,
    device: torch.device,
    precision: Precision,
) -> dict[str, torch.Tensor]:
    reference = copy.deepcopy(model)
    batches = [
        make_batch(1, rank, microbatch, device)
        for rank in range(world_size)
        for microbatch in range(accumulation)
    ]
    combined = Batch(
        torch.cat([batch.tokens for batch in batches]),
        torch.cat([batch.labels for batch in batches]),
        torch.cat([batch.mask for batch in batches]),
    )
    with torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=precision == "bf16"
    ):
        logits: object = reference(combined.tokens)
        if not isinstance(logits, torch.Tensor):
            raise TypeError("Reference output must be a tensor")
        loss = token_loss(logits, combined) / combined.mask.sum()
    loss.backward()
    return model_gradients(reference)


def compare_gradients(
    actual: dict[str, torch.Tensor],
    expected: dict[str, torch.Tensor],
    precision: Precision,
) -> dict[str, bool | float | int]:
    if actual.keys() != expected.keys():
        raise ValueError("Oracle parameter names differ")
    atol, rtol = (2e-6, 2e-4) if precision == "fp32" else (3e-3, 3e-2)
    maximum = 0.0
    passed = True
    for name, value in actual.items():
        reference = expected[name]
        if value.shape != reference.shape or value.dtype != reference.dtype:
            raise ValueError(f"Oracle metadata differs for {name}")
        if not bool(torch.isfinite(value).all()) or not bool(
            torch.isfinite(reference).all()
        ):
            raise ValueError(f"Nonfinite oracle gradient for {name}")
        maximum = max(maximum, float((value - reference).abs().max().item()))
        passed = passed and bool(torch.allclose(value, reference, atol=atol, rtol=rtol))
    return {
        "passed": passed,
        "max_abs_difference": maximum,
        "atol": atol,
        "rtol": rtol,
        "parameter_tensors": len(actual),
    }


def checkpoint_payload(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.StepLR,
    completed: int,
    rank: int,
    world_size: int,
    options: Options,
    device: torch.device,
) -> dict[str, StateValue]:
    return {
        "format_version": 1,
        "model": copy_state(model.state_dict()),
        "gradients": copy_state(model_gradients(model)),
        "optimizer": copy_state(optimizer.state_dict()),
        "scheduler": copy_state(scheduler.state_dict()),
        "cpu_rng": torch.get_rng_state().clone(),
        "cuda_rng": (
            torch.cuda.get_rng_state(device).cpu() if device.type == "cuda" else None
        ),
        "completed_steps": completed,
        "rank": rank,
        "world_size": world_size,
        "total_steps": options.steps,
        "precision": options.precision,
        "fault": options.fault,
        "accumulation": options.accumulation,
        "data_seed": SEED,
    }


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.StepLR,
    rank: int,
    world_size: int,
    options: Options,
    device: torch.device,
) -> int:
    raw: object = torch.load(path, map_location="cpu", weights_only=True)
    payload = string_mapping(raw, "checkpoint")
    expected: dict[str, int | str] = {
        "format_version": 1,
        "rank": rank,
        "world_size": world_size,
        "total_steps": options.steps,
        "precision": options.precision,
        "fault": options.fault,
        "accumulation": options.accumulation,
        "data_seed": SEED,
    }
    for key, expected_value in expected.items():
        if (
            key not in payload
            or not isinstance(payload[key], type(expected_value))
            or isinstance(payload[key], bool)
            or payload[key] != expected_value
        ):
            raise ValueError(f"Checkpoint field {key} does not match this run")
    completed = payload["completed_steps"]
    if (
        not isinstance(completed, int)
        or isinstance(completed, bool)
        or not 1 <= completed < options.stop_after
    ):
        raise ValueError(
            "Checkpoint completed_steps must precede this run's stop boundary"
        )
    # Validate the complete tree before handing the library its owned structures.
    copy_state(payload)
    model.load_state_dict(tensor_mapping(payload["model"], "model"), strict=True)
    optimizer.load_state_dict(string_mapping(payload["optimizer"], "optimizer"))
    scheduler.load_state_dict(string_mapping(payload["scheduler"], "scheduler"))
    cpu_rng = payload["cpu_rng"]
    if (
        not isinstance(cpu_rng, torch.Tensor)
        or cpu_rng.dtype != torch.uint8
        or cpu_rng.ndim != 1
    ):
        raise TypeError("cpu_rng must be a one-dimensional uint8 tensor")
    torch.set_rng_state(cpu_rng)
    cuda_rng = payload["cuda_rng"]
    if device.type == "cuda":
        if (
            not isinstance(cuda_rng, torch.Tensor)
            or cuda_rng.dtype != torch.uint8
            or cuda_rng.ndim != 1
        ):
            raise TypeError("cuda_rng must be a one-dimensional uint8 tensor")
        torch.cuda.set_rng_state(cuda_rng, device)
    elif cuda_rng is not None:
        raise ValueError("CPU checkpoint must not have CUDA RNG state")
    return completed


def parse_options(argv: list[str] | None = None) -> Options:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--accumulation", type=int, default=2)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--fault", choices=("none", "local-mean"), default="none")
    parser.add_argument(
        "--instrumentation", choices=("plain", "api", "subclass"), default="plain"
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args(argv)
    stop_after = args.steps if args.stop_after is None else args.stop_after
    if not 1 <= stop_after <= args.steps or not 1 <= args.accumulation <= 4:
        parser.error("Require 1 <= stop-after <= steps and 1 <= accumulation <= 4")
    return Options(
        args.output,
        args.resume,
        args.steps,
        stop_after,
        args.precision,
        args.fault,
        args.instrumentation,
        args.device,
        args.accumulation,
    )


def run(options: Options) -> None:
    from tools.distributed_probe.tracing import start_tracing

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if rank < 0 or world_size < 1 or rank >= world_size or local_rank < 0:
        raise ValueError("Invalid torchrun rank metadata")
    device = (
        torch.device("cuda", local_rank)
        if options.device == "cuda"
        else torch.device("cpu")
    )
    torch.set_num_threads(1)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(SEED)
    if world_size > 1:
        dist.init_process_group(
            "nccl" if device.type == "cuda" else "gloo",
            timeout=datetime.timedelta(seconds=60),
        )
    rank_output = options.output / f"rank{rank}"
    rank_output.mkdir(parents=True, exist_ok=False)
    model = TinyTransformer().to(device).train()
    reference = gradient_reference(
        model, world_size, options.accumulation, device, options.precision
    )
    session = start_tracing(options.instrumentation, rank_output / "trace", model)
    try:
        with session.suppress():
            wrapped: nn.Module = model
            if world_size > 1:
                wrapped = DistributedDataParallel(
                    model, device_ids=[local_rank] if device.type == "cuda" else None
                )
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, foreach=False)
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=2, gamma=0.8
            )
            completed = 0
            if options.resume is not None:
                completed = load_checkpoint(
                    options.resume / f"rank{rank}" / "checkpoint.pt",
                    model,
                    optimizer,
                    scheduler,
                    rank,
                    world_size,
                    options,
                    device,
                )
            initial = tensor_mapping(copy_state(model.state_dict()), "initial model")
            start_step = completed + 1
            batches = {
                step: [
                    make_batch(step, rank, microbatch, device)
                    for microbatch in range(options.accumulation)
                ]
                for step in range(start_step, options.stop_after + 1)
            }
        rows: list[dict[str, object]] = []
        oracle: dict[str, bool | float | int] | None = None
        started = time.perf_counter()
        for step in range(start_step, options.stop_after + 1):
            local_batches = batches[step]
            with session.suppress():
                local_tokens = sum(
                    int(batch.mask.sum().item()) for batch in local_batches
                )
                count_tensor = torch.tensor(
                    local_tokens, device=device, dtype=torch.int64
                )
                if world_size > 1:
                    dist.all_reduce(count_tensor)
                global_tokens = int(count_tensor.item())
                scale = loss_scale(
                    options.fault, world_size, global_tokens, local_tokens
                )
            losses: list[torch.Tensor] = []
            with session.step(step):
                optimizer.zero_grad(set_to_none=True)
                for microbatch, batch in enumerate(local_batches):
                    context = (
                        wrapped.no_sync()
                        if isinstance(wrapped, DistributedDataParallel)
                        and microbatch + 1 < len(local_batches)
                        else nullcontext()
                    )
                    with context:
                        with torch.autocast(
                            device_type=device.type,
                            dtype=torch.bfloat16,
                            enabled=options.precision == "bf16",
                        ):
                            logits: object = wrapped(batch.tokens)
                            if not isinstance(logits, torch.Tensor):
                                raise TypeError("Training output must be a Tensor")
                            loss = token_loss(logits, batch) * scale
                        loss.backward()
                        losses.append(loss.detach())
                if step == 1:
                    with session.suppress():
                        actual = model_gradients(model)
                        oracle = compare_gradients(actual, reference, options.precision)
                        torch.save(
                            {"actual": actual, "expected": reference},
                            rank_output / "oracle.pt",
                        )
                gradient_norm = nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0
                )
                optimizer.step()
                scheduler.step()
            with session.suppress():
                rows.append(
                    {
                        "step": step,
                        "local_valid_tokens": local_tokens,
                        "global_valid_tokens": global_tokens,
                        "scaled_local_loss": float(torch.stack(losses).sum().item()),
                        "gradient_norm_before_clip": float(gradient_norm.item()),
                        "learning_rate": scheduler.get_last_lr()[0],
                    }
                )
        with session.suppress():
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            training_seconds = time.perf_counter() - started
            final_state = tensor_mapping(copy_state(model.state_dict()), "final model")
            rank_maximum = 0.0
            if world_size > 1:
                for parameter in model.parameters():
                    if not bool(torch.isfinite(parameter).all()):
                        raise ValueError("Nonfinite final model parameter")
                    peer = parameter.detach().clone()
                    dist.broadcast(peer, src=0)
                    rank_maximum = max(
                        rank_maximum, float((peer - parameter).abs().max().item())
                    )
            payload = checkpoint_payload(
                model,
                optimizer,
                scheduler,
                options.stop_after,
                rank,
                world_size,
                options,
                device,
            )
            torch.save(payload, rank_output / "checkpoint.pt")
            metrics: dict[str, object] = {
                "rank": rank,
                "world_size": world_size,
                "device": str(device),
                "torch_version": torch.__version__,
                "precision": options.precision,
                "fault": options.fault,
                "instrumentation": options.instrumentation,
                "start_step": start_step,
                "completed_steps": options.stop_after,
                "total_steps": options.steps,
                "accumulation": options.accumulation,
                "parameter_count": sum(p.numel() for p in model.parameters()),
                "initial_model_sha256": tensor_hash(initial),
                "final_model_sha256": tensor_hash(final_state),
                "rank_parameter_max_abs_difference_from_rank0": rank_maximum,
                "gradient_oracle": oracle,
                "training_seconds_including_oracle_snapshot": training_seconds,
                "steps": rows,
                "limitations": [
                    "dropout=0; saved RNG is not exercised by stochastic model operations",
                    "synthetic deterministic absolute-step batches; no DataLoader cursor",
                    "BF16 autocast without GradScaler; no FP16 overflow coverage",
                    "update-boundary checkpoints only",
                    "saved gradients are comparison evidence; resumed updates zero gradients before use",
                    "all accumulation windows are full; partial-window behavior is not exercised",
                    "no inferred invariant or cross-rank TrainCheck checker in this workload",
                ],
            }
            (rank_output / "metrics.json").write_text(
                json.dumps(metrics, indent=2, allow_nan=False) + "\n"
            )
    finally:
        try:
            report = session.close()
            (rank_output / "trace-session.json").write_text(
                json.dumps(report, indent=2, allow_nan=False) + "\n"
            )
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()


def main() -> None:
    run(parse_options())


if __name__ == "__main__":
    main()
