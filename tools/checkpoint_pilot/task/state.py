"""Validated state boundaries shared by checkpoint backends."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias, TypedDict

import torch
import torch.nn as nn

TensorMap: TypeAlias = dict[str, torch.Tensor]
GroupScalar: TypeAlias = bool | int | float | str | None
Group: TypeAlias = dict[str, GroupScalar | list[int]]


class ProgressFields(TypedDict):
    completed_steps: int
    data_cursor: int
    seed: int


class RngFields(TypedDict):
    cpu: torch.Tensor
    cuda: torch.Tensor | None


class OptimizerFields(TypedDict):
    state: dict[int, TensorMap]
    param_groups: list[Group]


NamedGroup: TypeAlias = dict[str, GroupScalar | list[str]]


class Artifact(TypedDict):
    model: TensorMap
    optimizer_state: dict[str, TensorMap]
    optimizer_groups: list[NamedGroup]
    progress: ProgressFields
    rng: RngFields
    rank: int
    world_size: int


@dataclass(frozen=True)
class Progress:
    completed_steps: int
    data_cursor: int
    seed: int

    def fields(self) -> ProgressFields:
        if min(self.completed_steps, self.data_cursor, self.seed) < 0:
            raise ValueError("Progress fields must be nonnegative")
        if self.completed_steps != self.data_cursor:
            raise ValueError("This task consumes one complete batch per update")
        return {
            "completed_steps": self.completed_steps,
            "data_cursor": self.data_cursor,
            "seed": self.seed,
        }


def mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a mapping")
    result: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise TypeError(f"{label} keys must be strings")
        result[key] = item
    return result


def exact_keys(value: dict[str, object], keys: set[str], label: str) -> None:
    if set(value) != keys:
        raise ValueError(f"{label} fields differ: expected {sorted(keys)}")


def integer(value: object, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    if value < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    return value


def strings(value: object, label: str) -> list[str]:
    if not isinstance(value, list):
        raise TypeError(f"{label} must be a list")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item:
            raise TypeError(f"{label} entries must be nonempty strings")
        result.append(item)
    if len(set(result)) != len(result):
        raise ValueError(f"{label} entries must be unique")
    return result


def tensors(value: object, label: str) -> TensorMap:
    result: TensorMap = {}
    for name, item in mapping(value, label).items():
        if not isinstance(item, torch.Tensor):
            raise TypeError(f"{label}.{name} must be a tensor")
        if item.layout != torch.strided or item.is_quantized:
            raise ValueError(f"{label}.{name} must be a dense ordinary tensor")
        if not bool(torch.isfinite(item).all()):
            raise ValueError(f"{label}.{name} must be finite")
        result[name] = item
    return result


def tensor_snapshot(values: TensorMap) -> TensorMap:
    return {name: value.detach().cpu().clone() for name, value in values.items()}


def read(path: Path) -> dict[str, object]:
    raw: object = torch.load(path, map_location="cpu", weights_only=True)
    return mapping(raw, str(path))


def read_progress(value: object) -> Progress:
    fields = mapping(value, "progress")
    exact_keys(fields, {"completed_steps", "data_cursor", "seed"}, "progress")
    progress = Progress(
        integer(fields["completed_steps"], "completed_steps"),
        integer(fields["data_cursor"], "data_cursor"),
        integer(fields["seed"], "seed"),
    )
    progress.fields()
    return progress


def capture_rng(device: torch.device) -> RngFields:
    return {
        "cpu": torch.get_rng_state().clone(),
        "cuda": (
            torch.cuda.get_rng_state(device).clone() if device.type == "cuda" else None
        ),
    }


def read_rng(value: object, device: torch.device) -> RngFields:
    fields = mapping(value, "rng")
    exact_keys(fields, {"cpu", "cuda"}, "rng")
    cpu = fields["cpu"]
    cuda = fields["cuda"]
    for name, item in (("cpu", cpu), ("cuda", cuda)):
        if item is None and name == "cuda" and device.type == "cpu":
            continue
        if (
            not isinstance(item, torch.Tensor)
            or item.dtype != torch.uint8
            or item.ndim != 1
            or item.device.type != "cpu"
        ):
            raise TypeError(f"Invalid {name} RNG state")
    if not isinstance(cpu, torch.Tensor):
        raise TypeError("Missing CPU RNG state")
    if device.type == "cpu" and cuda is not None:
        raise ValueError("CPU checkpoint must not contain CUDA RNG state")
    if cuda is not None and not isinstance(cuda, torch.Tensor):
        raise TypeError("Invalid CUDA RNG state")
    return {"cpu": cpu, "cuda": cuda}


def restore_rng(value: RngFields, device: torch.device) -> None:
    torch.set_rng_state(value["cpu"])
    if device.type == "cuda":
        cuda = value["cuda"]
        if cuda is None:
            raise ValueError("CUDA checkpoint requires CUDA RNG state")
        torch.cuda.set_rng_state(cuda, device)


def groups(value: object, expected_count: int) -> list[Group]:
    if not isinstance(value, list) or len(value) != 1:
        raise ValueError("This task uses exactly one optimizer group")
    raw = mapping(value[0], "optimizer group")
    # This task uses the explicit SGD checkpoint schema from pinned PyTorch2.2.
    # Unknown settings require a deliberate compatibility review, not a fallback.
    exact_keys(
        raw,
        {
            "params",
            "lr",
            "momentum",
            "dampening",
            "weight_decay",
            "nesterov",
            "maximize",
            "foreach",
            "differentiable",
        },
        "optimizer group",
    )
    parameter_ids = raw["params"]
    if not isinstance(parameter_ids, list):
        raise TypeError("Optimizer params must be a list")
    ids = [integer(item, "parameter ID") for item in parameter_ids]
    if len(ids) != expected_count or len(set(ids)) != len(ids):
        raise ValueError("Optimizer parameter IDs are incomplete or duplicate")
    result: Group = {"params": ids}
    for name, item in raw.items():
        if name == "params":
            continue
        if item is not None and not isinstance(item, (bool, int, float, str)):
            raise TypeError(f"Unsupported optimizer group field {name}")
        if isinstance(item, float) and not math.isfinite(item):
            raise ValueError(f"Nonfinite optimizer group field {name}")
        result[name] = item
    for name in ("lr", "momentum", "dampening", "weight_decay"):
        item = result[name]
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise TypeError(f"Optimizer {name} must be numeric")
        if item < 0 or (name == "lr" and item == 0):
            raise ValueError(f"Optimizer {name} is outside the supported range")
    for name in ("nesterov", "maximize", "differentiable"):
        if not isinstance(result[name], bool):
            raise TypeError(f"Optimizer {name} must be boolean")
    foreach = result["foreach"]
    if foreach is not None and not isinstance(foreach, bool):
        raise TypeError("Optimizer foreach must be boolean or unavailable")
    if result["nesterov"] and (result["momentum"] == 0 or result["dampening"] != 0):
        raise ValueError("Nesterov requires positive momentum and zero dampening")
    return [result]


def parameter_ids(group_values: list[Group]) -> list[int]:
    assert len(group_values) == 1
    ids = group_values[0]["params"]
    assert isinstance(ids, list)
    return ids


def momentum(value: object, label: str) -> TensorMap:
    result = tensors(value, label)
    if set(result) != {"momentum_buffer"}:
        raise ValueError(f"{label} must contain exactly one momentum buffer")
    return result


def optimizer_snapshot(
    model: nn.Module, optimizer: torch.optim.Optimizer
) -> tuple[list[str], list[TensorMap], list[Group]]:
    names = optimizer_parameter_names(model, optimizer)
    raw: object = optimizer.state_dict()
    fields = mapping(raw, "optimizer")
    exact_keys(fields, {"state", "param_groups"}, "optimizer")
    group_values = groups(fields["param_groups"], len(names))
    raw_states = fields["state"]
    if not isinstance(raw_states, dict):
        raise TypeError("Optimizer state must be a mapping")
    states: dict[int, TensorMap] = {}
    for key, value in raw_states.items():
        index = integer(key, "optimizer state ID")
        states[index] = momentum(value, f"optimizer state {index}")
    ids = parameter_ids(group_values)
    if set(states) != set(ids):
        raise ValueError("Checkpoint requires initialized state for every parameter")
    return names, [tensor_snapshot(states[index]) for index in ids], group_values


def optimizer_parameter_names(
    model: nn.Module, optimizer: torch.optim.Optimizer
) -> list[str]:
    by_identity = {id(parameter): name for name, parameter in model.named_parameters()}
    live_groups: object = optimizer.param_groups
    if not isinstance(live_groups, list) or len(live_groups) != 1:
        raise ValueError("This task uses exactly one live optimizer group")
    group = mapping(live_groups[0], "live optimizer group")
    parameters = group["params"]
    if not isinstance(parameters, list):
        raise TypeError("Live optimizer params must be a list")
    names: list[str] = []
    for parameter in parameters:
        if not isinstance(parameter, torch.Tensor) or id(parameter) not in by_identity:
            raise ValueError("Optimizer references a parameter outside the model")
        names.append(by_identity[id(parameter)])
    if len(names) != len(by_identity) or len(set(names)) != len(names):
        raise ValueError("Optimizer must cover each model parameter exactly once")
    return names


def load_model(model: nn.Module, value: object) -> None:
    source = tensors(value, "model")
    expected: object = model.state_dict()
    target = tensors(expected, "live model")
    if source.keys() != target.keys():
        raise ValueError("Checkpoint model names differ")
    for name, item in source.items():
        if item.shape != target[name].shape or item.dtype != target[name].dtype:
            raise ValueError(f"Checkpoint model shape or dtype differs: {name}")
    model.load_state_dict(source, strict=True)


def load_optimizer(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    state_values: list[TensorMap],
    group_values: list[Group],
) -> None:
    by_name = dict(model.named_parameters())
    parameters = [by_name[name] for name in optimizer_parameter_names(model, optimizer)]
    ids = parameter_ids(group_values)
    if len(state_values) != len(parameters) or len(ids) != len(parameters):
        raise ValueError("Optimizer states do not cover every parameter")
    for parameter, state in zip(parameters, state_values, strict=True):
        buffer = momentum(state, "optimizer state")["momentum_buffer"]
        if buffer.shape != parameter.shape or buffer.dtype != parameter.dtype:
            raise ValueError("Optimizer state shape or dtype differs from parameter")
    payload: OptimizerFields = {
        "state": dict(zip(ids, state_values, strict=True)),
        "param_groups": group_values,
    }
    optimizer.load_state_dict(dict(payload))


def validate_rank(rank: int, world_size: int) -> None:
    integer(rank, "rank")
    integer(world_size, "world_size", 1)
    if world_size < 1 or rank < 0 or rank >= world_size:
        raise ValueError("Invalid distributed rank metadata")


def artifact(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    progress: Progress,
    rank: int,
    world_size: int,
    device: torch.device,
) -> Artifact:
    validate_rank(rank, world_size)
    names, states, group_values = optimizer_snapshot(model, optimizer)
    named_groups: list[NamedGroup] = []
    for group in group_values:
        named: NamedGroup = {}
        for key, value in group.items():
            if key == "params":
                named[key] = list(names)
            else:
                assert not isinstance(value, list)
                named[key] = value
        named_groups.append(named)
    raw_model: object = model.state_dict()
    return {
        "model": tensor_snapshot(tensors(raw_model, "model")),
        "optimizer_state": dict(zip(names, states, strict=True)),
        "optimizer_groups": named_groups,
        "progress": progress.fields(),
        "rng": capture_rng(device),
        "rank": rank,
        "world_size": world_size,
    }
