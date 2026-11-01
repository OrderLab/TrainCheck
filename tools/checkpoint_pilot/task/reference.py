"""Replicated checkpoint backend."""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import torch
import torch.nn as nn

from . import state


class Payload(TypedDict):
    schema_version: int
    model: state.TensorMap
    parameter_names: list[str]
    optimizer_states: list[state.TensorMap]
    optimizer_groups: list[state.Group]
    progress: state.ProgressFields
    rng: state.RngFields
    rank: int
    world_size: int


def save_checkpoint(
    directory: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    progress: state.Progress,
    rank: int,
    world_size: int,
    device: torch.device,
) -> None:
    state.validate_rank(rank, world_size)
    names, values, groups = state.optimizer_snapshot(model, optimizer)
    raw_model: object = model.state_dict()
    payload: Payload = {
        "schema_version": 1,
        "model": state.tensor_snapshot(state.tensors(raw_model, "model")),
        "parameter_names": names,
        "optimizer_states": values,
        "optimizer_groups": groups,
        "progress": progress.fields(),
        "rng": state.capture_rng(device),
        "rank": rank,
        "world_size": world_size,
    }
    torch.save(payload, directory / f"rank{rank}.pt")


def restore_checkpoint(
    directory: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    rank: int,
    world_size: int,
    device: torch.device,
) -> state.Progress:
    state.validate_rank(rank, world_size)
    payload = state.read(directory / f"rank{rank}.pt")
    state.exact_keys(
        payload,
        {
            "schema_version",
            "model",
            "parameter_names",
            "optimizer_states",
            "optimizer_groups",
            "progress",
            "rng",
            "rank",
            "world_size",
        },
        "replicated checkpoint",
    )
    if state.integer(payload["schema_version"], "schema_version") != 1:
        raise ValueError("Unsupported replicated checkpoint version")
    if (
        state.integer(payload["rank"], "rank") != rank
        or state.integer(payload["world_size"], "world_size", 1) != world_size
    ):
        raise ValueError("Checkpoint topology differs")
    names = state.strings(payload["parameter_names"], "parameter_names")
    if names != state.optimizer_parameter_names(model, optimizer):
        raise ValueError("Checkpoint optimizer parameter names differ")
    values = payload["optimizer_states"]
    if not isinstance(values, list) or len(values) != len(names):
        raise ValueError("Incomplete optimizer states")
    states = [
        state.momentum(value, name) for name, value in zip(names, values, strict=True)
    ]
    groups = state.groups(payload["optimizer_groups"], len(names))
    progress = state.read_progress(payload["progress"])
    rng = state.read_rng(payload["rng"], device)
    state.load_model(model, payload["model"])
    state.load_optimizer(model, optimizer, states, groups)
    state.restore_rng(rng, device)
    return progress
