"""Layer-sharded checkpoint backend."""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import torch
import torch.nn as nn

from . import state
from .model import LAYERS


class ParameterBundle(TypedDict):
    model: torch.Tensor
    optimizer: state.TensorMap


class LayerChunk(TypedDict):
    layer_name: str
    parameter_names: list[str]
    parameters: list[ParameterBundle]


class SharedPayload(TypedDict):
    schema_version: int
    parameter_names: list[str]
    optimizer_groups: list[state.Group]
    nonlayer_names: list[str]
    nonlayer_parameters: list[ParameterBundle]
    progress: state.ProgressFields
    world_size: int


class RankPayload(TypedDict):
    schema_version: int
    rank: int
    world_size: int
    layers: list[LayerChunk]
    progress: state.ProgressFields
    rng: state.RngFields


def layer_name(index: int) -> str:
    return f"encoder.layers.{index}"


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
    model_values = state.tensors(raw_model, "model")
    if set(model_values) != set(names):
        raise ValueError(
            "This checkpoint schema requires every model field to be a parameter"
        )
    by_name: dict[str, ParameterBundle] = {
        name: {"model": model_values[name].detach().cpu().clone(), "optimizer": value}
        for name, value in zip(names, values, strict=True)
    }
    nonlayer_names = [name for name in names if not name.startswith("encoder.layers.")]
    if rank == 0:
        shared: SharedPayload = {
            "schema_version": 2,
            "parameter_names": names,
            "optimizer_groups": groups,
            "nonlayer_names": nonlayer_names,
            "nonlayer_parameters": [by_name[name] for name in nonlayer_names],
            "progress": progress.fields(),
            "world_size": world_size,
        }
        torch.save(shared, directory / "shared.pt")
    chunks: list[LayerChunk] = []
    for index in range(rank, LAYERS, world_size):
        prefix = layer_name(index)
        selected = [name for name in names if name.startswith(prefix + ".")]
        if not selected:
            raise ValueError("Expected a nonempty Transformer layer")
        chunks.append(
            {
                "layer_name": prefix,
                "parameter_names": selected,
                "parameters": [by_name[name] for name in selected],
            }
        )
    shard: RankPayload = {
        "schema_version": 2,
        "rank": rank,
        "world_size": world_size,
        "layers": chunks,
        "progress": progress.fields(),
        "rng": state.capture_rng(device),
    }
    torch.save(shard, directory / f"rank{rank}.pt")


def read_bundle(value: object, label: str) -> ParameterBundle:
    fields = state.mapping(value, label)
    state.exact_keys(fields, {"model", "optimizer"}, label)
    model = state.tensors({"model": fields["model"]}, label)["model"]
    optimizer = state.momentum(fields["optimizer"], f"{label}.optimizer")
    momentum = optimizer["momentum_buffer"]
    if model.shape != momentum.shape or model.dtype != momentum.dtype:
        raise ValueError(f"{label} model and momentum shape or dtype differ")
    return {"model": model, "optimizer": optimizer}


def read_layer(value: object) -> LayerChunk:
    chunk = state.mapping(value, "layer")
    state.exact_keys(chunk, {"layer_name", "parameter_names", "parameters"}, "layer")
    prefix = chunk["layer_name"]
    if not isinstance(prefix, str) or prefix not in {
        layer_name(i) for i in range(LAYERS)
    }:
        raise ValueError("Unknown checkpoint layer")
    names = state.strings(chunk["parameter_names"], "layer parameter_names")
    values = chunk["parameters"]
    if not names or not all(name.startswith(prefix + ".") for name in names):
        raise ValueError("Layer parameter names differ")
    if not isinstance(values, list) or len(values) != len(names):
        raise ValueError("Incomplete layer state")
    return {
        "layer_name": prefix,
        "parameter_names": names,
        "parameters": [read_bundle(v, n) for n, v in zip(names, values, strict=True)],
    }


def restore_checkpoint(
    directory: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    rank: int,
    world_size: int,
    device: torch.device,
) -> state.Progress:
    state.validate_rank(rank, world_size)
    shared = state.read(directory / "shared.pt")
    state.exact_keys(
        shared,
        {
            "schema_version",
            "parameter_names",
            "optimizer_groups",
            "nonlayer_names",
            "nonlayer_parameters",
            "progress",
            "world_size",
        },
        "shared checkpoint",
    )
    if state.integer(shared["schema_version"], "schema_version") != 2:
        raise ValueError("Unsupported sharded checkpoint version")
    if state.integer(shared["world_size"], "world_size", 1) != world_size:
        raise ValueError("Checkpoint topology differs")
    names = state.strings(shared["parameter_names"], "parameter_names")
    if names != state.optimizer_parameter_names(model, optimizer):
        raise ValueError("Checkpoint optimizer parameter names differ")
    groups = state.groups(shared["optimizer_groups"], len(names))
    progress = state.read_progress(shared["progress"])
    chunks: list[LayerChunk] = []
    rank_rng: state.RngFields | None = None
    for source_rank in range(world_size):
        shard = state.read(directory / f"rank{source_rank}.pt")
        state.exact_keys(
            shard,
            {"schema_version", "rank", "world_size", "layers", "progress", "rng"},
            "rank shard",
        )
        if state.integer(shard["schema_version"], "schema_version") != 2:
            raise ValueError("Unsupported rank shard version")
        if (
            state.integer(shard["rank"], "rank") != source_rank
            or state.integer(shard["world_size"], "world_size", 1) != world_size
        ):
            raise ValueError("Shard topology differs")
        if state.read_progress(shard["progress"]) != progress:
            raise ValueError("Shard progress differs")
        layers = shard["layers"]
        if not isinstance(layers, list):
            raise TypeError("Shard layers must be a list")
        parsed = [read_layer(value) for value in layers]
        expected = [
            layer_name(index) for index in range(source_rank, LAYERS, world_size)
        ]
        if [chunk["layer_name"] for chunk in parsed] != expected:
            raise ValueError("Shard layer coverage differs")
        chunks.extend(parsed)
        rng = state.read_rng(shard["rng"], device)
        if source_rank == rank:
            rank_rng = rng
    assert rank_rng is not None
    nonlayer_names = state.strings(shared["nonlayer_names"], "nonlayer_names")
    nonlayer_values = shared["nonlayer_parameters"]
    expected_nonlayer = [
        name for name in names if not name.startswith("encoder.layers.")
    ]
    if (
        nonlayer_names != expected_nonlayer
        or not isinstance(nonlayer_values, list)
        or len(nonlayer_values) != len(nonlayer_names)
    ):
        raise ValueError("Nonlayer parameter coverage differs")
    restored = {
        name: read_bundle(value, name)
        for name, value in zip(nonlayer_names, nonlayer_values, strict=True)
    }
    layer_names = [name for name in names if name.startswith("encoder.layers.")]
    chunk_names = [name for chunk in chunks for name in chunk["parameter_names"]]
    if len(chunk_names) != len(layer_names) or set(chunk_names) != set(layer_names):
        raise ValueError("Layer parameter coverage differs")
    layer_values = [value for chunk in chunks for value in chunk["parameters"]]
    restored.update(zip(layer_names, layer_values, strict=True))
    state.load_model(model, {name: restored[name]["model"] for name in names})
    state.load_optimizer(
        model, optimizer, [restored[name]["optimizer"] for name in names], groups
    )
    state.restore_rng(rank_rng, device)
    return progress
