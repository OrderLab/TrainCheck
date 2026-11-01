"""Private synchronous qualification of native BF16 gradient clipping.

Install after audit_runtime so its model-step observer stays in the call chain.
This probe is intentionally limited to the native keyword-based clipping call;
it is not part of either solver arm or its performance measurements.
"""

from __future__ import annotations

import math
import os
from contextvars import ContextVar
from importlib import import_module
from pathlib import Path
from typing import TypedDict

import torch
import torch.nn as nn


class ClippingSnapshot(TypedDict):
    rank: int
    tp_rank: int
    dp_rank: int
    step: int
    max_norm: int | float
    global_norm: int | float
    selected: dict[str, bool]
    before: dict[str, torch.Tensor]
    after: dict[str, torch.Tensor]


_current_engine: ContextVar[object | None] = ContextVar(
    "clipping_audit_engine", default=None
)
_installed = False


def _nonnegative_integer(value: object, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{field} must be an integer")
    if value < 0:
        raise ValueError(f"{field} must be nonnegative")
    return value


def _positive_number(value: object, field: str) -> int | float:
    if type(value) not in (int, float):
        raise TypeError(f"{field} must be an int or float")
    assert isinstance(value, (int, float))
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{field} must be finite and positive")
    return value


def _tensor(value: object, field: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{field} must be a tensor")
    if value.layout != torch.strided or value.dtype != torch.float32:
        raise TypeError(f"{field} must be a dense FP32 gradient")
    if value.numel() == 0 or not bool(torch.isfinite(value).all()):
        raise ValueError(f"{field} must be nonempty and finite")
    return value


def _groups(value: object, field: str) -> list[list[torch.Tensor]]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{field} must contain tensor groups")
    groups: list[list[torch.Tensor]] = []
    for index, group in enumerate(value):
        if not isinstance(group, (list, tuple)):
            raise TypeError(f"{field}[{index}] must be a tensor group")
        tensors: list[torch.Tensor] = []
        for tensor in group:
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"{field}[{index}] contains a non-tensor")
            tensors.append(tensor)
        groups.append(tensors)
    return groups


def _replicated_gradients(
    engine: object,
) -> tuple[dict[str, torch.Tensor], set[int]]:
    module: object = getattr(engine, "module")
    optimizer: object = getattr(engine, "optimizer")
    if not isinstance(module, nn.Module):
        raise TypeError("Engine module must be a torch.nn.Module")
    names: dict[int, str] = {}
    required: set[str] = set()
    for name, parameter in module.named_parameters():
        if not isinstance(name, str) or not name:
            raise TypeError("Parameter names must be nonempty strings")
        if id(parameter) in names or name in names.values():
            raise ValueError("Duplicate named parameter identity or name")
        names[id(parameter)] = name
        # Megatron explicitly treats an absent flag as nonpartitioned.
        partitioned: object = getattr(parameter, "tensor_model_parallel", False)
        if not isinstance(partitioned, bool):
            raise TypeError(f"{name}.tensor_model_parallel must be a bool")
        if parameter.requires_grad and not partitioned:
            required.add(name)
    if not required:
        raise ValueError("No replicated trainable parameters were observed")

    low_groups = _groups(getattr(optimizer, "bf16_groups"), "bf16_groups")
    grad_groups = _groups(
        getattr(optimizer, "fp32_groups_gradients"), "fp32_groups_gradients"
    )
    if len(low_groups) != len(grad_groups):
        raise ValueError("BF16 parameter and FP32 gradient group counts differ")
    gradients: dict[str, torch.Tensor] = {}
    parameter_ids: set[int] = set()
    gradient_ids: set[int] = set()
    for group_index, (parameters, group_gradients) in enumerate(
        zip(low_groups, grad_groups)
    ):
        if len(parameters) != len(group_gradients):
            raise ValueError(
                f"Parameter/gradient lengths differ in group {group_index}"
            )
        for low_parameter, gradient in zip(parameters, group_gradients):
            parameter_id = id(low_parameter)
            if parameter_id not in names:
                raise ValueError("Optimizer parameter is absent from named_parameters")
            if parameter_id in parameter_ids or id(gradient) in gradient_ids:
                raise ValueError("Duplicate optimizer parameter or gradient identity")
            parameter_ids.add(parameter_id)
            gradient_ids.add(id(gradient))
            name = names[parameter_id]
            gradient = _tensor(gradient, f"{name}.gradient")
            # Native FP32 gradient views are flat, even for matrix parameters.
            if low_parameter.numel() != gradient.numel():
                raise ValueError(f"Parameter/gradient element counts differ for {name}")
            if name in required:
                gradients[name] = gradient
    missing = required.difference(gradients)
    if missing:
        raise ValueError(f"Missing replicated gradients: {sorted(missing)}")
    return gradients, gradient_ids


def _mpu_rank(mpu: object, method_name: str) -> int:
    method: object = getattr(mpu, method_name)
    if not callable(method):
        raise TypeError(f"mpu.{method_name} must be callable")
    value: object = method()
    return _nonnegative_integer(value, method_name)


def _snapshot(tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        name: _tensor(tensor, f"{name}.gradient").detach().cpu().clone()
        for name, tensor in tensors.items()
    }


def install() -> None:
    """Wrap the current native hooks once, without replacing prior observers."""
    global _installed
    if _installed:
        raise RuntimeError("Clipping audit is already installed")
    optimizer_module = import_module("deepspeed.runtime.bf16_optimizer")
    engine_module = import_module("deepspeed.runtime.engine")
    engine_class: object = getattr(engine_module, "DeepSpeedEngine")
    if not isinstance(engine_class, type):
        raise TypeError("DeepSpeedEngine must be a class")
    original_clip: object = getattr(optimizer_module, "clip_tensors_by_global_norm")
    original_step: object = getattr(engine_class, "_take_model_step")
    if not callable(original_clip) or not callable(original_step):
        raise TypeError("Native clipping and model-step hooks must be callable")

    def observed_clip(*args: object, **kwargs: object) -> object:
        if args:
            raise TypeError("Clipping qualification requires native keyword arguments")
        for field in ("input_tensors", "max_norm", "global_norm", "mpu"):
            if field not in kwargs:
                raise TypeError(f"Missing required clipping argument: {field}")
        engine = _current_engine.get()
        if engine is None:
            raise RuntimeError("Clipping was called outside a model step")
        input_tensors = kwargs["input_tensors"]
        if not isinstance(input_tensors, (list, tuple)):
            raise TypeError("input_tensors must be a list or tuple of tensors")
        selected_ids: set[int] = set()
        for tensor in input_tensors:
            tensor = _tensor(tensor, "input_tensors entry")
            if id(tensor) in selected_ids:
                raise ValueError("input_tensors contains a duplicate gradient object")
            selected_ids.add(id(tensor))
        gradients, known_gradient_ids = _replicated_gradients(engine)
        if not selected_ids.issubset(known_gradient_ids):
            raise ValueError("Selected gradient object is absent from optimizer state")
        step = _nonnegative_integer(getattr(engine, "global_steps"), "global_steps")
        rank = _nonnegative_integer(torch.distributed.get_rank(), "rank")
        output = os.environ["RUNTIME_CLIPPING_OUTPUT"]
        if not output:
            raise ValueError("RUNTIME_CLIPPING_OUTPUT must not be empty")
        destination = Path(output) / f"rank{rank}" / f"step{step + 1}.pt"
        if destination.exists():
            raise FileExistsError(destination)
        snapshot: ClippingSnapshot = {
            "rank": rank,
            # PipelineParallelGrid separates tensor slicing from its combined
            # pipeline/model rank. The latter is not a TP rank when PP > 1.
            "tp_rank": _mpu_rank(kwargs["mpu"], "get_slice_parallel_rank"),
            "dp_rank": _mpu_rank(kwargs["mpu"], "get_data_parallel_rank"),
            "step": step + 1,
            "max_norm": _positive_number(kwargs["max_norm"], "max_norm"),
            "global_norm": _positive_number(kwargs["global_norm"], "global_norm"),
            "selected": {
                name: id(gradient) in selected_ids
                for name, gradient in gradients.items()
            },
            "before": _snapshot(gradients),
            "after": {},
        }
        result: object = original_clip(*args, **kwargs)
        snapshot["after"] = _snapshot(gradients)
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation refuses duplicate evidence rather than overwriting it.
        with destination.open("xb") as stream:
            torch.save(snapshot, stream)
        return result

    def observed_step(engine: object, *args: object, **kwargs: object) -> object:
        token = _current_engine.set(engine)
        try:
            result: object = original_step(engine, *args, **kwargs)
            return result
        finally:
            _current_engine.reset(token)

    setattr(optimizer_module, "clip_tensors_by_global_norm", observed_clip)
    setattr(engine_class, "_take_model_step", observed_step)
    _installed = True
