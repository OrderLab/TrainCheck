"""Validate independently captured rank/step state without importing candidates.

The operator owns these artifacts and must freeze source and artifact manifests
outside this evaluator. ``weights_only=True`` limits deserialization; it does
not establish artifact provenance or defend against a candidate rewriting files.
Optimizer containers are native dict/list/tuple values. Their key/element order,
scalar types and tensor bytes remain significant. Model tolerances apply only
to floating/complex parameter comparisons, including replicated parameters.
"""

from __future__ import annotations

import math
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, TypeAlias

import torch

Optimizer: TypeAlias = (
    None
    | bool
    | int
    | float
    | str
    | torch.Tensor
    | list["Optimizer"]
    | tuple["Optimizer", ...]
    | dict[int | str, "Optimizer"]
)
Identity: TypeAlias = tuple[int, int]
Category: TypeAlias = Literal["model", "optimizer", "rng", "replicated"]


@dataclass(frozen=True)
class Snapshot:
    rank: int
    tp_rank: int
    dp_rank: int
    step: int
    parameters: dict[str, torch.Tensor]
    replicated_names: tuple[str, ...]
    optimizer: Optimizer
    rng: dict[str, torch.Tensor]


@dataclass(frozen=True)
class Evaluation:
    passed: bool
    world_size: int
    steps: tuple[int, ...]
    candidate_artifacts: int
    reference_artifacts: int
    tensor_parallel_size: int
    data_parallel_size: int
    model_checks: int
    optimizer_checks: int
    rng_checks: int
    replicated_checks: int
    replicated_status: Literal["passed", "violated", "not_applicable"]
    failed_checks: int
    failed_paths: tuple[str, ...]
    failed_paths_truncated: bool
    maximum_absolute_model_difference: float | None
    model_comparable_tensors: int
    model_absolute_difference_overflow: bool
    atol: float
    rtol: float


@dataclass
class _Checks:
    counts: dict[Category, int] = field(
        default_factory=lambda: {"model": 0, "optimizer": 0, "rng": 0, "replicated": 0}
    )
    failures: int = 0
    replicated_failures: int = 0
    paths: list[str] = field(default_factory=list)
    maximum_model_difference: float | None = None
    comparable_model_tensors: int = 0
    model_difference_overflow: bool = False

    def record(self, category: Category, path: str, passed: bool) -> None:
        self.counts[category] += 1
        if not passed:
            self.failures += 1
            if category == "replicated":
                self.replicated_failures += 1
            if len(self.paths) < 100:
                self.paths.append(path)


def _integer(value: object, path: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{path}: expected integer, excluding bool")
    if value < (1 if positive else 0):
        raise ValueError(f"{path}: integer is out of range")
    return value


def _tolerance(value: object, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{path}: expected finite nonnegative number")
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{path}: expected finite nonnegative number")
    return float(value)


def _mapping(value: object, path: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"{path}: expected dict")
    result: dict[str, object] = {}
    for key, child in value.items():
        if not isinstance(key, str) or not key:
            raise TypeError(f"{path}: keys must be nonempty strings")
        result[key] = child
    return result


def _tensor(value: object, path: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{path}: expected tensor")
    if value.layout != torch.strided or value.is_quantized:
        raise TypeError(f"{path}: expected dense nonquantized tensor")
    if value.device.type != "cpu":
        raise ValueError(f"{path}: deserialized tensor is not on CPU")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{path}: tensor contains nonfinite values")
    return value


def _tensor_mapping(value: object, path: str) -> dict[str, torch.Tensor]:
    mapping = _mapping(value, path)
    if not mapping:
        raise ValueError(f"{path}: empty mapping does not establish coverage")
    return {name: _tensor(item, f"{path}[{name!r}]") for name, item in mapping.items()}


def _optimizer(value: object, path: str, active: set[int]) -> Optimizer:
    if isinstance(value, torch.Tensor):
        return _tensor(value, path)
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path}: nonfinite scalar")
        return value
    # Native container types are the artifact contract. Reject subclasses rather
    # than silently normalizing their type (e.g. OrderedDict) into a plain dict.
    if type(value) not in (dict, list, tuple):
        raise TypeError(f"{path}: unsupported optimizer value {type(value).__name__}")
    identity = id(value)
    if identity in active:
        raise ValueError(f"{path}: cyclic optimizer containers are unsupported")
    active.add(identity)
    try:
        if isinstance(value, list):
            return [
                _optimizer(child, f"{path}[{i}]", active)
                for i, child in enumerate(value)
            ]
        if isinstance(value, tuple):
            return tuple(
                _optimizer(child, f"{path}[{i}]", active)
                for i, child in enumerate(value)
            )
        assert isinstance(value, dict)
        result: dict[int | str, Optimizer] = {}
        for key, child in value.items():
            if isinstance(key, bool) or not isinstance(key, (int, str)):
                raise TypeError(
                    f"{path}: optimizer dict keys must be int or str, excluding bool"
                )
            result[key] = _optimizer(child, f"{path}[{key!r}]", active)
        return result
    finally:
        active.remove(identity)


def _snapshot(value: object, path: Path) -> Snapshot:
    record = _mapping(value, str(path))
    required = {
        "rank",
        "tp_rank",
        "dp_rank",
        "step",
        "parameters",
        "replicated_names",
        "optimizer",
        "rng",
    }
    if set(record) != required:
        raise ValueError(
            f"{path}: snapshot fields differ; missing={sorted(required - set(record))}, extra={sorted(set(record) - required)}"
        )
    parameters = _tensor_mapping(record["parameters"], f"{path}.parameters")
    names = record["replicated_names"]
    if not isinstance(names, list):
        raise TypeError(f"{path}.replicated_names: expected list")
    replicated: list[str] = []
    for name in names:
        if not isinstance(name, str) or name not in parameters:
            raise ValueError(
                f"{path}.replicated_names: unknown parameter name {name!r}"
            )
        if name in replicated:
            raise ValueError(f"{path}.replicated_names: duplicate parameter {name!r}")
        replicated.append(name)
    rng = _tensor_mapping(record["rng"], f"{path}.rng")
    for name, tensor in rng.items():
        if tensor.dtype != torch.uint8 or tensor.ndim != 1 or tensor.numel() == 0:
            raise ValueError(f"{path}.rng[{name!r}]: expected nonempty 1-D byte tensor")
    return Snapshot(
        rank=_integer(record["rank"], f"{path}.rank"),
        tp_rank=_integer(record["tp_rank"], f"{path}.tp_rank"),
        dp_rank=_integer(record["dp_rank"], f"{path}.dp_rank"),
        step=_integer(record["step"], f"{path}.step"),
        parameters=parameters,
        replicated_names=tuple(replicated),
        optimizer=_optimizer(record["optimizer"], f"{path}.optimizer", set()),
        rng=rng,
    )


def _load(
    directory: Path, world_size: int, steps: tuple[int, ...]
) -> dict[Identity, Snapshot]:
    if not directory.is_dir() or directory.is_symlink():
        raise ValueError(f"{directory}: expected existing nonsymlink directory")
    snapshots: dict[Identity, Snapshot] = {}
    for path in sorted(directory.glob("rank*/step*.pt")):
        if path.is_symlink() or path.parent.is_symlink() or not path.is_file():
            raise ValueError(f"{path}: expected regular nonsymlink artifact")
        rank_match = re.fullmatch(r"rank([0-9]+)", path.parent.name)
        step_match = re.fullmatch(r"step([0-9]+)\.pt", path.name)
        if rank_match is None or step_match is None:
            raise ValueError(f"{path}: malformed rank/step filename")
        raw: object = torch.load(path, map_location="cpu", weights_only=True)
        state = _snapshot(raw, path)
        identity = (state.rank, state.step)
        if identity in snapshots:
            raise ValueError(f"{path}: duplicate rank/step identity {identity}")
        if identity != (int(rank_match[1]), int(step_match[1])):
            raise ValueError(f"{path}: path disagrees with recorded rank/step")
        snapshots[identity] = state
    expected = {(rank, step) for rank in range(world_size) for step in steps}
    if set(snapshots) != expected:
        raise ValueError(
            f"{directory}: incomplete or unexpected rank/step coverage; missing={sorted(expected - set(snapshots))}, extra={sorted(set(snapshots) - expected)}"
        )
    return snapshots


def _topology(
    states: dict[Identity, Snapshot], world_size: int, steps: tuple[int, ...]
) -> tuple[int, int]:
    first = {
        rank: (states[rank, steps[0]].dp_rank, states[rank, steps[0]].tp_rank)
        for rank in range(world_size)
    }
    dp_size = max(dp for dp, _ in first.values()) + 1
    tp_size = max(tp for _, tp in first.values()) + 1
    if dp_size * tp_size != world_size:
        raise ValueError("Incomplete or duplicated data/tensor parallel topology")
    expected = {(dp, tp) for dp in range(dp_size) for tp in range(tp_size)}
    if len(expected) != world_size or set(first.values()) != expected:
        raise ValueError("Incomplete or duplicated data/tensor parallel topology")
    for (rank, _), state in states.items():
        if (state.dp_rank, state.tp_rank) != first[rank]:
            raise ValueError("Rank topology changed across observed steps")
    for step in steps:
        for dp in range(dp_size):
            group = [
                state
                for state in states.values()
                if state.step == step and state.dp_rank == dp
            ]
            names = group[0].replicated_names
            if any(state.replicated_names != names for state in group):
                raise ValueError("Replicated parameter names differ within a TP group")
            if tp_size > 1 and not names:
                raise ValueError(
                    "Empty replicated parameter coverage for a multi-rank TP group"
                )
            if tp_size > 1 and any(
                state.parameters[name].numel() == 0 for state in group for name in names
            ):
                raise ValueError("Empty replicated tensor does not establish coverage")
    return tp_size, dp_size


def _same_tensor_bytes(left: torch.Tensor, right: torch.Tensor) -> bool:
    return torch.equal(
        left.contiguous().reshape(-1).view(torch.uint8),
        right.contiguous().reshape(-1).view(torch.uint8),
    )


def _maximum_difference(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.numel() == 0:
        return 0.0
    if left.is_floating_point() or left.is_complex():
        dtype = torch.complex128 if left.is_complex() else torch.float64
        return float((left.to(dtype) - right.to(dtype)).abs().max().item())
    # Python integer subtraction avoids int64 overflow and float64 rounding
    # that could otherwise report zero for unequal large integer parameters.
    left_values: object = left.reshape(-1).tolist()
    right_values: object = right.reshape(-1).tolist()
    assert isinstance(left_values, list) and isinstance(right_values, list)
    maximum = 0
    for left_item, right_item in zip(left_values, right_values):
        assert isinstance(left_item, (bool, int)) and isinstance(
            right_item, (bool, int)
        )
        maximum = max(maximum, abs(int(left_item) - int(right_item)))
    return float(maximum)


def _compare_tensor(
    candidate: torch.Tensor,
    reference: torch.Tensor,
    path: str,
    category: Category,
    checks: _Checks,
    atol: float,
    rtol: float,
) -> None:
    comparable = (
        candidate.dtype == reference.dtype and candidate.shape == reference.shape
    )
    checks.record(category, path + ".dtype_shape", comparable)
    if not comparable:
        return
    if category == "model":
        checks.comparable_model_tensors += 1
        difference = _maximum_difference(candidate, reference)
        if math.isfinite(difference):
            previous = checks.maximum_model_difference
            checks.maximum_model_difference = (
                difference if previous is None else max(previous, difference)
            )
        else:
            checks.model_difference_overflow = True
    if (atol or rtol) and (candidate.is_floating_point() or candidate.is_complex()):
        same = bool(torch.allclose(candidate, reference, atol=atol, rtol=rtol))
    else:
        same = _same_tensor_bytes(candidate, reference)
    checks.record(category, path + ".value", same)


def _compare_optimizer(
    candidate: Optimizer, reference: Optimizer, path: str, checks: _Checks
) -> None:
    checks.record("optimizer", path + ".type", type(candidate) is type(reference))
    if type(candidate) is not type(reference):
        return
    if isinstance(candidate, torch.Tensor):
        assert isinstance(reference, torch.Tensor)
        _compare_tensor(candidate, reference, path, "optimizer", checks, 0.0, 0.0)
    elif isinstance(candidate, dict):
        assert isinstance(reference, dict)
        candidate_keys = [(type(key), key) for key in candidate]
        reference_keys = [(type(key), key) for key in reference]
        same = candidate_keys == reference_keys
        checks.record("optimizer", path + ".ordered_keys", same)
        if same:
            for key in reference:
                _compare_optimizer(
                    candidate[key], reference[key], f"{path}[{key!r}]", checks
                )
    elif isinstance(candidate, (list, tuple)):
        assert isinstance(reference, (list, tuple))
        checks.record("optimizer", path + ".length", len(candidate) == len(reference))
        for index, (left, right) in enumerate(zip(candidate, reference)):
            _compare_optimizer(left, right, f"{path}[{index}]", checks)
    elif isinstance(candidate, float):
        assert isinstance(reference, float)
        checks.record(
            "optimizer",
            path + ".value",
            struct.pack("!d", candidate) == struct.pack("!d", reference),
        )
    else:
        checks.record("optimizer", path + ".value", candidate == reference)


def evaluate(
    candidate_directory: Path,
    reference_directory: Path,
    world_size: int,
    steps: list[int],
    atol: float = 0.0,
    rtol: float = 0.0,
) -> Evaluation:
    """Compare complete coverage; malformed/missing evidence raises an error.

    Replication checks use each step's TP ranks within one DP group. A TP1
    topology reports ``not_applicable`` even if several DP replicas exist.
    References must independently satisfy the same replication invariant.
    """
    world_size = _integer(world_size, "world_size", positive=True)
    if not isinstance(steps, list) or not steps:
        raise ValueError("steps must be a nonempty list")
    expected_steps = tuple(_integer(step, "step") for step in steps)
    if tuple(sorted(set(expected_steps))) != expected_steps:
        raise ValueError("steps must be strictly increasing and unique")
    atol = _tolerance(atol, "atol")
    rtol = _tolerance(rtol, "rtol")
    candidate = _load(candidate_directory, world_size, expected_steps)
    reference = _load(reference_directory, world_size, expected_steps)
    topology = _topology(candidate, world_size, expected_steps)
    if topology != _topology(reference, world_size, expected_steps):
        raise ValueError("Candidate and reference parallel topology differ")
    checks = _Checks()
    for identity in sorted(reference):
        left, right = candidate[identity], reference[identity]
        path = f"rank{identity[0]}/step{identity[1]}"
        if (left.dp_rank, left.tp_rank) != (right.dp_rank, right.tp_rank):
            raise ValueError(f"{path}: candidate/reference rank topology differs")
        if left.replicated_names != right.replicated_names:
            raise ValueError(f"{path}: candidate/reference replicated metadata differs")
        checks.record(
            "model",
            path + ".parameter_names",
            list(left.parameters) == list(right.parameters),
        )
        for name in right.parameters:
            if name in left.parameters:
                _compare_tensor(
                    left.parameters[name],
                    right.parameters[name],
                    f"{path}.parameters[{name!r}]",
                    "model",
                    checks,
                    atol,
                    rtol,
                )
        _compare_optimizer(left.optimizer, right.optimizer, path + ".optimizer", checks)
        checks.record("rng", path + ".rng_names", list(left.rng) == list(right.rng))
        for name in right.rng:
            if name in left.rng:
                _compare_tensor(
                    left.rng[name],
                    right.rng[name],
                    f"{path}.rng[{name!r}]",
                    "rng",
                    checks,
                    0.0,
                    0.0,
                )
    tp_size, dp_size = topology
    if tp_size > 1:
        for step in expected_steps:
            for dp in range(dp_size):
                group = sorted(
                    (
                        state
                        for state in candidate.values()
                        if state.step == step and state.dp_rank == dp
                    ),
                    key=lambda state: state.tp_rank,
                )
                base = group[0]
                for state in group[1:]:
                    for name in base.replicated_names:
                        path = (
                            f"step{step}/dp{dp}/tp{state.tp_rank}.replicated[{name!r}]"
                        )
                        _compare_tensor(
                            state.parameters[name],
                            base.parameters[name],
                            path,
                            "replicated",
                            checks,
                            atol,
                            rtol,
                        )
                        reference_base = reference[base.rank, step].parameters[name]
                        reference_other = reference[state.rank, step].parameters[name]
                        reference_checks = _Checks()
                        _compare_tensor(
                            reference_other,
                            reference_base,
                            path,
                            "replicated",
                            reference_checks,
                            atol,
                            rtol,
                        )
                        if reference_checks.failures:
                            raise ValueError(
                                f"{path}: known-correct reference violates replication"
                            )
        assert checks.counts["replicated"] > 0
    status: Literal["passed", "violated", "not_applicable"] = (
        "not_applicable"
        if tp_size == 1
        else "violated" if checks.replicated_failures else "passed"
    )
    return Evaluation(
        passed=checks.failures == 0,
        world_size=world_size,
        steps=expected_steps,
        candidate_artifacts=len(candidate),
        reference_artifacts=len(reference),
        tensor_parallel_size=tp_size,
        data_parallel_size=dp_size,
        model_checks=checks.counts["model"],
        optimizer_checks=checks.counts["optimizer"],
        rng_checks=checks.counts["rng"],
        replicated_checks=checks.counts["replicated"],
        replicated_status=status,
        failed_checks=checks.failures,
        failed_paths=tuple(checks.paths),
        failed_paths_truncated=checks.failures > len(checks.paths),
        maximum_absolute_model_difference=(
            None
            if checks.model_difference_overflow
            else checks.maximum_model_difference
        ),
        model_comparable_tensors=checks.comparable_model_tensors,
        model_absolute_difference_overflow=checks.model_difference_overflow,
        atol=atol,
        rtol=rtol,
    )
