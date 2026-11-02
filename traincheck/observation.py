"""Explicit model snapshots and offline comparison, without installing hooks.

Snapshots reuse model sampling's enumeration and the tensor tracker's existing
SHA-256 implementation. They fingerprint *all* logical tensor bytes, not the
legacy quantized, first-row parallel hash. Equal digests are fingerprint
evidence, not a collision-free proof. No model or tensor is retained.

Capture at a quiescent boundary: concurrent mutation is unsupported. For FSDP,
the caller must enter its framework's full-parameter context on every required
rank and pass the unwrapped module. There are no hidden collectives, automatic
shard gathering, implicit rank inference, or inference-engine requirements.
"""

from __future__ import annotations

import inspect
import json
import sys
import time
from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import ContextManager, Literal

import torch
from torch import nn

from traincheck._model_state import TensorKind, iter_model_tensors
from traincheck._tensor_fingerprint import exact_tensor_hash

_SUPPORTED_DTYPES: frozenset[str] = frozenset(
    str(dtype)
    for dtype in (
        torch.bool,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
        torch.complex64,
        torch.complex128,
    )
)


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a nonempty string")
    return value


def _integer(value: object, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{field} must be a nonnegative integer")
    return value


@dataclass(frozen=True)
class TensorSnapshot:
    """One registered tensor name; aliases reference the first name of the object."""

    name: str
    kind: TensorKind
    shape: tuple[int, ...]
    dtype: str
    device: str
    sha256: str
    requires_grad: bool
    object_alias_of: str | None

    def __post_init__(self) -> None:
        _text(self.name, "tensor.name")
        if self.kind not in ("parameter", "buffer"):
            raise ValueError("tensor.kind must be parameter or buffer")
        if not isinstance(self.shape, tuple):
            raise ValueError("tensor.shape must be a tuple")
        for dimension in self.shape:
            _integer(dimension, "tensor.shape dimension")
        if _text(self.dtype, "tensor.dtype") not in _SUPPORTED_DTYPES:
            raise ValueError(f"unsupported tensor dtype: {self.dtype}")
        _text(self.device, "tensor.device")
        if (
            not isinstance(self.sha256, str)
            or len(self.sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.sha256)
        ):
            raise ValueError("tensor.sha256 must be a lowercase SHA-256 hex digest")
        if not isinstance(self.requires_grad, bool):
            raise ValueError("tensor.requires_grad must be bool")
        if self.object_alias_of is not None:
            _text(self.object_alias_of, "tensor.object_alias_of")


@dataclass(frozen=True)
class ModelSnapshot:
    schema_version: Literal[1]
    run_id: str
    model_name: str
    rank: int
    step: int
    phase: str
    time_ns: int
    tensors: tuple[TensorSnapshot, ...]

    def __post_init__(self) -> None:
        if _integer(self.schema_version, "schema_version") != 1:
            raise ValueError("unsupported snapshot schema_version")
        _text(self.run_id, "run_id")
        _text(self.model_name, "model_name")
        _text(self.phase, "phase")
        _integer(self.rank, "rank")
        _integer(self.step, "step")
        _integer(self.time_ns, "time_ns")
        if not isinstance(self.tensors, tuple) or not self.tensors:
            raise ValueError("snapshot must contain a nonempty tuple of tensors")
        seen: dict[str, TensorSnapshot] = {}
        for tensor in self.tensors:
            if not isinstance(tensor, TensorSnapshot):
                raise ValueError("snapshot.tensors must contain TensorSnapshot values")
            if tensor.name in seen:
                raise ValueError(f"duplicate tensor name: {tensor.name}")
            alias = tensor.object_alias_of
            if alias is not None:
                if alias not in seen or seen[alias].object_alias_of is not None:
                    raise ValueError(f"invalid object alias for {tensor.name}: {alias}")
                original = seen[alias]
                if (
                    tensor.shape,
                    tensor.dtype,
                    tensor.device,
                    tensor.sha256,
                    tensor.requires_grad,
                ) != (
                    original.shape,
                    original.dtype,
                    original.device,
                    original.sha256,
                    original.requires_grad,
                ):
                    raise ValueError(f"inconsistent object alias for {tensor.name}")
            seen[tensor.name] = tensor


@dataclass(frozen=True)
class TensorDifference:
    name: str
    fields: tuple[str, ...]


@dataclass(frozen=True)
class SnapshotComparison:
    reference_rank: int
    candidate_rank: int
    tensors_compared: int
    differences: tuple[TensorDifference, ...]

    @property
    def equal(self) -> bool:
        return not self.differences


@dataclass(frozen=True)
class RankComparison:
    reference_rank: int
    ranks: tuple[int, ...]
    comparisons: tuple[SnapshotComparison, ...]

    @property
    def equal(self) -> bool:
        return all(comparison.equal for comparison in self.comparisons)


def _validate_tensor(tensor: torch.Tensor, name: str) -> None:
    # Do not silently call DTensor.to_local(), accept a fake tensor, or execute
    # arbitrary subclass dispatch. ProxyParameter has ordinary Tensor dispatch.
    if type(tensor) not in (torch.Tensor, nn.Parameter) and not _is_tracked_parameter(
        tensor
    ):
        raise TypeError(f"unsupported tensor subclass at {name}: {type(tensor)}")
    if tensor.layout != torch.strided or tensor.is_quantized or tensor.is_meta:
        raise ValueError(f"{name} requires a materialized, dense, unquantized tensor")
    if tensor.device.type not in ("cpu", "cuda", "mps"):
        raise ValueError(f"unsupported tensor device at {name}: {tensor.device}")
    if str(tensor.dtype) not in _SUPPORTED_DTYPES:
        raise ValueError(f"unsupported tensor dtype at {name}: {tensor.dtype}")
    if tensor.__dict__.get("_is_flat_param", False):
        raise ValueError(
            f"{name} is a flat parameter; expose complete named parameters"
        )


def _is_tracked_parameter(tensor: torch.Tensor) -> bool:
    """Recognize the exact existing tracker type without installing that tracker."""
    name = "traincheck.instrumentor.proxy_wrapper.subclass"
    if name not in sys.modules:  # Absence means the subclass tracker is inactive.
        return False
    module = sys.modules[name]
    if not isinstance(module, ModuleType):
        raise TypeError("the loaded subclass tracker must be a Python module")
    parameter_type: object = vars(module).get("ProxyParameter")
    if (
        not isinstance(parameter_type, type)
        or parameter_type.__module__ != name
        or not issubclass(parameter_type, nn.Parameter)
    ):
        raise TypeError("the loaded tracker has an invalid ProxyParameter class")
    return type(tensor) is parameter_type


def _suppress_existing_tracer() -> ContextManager[object]:
    """Use the installed recorder's guard, without importing an inactive recorder."""
    name = "traincheck.instrumentor.tracer"
    if name not in sys.modules:
        return nullcontext()
    module = sys.modules[name]
    if not isinstance(module, ModuleType):
        raise TypeError("the loaded tracer must be a Python module")
    # A missing guard in an already-loaded recorder must propagate as an error.
    from traincheck.instrumentor.tracer import suppress_tracing

    return suppress_tracing()


def _validate_model_enumeration(model: nn.Module) -> None:
    """Reject custom enumeration that could silently omit registered state."""
    visited: set[int] = set()
    active: set[int] = set()

    def visit(module: nn.Module) -> None:
        if id(module) in active:
            raise ValueError("cyclic module registration is unsupported")
        if id(module) in visited:
            return
        visited.add(id(module))
        active.add(id(module))
        for name in (
            "named_parameters",
            "named_buffers",
            "named_modules",
            "_named_members",
        ):
            if inspect.getattr_static(module, name) is not inspect.getattr_static(
                nn.Module, name
            ):
                raise TypeError(
                    f"custom {name} enumeration is unsupported for snapshots"
                )
        children: object = vars(module)["_modules"]
        if not isinstance(children, dict):
            raise TypeError("module._modules must be a dictionary")
        for child in children.values():
            if child is not None:
                if not isinstance(child, nn.Module):
                    raise TypeError("module children must be nn.Module or None")
                visit(child)
        active.remove(id(module))

    visit(model)


def snapshot_model(
    model: nn.Module,
    *,
    run_id: str,
    rank: int,
    phase: str,
    step: int = 0,
    model_name: str = "model",
) -> ModelSnapshot:
    """Fingerprint parameters and all buffers at an explicit observation boundary.

    Every registered alias is retained. Captured values do not share storage with
    the model; requires_grad, gradients, optimizer references and RNG are not
    changed. None registrations are not tensors. Device is retained as provenance
    but excluded from cross-device comparison. Object aliasing is recorded;
    storage aliasing between distinct tensor objects is not assessed.

    CPU transfer and hashing are synchronous. This is occasional observation,
    not a zero-overhead hook or an atomic snapshot of concurrent training.
    """
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    _text(run_id, "run_id")
    _text(model_name, "model_name")
    _text(phase, "phase")
    _integer(rank, "rank")
    _integer(step, "step")
    with _suppress_existing_tracer(), torch.no_grad():
        started = time.monotonic_ns()
        _validate_model_enumeration(model)
        entries = tuple(
            iter_model_tensors(model, include_buffers=True, remove_duplicate=False)
        )
        for _, name, tensor in entries:
            _validate_tensor(tensor, name)
        aliases: dict[int, str] = {}
        records: list[TensorSnapshot] = []
        for kind, name, tensor in entries:
            # Resolve lazy conjugate/negative views and hash complete logical
            # bytes. The fresh byte view cannot carry an application-cached hash.
            value = tensor.detach().resolve_conj().resolve_neg().contiguous().cpu()
            raw = value.reshape(-1).view(torch.uint8)
            digest = exact_tensor_hash(raw)
            alias = aliases.get(id(tensor))  # Absent means this object's first name.
            records.append(
                TensorSnapshot(
                    name=name,
                    kind=kind,
                    shape=tuple(tensor.shape),
                    dtype=str(tensor.dtype),
                    device=str(tensor.device),
                    sha256=f"{digest:064x}",
                    requires_grad=tensor.requires_grad,
                    object_alias_of=alias,
                )
            )
            if alias is None:
                aliases[id(tensor)] = name
        return ModelSnapshot(
            1, run_id, model_name, rank, step, phase, started, tuple(records)
        )


def write_snapshot(snapshot: ModelSnapshot, path: Path) -> None:
    """Write one immutable observation; an existing file is an error."""
    if not isinstance(snapshot, ModelSnapshot):
        raise TypeError("snapshot must be ModelSnapshot")
    with path.open("x", encoding="utf-8") as handle:
        json.dump(asdict(snapshot), handle, allow_nan=False, separators=(",", ":"))
        handle.write("\n")


def _object(value: object, fields: set[str], name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    result: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ValueError(f"{name} keys must be strings")
        result[key] = item
    if set(result) != fields:
        raise ValueError(f"{name} fields must be exactly {sorted(fields)}")
    return result


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _read_tensor(value: object) -> TensorSnapshot:
    data = _object(
        value,
        {
            "name",
            "kind",
            "shape",
            "dtype",
            "device",
            "sha256",
            "requires_grad",
            "object_alias_of",
        },
        "tensor",
    )
    shape = data["shape"]
    if not isinstance(shape, list):
        raise ValueError("tensor.shape must be a JSON array")
    dimensions = tuple(_integer(size, "tensor.shape dimension") for size in shape)
    kind = data["kind"]
    if kind not in ("parameter", "buffer"):
        raise ValueError("tensor.kind must be parameter or buffer")
    # The explicit branch narrows the untrusted JSON value to its literal contract.
    tensor_kind: TensorKind = "parameter" if kind == "parameter" else "buffer"
    requires_grad = data["requires_grad"]
    if not isinstance(requires_grad, bool):
        raise ValueError("tensor.requires_grad must be bool")
    alias = data["object_alias_of"]
    if alias is not None:
        alias = _text(alias, "tensor.object_alias_of")
    return TensorSnapshot(
        _text(data["name"], "tensor.name"),
        tensor_kind,
        dimensions,
        _text(data["dtype"], "tensor.dtype"),
        _text(data["device"], "tensor.device"),
        _text(data["sha256"], "tensor.sha256"),
        requires_grad,
        alias,
    )


def read_snapshot(path: Path) -> ModelSnapshot:
    """Read and validate an observation, rejecting missing/duplicate fields."""
    with path.open(encoding="utf-8") as handle:
        raw: object = json.load(handle, object_pairs_hook=_unique_object)
    data = _object(
        raw,
        {
            "schema_version",
            "run_id",
            "model_name",
            "rank",
            "step",
            "phase",
            "time_ns",
            "tensors",
        },
        "snapshot",
    )
    if _integer(data["schema_version"], "schema_version") != 1:
        raise ValueError("unsupported snapshot schema_version")
    tensors = data["tensors"]
    if not isinstance(tensors, list):
        raise ValueError("snapshot.tensors must be a JSON array")
    return ModelSnapshot(
        1,
        _text(data["run_id"], "run_id"),
        _text(data["model_name"], "model_name"),
        _integer(data["rank"], "rank"),
        _integer(data["step"], "step"),
        _text(data["phase"], "phase"),
        _integer(data["time_ns"], "time_ns"),
        tuple(_read_tensor(tensor) for tensor in tensors),
    )


def compare_model_snapshots(
    reference: ModelSnapshot, candidate: ModelSnapshot
) -> SnapshotComparison:
    """Compare an explicitly selected pair, allowing different phases/steps/ranks.

    Both observations must identify the same run and model. All tensor names,
    registration order, kinds, shapes, dtypes, contents and object aliases are
    compared. Physical devices and observation times are provenance, not equality
    conditions. A matching SHA-256 is fingerprint evidence, not a proof of equal
    values. Missing tensors are differences, not a reduced comparison scope.
    """
    if not isinstance(reference, ModelSnapshot) or not isinstance(
        candidate, ModelSnapshot
    ):
        raise TypeError("comparison requires ModelSnapshot values")
    if (reference.run_id, reference.model_name) != (
        candidate.run_id,
        candidate.model_name,
    ):
        raise ValueError("snapshots must identify the same run and model")
    left = {tensor.name: tensor for tensor in reference.tensors}
    right = {tensor.name: tensor for tensor in candidate.tensors}
    differences: list[TensorDifference] = []
    if tuple(left) != tuple(right):
        differences.append(TensorDifference("<model>", ("tensor_names_or_order",)))
    compared = 0
    for name, tensor in left.items():
        if name not in right:
            differences.append(TensorDifference(name, ("missing",)))
            continue
        other = right[name]
        compared += 1
        fields: list[str] = []
        for field in (
            "kind",
            "shape",
            "dtype",
            "sha256",
            "requires_grad",
            "object_alias_of",
        ):
            if getattr(tensor, field) != getattr(other, field):
                fields.append(field)
        if fields:
            differences.append(TensorDifference(name, tuple(fields)))
    for name in right:
        if name not in left:
            differences.append(TensorDifference(name, ("unexpected",)))
    return SnapshotComparison(
        reference.rank, candidate.rank, compared, tuple(differences)
    )


def compare_ranks(
    snapshots: Sequence[ModelSnapshot],
    *,
    expected_ranks: Sequence[int],
    reference_rank: int = 0,
) -> RankComparison:
    """Compare complete rank coverage at one phase/step; missing ranks are errors."""
    ranks = tuple(expected_ranks)
    if len(ranks) < 2 or len(set(ranks)) != len(ranks):
        raise ValueError("expected_ranks must contain at least two distinct ranks")
    for rank in ranks:
        _integer(rank, "expected rank")
    _integer(reference_rank, "reference_rank")
    if reference_rank not in ranks:
        raise ValueError("reference_rank must be in expected_ranks")
    by_rank: dict[int, ModelSnapshot] = {}
    for snapshot in snapshots:
        if not isinstance(snapshot, ModelSnapshot):
            raise TypeError("snapshots must contain ModelSnapshot values")
        if snapshot.rank in by_rank:
            raise ValueError(f"duplicate snapshot rank: {snapshot.rank}")
        by_rank[snapshot.rank] = snapshot
    if set(by_rank) != set(ranks):
        raise ValueError(
            f"rank coverage mismatch: expected {ranks}, observed {tuple(by_rank)}"
        )
    reference = by_rank[reference_rank]
    comparisons: list[SnapshotComparison] = []
    for rank in ranks:
        snapshot = by_rank[rank]
        if (snapshot.phase, snapshot.step) != (reference.phase, reference.step):
            raise ValueError("rank snapshots must share phase and step")
        if rank != reference_rank:
            comparisons.append(compare_model_snapshots(reference, snapshot))
    return RankComparison(reference_rank, ranks, tuple(comparisons))
