"""Operator-only comparison of independently produced checkpoint run artifacts.

This module never imports a candidate. The operator must produce the reference
before running candidate code, freeze manifests, and run each prefix/resume in a
fresh process. The evaluator checks the recorded evidence; it is not a sandbox
against a candidate that can rewrite the operator's reference or manifests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias, TypedDict

import torch

State: TypeAlias = (
    torch.Tensor
    | None
    | bool
    | int
    | float
    | str
    | list["State"]
    | tuple["State", ...]
    | dict[str, "State"]
)


@dataclass(frozen=True)
class Contract:
    world_size: int
    seed: int
    eval_seed: int
    prefix_steps: int
    suffix_steps: int
    accumulation: int
    learning_rate: float
    cuda_expected: bool
    trajectory_atol: float = 2e-6
    trajectory_rtol: float = 2e-5

    def __post_init__(self) -> None:
        for name in ("world_size", "prefix_steps", "suffix_steps", "accumulation"):
            if integer(getattr(self, name), name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("seed", "eval_seed"):
            if integer(getattr(self, name), name) < 0:
                raise ValueError(f"{name} must be nonnegative")
        if number(self.learning_rate, "learning_rate") <= 0:
            raise ValueError("learning_rate must be positive")
        for name in ("trajectory_atol", "trajectory_rtol"):
            if number(getattr(self, name), name) < 0:
                raise ValueError(f"{name} must be nonnegative")
        if not isinstance(self.cuda_expected, bool):
            raise TypeError("cuda_expected must be a boolean")


@dataclass(frozen=True)
class RunEvidence:
    phase_directory: Path
    source_directory: Path
    source_sha256: dict[str, str]
    artifact_sha256: dict[str, str]


class Comparison(TypedDict):
    path: str
    kind: Literal["tensor", "scalar", "mapping_keys", "sequence_type", "length"]
    passed: bool
    reference: str
    candidate: str
    reference_sha256: str | None
    candidate_sha256: str | None
    maximum_absolute_error: float | None


class RankResult(TypedDict):
    rank: int
    prefix_passed: bool
    restore_passed: bool
    trajectory_passed: bool
    prefix: list[Comparison]
    restore: list[Comparison]
    terminal: list[Comparison]
    curves: list[Comparison]


class Error(TypedDict):
    type: str
    message: str


class Evaluation(TypedDict):
    status: Literal["passed", "violated", "error"]
    contract_passed: bool | None
    prefix_passed: bool | None
    restore_passed: bool | None
    trajectory_passed: bool | None
    ranks: list[RankResult] | None
    source_and_artifact_manifests_verified: bool | None
    error: Error | None
    trajectory_atol: float
    trajectory_rtol: float
    limitations: list[str]


def record(value: object, context: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"{context} must be a mapping")
    result: dict[str, object] = {}
    for key, child in value.items():
        if not isinstance(key, str):
            raise TypeError(f"{context} must have string keys")
        result[key] = child
    return result


def integer(value: object, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{context} must be an integer")
    return value


def number(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{context} must be numeric")
    if not math.isfinite(value):
        raise ValueError(f"{context} must be finite")
    return float(value)


def string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{context} must be a nonempty string")
    return value


def array(value: object, context: str) -> list[object]:
    if not isinstance(value, list):
        raise TypeError(f"{context} must be an array")
    return list(value)


def require(data: dict[str, object], key: str, context: str) -> object:
    if key not in data:
        raise ValueError(f"{context} missing required field {key}")
    return data[key]


def read_json(path: Path) -> dict[str, object]:
    value: object = json.loads(path.read_text())
    return record(value, str(path))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_manifest(value: object, context: str) -> dict[str, str]:
    mapping = record(value, context)
    if not mapping:
        raise ValueError(f"{context} cannot be empty")
    result: dict[str, str] = {}
    for relative, raw in mapping.items():
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or str(path) != relative:
            raise ValueError(f"{context} contains an invalid relative path")
        digest = string(raw, context)
        if re.fullmatch("[0-9a-f]{64}", digest) is None:
            raise ValueError(f"{context} contains an invalid SHA-256")
        result[relative] = digest
    return result


def verify_files(root: Path, manifest: dict[str, str]) -> None:
    resolved_root = root.resolve(strict=True)
    for relative, expected in manifest.items():
        path = root / relative
        resolved = path.resolve(strict=True)
        if path.is_symlink() or not resolved.is_relative_to(resolved_root):
            raise ValueError(f"Evidence path escapes root: {path}")
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Evidence hash mismatch: {path}")


def validate_run(run: RunEvidence, contract: Contract, stage: str) -> None:
    source = validate_manifest(run.source_sha256, "source_sha256")
    artifacts = validate_manifest(run.artifact_sha256, "artifact_sha256")
    verify_files(run.source_directory, source)
    actual_sources = {
        str(path.relative_to(run.source_directory))
        for path in run.source_directory.rglob("*.py")
        if path.is_file()
    }
    manifest_sources = {name for name in source if name.endswith(".py")}
    if actual_sources != manifest_sources:
        raise ValueError("Source manifest does not cover the exact Python source set")
    required = {"phase-result.json", "invocation.json"}
    for rank in range(contract.world_size):
        required.update(
            {
                f"payload/rank{rank}/metrics.json",
                f"payload/rank{rank}/terminal_state.pt",
            }
        )
        if stage == "resume":
            required.add(f"payload/rank{rank}/restored_state.pt")
    if not required.issubset(artifacts):
        raise ValueError("Artifact manifest lacks a required phase/rank artifact")
    verify_files(run.phase_directory, artifacts)
    phase = read_json(run.phase_directory / "phase-result.json")
    if (
        integer(require(phase, "exit_code", "phase"), "exit_code") != 0
        or require(phase, "stop_reason", "phase") is not None
        or require(phase, "source_unchanged", "phase") is not True
    ):
        raise ValueError("Training phase did not complete successfully")
    invocation = read_json(run.phase_directory / "invocation.json")
    invocation_sources = validate_manifest(
        require(invocation, "source_sha256", "invocation"), "invocation source_sha256"
    )
    expected_sources = {name: source[name] for name in manifest_sources}
    if invocation_sources != expected_sources:
        raise ValueError("Run invocation does not match frozen Python source manifest")
    rank_directories = {
        path.name
        for path in (run.phase_directory / "payload").iterdir()
        if path.is_dir() and path.name.startswith("rank")
    }
    if rank_directories != {f"rank{rank}" for rank in range(contract.world_size)}:
        raise ValueError("Incomplete or unexpected rank directories")


def snapshot(value: object, path: str) -> State:
    if isinstance(value, torch.Tensor):
        if value.layout != torch.strided or value.is_quantized:
            raise TypeError(f"{path}: only dense nonquantized tensors are supported")
        if (value.is_floating_point() or value.is_complex()) and not bool(
            torch.isfinite(value).all()
        ):
            raise ValueError(f"{path}: nonfinite tensor")
        return value.detach().cpu().clone()
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path}: nonfinite scalar")
        return value
    if isinstance(value, list):
        return [snapshot(item, f"{path}[{i}]") for i, item in enumerate(value)]
    if isinstance(value, tuple):
        return tuple(snapshot(item, f"{path}[{i}]") for i, item in enumerate(value))
    if isinstance(value, dict):
        return {
            key: snapshot(item, f"{path}[{key!r}]")
            for key, item in record(value, path).items()
        }
    raise TypeError(f"{path}: unsupported value {type(value).__name__}")


def tensor_digest(value: torch.Tensor) -> str:
    raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    return hashlib.sha256(raw.numpy().tobytes()).hexdigest()


def comparison(
    path: str,
    kind: Literal["scalar", "mapping_keys", "sequence_type", "length"],
    left: object,
    right: object,
    passed: bool,
) -> Comparison:
    return {
        "path": path,
        "kind": kind,
        "passed": passed,
        "reference": repr(left),
        "candidate": repr(right),
        "reference_sha256": None,
        "candidate_sha256": None,
        "maximum_absolute_error": None,
    }


def compare_states(
    left: State,
    right: State,
    path: str,
    *,
    exact: bool,
    atol: float,
    rtol: float,
) -> list[Comparison]:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        metadata_equal = left.shape == right.shape and left.dtype == right.dtype
        delta: float | None = None
        if metadata_equal and (left.is_floating_point() or left.is_complex()):
            delta = float((left - right).abs().max().item()) if left.numel() else 0.0
            passed = (
                tensor_digest(left) == tensor_digest(right)
                if exact
                else bool(torch.allclose(left, right, atol=atol, rtol=rtol))
            )
        else:
            passed = metadata_equal and bool(torch.equal(left, right))
        return [
            {
                "path": path,
                "kind": "tensor",
                "passed": passed,
                "reference": f"shape={tuple(left.shape)},dtype={left.dtype}",
                "candidate": f"shape={tuple(right.shape)},dtype={right.dtype}",
                "reference_sha256": tensor_digest(left),
                "candidate_sha256": tensor_digest(right),
                "maximum_absolute_error": delta,
            }
        ]
    if isinstance(left, dict) and isinstance(right, dict):
        # Names bind state to logical parameters. Dictionary insertion order is
        # retained as evidence but is not itself parameter identity; ordered
        # parameter-group lists are compared separately, without reordering.
        reports = [
            comparison(
                path + ".keys",
                "mapping_keys",
                list(left),
                list(right),
                set(left) == set(right),
            )
        ]
        for key in left:
            if key in right:
                reports.extend(
                    compare_states(
                        left[key],
                        right[key],
                        f"{path}[{key!r}]",
                        exact=exact,
                        atol=atol,
                        rtol=rtol,
                    )
                )
        return reports
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        reports = [
            comparison(
                path + ".type",
                "sequence_type",
                type(left).__name__,
                type(right).__name__,
                type(left) is type(right),
            ),
            comparison(
                path + ".length",
                "length",
                len(left),
                len(right),
                len(left) == len(right),
            ),
        ]
        for index, (a, b) in enumerate(zip(left, right)):
            reports.extend(
                compare_states(
                    a, b, f"{path}[{index}]", exact=exact, atol=atol, rtol=rtol
                )
            )
        return reports
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return [
            comparison(
                path + ".type",
                "scalar",
                type(left).__name__,
                type(right).__name__,
                False,
            )
        ]
    passed = type(left) is type(right) and left == right
    if isinstance(left, float) and isinstance(right, float):
        passed = struct.pack("!d", left) == struct.pack("!d", right)
    return [comparison(path, "scalar", left, right, passed is True)]


def load_state(
    path: Path, contract: Contract, rank: int, completed_steps: int
) -> State:
    # These are operator-owned workload outputs, not arbitrary downloaded pickle
    # files. weights_only also rejects custom Python objects before validation.
    raw: object = torch.load(path, map_location="cpu", weights_only=True)
    data = record(raw, str(path))
    expected_fields = {
        "model",
        "optimizer_state",
        "optimizer_groups",
        "progress",
        "rng",
        "rank",
        "world_size",
    }
    if set(data) != expected_fields:
        raise ValueError("Unexpected or missing state fields")
    for key, expected in (("rank", rank), ("world_size", contract.world_size)):
        if integer(data[key], key) != expected:
            raise ValueError(f"Unexpected state {key}")
    progress = record(data["progress"], "progress")
    if set(progress) != {"completed_steps", "data_cursor", "seed"}:
        raise ValueError("Unexpected progress schema")
    for key, expected in (
        ("completed_steps", completed_steps),
        ("seed", contract.seed),
    ):
        if integer(progress[key], key) != expected:
            raise ValueError(f"Unexpected progress {key}")
    if integer(progress["data_cursor"], "data_cursor") < 0:
        raise ValueError("Negative data cursor")
    model = record(data["model"], "model")
    if not model or not all(
        isinstance(value, torch.Tensor) for value in model.values()
    ):
        raise TypeError("Model state must contain named tensors")
    optimizer = record(data["optimizer_state"], "optimizer_state")
    if not optimizer:
        raise ValueError("Missing optimizer state")
    for name, value in optimizer.items():
        if name not in model or not record(value, "parameter optimizer state"):
            raise ValueError("Optimizer state has unknown or empty parameter state")
    names: list[str] = []
    for raw_group in array(data["optimizer_groups"], "optimizer_groups"):
        group = record(raw_group, "optimizer group")
        names.extend(
            string(name, "parameter name")
            for name in array(
                require(group, "params", "optimizer group"), "group params"
            )
        )
    if not names or len(set(names)) != len(names) or set(names) != set(optimizer):
        raise ValueError(
            "Optimizer groups do not bind all states to unique parameter names"
        )
    rng = record(data["rng"], "rng")
    if set(rng) != {"cpu", "cuda"}:
        raise ValueError("Unexpected RNG schema")
    for name in ("cpu", "cuda"):
        value = rng[name]
        if name == "cuda" and not contract.cuda_expected:
            if value is not None:
                raise ValueError("Unexpected CUDA RNG in CPU scenario")
            continue
        if (
            not isinstance(value, torch.Tensor)
            or value.dtype != torch.uint8
            or value.ndim != 1
            or value.numel() == 0
        ):
            raise TypeError(
                f"{name} RNG must be a nonempty one-dimensional byte tensor"
            )
    return snapshot(data, str(path))


def curves(path: Path, contract: Contract, rank: int, stage: str) -> State:
    data = read_json(path)
    start = 0 if stage == "prefix" else contract.prefix_steps
    count = contract.prefix_steps if stage == "prefix" else contract.suffix_steps
    expected: dict[str, object] = {
        "stage": stage,
        "rank": rank,
        "world_size": contract.world_size,
        "seed": contract.seed,
        "eval_seed": contract.eval_seed,
        "accumulation": contract.accumulation,
        "examples_per_rank_update": 8,
        "tokens_per_global_update": contract.world_size * 64,
        "requested_updates": count,
        "completed_steps": start + count,
        "device": f"cuda:{rank}" if contract.cuda_expected else "cpu",
    }
    for key, value in expected.items():
        actual = require(data, key, "metrics")
        if type(actual) is not type(value) or actual != value:
            raise ValueError(f"Unexpected metrics {key}")
    string(require(data, "backend", "metrics"), "backend")
    if number(require(data, "lr", "metrics"), "lr") != contract.learning_rate:
        raise ValueError("Unexpected learning rate")
    parameter_count = integer(
        require(data, "parameter_count", "metrics"), "parameter_count"
    )
    if parameter_count <= 0:
        raise ValueError("Invalid parameter count")
    torch_version = string(require(data, "torch_version", "metrics"), "torch_version")
    if (
        number(
            require(data, "training_seconds_including_validation", "metrics"),
            "training seconds",
        )
        < 0
    ):
        raise ValueError("Negative training time")
    training = array(require(data, "steps", "metrics"), "steps")
    if len(training) != count:
        raise ValueError("Incomplete training curve")
    for offset, item in enumerate(training, 1):
        row = record(item, "training row")
        if (
            set(row) != {"step", "train_token_loss"}
            or integer(row["step"], "step") != start + offset
        ):
            raise ValueError("Unexpected training curve step/schema")
        number(row["train_token_loss"], "training loss")
    validation = array(require(data, "validation_curve", "metrics"), "validation_curve")
    expected_steps = [start] + [
        step
        for step in range(start + 1, start + count + 1)
        if step % 4 == 0 or step == start + count
    ]
    if len(validation) != len(expected_steps):
        raise ValueError("Incomplete validation curve")
    for item, step in zip(validation, expected_steps):
        row = record(item, "validation row")
        if (
            set(row) != {"step", "validation_loss"}
            or integer(row["step"], "step") != step
        ):
            raise ValueError("Unexpected validation curve step/schema")
        number(row["validation_loss"], "validation loss")
    # Convert only measured numerical curve values to tensors so declared
    # trajectory tolerance applies. Keep step positions as exact integer lists.
    return {
        "parameter_count": parameter_count,
        "torch_version": torch_version,
        "training_steps": [
            integer(record(row, "row")["step"], "step") for row in training
        ],
        "training_losses": torch.tensor(
            [
                number(record(row, "row")["train_token_loss"], "loss")
                for row in training
            ],
            dtype=torch.float64,
        ),
        "validation_steps": [
            integer(record(row, "row")["step"], "step") for row in validation
        ],
        "validation_losses": torch.tensor(
            [
                number(record(row, "row")["validation_loss"], "loss")
                for row in validation
            ],
            dtype=torch.float64,
        ),
    }


def _evaluate_pair(
    reference_prefix: RunEvidence,
    reference_resume: RunEvidence,
    candidate_prefix: RunEvidence,
    candidate_resume: RunEvidence,
    contract: Contract,
) -> list[RankResult]:
    runs = (
        (reference_prefix, "prefix"),
        (reference_resume, "resume"),
        (candidate_prefix, "prefix"),
        (candidate_resume, "resume"),
    )
    for run, stage in runs:
        validate_run(run, contract, stage)
    results: list[RankResult] = []
    for rank in range(contract.world_size):
        directories = [
            run.phase_directory / "payload" / f"rank{rank}" for run, _ in runs
        ]
        reference_start = load_state(
            directories[0] / "terminal_state.pt", contract, rank, contract.prefix_steps
        )
        reference_restored = load_state(
            directories[1] / "restored_state.pt", contract, rank, contract.prefix_steps
        )
        reference_end = load_state(
            directories[1] / "terminal_state.pt",
            contract,
            rank,
            contract.prefix_steps + contract.suffix_steps,
        )
        candidate_start = load_state(
            directories[2] / "terminal_state.pt", contract, rank, contract.prefix_steps
        )
        candidate_restored = load_state(
            directories[3] / "restored_state.pt", contract, rank, contract.prefix_steps
        )
        candidate_end = load_state(
            directories[3] / "terminal_state.pt",
            contract,
            rank,
            contract.prefix_steps + contract.suffix_steps,
        )
        options = {"atol": contract.trajectory_atol, "rtol": contract.trajectory_rtol}
        baseline = compare_states(
            reference_start,
            reference_restored,
            "reference_restore",
            exact=True,
            **options,
        )
        if not all(row["passed"] for row in baseline):
            raise ValueError(
                "Reference restoration does not match its own checkpoint state"
            )
        prefix = compare_states(
            reference_start, candidate_start, "prefix", exact=True, **options
        )
        restored = compare_states(
            reference_start, candidate_restored, "restore", exact=True, **options
        )
        terminal = compare_states(
            reference_end, candidate_end, "terminal", exact=False, **options
        )
        curve_values = [
            curves(directory / "metrics.json", contract, rank, stage)
            for directory, (_, stage) in zip(directories, runs)
        ]
        prefix.extend(
            compare_states(
                curve_values[0], curve_values[2], "prefix_curves", exact=True, **options
            )
        )
        trajectory = compare_states(
            curve_values[1], curve_values[3], "resume_curves", exact=False, **options
        )
        results.append(
            {
                "rank": rank,
                "prefix_passed": all(row["passed"] for row in prefix),
                "restore_passed": all(row["passed"] for row in restored),
                "trajectory_passed": all(
                    row["passed"] for row in terminal + trajectory
                ),
                "prefix": prefix,
                "restore": restored,
                "terminal": terminal,
                "curves": trajectory,
            }
        )
    # Recheck after all reads so changes during evaluation cannot be accepted.
    for run, stage in runs:
        validate_run(run, contract, stage)
    return results


def evaluate_pair(
    reference_prefix: RunEvidence,
    reference_resume: RunEvidence,
    candidate_prefix: RunEvidence,
    candidate_resume: RunEvidence,
    contract: Contract,
) -> Evaluation:
    limits = [
        "Trusted operator manifests and independently generated reference are required; this is not an adversarial execution sandbox",
        "Exact restore compares all saved values and logical parameter identity, not storage aliasing or original device placement",
        "Named mapping insertion order is recorded but nonsemantic; parameter-group and all sequence orders are semantic",
        "Finite post-resume tensors and curves use declared tolerances; RNG, scalar state and progress remain exact",
        "Passing finite scenarios does not certify arbitrary collective behavior, long-run training, or untested checkpoint formats",
    ]
    try:
        ranks = _evaluate_pair(
            reference_prefix,
            reference_resume,
            candidate_prefix,
            candidate_resume,
            contract,
        )
    except Exception as error:
        # IO, deserialization, and validation failures are explicit unknown
        # outcomes. The caller must inspect status/contract_passed and CLI exits2.
        return {
            "status": "error",
            "contract_passed": None,
            "prefix_passed": None,
            "restore_passed": None,
            "trajectory_passed": None,
            "ranks": None,
            "source_and_artifact_manifests_verified": None,
            "error": {"type": type(error).__name__, "message": str(error)},
            "trajectory_atol": contract.trajectory_atol,
            "trajectory_rtol": contract.trajectory_rtol,
            "limitations": limits,
        }
    prefix = all(row["prefix_passed"] for row in ranks)
    restored = all(row["restore_passed"] for row in ranks)
    trajectory = all(row["trajectory_passed"] for row in ranks)
    passed = prefix and restored and trajectory
    return {
        "status": "passed" if passed else "violated",
        "contract_passed": passed,
        "prefix_passed": prefix,
        "restore_passed": restored,
        "trajectory_passed": trajectory,
        "ranks": ranks,
        "source_and_artifact_manifests_verified": True,
        "error": None,
        "trajectory_atol": contract.trajectory_atol,
        "trajectory_rtol": contract.trajectory_rtol,
        "limitations": limits,
    }


def run_from_manifest(path: Path) -> Evaluation:
    manifest = read_json(path)
    if integer(require(manifest, "schema_version", "manifest"), "schema_version") != 1:
        raise ValueError("Unsupported manifest schema")
    raw = record(require(manifest, "contract", "manifest"), "contract")
    cuda = require(raw, "cuda_expected", "contract")
    if not isinstance(cuda, bool):
        raise TypeError("cuda_expected must be boolean")
    contract = Contract(
        world_size=integer(require(raw, "world_size", "contract"), "world_size"),
        seed=integer(require(raw, "seed", "contract"), "seed"),
        eval_seed=integer(require(raw, "eval_seed", "contract"), "eval_seed"),
        prefix_steps=integer(require(raw, "prefix_steps", "contract"), "prefix_steps"),
        suffix_steps=integer(require(raw, "suffix_steps", "contract"), "suffix_steps"),
        accumulation=integer(require(raw, "accumulation", "contract"), "accumulation"),
        learning_rate=number(
            require(raw, "learning_rate", "contract"), "learning_rate"
        ),
        cuda_expected=cuda,
        trajectory_atol=number(
            require(raw, "trajectory_atol", "contract"), "trajectory_atol"
        ),
        trajectory_rtol=number(
            require(raw, "trajectory_rtol", "contract"), "trajectory_rtol"
        ),
    )
    runs: list[RunEvidence] = []
    for name in (
        "reference_prefix",
        "reference_resume",
        "candidate_prefix",
        "candidate_resume",
    ):
        run = record(require(manifest, name, "manifest"), name)
        runs.append(
            RunEvidence(
                Path(string(require(run, "phase_directory", name), "phase_directory")),
                Path(
                    string(require(run, "source_directory", name), "source_directory")
                ),
                validate_manifest(require(run, "source_sha256", name), "source_sha256"),
                validate_manifest(
                    require(run, "artifact_sha256", name), "artifact_sha256"
                ),
            )
        )
    return evaluate_pair(runs[0], runs[1], runs[2], runs[3], contract)


class Options(argparse.Namespace):
    manifest: Path
    output: Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    options = parser.parse_args(namespace=Options())
    try:
        result: object = run_from_manifest(options.manifest)
    except Exception as error:
        # A malformed operator manifest has no evaluable contract. Preserve an
        # explicit error record rather than inventing defaults for its fields.
        result = {
            "status": "error",
            "contract_passed": None,
            "error": {"type": type(error).__name__, "message": str(error)},
        }
    options.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    status = record(result, "evaluation")["status"]
    return 0 if status == "passed" else 1 if status == "violated" else 2


if __name__ == "__main__":
    raise SystemExit(main())
