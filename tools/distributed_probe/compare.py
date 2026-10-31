"""Exact, complete comparison of trusted probe checkpoint artifacts.

The weights-only loader maps storage to CPU. Tensor comparison preserves dtype,
shape and every stored value bit (including signed zero/NaN payloads); device
placement and storage aliasing are not reconstructed by this artifact comparison.
Dictionary order and key types, sequence types, and primitive types also matter.
No checkpoint field is excluded. Missing artifacts are errors, not equality.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypedDict

import torch


@dataclass
class Difference:
    path: str
    reason: str


@dataclass
class Comparison:
    tensor_pairs_compared: int = 0
    scalar_pairs_compared: int = 0
    container_pairs_compared: int = 0
    differences: list[Difference] = field(default_factory=list)


class FileComparison(TypedDict):
    left: str
    right: str
    left_sha256: str
    right_sha256: str
    equal: bool
    tensor_pairs_compared: int
    scalar_pairs_compared: int
    container_pairs_compared: int
    differences: list[dict[str, str]]


class PathComparison(TypedDict):
    equal: bool
    pairs: list[FileComparison]
    scope: str


class Options(argparse.Namespace):
    left: Path
    right: Path
    output: Path


def validate_state(value: object, path: str = "$") -> None:
    """Reject unsupported state rather than coercing it for a comparison."""
    if isinstance(value, torch.Tensor):
        if value.layout != torch.strided or value.is_quantized:
            raise TypeError(f"{path}: only dense nonquantized tensors are supported")
        return
    if value is None or type(value) in (bool, int, float, str):
        return
    if isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            validate_state(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if type(key) not in (int, str):
                raise TypeError(f"{path}: unsupported mapping key type")
            validate_state(item, f"{path}[{key!r}]")
        return
    raise TypeError(f"{path}: unsupported state type {type(value).__name__}")


def tensor_bytes(value: torch.Tensor) -> bytes:
    return (
        value.detach()
        .cpu()
        .contiguous()
        .reshape(-1)
        .view(torch.uint8)
        .numpy()
        .tobytes()
    )


def compare_values(
    left: object, right: object, result: Comparison, path: str = "$"
) -> None:
    """Compare values after boundary validation; collect differences without truncation."""
    if type(left) is not type(right):
        result.differences.append(Difference(path, "types differ"))
    elif isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        result.tensor_pairs_compared += 1
        if left.dtype != right.dtype or left.shape != right.shape:
            result.differences.append(Difference(path, "tensor shape/dtype differ"))
        elif tensor_bytes(left) != tensor_bytes(right):
            result.differences.append(Difference(path, "tensor value bits differ"))
    elif isinstance(left, dict) and isinstance(right, dict):
        result.container_pairs_compared += 1
        left_keys = [(type(key), key) for key in left]
        right_keys = [(type(key), key) for key in right]
        if left_keys != right_keys:
            result.differences.append(
                Difference(path, "mapping keys/order/types differ")
            )
        for key in left:
            if key in right and (type(key), key) in right_keys:
                compare_values(left[key], right[key], result, f"{path}[{key!r}]")
    elif isinstance(left, (tuple, list)) and isinstance(right, (tuple, list)):
        result.container_pairs_compared += 1
        if len(left) != len(right):
            result.differences.append(Difference(path, "sequence lengths differ"))
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            compare_values(left_item, right_item, result, f"{path}[{index}]")
    else:
        result.scalar_pairs_compared += 1
        if isinstance(left, float) and isinstance(right, float):
            equal = struct.pack("!d", left) == struct.pack("!d", right)
        else:
            equal = left == right
        if not equal:
            result.differences.append(Difference(path, "scalar value differs"))


def compare_files(left: Path, right: Path) -> FileComparison:
    left_value: object = torch.load(left, map_location="cpu", weights_only=True)
    right_value: object = torch.load(right, map_location="cpu", weights_only=True)
    validate_state(left_value)
    validate_state(right_value)
    result = Comparison()
    compare_values(left_value, right_value, result)
    return {
        "left": str(left.resolve()),
        "right": str(right.resolve()),
        "left_sha256": hashlib.sha256(left.read_bytes()).hexdigest(),
        "right_sha256": hashlib.sha256(right.read_bytes()).hexdigest(),
        "equal": not result.differences,
        "tensor_pairs_compared": result.tensor_pairs_compared,
        "scalar_pairs_compared": result.scalar_pairs_compared,
        "container_pairs_compared": result.container_pairs_compared,
        "differences": [
            {"path": item.path, "reason": item.reason} for item in result.differences
        ],
    }


def checkpoint_files(path: Path) -> dict[str, Path]:
    """Accept a checkpoint, worker directory, or phase/payload with rank directories."""
    if path.is_file():
        return {"checkpoint.pt": path}
    if not path.is_dir():
        raise FileNotFoundError(path)
    if (path / "checkpoint.pt").is_file():
        return {"checkpoint.pt": path / "checkpoint.pt"}
    direct = list(path.glob("rank*/checkpoint.pt"))
    nested = list((path / "payload").glob("rank*/checkpoint.pt"))
    if direct and nested:
        raise ValueError("Both direct and payload checkpoint sets exist")
    paths = direct or nested
    if not paths:
        raise ValueError(f"No rank checkpoints found under {path}")
    return {item.parent.name: item for item in sorted(paths)}


def compare_paths(left: Path, right: Path) -> PathComparison:
    left_files, right_files = checkpoint_files(left), checkpoint_files(right)
    if left_files.keys() != right_files.keys():
        raise ValueError("Checkpoint rank/artifact sets differ")
    comparisons = [
        compare_files(left_files[key], right_files[key]) for key in left_files
    ]
    return {
        "equal": all(item["equal"] is True for item in comparisons),
        "pairs": comparisons,
        "scope": "complete stored checkpoint; CPU mapped; storage aliasing not checked",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(namespace=Options())
    report = compare_paths(args.left, args.right)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["equal"] is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
