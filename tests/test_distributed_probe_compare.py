"""Checkpoint parity must include optimizer/RNG state and exact scalar contracts."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tools.distributed_probe.compare import (
    Comparison,
    checkpoint_files,
    compare_files,
    compare_paths,
    compare_values,
    validate_state,
)


def state() -> dict[str, object]:
    return {
        "model": {"weight": torch.tensor([1.0, 2.0])},
        "optimizer": {"state": {0: {"moment": torch.tensor([0.1, 0.2])}}},
        "rng": torch.tensor([1, 2, 3], dtype=torch.uint8),
        "step": 4,
        "options": (False, ["fp32", None]),
    }


def test_complete_state_parity_and_optimizer_difference(tmp_path: Path) -> None:
    left = tmp_path / "left.pt"
    right = tmp_path / "right.pt"
    torch.save(state(), left)
    torch.save(state(), right)
    report = compare_files(left, right)
    assert report["equal"] is True
    assert report["tensor_pairs_compared"] == 3
    changed = state()
    changed["optimizer"] = {"state": {0: {"moment": torch.tensor([0.1, 0.3])}}}
    torch.save(changed, right)
    report = compare_files(left, right)
    assert report["equal"] is False
    assert report["differences"] == [
        {
            "path": "$['optimizer']['state'][0]['moment']",
            "reason": "tensor value bits differ",
        }
    ]


@pytest.mark.parametrize(
    ("left", "right"),
    [
        (1, True),
        ([1], (1,)),
        ({"a": 1}, {}),
        ({"a": 1, "b": 2}, {"b": 2, "a": 1}),
        (0.0, -0.0),
        (torch.tensor([0.0]), torch.tensor([-0.0])),
        (torch.tensor([1]), torch.tensor([1.0])),
        (torch.tensor([1]), torch.tensor([[1]])),
        ([1, 2], [1]),
    ],
)
def test_meaningful_differences_are_preserved(left: object, right: object) -> None:
    validate_state(left)
    validate_state(right)
    report = Comparison()
    compare_values(left, right, report)
    assert report.differences


def test_identical_nan_payload_is_byte_equal_not_a_finiteness_claim() -> None:
    left = torch.tensor([float("nan")])
    report = Comparison()
    compare_values(left, left.clone(), report)
    assert not report.differences


@pytest.mark.parametrize("value", [{True: 1}, {"value": object()}, {1, 2}])
def test_unsupported_state_is_rejected(value: object) -> None:
    with pytest.raises(TypeError):
        validate_state(value)


def test_phase_payload_discovery_requires_matching_rank_sets(tmp_path: Path) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    for rank in (0, 1):
        for root in (left, right / "payload"):
            path = root / f"rank{rank}" / "checkpoint.pt"
            path.parent.mkdir(parents=True)
            torch.save(state(), path)
    assert compare_paths(left, right)["equal"] is True
    (right / "payload" / "rank1" / "checkpoint.pt").unlink()
    with pytest.raises(ValueError, match="sets differ"):
        compare_paths(left, right)
    with pytest.raises(ValueError, match="No rank checkpoints"):
        checkpoint_files(tmp_path)
    with pytest.raises(FileNotFoundError):
        checkpoint_files(tmp_path / "absent")
