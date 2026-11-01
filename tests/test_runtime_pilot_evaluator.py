"""Operator-only runtime evidence tests; do not export to solver packets."""

from __future__ import annotations

import shutil
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path

import pytest
import torch

from tools.runtime_pilot.evaluate import Evaluation, evaluate


def artifact(rank: int, step: int, tp_size: int = 2) -> dict[str, object]:
    return {
        "rank": rank,
        "tp_rank": rank % tp_size,
        "dp_rank": rank // tp_size,
        "step": step,
        "parameters": {
            "norm": torch.tensor([1.0, 2.0]) + step,
            "partition": torch.tensor([float(rank), 3.0]) + step,
        },
        "replicated_names": ["norm"],
        "optimizer": {
            "state": {0: {"momentum": torch.tensor([rank + 0.25, 2.0])}},
            "groups": [{"lr": 0.125, "params": (0,)}],
            "active": True,
            "optional": None,
        },
        "rng": {"cpu": torch.tensor([1, rank + 3, step], dtype=torch.uint8)},
    }


def save(root: Path, rank: int, step: int, state: dict[str, object]) -> Path:
    path = root / f"rank{rank}" / f"step{step}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    return path


def runs(
    root: Path, world_size: int = 2, tp_size: int = 2, steps: tuple[int, ...] = (0, 1)
) -> tuple[Path, Path]:
    candidate, reference = root / "candidate", root / "reference"
    for directory in (candidate, reference):
        for rank in range(world_size):
            for step in steps:
                save(directory, rank, step, artifact(rank, step, tp_size))
    return candidate, reference


def change(
    root: Path,
    update: Callable[[dict[str, object]], None],
    rank: int = 0,
    step: int = 1,
) -> None:
    path = root / f"rank{rank}" / f"step{step}.pt"
    raw: object = torch.load(path, map_location="cpu", weights_only=True)
    assert isinstance(raw, dict)
    state: dict[str, object] = {}
    for key, value in raw.items():
        assert isinstance(key, str)
        state[key] = value
    update(state)
    torch.save(state, path)


def change_field(
    root: Path, name: str, value: object, rank: int = 0, step: int = 1
) -> None:
    def update(state: dict[str, object]) -> None:
        state[name] = value

    change(root, update, rank, step)


def check(
    paths: tuple[Path, Path],
    world_size: int = 2,
    steps: list[int] | None = None,
    atol: float = 0.0,
    rtol: float = 0.0,
) -> Evaluation:
    return evaluate(*paths, world_size, [0, 1] if steps is None else steps, atol, rtol)


def test_complete_partitioned_state_passes(tmp_path: Path) -> None:
    report = check(runs(tmp_path))
    assert report.passed
    assert report.candidate_artifacts == report.reference_artifacts == 4
    assert report.tensor_parallel_size == 2 and report.data_parallel_size == 1
    assert report.replicated_status == "passed" and report.replicated_checks > 0
    assert report.model_comparable_tensors == 8
    assert report.maximum_absolute_model_difference == 0
    assert report.failed_checks == 0 and report.failed_paths == ()


def test_partitioned_parameter_is_still_compared_to_reference(tmp_path: Path) -> None:
    paths = runs(tmp_path)
    change_field(
        paths[0],
        "parameters",
        {"norm": torch.tensor([2.0, 3.0]), "partition": torch.tensor([9.0, 4.0])},
        rank=1,
    )
    report = check(paths)
    assert not report.passed
    assert report.replicated_status == "passed"
    assert report.maximum_absolute_model_difference == 7.0
    assert any("partition" in path for path in report.failed_paths)


def test_replicated_parameter_failure(tmp_path: Path) -> None:
    paths = runs(tmp_path)
    change_field(
        paths[0],
        "parameters",
        {"norm": torch.tensor([2.5, 3.0]), "partition": torch.tensor([2.0, 4.0])},
        rank=1,
    )
    report = check(paths)
    assert not report.passed and report.replicated_status == "violated"
    assert report.maximum_absolute_model_difference == 0.5


@pytest.mark.parametrize(
    "field,value",
    [
        (
            "optimizer",
            {
                "state": {0: {"momentum": torch.tensor([9.0, 2.0])}},
                "groups": [{"lr": 0.125, "params": (0,)}],
                "active": True,
                "optional": None,
            },
        ),
        ("rng", {"cpu": torch.tensor([1, 99, 1], dtype=torch.uint8)}),
    ],
)
def test_nonmodel_state_failure(tmp_path: Path, field: str, value: object) -> None:
    paths = runs(tmp_path)
    change_field(paths[0], field, value)
    report = check(paths, atol=100.0, rtol=100.0)
    assert not report.passed
    assert report.maximum_absolute_model_difference == 0
    assert report.replicated_status == "passed"
    assert all(field in path for path in report.failed_paths)


@pytest.mark.parametrize("remove_rank", [False, True])
def test_missing_rank_or_step_is_error(tmp_path: Path, remove_rank: bool) -> None:
    paths = runs(tmp_path)
    if remove_rank:
        shutil.rmtree(paths[0] / "rank1")
    else:
        (paths[0] / "rank1/step1.pt").unlink()
    with pytest.raises(ValueError, match="coverage"):
        check(paths)


def test_duplicate_rank_step_is_error(tmp_path: Path) -> None:
    paths = runs(tmp_path)
    shutil.copyfile(paths[0] / "rank0/step1.pt", paths[0] / "rank0/step01.pt")
    with pytest.raises(ValueError, match="duplicate rank/step"):
        check(paths)


def test_filename_identity_must_match(tmp_path: Path) -> None:
    paths = runs(tmp_path)
    change_field(paths[0], "rank", 1)
    with pytest.raises(ValueError, match="path disagrees"):
        check(paths)


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("rank", True, TypeError),
        ("step", "1", TypeError),
        ("tp_rank", -1, ValueError),
        ("dp_rank", 0.0, TypeError),
        ("parameters", {}, ValueError),
        ("parameters", {"norm": [1, 2]}, TypeError),
        ("parameters", {"norm": torch.tensor([float("nan")])}, ValueError),
        ("parameters", {"norm": torch.tensor([float("inf")])}, ValueError),
        (
            "parameters",
            {
                "norm": torch.sparse_coo_tensor(
                    torch.tensor([[0]]), torch.tensor([1.0]), (2,)
                )
            },
            TypeError,
        ),
        ("replicated_names", "norm", TypeError),
        ("replicated_names", ["missing"], ValueError),
        ("replicated_names", ["norm", "norm"], ValueError),
        ("optimizer", {True: 1}, TypeError),
        ("optimizer", {"x": float("nan")}, ValueError),
        ("optimizer", OrderedDict([("x", 1)]), TypeError),
        ("rng", {}, ValueError),
        ("rng", {"cpu": torch.tensor([1.0])}, ValueError),
        ("rng", {"cpu": torch.zeros((1, 2), dtype=torch.uint8)}, ValueError),
        ("rng", {"cpu": torch.empty(0, dtype=torch.uint8)}, ValueError),
    ],
)
def test_malformed_state_rejected(
    tmp_path: Path, field: str, value: object, error: type[Exception]
) -> None:
    paths = runs(tmp_path)
    change_field(paths[0], field, value)
    with pytest.raises(error):
        check(paths)


@pytest.mark.parametrize("extra", [False, True])
def test_exact_required_fields(tmp_path: Path, extra: bool) -> None:
    paths = runs(tmp_path)

    def update(state: dict[str, object]) -> None:
        if extra:
            state["unexpected"] = 1
        else:
            del state["optimizer"]

    change(paths[0], update)
    with pytest.raises(ValueError, match="snapshot fields differ"):
        check(paths)


@pytest.mark.parametrize(
    "tensor",
    [torch.tensor([2.0, 3.0], dtype=torch.float64), torch.tensor([[2.0, 3.0]])],
)
def test_shape_dtype_failure_is_not_value_comparison(
    tmp_path: Path, tensor: torch.Tensor
) -> None:
    paths = runs(tmp_path)
    change_field(
        paths[0], "parameters", {"norm": tensor, "partition": torch.tensor([1.0, 4.0])}
    )
    report = check(paths)
    assert not report.passed
    assert report.model_comparable_tensors == 7
    assert any("dtype_shape" in path for path in report.failed_paths)


def test_parameter_tolerances_do_not_change_exact_state_checks(tmp_path: Path) -> None:
    paths = runs(tmp_path)
    change_field(
        paths[0],
        "parameters",
        {"norm": torch.tensor([2.001, 3.0]), "partition": torch.tensor([1.0, 4.0])},
    )
    assert not check(paths).passed
    tolerant = check(paths, atol=0.002)
    assert tolerant.passed and tolerant.replicated_status == "passed"
    assert tolerant.maximum_absolute_model_difference is not None
    assert tolerant.maximum_absolute_model_difference > 0


def test_empty_replicated_coverage_is_error(tmp_path: Path) -> None:
    paths = runs(tmp_path)
    for directory in paths:
        for rank in range(2):
            for step in (0, 1):
                change_field(directory, "replicated_names", [], rank, step)
    with pytest.raises(ValueError, match="Empty replicated"):
        check(paths)


def test_tp1_explicitly_not_applicable_with_multiple_dp_ranks(tmp_path: Path) -> None:
    paths = runs(tmp_path, world_size=2, tp_size=1)
    for directory in paths:
        for rank in range(2):
            for step in (0, 1):
                change_field(directory, "replicated_names", [], rank, step)
    report = check(paths)
    assert report.passed and report.tensor_parallel_size == 1
    assert report.data_parallel_size == 2
    assert (
        report.replicated_status == "not_applicable" and report.replicated_checks == 0
    )


def test_replication_compares_only_within_same_dp_group(tmp_path: Path) -> None:
    paths = runs(tmp_path, world_size=4, tp_size=2)
    for directory in paths:
        for rank in (2, 3):
            for step in (0, 1):
                change_field(
                    directory,
                    "parameters",
                    {
                        "norm": torch.tensor([10.0, 20.0]),
                        "partition": torch.tensor([float(rank), float(step)]),
                    },
                    rank,
                    step,
                )
    assert check(paths, world_size=4).passed


def test_reference_must_satisfy_replication(tmp_path: Path) -> None:
    paths = runs(tmp_path)
    change_field(
        paths[1],
        "parameters",
        {"norm": torch.tensor([99.0, 3.0]), "partition": torch.tensor([2.0, 4.0])},
        rank=1,
    )
    with pytest.raises(ValueError, match="reference violates replication"):
        check(paths)


@pytest.mark.parametrize(
    "optimizer",
    [
        {
            "active": True,
            "groups": [{"lr": 0.125, "params": (0,)}],
            "state": {0: {"momentum": torch.tensor([0.25, 2.0])}},
            "optional": None,
        },
        {
            "state": {0: {"momentum": torch.tensor([0.25, 2.0])}},
            "groups": [{"lr": 0.125, "params": [0]}],
            "active": True,
            "optional": None,
        },
        {
            "state": {"0": {"momentum": torch.tensor([0.25, 2.0])}},
            "groups": [{"lr": 0.125, "params": (0,)}],
            "active": True,
            "optional": None,
        },
        {
            "state": {0: {"momentum": torch.tensor([0.25, 2.0])}},
            "groups": [{"lr": 0.125, "params": (0,)}],
            "active": 1,
            "optional": None,
        },
    ],
)
def test_optimizer_order_key_and_value_types_are_significant(
    tmp_path: Path, optimizer: object
) -> None:
    paths = runs(tmp_path)
    change_field(paths[0], "optimizer", optimizer)
    report = check(paths)
    assert not report.passed
    assert all("optimizer" in path for path in report.failed_paths)


def test_optimizer_signed_zero_tensor_and_scalar_are_exact(tmp_path: Path) -> None:
    paths = runs(tmp_path)
    for directory in paths:
        change_field(directory, "optimizer", [0.0, torch.tensor([0.0])])
    change_field(paths[0], "optimizer", [-0.0, torch.tensor([-0.0])])
    report = check(paths, atol=1.0)
    assert not report.passed and report.failed_checks == 2


def test_cyclic_optimizer_rejected(tmp_path: Path) -> None:
    paths = runs(tmp_path)
    cycle: list[object] = []
    cycle.append(cycle)
    change_field(paths[0], "optimizer", cycle)
    with pytest.raises(ValueError, match="cyclic"):
        check(paths)


def test_model_integer_difference_is_not_rounded_to_zero(tmp_path: Path) -> None:
    paths = runs(tmp_path)
    for directory in paths:
        change_field(
            directory,
            "parameters",
            {
                "norm": torch.tensor([2.0, 3.0]),
                "partition": torch.tensor([2**62], dtype=torch.int64),
            },
        )
    change_field(
        paths[0],
        "parameters",
        {
            "norm": torch.tensor([2.0, 3.0]),
            "partition": torch.tensor([2**62 + 1], dtype=torch.int64),
        },
    )
    report = check(paths, atol=10.0)
    assert not report.passed and report.maximum_absolute_model_difference == 1.0


def test_total_failures_are_not_truncated(tmp_path: Path) -> None:
    paths = runs(tmp_path, steps=(0,))
    for rank in range(2):
        change_field(paths[0], "optimizer", [1] * 120, rank, 0)
        change_field(paths[1], "optimizer", [0] * 120, rank, 0)
    report = check(paths, steps=[0])
    assert report.failed_checks == 240
    assert len(report.failed_paths) == 100 and report.failed_paths_truncated


@pytest.mark.parametrize("steps", [[], [1, 0], [0, 0], [True], [-1]])
def test_invalid_expected_steps(tmp_path: Path, steps: list[int]) -> None:
    paths = runs(tmp_path)
    with pytest.raises((ValueError, TypeError)):
        check(paths, steps=steps)


def test_unexpected_step_and_symlink_are_rejected(tmp_path: Path) -> None:
    paths = runs(tmp_path)
    path = save(paths[0], 0, 9, artifact(0, 9))
    with pytest.raises(ValueError, match="coverage"):
        check(paths)
    path.unlink()
    path = paths[0] / "rank0/step1.pt"
    path.unlink()
    path.symlink_to(paths[1] / "rank0/step1.pt")
    with pytest.raises(ValueError, match="nonsymlink"):
        check(paths)


def test_topology_changes_and_duplicate_coordinates_are_errors(tmp_path: Path) -> None:
    paths = runs(tmp_path)
    change_field(paths[0], "tp_rank", 0, rank=1, step=0)
    with pytest.raises(ValueError, match="topology"):
        check(paths)
    change_field(paths[0], "tp_rank", 1, rank=1, step=0)
    change_field(paths[0], "dp_rank", 1, rank=1, step=1)
    with pytest.raises(ValueError, match="topology"):
        check(paths)


def test_extreme_topology_rejected_before_constructing_rank_grid(
    tmp_path: Path,
) -> None:
    paths = runs(tmp_path)
    change_field(paths[0], "tp_rank", 10**20, rank=1, step=0)
    with pytest.raises(ValueError, match="topology"):
        check(paths)


def test_empty_replicated_tensors_are_not_exercised_checks(tmp_path: Path) -> None:
    paths = runs(tmp_path)
    for directory in paths:
        for rank in range(2):
            for step in (0, 1):
                change_field(
                    directory,
                    "parameters",
                    {"norm": torch.empty(0), "partition": torch.tensor([1.0])},
                    rank,
                    step,
                )
    with pytest.raises(ValueError, match="Empty replicated tensor"):
        check(paths)


def test_finite_values_with_unrepresentable_difference_are_explicit(
    tmp_path: Path,
) -> None:
    paths = runs(tmp_path)
    for directory, value in zip(paths, (1.7e308, -1.7e308)):
        change_field(
            directory,
            "parameters",
            {
                "norm": torch.tensor([2.0, 3.0]),
                "partition": torch.tensor([value], dtype=torch.float64),
            },
        )
    report = check(paths)
    assert not report.passed
    assert report.model_absolute_difference_overflow
    assert report.maximum_absolute_model_difference is None
