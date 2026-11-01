"""Private workload qualification; never included in solver packets."""

from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path
from types import FunctionType

import pytest
import torch

from tools.checkpoint_pilot.task import current, reference, state
from tools.checkpoint_pilot.task.engine import Options, parse_options, run, run_update
from tools.checkpoint_pilot.task.model import (
    LAYERS,
    Transformer,
    evaluate,
    make_batch,
    microbatches,
)

CPU = torch.device("cpu")


def test_layers_start_with_distinct_matrix_weights_and_identical_shapes() -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(73)
        model = Transformer()
        layers = [
            state.tensors(layer.state_dict(), "layer") for layer in model.encoder.layers
        ]
        assert len(layers) == LAYERS
        matrix_names = [name for name, value in layers[0].items() if value.ndim > 1]
        assert matrix_names
        for index, layer in enumerate(layers):
            assert list(layer) == list(layers[0])
            for name, value in layer.items():
                assert value.shape == layers[0][name].shape
                assert value.dtype == layers[0][name].dtype
            for other in layers[:index]:
                assert all(
                    not torch.equal(layer[name], other[name]) for name in matrix_names
                )
        torch.manual_seed(73)
        repeated = Transformer()
        assert equal(model.state_dict(), repeated.state_dict())


def initialized() -> tuple[Transformer, torch.optim.SGD, state.Progress]:
    torch.set_num_threads(1)
    torch.manual_seed(43)
    model = Transformer().train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.15, momentum=0.9)
    for step in range(1, 4):
        run_update(model, optimizer, microbatches(make_batch(43, step, 0, CPU), 2))
    return model, optimizer, state.Progress(3, 3, 43)


def fresh() -> tuple[Transformer, torch.optim.SGD]:
    torch.manual_seed(877)
    model = Transformer().train()
    return model, torch.optim.SGD(model.parameters(), lr=0.15, momentum=0.9)


def equal(left: object, right: object) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return left.dtype == right.dtype and torch.equal(left, right)
    if isinstance(left, dict) and isinstance(right, dict):
        return list(left) == list(right) and all(
            equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (tuple, list)) and isinstance(right, (tuple, list)):
        return (
            type(left) is type(right)
            and len(left) == len(right)
            and all(equal(a, b) for a, b in zip(left, right, strict=True))
        )
    return type(left) is type(right) and left == right


def sharded_checkpoint(directory: Path, world_size: int) -> state.Artifact:
    model, optimizer, progress = initialized()
    directory.mkdir()
    result = state.artifact(model, optimizer, progress, 0, world_size, CPU)
    for rank in range(world_size):
        if rank:
            torch.manual_seed(100 + rank)
        current.save_checkpoint(
            directory, model, optimizer, progress, rank, world_size, CPU
        )
    return result


@pytest.mark.parametrize("backend", [reference, current])
def test_single_rank_checkpoint_round_trip_preserves_complete_state(
    tmp_path: Path, backend: object
) -> None:
    model, optimizer, progress = initialized()
    directory = tmp_path / "checkpoint"
    directory.mkdir()
    before = state.artifact(model, optimizer, progress, 0, 1, CPU)
    if backend is reference:
        reference.save_checkpoint(directory, model, optimizer, progress, 0, 1, CPU)
    else:
        assert backend is current
        current.save_checkpoint(directory, model, optimizer, progress, 0, 1, CPU)
    resumed, resumed_optimizer = fresh()
    if backend is reference:
        restored = reference.restore_checkpoint(
            directory, resumed, resumed_optimizer, 0, 1, CPU
        )
    else:
        restored = current.restore_checkpoint(
            directory, resumed, resumed_optimizer, 0, 1, CPU
        )
    after = state.artifact(resumed, resumed_optimizer, restored, 0, 1, CPU)
    assert equal(before, after)


def test_two_rank_shards_preserve_other_state_but_permute_layer_momentum(
    tmp_path: Path,
) -> None:
    before = sharded_checkpoint(tmp_path / "checkpoint", 2)
    model, optimizer = fresh()
    progress = current.restore_checkpoint(
        tmp_path / "checkpoint", model, optimizer, 0, 2, CPU
    )
    after = state.artifact(model, optimizer, progress, 0, 2, CPU)
    preserved: list[tuple[object, object]] = [
        (before["model"], after["model"]),
        (before["optimizer_groups"], after["optimizer_groups"]),
        (before["progress"], after["progress"]),
        (before["rng"], after["rng"]),
        (before["rank"], after["rank"]),
        (before["world_size"], after["world_size"]),
    ]
    assert all(equal(left, right) for left, right in preserved)
    states_before = before["optimizer_state"]
    states_after = after["optimizer_state"]
    for name in states_before:
        source_name = name.replace("encoder.layers.1.", "encoder.layers.2.")
        if name.startswith("encoder.layers.2."):
            source_name = name.replace("encoder.layers.2.", "encoder.layers.1.")
        assert equal(states_before[source_name], states_after[name])
    assert not equal(states_before, states_after)


def test_private_name_binding_revision_restores_complete_existing_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    before = sharded_checkpoint(tmp_path / "checkpoint", 2)
    source = Path(current.__file__).read_text()
    previous = "restored.update(zip(layer_names, layer_values, strict=True))"
    corrected = "restored.update(zip(chunk_names, layer_values, strict=True))"
    assert source.count(previous) == 1
    path = tmp_path / "operator_reference.py"
    path.write_text(source.replace(previous, corrected))
    name = "tools.checkpoint_pilot.task._operator_reference"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    restore: object = vars(module)["restore_checkpoint"]
    assert isinstance(restore, FunctionType)
    model, optimizer = fresh()
    progress: object = restore(tmp_path / "checkpoint", model, optimizer, 0, 2, CPU)
    assert isinstance(progress, state.Progress)
    after = state.artifact(model, optimizer, progress, 0, 2, CPU)
    assert equal(before, after)


def test_each_rank_restores_its_own_rng(tmp_path: Path) -> None:
    sharded_checkpoint(tmp_path / "checkpoint", 2)
    recorded: list[torch.Tensor] = []
    for rank in (0, 1):
        model, optimizer = fresh()
        current.restore_checkpoint(
            tmp_path / "checkpoint", model, optimizer, rank, 2, CPU
        )
        actual = torch.get_rng_state()
        expected = state.read_rng(
            state.read(tmp_path / "checkpoint" / f"rank{rank}.pt")["rng"], CPU
        )
        assert torch.equal(actual, expected["cpu"])
        recorded.append(actual)
    assert not torch.equal(recorded[0], recorded[1])


def test_named_optimizer_export_uses_live_identity_not_parameter_ordinal() -> None:
    model, optimizer, progress = initialized()
    before = state.artifact(model, optimizer, progress, 0, 1, CPU)
    optimizer.param_groups[0]["params"].reverse()
    after = state.artifact(model, optimizer, progress, 0, 1, CPU)
    assert set(before["optimizer_state"]) == set(after["optimizer_state"])
    for name in before["optimizer_state"]:
        assert equal(before["optimizer_state"][name], after["optimizer_state"][name])
    assert after["optimizer_groups"][0]["params"] == list(
        reversed(before["optimizer_state"])
    )


def test_reference_split_run_equals_uninterrupted_with_dropout(tmp_path: Path) -> None:
    run(
        Options(
            "prefix",
            "reference",
            tmp_path / "short-checkpoint",
            tmp_path / "short",
            steps=2,
            seed=53,
            device="cpu",
        )
    )
    run(
        Options(
            "resume",
            "reference",
            tmp_path / "short-checkpoint",
            tmp_path / "resumed",
            steps=3,
            seed=53,
            device="cpu",
        )
    )
    run(
        Options(
            "prefix",
            "reference",
            tmp_path / "full-checkpoint",
            tmp_path / "full",
            steps=5,
            seed=53,
            device="cpu",
        )
    )
    prefix = state.read(tmp_path / "short/rank0/terminal_state.pt")
    restored = state.read(tmp_path / "resumed/rank0/restored_state.pt")
    resumed = state.read(tmp_path / "resumed/rank0/terminal_state.pt")
    full = state.read(tmp_path / "full/rank0/terminal_state.pt")
    assert equal(prefix, restored)
    assert equal(resumed, full)


def test_evaluation_preserves_rng_gradients_state_and_module_modes() -> None:
    model, optimizer, progress = initialized()
    model.embedding.eval()
    before = state.artifact(model, optimizer, progress, 0, 1, CPU)
    gradients = [
        parameter.grad.clone()
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    modes = [module.training for module in model.modules()]
    score = evaluate(model, 777, CPU)
    assert score == evaluate(model, 777, CPU)
    assert equal(before, state.artifact(model, optimizer, progress, 0, 1, CPU))
    assert modes == [module.training for module in model.modules()]
    assert equal(gradients, [parameter.grad for parameter in model.parameters()])


def test_complete_batch_is_independent_of_microbatch_partition_and_rng() -> None:
    rng = torch.get_rng_state()
    batch = make_batch(7, 3, 1, CPU)
    assert torch.equal(rng, torch.get_rng_state())
    assert not torch.equal(batch.tokens, make_batch(7, 3, 0, CPU).tokens)
    for accumulation in (1, 2, 4, 8):
        parts = microbatches(batch, accumulation)
        assert torch.equal(torch.cat([part.tokens for part in parts]), batch.tokens)
        assert torch.equal(torch.cat([part.labels for part in parts]), batch.labels)


@pytest.mark.parametrize(
    "field,value",
    [
        ("world_size", True),
        ("world_size", 3),
        ("schema_version", 2),
        ("parameter_names", []),
        ("nonlayer_names", ["unknown"]),
    ],
)
def test_sharded_restore_rejects_malformed_shared_metadata(
    tmp_path: Path, field: str, value: object
) -> None:
    directory = tmp_path / "checkpoint"
    sharded_checkpoint(directory, 2)
    payload = state.read(directory / "shared.pt")
    payload[field] = value
    torch.save(payload, directory / "shared.pt")
    model, optimizer = fresh()
    with pytest.raises((ValueError, TypeError)):
        current.restore_checkpoint(directory, model, optimizer, 0, 2, CPU)


def test_missing_required_field_is_rejected(tmp_path: Path) -> None:
    directory = tmp_path / "checkpoint"
    sharded_checkpoint(directory, 2)
    payload = state.read(directory / "shared.pt")
    del payload["optimizer_groups"]
    torch.save(payload, directory / "shared.pt")
    model, optimizer = fresh()
    with pytest.raises(ValueError, match="fields differ"):
        current.restore_checkpoint(directory, model, optimizer, 0, 2, CPU)


def test_missing_shard_is_not_silently_replaced(tmp_path: Path) -> None:
    directory = tmp_path / "checkpoint"
    sharded_checkpoint(directory, 2)
    (directory / "rank1.pt").unlink()
    model, optimizer = fresh()
    with pytest.raises(FileNotFoundError):
        current.restore_checkpoint(directory, model, optimizer, 0, 2, CPU)


def test_duplicate_layer_and_wrong_progress_are_rejected(tmp_path: Path) -> None:
    directory = tmp_path / "checkpoint"
    sharded_checkpoint(directory, 2)
    original = state.read(directory / "rank0.pt")
    duplicate = copy.deepcopy(original)
    layers = duplicate["layers"]
    assert isinstance(layers, list)
    layers.append(copy.deepcopy(layers[0]))
    torch.save(duplicate, directory / "rank0.pt")
    model, optimizer = fresh()
    with pytest.raises(ValueError, match="layer coverage"):
        current.restore_checkpoint(directory, model, optimizer, 0, 2, CPU)
    original["progress"] = {"completed_steps": 3, "data_cursor": 2, "seed": 43}
    torch.save(original, directory / "rank0.pt")
    with pytest.raises(ValueError, match="one complete batch"):
        current.restore_checkpoint(directory, model, optimizer, 0, 2, CPU)


@pytest.mark.parametrize(
    "arguments",
    [
        ["--steps", "0"],
        ["--steps", "129"],
        ["--accumulation", "3"],
        ["--seed", "-1"],
        ["--lr", "nan"],
    ],
)
def test_cli_rejects_invalid_configuration(arguments: list[str]) -> None:
    with pytest.raises(SystemExit):
        parse_options(
            [
                "prefix",
                "--backend",
                "reference",
                "--checkpoint",
                "/unused",
                "--output",
                "/unused",
                *arguments,
            ]
        )


def test_optimizer_shape_nonfinite_and_foreign_parameters_are_rejected() -> None:
    model, optimizer, _ = initialized()
    _, states, groups = state.optimizer_snapshot(model, optimizer)
    wrong_shape = copy.deepcopy(states)
    wrong_shape[0]["momentum_buffer"] = torch.zeros(1)
    with pytest.raises(ValueError, match="shape or dtype"):
        state.load_optimizer(model, optimizer, wrong_shape, groups)
    states[0]["momentum_buffer"].fill_(float("nan"))
    with pytest.raises(ValueError, match="finite"):
        state.load_optimizer(model, optimizer, states, groups)
    foreign = torch.optim.SGD(Transformer().parameters(), lr=0.15, momentum=0.9)
    with pytest.raises(ValueError, match="outside the model"):
        state.optimizer_parameter_names(model, foreign)


@pytest.mark.parametrize(
    "field,value",
    [
        ("lr", -1.0),
        ("momentum", True),
        ("dampening", "zero"),
        ("nesterov", 1),
        ("foreach", "automatic"),
        ("weight_decay", float("inf")),
    ],
)
def test_optimizer_group_values_are_validated(field: str, value: object) -> None:
    model, optimizer, _ = initialized()
    _, _, groups = state.optimizer_snapshot(model, optimizer)
    fields: dict[str, object] = dict(groups[0])
    fields[field] = value
    with pytest.raises((ValueError, TypeError)):
        state.groups([fields], len(list(model.parameters())))


def test_missing_optimizer_group_field_is_rejected_early() -> None:
    model, optimizer, _ = initialized()
    _, _, groups = state.optimizer_snapshot(model, optimizer)
    del groups[0]["dampening"]
    with pytest.raises(ValueError, match="fields differ"):
        state.groups(groups, len(list(model.parameters())))
