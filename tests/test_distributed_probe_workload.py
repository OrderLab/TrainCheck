from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from tools.distributed_probe.workload import (
    Fault,
    Options,
    TinyTransformer,
    checkpoint_payload,
    compare_gradients,
    copy_state,
    gradient_reference,
    load_checkpoint,
    loss_scale,
    make_batch,
    model_gradients,
    parse_options,
    token_loss,
)


def test_global_token_oracle_distinguishes_local_mean_fault() -> None:
    torch.set_num_threads(1)
    torch.manual_seed(41)
    model = TinyTransformer()
    device = torch.device("cpu")
    expected = gradient_reference(model, 2, 2, device, "fp32")
    rank_batches = [
        [make_batch(1, rank, microbatch, device) for microbatch in range(2)]
        for rank in range(2)
    ]
    local_counts = [sum(int(b.mask.sum()) for b in batches) for batches in rank_batches]
    assert local_counts[0] != local_counts[1]
    faults: tuple[Fault, ...] = ("none", "local-mean")
    for fault in faults:
        rank_gradients: list[dict[str, torch.Tensor]] = []
        for rank, batches in enumerate(rank_batches):
            replica = copy.deepcopy(model)
            scale = loss_scale(fault, 2, sum(local_counts), local_counts[rank])
            for batch in batches:
                (token_loss(replica(batch.tokens), batch) * scale).backward()
            rank_gradients.append(model_gradients(replica))
        averaged = {
            name: (rank_gradients[0][name] + rank_gradients[1][name]) / 2
            for name in expected
        }
        comparison = compare_gradients(averaged, expected, "fp32")
        assert comparison["passed"] is (fault == "none")


def test_state_snapshot_is_independent_and_rejects_unsupported_values() -> None:
    source = torch.tensor([1.0, 2.0])
    snapshot = copy_state({"tensor": source, "empty": [], "optional": None})
    source.add_(3)
    assert isinstance(snapshot, dict)
    assert isinstance(snapshot["tensor"], torch.Tensor)
    assert torch.equal(snapshot["tensor"], torch.tensor([1.0, 2.0]))
    with pytest.raises(TypeError, match="unsupported"):
        copy_state({"bad": object()})
    with pytest.raises(TypeError, match="dictionary key"):
        copy_state({True: 1})


def test_oracle_rejects_nonfinite_gradients() -> None:
    with pytest.raises(ValueError, match="Nonfinite"):
        compare_gradients(
            {"p": torch.tensor([float("nan")])}, {"p": torch.zeros(1)}, "fp32"
        )


def checkpoint_fixture(
    path: Path,
) -> tuple[nn.Module, torch.optim.AdamW, torch.optim.lr_scheduler.StepLR, Options]:
    model = nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), foreach=False)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2)
    model(torch.ones(2, 3)).sum().backward()
    optimizer.step()
    scheduler.step()
    options = Options(path, None, 4, 4, "fp32", "none", "plain", "cpu", 2)
    return model, optimizer, scheduler, options


def test_checkpoint_restores_model_optimizer_scheduler_and_rng(tmp_path: Path) -> None:
    torch.manual_seed(7)
    model, optimizer, scheduler, options = checkpoint_fixture(tmp_path)
    payload = checkpoint_payload(
        model, optimizer, scheduler, 2, 0, 2, options, torch.device("cpu")
    )
    path = tmp_path / "checkpoint.pt"
    torch.save(payload, path)
    expected_rng = torch.rand(5)
    expected_weights = model.weight.detach().clone()
    with torch.no_grad():
        model.weight.add_(10)
    optimizer.state.clear()
    scheduler.last_epoch = 90
    loaded = load_checkpoint(
        path, model, optimizer, scheduler, 0, 2, options, torch.device("cpu")
    )
    assert loaded == 2
    assert torch.equal(model.weight, expected_weights)
    assert torch.equal(torch.rand(5), expected_rng)
    assert len(optimizer.state) == 2
    assert scheduler.last_epoch == 1


@pytest.mark.parametrize(
    "field,value",
    [("rank", 1), ("rank", False), ("precision", "bf16"), ("completed_steps", 4)],
)
def test_checkpoint_rejects_incompatible_provenance(
    tmp_path: Path, field: str, value: object
) -> None:
    model, optimizer, scheduler, options = checkpoint_fixture(tmp_path)
    raw: dict[str, object] = dict(
        checkpoint_payload(
            model, optimizer, scheduler, 2, 0, 2, options, torch.device("cpu")
        )
    )
    raw[field] = value
    path = tmp_path / "checkpoint.pt"
    torch.save(raw, path)
    with pytest.raises(ValueError, match="Checkpoint"):
        load_checkpoint(
            path, model, optimizer, scheduler, 0, 2, options, torch.device("cpu")
        )


def test_checkpoint_missing_required_state_is_an_error(tmp_path: Path) -> None:
    model, optimizer, scheduler, options = checkpoint_fixture(tmp_path)
    payload = checkpoint_payload(
        model, optimizer, scheduler, 2, 0, 2, options, torch.device("cpu")
    )
    del payload["optimizer"]
    path = tmp_path / "checkpoint.pt"
    torch.save(payload, path)
    with pytest.raises(KeyError, match="optimizer"):
        load_checkpoint(
            path, model, optimizer, scheduler, 0, 2, options, torch.device("cpu")
        )


def test_cli_rejects_invalid_update_boundary() -> None:
    with pytest.raises(SystemExit):
        parse_options(["--output", "/unused", "--steps", "2", "--stop-after", "3"])
