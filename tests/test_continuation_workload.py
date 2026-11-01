"""Operator-only qualification tests; excluded from the agent's task packet."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from tools.continuation_pilot.training_task.current import run_update as current_update
from tools.continuation_pilot.training_task.engine import (
    Batch,
    Options,
    TinyTransformer,
    evaluate,
    make_batch,
    parse_options,
    run,
    split_batch,
)
from tools.continuation_pilot.training_task.reference import (
    run_update as reference_update,
)


def test_batch_is_reproducible_independent_of_accumulation_and_global_rng() -> None:
    torch.manual_seed(17)
    before = torch.get_rng_state().clone()
    device = torch.device("cpu")
    original = make_batch(123, 7, 0, device)
    repeated = make_batch(123, 7, 0, device)
    assert torch.equal(original.tokens, repeated.tokens)
    assert torch.equal(original.labels, repeated.labels)
    assert torch.equal(torch.get_rng_state(), before)
    assert not torch.equal(original.tokens, make_batch(123, 7, 1, device).tokens)
    for accumulation in (1, 2, 4, 8):
        parts = split_batch(original, accumulation)
        assert torch.equal(torch.cat([part.tokens for part in parts]), original.tokens)
        assert torch.equal(torch.cat([part.labels for part in parts]), original.labels)


@pytest.mark.parametrize("accumulation", [1, 2, 4])
def test_current_gradient_differs_by_one_accumulation_factor(
    accumulation: int,
) -> None:
    torch.set_num_threads(1)
    torch.manual_seed(31)
    reference = TinyTransformer().train()
    current = copy.deepcopy(reference)
    batch = make_batch(444, 1, 0, torch.device("cpu"))
    parts = split_batch(batch, accumulation)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01, momentum=0.9)
    current_optimizer = torch.optim.SGD(current.parameters(), lr=0.01, momentum=0.9)
    reference_loss = reference_update(reference, reference_optimizer, parts)
    current_loss = current_update(current, current_optimizer, parts)
    assert current_loss == pytest.approx(reference_loss, rel=1e-6, abs=1e-6)
    nonzero = False
    for reference_parameter, current_parameter in zip(
        reference.parameters(), current.parameters(), strict=True
    ):
        assert (
            reference_parameter.grad is not None and current_parameter.grad is not None
        )
        assert torch.allclose(
            current_parameter.grad * accumulation,
            reference_parameter.grad,
            rtol=1e-5,
            atol=2e-6,
        )
        nonzero = nonzero or bool(reference_parameter.grad.abs().max() > 1e-4)
    assert nonzero
    if accumulation > 1:
        assert any(
            not torch.equal(left, right)
            for left, right in zip(
                reference.parameters(), current.parameters(), strict=True
            )
        )


def test_evaluation_preserves_rng_parameters_gradients_and_module_modes() -> None:
    torch.set_num_threads(1)
    torch.manual_seed(43)
    model = TinyTransformer().train()
    model.embedding.eval()
    model(torch.tensor([[0, 1]])).sum().backward()
    state = {key: value.clone() for key, value in model.state_dict().items()}
    gradients = [
        parameter.grad.clone()
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    modes = [module.training for module in model.modules()]
    before = torch.get_rng_state().clone()
    score = evaluate(model, torch.device("cpu"), 127)
    assert math.isfinite(score) and score > 0
    assert evaluate(model, torch.device("cpu"), 127) == score
    assert torch.equal(torch.get_rng_state(), before)
    assert [module.training for module in model.modules()] == modes
    for name, parameter in model.state_dict().items():
        assert torch.equal(parameter, state[name])
    actual_gradients = [
        parameter.grad for parameter in model.parameters() if parameter.grad is not None
    ]
    assert len(actual_gradients) == len(gradients)
    for actual, expected in zip(actual_gradients, gradients, strict=True):
        assert torch.equal(actual, expected)


class UniformModel(nn.Module):
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return torch.zeros((*tokens.shape, 32), device=tokens.device)


def test_evaluation_reports_unscaled_token_cross_entropy() -> None:
    value = evaluate(UniformModel(), torch.device("cpu"))
    assert value == pytest.approx(math.log(32), abs=1e-6)


@pytest.mark.parametrize(
    "arguments",
    [
        ["--steps", "129"],
        ["--steps", "0"],
        ["--accumulation", "3"],
        ["--accumulation", "0"],
        ["--seed", "-1"],
        ["--eval-seed", "-1"],
        ["--lr", "nan"],
        ["--lr", "0"],
    ],
)
def test_cli_rejects_invalid_experiment_configuration(arguments: list[str]) -> None:
    with pytest.raises(SystemExit):
        parse_options(["--pipeline", "current", "--output", "/unused", *arguments])


def test_invalid_complete_batch_shape_is_rejected() -> None:
    malformed = Batch(torch.ones((7, 8)), torch.ones((7, 8)))
    with pytest.raises(ValueError, match="shape"):
        split_batch(malformed, 2)


@pytest.mark.parametrize("pipeline", ["reference", "current"])
def test_short_cpu_run_writes_real_curve_and_checkpoint(
    tmp_path: Path, pipeline: str
) -> None:
    output = tmp_path / pipeline
    options = parse_options(
        [
            "--pipeline",
            pipeline,
            "--output",
            str(output),
            "--steps",
            "2",
            "--device",
            "cpu",
            "--accumulation",
            "4",
            "--seed",
            "117",
            "--eval-seed",
            "719",
        ]
    )
    run(options)
    raw: object = json.loads((output / "rank0/metrics.json").read_text())
    assert isinstance(raw, dict)
    assert raw["optimizer_updates"] == 2
    assert raw["tokens_per_global_update"] == 64
    assert raw["eval_seed"] == 719
    curve = raw["validation_curve"]
    assert isinstance(curve, list)
    assert [item["step"] for item in curve] == [0, 2]
    checkpoint: object = torch.load(output / "rank0/checkpoint.pt", weights_only=True)
    assert isinstance(checkpoint, dict)
    assert checkpoint["completed_steps"] == 2
    assert isinstance(checkpoint["model"], dict)
    assert isinstance(checkpoint["optimizer"], dict)


def test_programmatic_boolean_count_is_rejected_before_output(tmp_path: Path) -> None:
    options = Options("current", tmp_path / "run", device="cpu")
    object.__setattr__(options, "steps", True)
    with pytest.raises(TypeError, match="steps"):
        run(options)
    assert not options.output.exists()
