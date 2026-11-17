"""Regression checks for the private oracle, using fixed tiny manually grouped updates."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "public"))
sys.path.insert(0, str(BASE / "private"))

from worker import SCENARIOS, compare_map, gradients, momentum, reference, values
from workload import IGNORE_INDEX, TokenModel, batches, make_optimizer


def test_overshoot_flush_does_not_carry_excess_and_partial_tail_is_applied() -> None:
    device = torch.device("cpu")
    torch.manual_seed(30)
    model = TokenModel()
    check_model = copy.deepcopy(model)
    opt = make_optimizer(check_model)
    counts = ((4, 17, 3, 8, 5, 2), (8, 4, 19, 6, 5, 3))
    data = [batches(c, r, 442, device) for r, c in enumerate(counts)]
    steps = reference(model, data, 29)
    # Global counts [12, 21, 22, 14, 10, 5] make windows 33, 36 and 15.
    assert len(steps) == 3
    for expected, indices in zip(steps, ((0, 1), (2, 3), (4, 5)), strict=True):
        opt.zero_grad(set_to_none=True)
        # Token-by-token mean over the manually chosen active examples.
        inputs, labels = [], []
        for rank in range(2):
            for index in indices:
                source, targets = data[rank][index]
                keep = targets.ne(IGNORE_INDEX)
                inputs.append(source[keep])
                labels.append(targets[keep])
        logits = check_model(torch.cat(inputs))
        torch.autograd.backward(F.cross_entropy(logits, torch.cat(labels)))
        assert not compare_map("grads", expected.gradients, gradients(check_model))
        opt.step()
        assert not compare_map("params", expected.parameters, values(check_model))
        assert not compare_map(
            "momentum", expected.momentum, momentum(check_model, opt)
        )


def test_equal_step_counts_do_not_hide_incorrect_gradients() -> None:
    expected = {"weight": torch.tensor([0.12, 0.42])}
    swapped = {"weight": torch.tensor([0.42, 0.12])}
    assert compare_map("gradient", expected, swapped)
    assert compare_map("gradient", expected, {})
    assert compare_map(
        "gradient", expected, {"weight": torch.tensor([float("nan"), 0.42])}
    )
    assert compare_map("gradient", expected, {"weight": expected["weight"].double()})
    assert compare_map(
        "gradient", expected, {"weight": expected["weight"].reshape(1, 2)}
    )


@pytest.mark.parametrize(
    "case_index,expected_steps", [(0, 3), (1, 2), (2, 3), (3, 3), (4, 1), (5, 3)]
)
def test_all_frozen_schedules(case_index: int, expected_steps: int) -> None:
    case = SCENARIOS[case_index]
    data = [
        batches(c, r, case.seed, torch.device("cpu")) for r, c in enumerate(case.counts)
    ]
    steps = reference(TokenModel(), data, case.budget)
    assert len(steps) == expected_steps
    assert all(
        torch.isfinite(t).all() for step in steps for t in step.gradients.values()
    )
