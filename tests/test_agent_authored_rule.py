"""Validate the frozen author artifact without editing its predicate."""

from dataclasses import replace

import pytest

from tools.agent_invariants.authored_rule import Observation, evaluate


def observation() -> Observation:
    return Observation(0, 2, 1, 0, 2, 10, 32, 0.0625)


@pytest.mark.parametrize(
    "world_size,local_tokens,global_tokens,accumulation",
    [(1, 5, 5, 1), (2, 5, 16, 1), (2, 10, 32, 2), (2, 15, 48, 3), (4, 7, 37, 4)],
)
def test_rule_uses_window_counts_instead_of_a_fixed_scale(
    world_size: int, local_tokens: int, global_tokens: int, accumulation: int
) -> None:
    item = Observation(
        0,
        world_size,
        1,
        0,
        accumulation,
        local_tokens,
        global_tokens,
        world_size / global_tokens,
    )
    assert evaluate(item).passed
    assert not evaluate(replace(item, applied_scale=item.applied_scale / 2)).passed


@pytest.mark.parametrize("scale", [0.1, 1.0 / 22, 0.0, -0.0625, 0.0625 / 2])
def test_finite_wrong_multiplier_is_a_violation(scale: float) -> None:
    result = evaluate(replace(observation(), applied_scale=scale))
    assert not result.passed
    assert result.observed == scale
    assert result.expected == 0.0625


@pytest.mark.parametrize(
    "field,value",
    [
        ("rank", True),
        ("world_size", "2"),
        ("step", 1.0),
        ("microbatch", False),
        ("accumulation", None),
        ("local_tokens", 10.0),
        ("global_tokens", True),
        ("applied_scale", 1),
        ("applied_scale", True),
    ],
)
def test_wrong_types_are_errors_instead_of_passes(field: str, value: object) -> None:
    # Simulate a malformed runtime payload without weakening the public signature.
    item = observation()
    object.__setattr__(item, field, value)
    with pytest.raises(TypeError):
        evaluate(item)


@pytest.mark.parametrize(
    "field,value",
    [
        ("rank", -1),
        ("rank", 2),
        ("world_size", 0),
        ("step", 0),
        ("microbatch", 2),
        ("accumulation", 0),
        ("local_tokens", 0),
        ("global_tokens", 10),
        ("global_tokens", 10**400),
        ("applied_scale", float("nan")),
        ("applied_scale", float("inf")),
    ],
)
def test_unsupported_domains_are_errors(field: str, value: object) -> None:
    item = observation()
    object.__setattr__(item, field, value)
    with pytest.raises(ValueError):
        evaluate(item)


def test_single_rank_and_relative_tolerance_boundary() -> None:
    with pytest.raises(ValueError, match="single-rank"):
        evaluate(replace(observation(), world_size=1))
    assert evaluate(replace(observation(), applied_scale=0.0625 * (1 + 1e-13))).passed
    assert not evaluate(
        replace(observation(), applied_scale=0.0625 * (1 + 1e-10))
    ).passed
