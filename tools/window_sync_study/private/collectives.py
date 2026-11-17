"""Decode native ProcessGroupNCCL all-reduces against training boundaries."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from typing import Protocol

BACKWARD = "traincheck.window.backward"
STEP = "traincheck.window.optimizer_step"


class TimeRange(Protocol):
    @property
    def start(self) -> float: ...


class Event(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def time_range(self) -> TimeRange: ...

    @property
    def input_shapes(self) -> Sequence[Sequence[int]]: ...


@dataclass(frozen=True)
class Reduction:
    backward: int
    update: int
    elements: int


def audit(
    events: Iterable[Event], expected_ends: Sequence[int], gradient_elements: int
) -> dict[str, object]:
    """Count launched all-reduces. This workload's sole scalar payload is counts."""
    backwards = 0
    update = 0
    scalar_calls = 0
    reductions: list[Reduction] = []
    issues: list[str] = []
    observed_ends: list[int] = []
    for event in sorted(events, key=lambda entry: entry.time_range.start):
        if event.name == BACKWARD:
            backwards += 1
        elif event.name == STEP:
            observed_ends.append(backwards)
            update += 1
        elif event.name == "nccl:all_reduce":
            shapes = event.input_shapes
            if len(shapes) != 1 or any(n <= 0 for n in shapes[0]):
                raise ValueError(f"Missing native NCCL payload dimension: {shapes!r}")
            elements = math.prod(shapes[0])
            if elements == 1:
                scalar_calls += 1
            else:
                reductions.append(Reduction(backwards, update, elements))
    if list(expected_ends) != observed_ends:
        issues.append(
            f"Expected update ends {list(expected_ends)}, saw {observed_ends}"
        )
    for i, end in enumerate(expected_ends):
        group = [r for r in reductions if r.update == i]
        if any(r.backward != end for r in group):
            issues.append(
                f"Update {i} issued gradient communication before its last backward"
            )
        elements = sum(r.elements for r in group)
        if elements != gradient_elements:
            issues.append(
                f"Update {i} reduced {elements} gradient elements; expected {gradient_elements}"
            )
    if any(r.update >= len(expected_ends) for r in reductions):
        issues.append("Gradient communication occurred after the expected final update")
    return {
        "issues": issues,
        "scalar_all_reduce_calls": scalar_calls,
        "gradient_all_reduce_calls": len(reductions),
        "gradient_elements_reduced": sum(r.elements for r in reductions),
        "expected_gradient_elements_per_update": gradient_elements,
        "expected_end_microbatches": list(expected_ends),
        "observed_end_microbatches": observed_ends,
        "gradient_reductions": [asdict(r) for r in reductions],
    }
