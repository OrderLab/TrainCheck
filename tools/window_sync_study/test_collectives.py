from dataclasses import dataclass

from tools.window_sync_study.private.collectives import BACKWARD, STEP, audit


@dataclass(frozen=True)
class Interval:
    start: float


@dataclass(frozen=True)
class Record:
    name: str
    time_range: Interval
    input_shapes: list[list[int]]


def trace(sizes: list[tuple[str, int]]) -> list[Record]:
    return [
        Record(name, Interval(index), [[size]] if size else [])
        for index, (name, size) in enumerate(sizes)
    ]


def test_full_and_partial_windows_with_separate_gradient_buckets() -> None:
    events = trace(
        [
            (BACKWARD, 0),
            ("nccl:all_reduce", 1),
            (BACKWARD, 0),
            ("nccl:all_reduce", 5),
            ("nccl:all_reduce", 7),
            (STEP, 0),
            (BACKWARD, 0),
            ("nccl:all_reduce", 12),
            (STEP, 0),
        ]
    )
    result = audit(events, [2, 3], 12)
    assert result["issues"] == []
    assert result["scalar_all_reduce_calls"] == 1
    assert result["gradient_all_reduce_calls"] == 3


def test_count_alone_does_not_pass_early_or_missing_tail_sync() -> None:
    events = trace(
        [
            (BACKWARD, 0),
            ("nccl:all_reduce", 12),
            (BACKWARD, 0),
            (STEP, 0),
            (BACKWARD, 0),
            (STEP, 0),
            ("nccl:all_reduce", 12),
        ]
    )
    result = audit(events, [2, 3], 12)
    assert result["gradient_all_reduce_calls"] == 2
    assert result["issues"] == [
        "Update 0 issued gradient communication before its last backward",
        "Update 1 reduced 0 gradient elements; expected 12",
        "Gradient communication occurred after the expected final update",
    ]
