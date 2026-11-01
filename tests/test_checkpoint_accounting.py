"""Regressions for unknown usage, cumulative counters, and cost boundaries."""

from __future__ import annotations

import json
from collections.abc import Mapping

import pytest

from tools.checkpoint_pilot.accounting import (
    CompletedTurn,
    FailedExecutionError,
    IncompleteExecutionError,
    Interval,
    TokenUsage,
    UsageCheckpoint,
    UsageMode,
    account_turn,
    blocked_seconds,
    gpu_allocation_seconds,
    parse_exec_jsonl,
)


def events(*rows: Mapping[str, object]) -> str:
    return "\n".join(json.dumps(dict(row)) for row in rows) + "\n"


def stream(usage: Mapping[str, object], thread_id: str = "session-a") -> str:
    return events(
        {"type": "thread.started", "thread_id": thread_id},
        {"type": "turn.started"},
        {"type": "item.completed", "item": {"type": "agent_message", "text": "done"}},
        {"type": "turn.completed", "usage": dict(usage)},
    )


def minimal_usage() -> dict[str, object]:
    return {"input_tokens": 100, "cached_input_tokens": 40, "output_tokens": 20}


def test_actual_cli_fields_preserve_usage_and_subset_accounting() -> None:
    usage = {
        "input_tokens": 20766,
        "cached_input_tokens": 12160,
        "cache_write_input_tokens": 0,
        "output_tokens": 46,
        "reasoning_output_tokens": 39,
    }
    result = parse_exec_jsonl(stream(usage, "01a1978f-ef8b-77d0-9c02-bce593f7edef"))
    assert result.thread_id == "01a1978f-ef8b-77d0-9c02-bce593f7edef"
    assert result.usage == TokenUsage(20766, 12160, 46, 39, 0)
    assert result.usage.total_tokens == 20812


def test_absent_optional_usage_is_unknown_not_zero() -> None:
    result = parse_exec_jsonl(stream(minimal_usage()))
    assert result.usage.reasoning_output_tokens is None
    assert result.usage.cache_write_input_tokens is None


def test_explicit_successful_zero_usage_remains_distinct_from_unknown() -> None:
    usage = {"input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0}
    assert parse_exec_jsonl(stream(usage)).usage == TokenUsage(0, 0, 0)


def test_unknown_event_and_usage_fields_are_extensible() -> None:
    usage = minimal_usage()
    usage["future_counter"] = {"arbitrary": "metadata"}
    text = events(
        {"type": "thread.started", "thread_id": "session-a"},
        {"type": "future.event", "value": [1, 2]},
        {"type": "turn.completed", "usage": usage},
    )
    assert parse_exec_jsonl(text).usage == TokenUsage(100, 40, 20)


@pytest.mark.parametrize(
    "field", ["input_tokens", "cached_input_tokens", "output_tokens"]
)
def test_missing_required_usage_is_rejected(field: str) -> None:
    usage = minimal_usage()
    del usage[field]
    with pytest.raises(ValueError, match="Missing required field"):
        parse_exec_jsonl(stream(usage))


@pytest.mark.parametrize(
    "field",
    [
        "input_tokens",
        "cached_input_tokens",
        "output_tokens",
        "reasoning_output_tokens",
        "cache_write_input_tokens",
    ],
)
@pytest.mark.parametrize("value", [True, -1, 1.5, "1", None])
def test_invalid_present_usage_is_rejected(field: str, value: object) -> None:
    usage = minimal_usage()
    usage[field] = value
    with pytest.raises(ValueError, match="nonnegative integer"):
        parse_exec_jsonl(stream(usage))


def test_cached_tokens_cannot_exceed_input() -> None:
    usage = minimal_usage()
    usage["cached_input_tokens"] = 101
    with pytest.raises(ValueError, match="exceeds input"):
        parse_exec_jsonl(stream(usage))


@pytest.mark.parametrize(
    "text", ["", "\n", events({"type": "thread.started", "thread_id": "x"})]
)
def test_incomplete_execution_never_has_zero_usage(text: str) -> None:
    with pytest.raises(IncompleteExecutionError):
        parse_exec_jsonl(text)


@pytest.mark.parametrize("kind", ["error", "turn.failed"])
def test_failed_execution_rejected_even_if_followed_by_completion(kind: str) -> None:
    text = events(
        {"type": "thread.started", "thread_id": "x"},
        {"type": kind, "message": "transport failed"},
        {"type": "turn.completed", "usage": minimal_usage()},
    )
    with pytest.raises(FailedExecutionError):
        parse_exec_jsonl(text)


@pytest.mark.parametrize(
    "text",
    [
        "not json",
        "[]",
        events({"type": 2}),
        events({"type": "thread.started"}),
        events({"type": "thread.started", "thread_id": "   "}),
        events({"type": "turn.completed", "usage": minimal_usage()}),
        events({"type": "turn.started"}),
        events(
            {"type": "thread.started", "thread_id": "x"}, {"type": "turn.completed"}
        ),
        events(
            {"type": "thread.started", "thread_id": "x"},
            {"type": "turn.completed", "usage": []},
        ),
        '{"type":"thread.started","thread_id":"a","thread_id":"b"}',
    ],
)
def test_malformed_json_or_identity_is_rejected(text: str) -> None:
    with pytest.raises(ValueError):
        parse_exec_jsonl(text)


def test_duplicate_starts_completions_and_trailing_events_are_rejected() -> None:
    start = {"type": "thread.started", "thread_id": "x"}
    with pytest.raises(ValueError, match="Duplicate"):
        parse_exec_jsonl(events(start, start))
    with pytest.raises(ValueError, match="Ambiguous"):
        parse_exec_jsonl(
            events(start, {"type": "turn.started"}, {"type": "turn.started"})
        )
    tails: tuple[dict[str, object], ...] = (
        {"type": "turn.completed", "usage": minimal_usage()},
        {"type": "future.event"},
    )
    for tail in tails:
        with pytest.raises(ValueError, match="after terminal"):
            parse_exec_jsonl(stream(minimal_usage()) + events(tail))


@pytest.mark.parametrize("mode", ["per_turn", "cumulative"])
def test_first_turn_requires_explicit_initial_assertion(mode: UsageMode) -> None:
    turn = CompletedTurn("s", TokenUsage(100, 40, 20))
    with pytest.raises(ValueError, match="Missing prior"):
        account_turn(turn, mode=mode)
    first = account_turn(turn, mode=mode, initial_turn=True)
    assert first.delta == turn.usage
    assert first.checkpoint == UsageCheckpoint("s", mode, turn.usage)
    with pytest.raises(ValueError, match="Invalid prior"):
        account_turn(turn, mode=mode, previous=first.checkpoint, initial_turn=True)


def test_per_turn_counters_sum_even_when_second_turn_is_smaller() -> None:
    first = account_turn(
        CompletedTurn("s", TokenUsage(100, 40, 20, 12, 4)),
        mode="per_turn",
        initial_turn=True,
    )
    second = account_turn(
        CompletedTurn("s", TokenUsage(30, 20, 10, 7, 2)),
        mode="per_turn",
        previous=first.checkpoint,
    )
    assert second.delta == TokenUsage(30, 20, 10, 7, 2)
    assert second.checkpoint.cumulative_usage == TokenUsage(130, 60, 30, 19, 6)


def test_cumulative_counters_compute_validated_delta() -> None:
    first = account_turn(
        CompletedTurn("s", TokenUsage(100, 40, 20, 12, 4)),
        mode="cumulative",
        initial_turn=True,
    )
    second = account_turn(
        CompletedTurn("s", TokenUsage(130, 60, 30, 19, 6)),
        mode="cumulative",
        previous=first.checkpoint,
    )
    assert second.delta == TokenUsage(30, 20, 10, 7, 2)
    assert second.checkpoint.cumulative_usage == TokenUsage(130, 60, 30, 19, 6)


def test_actual_cli_resume_reports_cumulative_counters() -> None:
    first = account_turn(
        CompletedTurn(
            "01a1978f-ef8b-77d0-9c02-bce593f7edef", TokenUsage(20766, 12160, 46, 39, 0)
        ),
        mode="cumulative",
        initial_turn=True,
    )
    resumed = account_turn(
        CompletedTurn(
            "01a1978f-ef8b-77d0-9c02-bce593f7edef", TokenUsage(42606, 32768, 52, 39, 0)
        ),
        mode="cumulative",
        previous=first.checkpoint,
    )
    assert resumed.delta == TokenUsage(21840, 20608, 6, 0, 0)
    assert resumed.checkpoint.cumulative_usage.total_tokens == 42658


@pytest.mark.parametrize("mode", ["per_turn", "cumulative"])
def test_session_identity_and_mode_must_match(mode: UsageMode) -> None:
    turn = CompletedTurn("s", TokenUsage(100, 40, 20))
    wrong_thread = UsageCheckpoint("different", mode, turn.usage)
    with pytest.raises(ValueError, match="different thread"):
        account_turn(turn, mode=mode, previous=wrong_thread)
    other_mode: UsageMode = "cumulative" if mode == "per_turn" else "per_turn"
    wrong_mode = UsageCheckpoint("s", other_mode, turn.usage)
    with pytest.raises(ValueError, match="mode changed"):
        account_turn(turn, mode=mode, previous=wrong_mode)


@pytest.mark.parametrize(
    "current",
    [
        TokenUsage(99, 40, 20, 12, 4),
        TokenUsage(100, 39, 20, 12, 4),
        TokenUsage(100, 40, 19, 12, 4),
        TokenUsage(100, 40, 20, 11, 4),
        TokenUsage(100, 40, 20, 12, 3),
    ],
)
def test_decreasing_cumulative_counters_are_rejected(current: TokenUsage) -> None:
    previous = UsageCheckpoint("s", "cumulative", TokenUsage(100, 40, 20, 12, 4))
    with pytest.raises(ValueError, match="decreased"):
        account_turn(CompletedTurn("s", current), mode="cumulative", previous=previous)


def test_cumulative_cache_delta_is_also_a_subset() -> None:
    previous = UsageCheckpoint("s", "cumulative", TokenUsage(100, 40, 20))
    with pytest.raises(ValueError, match="exceeds input"):
        account_turn(
            CompletedTurn("s", TokenUsage(101, 45, 22)),
            mode="cumulative",
            previous=previous,
        )


def test_optional_unknown_counter_cannot_be_reconstructed_from_one_endpoint() -> None:
    previous = UsageCheckpoint("s", "cumulative", TokenUsage(100, 40, 20))
    current = CompletedTurn("s", TokenUsage(130, 60, 30, 7, 2))
    result = account_turn(current, mode="cumulative", previous=previous)
    assert result.delta.reasoning_output_tokens is None
    assert result.delta.cache_write_input_tokens is None
    assert result.checkpoint.cumulative_usage.reasoning_output_tokens == 7
    per_turn = account_turn(
        current,
        mode="per_turn",
        previous=UsageCheckpoint("s", "per_turn", previous.cumulative_usage),
    )
    assert per_turn.checkpoint.cumulative_usage.reasoning_output_tokens is None
    assert per_turn.checkpoint.cumulative_usage.cache_write_input_tokens is None


def test_optional_counter_disappearing_makes_delta_and_new_total_unknown() -> None:
    previous = UsageCheckpoint("s", "cumulative", TokenUsage(100, 40, 20, 12, 4))
    result = account_turn(
        CompletedTurn("s", TokenUsage(130, 60, 30)),
        mode="cumulative",
        previous=previous,
    )
    assert result.delta.reasoning_output_tokens is None
    assert result.delta.cache_write_input_tokens is None
    assert result.checkpoint.cumulative_usage.reasoning_output_tokens is None
    assert result.checkpoint.cumulative_usage.cache_write_input_tokens is None


def test_blocked_time_uses_union_including_duplicates_and_touching_intervals() -> None:
    trial = Interval(10, 40)
    blocked = [
        Interval(20, 25),
        Interval(11, 15),
        Interval(14, 19),
        Interval(19, 20),
        Interval(11, 15),
        Interval(28, 28),
        Interval(30, 32),
    ]
    assert blocked_seconds(trial, blocked) == 16
    assert blocked_seconds(trial, []) == 0
    assert blocked_seconds(trial, [trial]) == trial.seconds
    assert trial.seconds - blocked_seconds(trial, blocked) == 14


@pytest.mark.parametrize("interval", [Interval(9, 11), Interval(39, 41)])
def test_blocked_intervals_are_not_silently_clipped(interval: Interval) -> None:
    with pytest.raises(ValueError, match="outside trial"):
        blocked_seconds(Interval(10, 40), [interval])


@pytest.mark.parametrize(
    "start,end", [(-1, 2), (2, 1), (float("nan"), 2), (0, float("inf")), (True, 2)]
)
def test_invalid_clock_intervals_are_rejected(start: float, end: float) -> None:
    with pytest.raises(ValueError):
        Interval(start, end)


@pytest.mark.parametrize("count", [0, -1, True])
def test_gpu_count_must_be_positive_integer(count: int) -> None:
    with pytest.raises(ValueError):
        gpu_allocation_seconds(count, Interval(0, 1))


def test_gpu_cost_is_allocated_device_time_not_request_count() -> None:
    assert gpu_allocation_seconds(2, Interval(10, 40)) == 60
    assert gpu_allocation_seconds(2, Interval(10, 10)) == 0


def test_unrepresentable_clock_and_gpu_durations_are_rejected() -> None:
    with pytest.raises(ValueError, match="clock range"):
        Interval(0, 10**1000)
    with pytest.raises(ValueError, match="supported range"):
        gpu_allocation_seconds(2, Interval(0, 1e308))
    with pytest.raises(ValueError, match="supported range"):
        gpu_allocation_seconds(10**1000, Interval(0, 1.0))
