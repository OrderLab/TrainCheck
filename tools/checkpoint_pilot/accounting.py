"""Validated CLI token accounting and nonoverlapping experiment costs.

Usage mode is an operator-verified protocol choice, never inferred from counter
sizes. Cached input is a subset of input tokens; optional counters remain unknown
when the CLI does not report them. Durations use one monotonic clock per interval
set. These helpers do not estimate token prices or infer active GPU utilization.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

UsageMode = Literal["per_turn", "cumulative"]


class IncompleteExecutionError(ValueError):
    """The stream does not establish a completed turn with known usage."""


class FailedExecutionError(ValueError):
    """The CLI explicitly reported an execution error or failed turn."""


def _count(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a nonnegative integer")
    return value


def _identity(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("thread_id must be a nonempty string")
    return value


def _mode(value: object) -> UsageMode:
    if not isinstance(value, str) or value not in ("per_turn", "cumulative"):
        raise ValueError("usage mode must be per_turn or cumulative")
    if value == "per_turn":
        return "per_turn"
    return "cumulative"


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    reasoning_output_tokens: int | None = None
    cache_write_input_tokens: int | None = None

    def __post_init__(self) -> None:
        for field, value in (
            ("input_tokens", self.input_tokens),
            ("cached_input_tokens", self.cached_input_tokens),
            ("output_tokens", self.output_tokens),
        ):
            _count(value, field)
        if self.cached_input_tokens > self.input_tokens:
            raise ValueError("cached_input_tokens exceeds input_tokens")
        for field, optional_value in (
            ("reasoning_output_tokens", self.reasoning_output_tokens),
            ("cache_write_input_tokens", self.cache_write_input_tokens),
        ):
            if optional_value is not None:
                _count(optional_value, field)

    @property
    def total_tokens(self) -> int:
        """Reported input plus output; do not add cached or reasoning subsets."""
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True)
class CompletedTurn:
    thread_id: str
    usage: TokenUsage

    def __post_init__(self) -> None:
        _identity(self.thread_id)
        if not isinstance(self.usage, TokenUsage):
            raise ValueError("usage must be validated TokenUsage")


@dataclass(frozen=True)
class UsageCheckpoint:
    thread_id: str
    mode: UsageMode
    cumulative_usage: TokenUsage

    def __post_init__(self) -> None:
        _identity(self.thread_id)
        _mode(self.mode)
        if not isinstance(self.cumulative_usage, TokenUsage):
            raise ValueError("cumulative_usage must be validated TokenUsage")


@dataclass(frozen=True)
class AccountedTurn:
    delta: TokenUsage
    checkpoint: UsageCheckpoint


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field: {key}")
        result[key] = value
    return result


def _object(value: object, context: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a JSON object")
    result: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ValueError(f"{context} must have string keys")
        result[key] = item
    return result


def _required(record: dict[str, object], field: str) -> object:
    if field not in record:
        raise ValueError(f"Missing required field: {field}")
    return record[field]


def _optional_count(record: dict[str, object], field: str) -> int | None:
    # Absence means the provider did not expose this counter. A present null or
    # malformed value is not a supported substitute for an integer counter.
    return _count(record[field], field) if field in record else None


def parse_exec_jsonl(text: str) -> CompletedTurn:
    """Parse one CLI exec/resume invocation, ending in exactly one completion.

    Unknown event kinds are accepted before completion for extensibility. Error
    events, duplicate starts/completions, trailing events, and malformed usage
    never become successful zero-cost turns. Raw JSONL should also be retained.
    """
    thread_id: str | None = None
    completed: CompletedTurn | None = None
    started = False
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        value: object = json.loads(line, object_pairs_hook=_unique_object)
        event = _object(value, f"line {line_number}")
        kind = _required(event, "type")
        if not isinstance(kind, str) or not kind:
            raise ValueError(f"line {line_number}: type must be a nonempty string")
        if completed is not None:
            raise ValueError("Event after terminal turn.completed")
        if kind in ("error", "turn.failed"):
            raise FailedExecutionError(f"CLI reported {kind} at line {line_number}")
        if kind == "thread.started":
            if thread_id is not None or started:
                raise ValueError("Duplicate or late thread.started")
            thread_id = _identity(_required(event, "thread_id"))
        elif kind == "turn.started":
            if thread_id is None or started:
                raise ValueError("Ambiguous turn.started or missing thread.started")
            started = True
        elif kind == "turn.completed":
            if thread_id is None:
                raise ValueError("Completion has no thread.started identity")
            usage = _object(_required(event, "usage"), "usage")
            completed = CompletedTurn(
                thread_id,
                TokenUsage(
                    input_tokens=_count(
                        _required(usage, "input_tokens"), "input_tokens"
                    ),
                    cached_input_tokens=_count(
                        _required(usage, "cached_input_tokens"), "cached_input_tokens"
                    ),
                    output_tokens=_count(
                        _required(usage, "output_tokens"), "output_tokens"
                    ),
                    reasoning_output_tokens=_optional_count(
                        usage, "reasoning_output_tokens"
                    ),
                    cache_write_input_tokens=_optional_count(
                        usage, "cache_write_input_tokens"
                    ),
                ),
            )
    if completed is None:
        raise IncompleteExecutionError("No terminal turn.completed with known usage")
    return completed


def _optional_sum(left: int | None, right: int | None) -> int | None:
    return None if left is None or right is None else left + right


def _difference(current: int, previous: int, field: str) -> int:
    if current < previous:
        raise ValueError(f"Cumulative {field} decreased")
    return current - previous


def _optional_difference(
    current: int | None, previous: int | None, field: str
) -> int | None:
    if current is None or previous is None:
        return None
    return _difference(current, previous, field)


def account_turn(
    turn: CompletedTurn,
    *,
    mode: UsageMode,
    previous: UsageCheckpoint | None = None,
    initial_turn: bool = False,
) -> AccountedTurn:
    """Accumulate a completed turn using empirically verified CLI semantics.

    With no prior checkpoint, initial_turn=True explicitly asserts this is the
    first observed turn of a newly created session. Never use it to invent a
    zero baseline for an existing session with unobserved prior usage.
    """
    mode = _mode(mode)
    if not isinstance(turn, CompletedTurn):
        raise ValueError("turn must be a validated CompletedTurn")
    if not isinstance(initial_turn, bool):
        raise ValueError("initial_turn must be boolean")
    if previous is None:
        if not initial_turn:
            raise ValueError("Missing prior checkpoint; initial turn must be explicit")
        return AccountedTurn(
            turn.usage, UsageCheckpoint(turn.thread_id, mode, turn.usage)
        )
    if initial_turn or not isinstance(previous, UsageCheckpoint):
        raise ValueError("Invalid prior checkpoint for this turn")
    if previous.thread_id != turn.thread_id:
        raise ValueError("Prior checkpoint belongs to a different thread")
    if previous.mode != mode:
        raise ValueError("Usage mode changed within a session")
    before, current = previous.cumulative_usage, turn.usage
    if mode == "per_turn":
        cumulative = TokenUsage(
            before.input_tokens + current.input_tokens,
            before.cached_input_tokens + current.cached_input_tokens,
            before.output_tokens + current.output_tokens,
            _optional_sum(
                before.reasoning_output_tokens, current.reasoning_output_tokens
            ),
            _optional_sum(
                before.cache_write_input_tokens, current.cache_write_input_tokens
            ),
        )
        return AccountedTurn(current, UsageCheckpoint(turn.thread_id, mode, cumulative))
    delta = TokenUsage(
        _difference(current.input_tokens, before.input_tokens, "input_tokens"),
        _difference(
            current.cached_input_tokens,
            before.cached_input_tokens,
            "cached_input_tokens",
        ),
        _difference(current.output_tokens, before.output_tokens, "output_tokens"),
        _optional_difference(
            current.reasoning_output_tokens,
            before.reasoning_output_tokens,
            "reasoning_output_tokens",
        ),
        _optional_difference(
            current.cache_write_input_tokens,
            before.cache_write_input_tokens,
            "cache_write_input_tokens",
        ),
    )
    return AccountedTurn(delta, UsageCheckpoint(turn.thread_id, mode, current))


def _timestamp(value: object, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite nonnegative timestamp")
    try:
        finite = math.isfinite(value)
    except OverflowError as error:
        raise ValueError(f"{field} exceeds the supported clock range") from error
    if value < 0 or not finite:
        raise ValueError(f"{field} must be a finite nonnegative timestamp")


@dataclass(frozen=True)
class Interval:
    """Start and end in seconds on the same nonnegative monotonic clock."""

    start: float
    end: float

    def __post_init__(self) -> None:
        _timestamp(self.start, "start")
        _timestamp(self.end, "end")
        if self.end < self.start:
            raise ValueError("Interval ends before it starts")

    @property
    def seconds(self) -> float:
        return self.end - self.start


def blocked_seconds(trial: Interval, intervals: Iterable[Interval]) -> float:
    """Measure the union of genuinely blocked intervals, without clipping them."""
    if not isinstance(trial, Interval):
        raise ValueError("trial must be a validated Interval")
    ordered: list[Interval] = []
    for interval in intervals:
        if not isinstance(interval, Interval):
            raise ValueError("blocked intervals must be validated Intervals")
        if interval.start < trial.start or interval.end > trial.end:
            raise ValueError("Blocked interval lies outside trial")
        ordered.append(interval)
    ordered.sort(key=lambda interval: (interval.start, interval.end))
    if not ordered:
        return 0.0
    total = 0.0
    start, end = ordered[0].start, ordered[0].end
    for interval in ordered[1:]:
        if interval.start <= end:
            end = max(end, interval.end)
        else:
            total += end - start
            start, end = interval.start, interval.end
    return total + end - start


def gpu_allocation_seconds(gpu_count: int, interval: Interval) -> float:
    """Allocated device-seconds, including idle time inside the allocation."""
    if _count(gpu_count, "gpu_count") == 0:
        raise ValueError("gpu_count must be positive")
    if not isinstance(interval, Interval):
        raise ValueError("interval must be a validated Interval")
    try:
        seconds = gpu_count * interval.seconds
    except OverflowError as error:
        raise ValueError(
            "GPU allocation duration exceeds the supported range"
        ) from error
    if not math.isfinite(seconds):
        raise ValueError("GPU allocation duration exceeds the supported range")
    return seconds
