"""Evaluate a frozen agent-authored predicate on native TrainCheck API records.

This experiment checker is not TrainCheck's built-in inferred-invariant engine.
It requires complete, successful observation calls for an explicitly declared run.
Missing coverage and malformed data raise errors; neither becomes a passing rule.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypedDict

from tools.agent_invariants import authored_rule
from tools.agent_invariants.authored_rule import Observation

FUNCTION = "tools.agent_invariants.observation.apply_loss_scale"
PRE = "function_call (pre)"
POST = "function_call (post)"
EXCEPTION = "function_call (post) (exception)"
INTEGER_FIELDS = (
    "rank",
    "world_size",
    "step",
    "microbatch",
    "accumulation",
    "local_tokens",
    "global_tokens",
)
FIELDS = frozenset((*INTEGER_FIELDS, "applied_scale"))


@dataclass(frozen=True)
class ExpectedRun:
    world_size: int
    start_step: int
    steps: int
    accumulation: int

    def validate(self) -> None:
        for name, value in self.to_dict().items():
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"expected {name} must be an integer")
            if value < 1:
                raise ValueError(f"expected {name} must be positive")

    def to_dict(self) -> dict[str, int]:
        return {
            "world_size": self.world_size,
            "start_step": self.start_step,
            "steps": self.steps,
            "accumulation": self.accumulation,
        }


@dataclass(frozen=True)
class Event:
    source: str
    line: int
    rank_directory: int
    process_id: int
    thread_id: int
    func_call_id: str
    timestamp: int
    step: int
    kind: str
    raw: dict[str, object]
    observation: Observation | None

    def identity(self) -> tuple[int, int, int, str]:
        return (
            self.rank_directory,
            self.process_id,
            self.thread_id,
            self.func_call_id,
        )


class TraceFile(TypedDict):
    path: str
    bytes: int
    sha256: str


class EventEvidence(TypedDict):
    path: str
    line: int
    record: dict[str, object]


class DecisionRecord(TypedDict):
    rule_id: str
    passed: bool
    observed: float
    expected: float
    explanation: str
    observation: dict[str, int | float]
    pre: EventEvidence
    post: EventEvidence


class CheckReport(TypedDict):
    status: Literal["passed", "violated"]
    scope: str
    expected_run: dict[str, int]
    expected_observations: int
    completed_observations: int
    violations: int
    rule_source: TraceFile
    trace_files: list[TraceFile]
    decisions: list[DecisionRecord]


def mapping(value: object, context: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"{context} must be an object")
    result: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise TypeError(f"{context} must have string keys")
        result[key] = item
    return result


def integer(value: object, context: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{context} must be an integer")
    return value


def text(value: object, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{context} must be a nonempty string")
    return value


def unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field: {key}")
        result[key] = value
    return result


def reject_json_constant(value: str) -> object:
    raise ValueError(f"Nonfinite JSON constant: {value}")


def observation_from_record(record: dict[str, object]) -> Observation:
    positional = mapping(record["args"], "args")
    if positional:
        raise ValueError("Observation calls must record only named scalar arguments")
    keywords = mapping(record["kwargs"], "kwargs")
    if set(keywords) != FIELDS:
        raise ValueError("Observation kwargs do not match the required fields")
    values: dict[str, int] = {}
    for field in INTEGER_FIELDS:
        payload = mapping(keywords[field], f"kwargs.{field}")
        if set(payload) != {"int"}:
            raise TypeError(f"kwargs.{field} must have the native int payload")
        values[field] = integer(payload["int"], f"kwargs.{field}.int")
    scale_payload = mapping(keywords["applied_scale"], "kwargs.applied_scale")
    if set(scale_payload) != {"float"}:
        raise TypeError("kwargs.applied_scale must have the native float payload")
    scale = scale_payload["float"]
    if not isinstance(scale, float):
        raise TypeError("kwargs.applied_scale.float must be a float")
    if not math.isfinite(scale):
        raise ValueError("kwargs.applied_scale.float must be finite")
    if values["local_tokens"] < 1 or values["global_tokens"] < values["local_tokens"]:
        raise ValueError("Observation requires positive, ordered token counts")
    return Observation(
        rank=values["rank"],
        world_size=values["world_size"],
        step=values["step"],
        microbatch=values["microbatch"],
        accumulation=values["accumulation"],
        local_tokens=values["local_tokens"],
        global_tokens=values["global_tokens"],
        applied_scale=scale,
    )


def parse_event(record: dict[str, object], path: Path, line: int, rank: int) -> Event:
    kind = text(record["type"], "type")
    if kind not in (PRE, POST, EXCEPTION):
        raise ValueError(f"Unexpected observation event type: {kind}")
    process_id = integer(record["process_id"], "process_id")
    thread_id = integer(record["thread_id"], "thread_id")
    timestamp = integer(record["time"], "time")
    if process_id < 1 or thread_id < 1 or timestamp < 0:
        raise ValueError("Invalid observation event identity or timestamp")
    meta = mapping(record["meta_vars"], "meta_vars")
    step = integer(meta["step"], "meta_vars.step")
    if record["is_bound_method"] is not False or record["obj_id"] is not None:
        raise ValueError("Observation helper must be an unbound function")
    observation = observation_from_record(record) if kind == PRE else None
    if kind != PRE and ("args" in record or "kwargs" in record):
        raise ValueError("Observation post record must not repeat argument fields")
    if observation is not None and (
        observation.rank != rank or observation.step != step
    ):
        raise ValueError("Observation rank/step disagrees with its trace context")
    return Event(
        source=str(path.resolve()),
        line=line,
        rank_directory=rank,
        process_id=process_id,
        thread_id=thread_id,
        func_call_id=text(record["func_call_id"], "func_call_id"),
        timestamp=timestamp,
        step=step,
        kind=kind,
        raw=record,
        observation=observation,
    )


def file_identity(path: Path) -> TraceFile:
    data = path.read_bytes()
    return {
        "path": str(path.resolve()),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def read_events(
    root: Path, expected: ExpectedRun
) -> tuple[list[Event], list[TraceFile]]:
    if not root.is_dir():
        raise FileNotFoundError(root)
    rank_paths = sorted(root.glob("rank*"))
    if {path.name for path in rank_paths} != {
        f"rank{rank}" for rank in range(expected.world_size)
    } or any(not path.is_dir() for path in rank_paths):
        raise ValueError("Run rank directories do not match expected world_size")
    events: list[Event] = []
    files: list[TraceFile] = []
    for rank in range(expected.world_size):
        paths = sorted((root / f"rank{rank}" / "trace").glob("trace_API_*.log"))
        if not paths:
            raise ValueError(f"No API trace files for rank {rank}")
        for path in paths:
            data = path.read_bytes()
            files.append(
                {
                    "path": str(path.resolve()),
                    "bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
            )
            for line, content in enumerate(data.decode("utf-8").splitlines(), 1):
                decoded: object = json.loads(
                    content,
                    object_pairs_hook=unique_json_object,
                    parse_constant=reject_json_constant,
                )
                record = mapping(decoded, f"{path}:{line}")
                function = text(record["function"], "function")
                if function == FUNCTION:
                    events.append(parse_event(record, path, line, rank))
    return events, files


def complete_calls(events: list[Event]) -> list[tuple[Event, Event]]:
    calls: dict[tuple[int, int, int, str], dict[str, Event]] = {}
    for event in events:
        call = calls.setdefault(event.identity(), {})
        if event.kind in call:
            raise ValueError(f"Duplicate observation event: {event.identity()}")
        call[event.kind] = event
    paired: list[tuple[Event, Event]] = []
    for identity, call in calls.items():
        if EXCEPTION in call:
            raise ValueError(f"Observation call raised an exception: {identity}")
        if set(call) != {PRE, POST}:
            raise ValueError(f"Incomplete observation call: {identity}")
        before, after = call[PRE], call[POST]
        if before.timestamp > after.timestamp:
            raise ValueError("Observation post timestamp precedes its pre timestamp")
        if before.raw["meta_vars"] != after.raw["meta_vars"]:
            raise ValueError("Observation pre/post metadata differs")
        paired.append((before, after))
    return paired


def validate_coverage(calls: list[tuple[Event, Event]], expected: ExpectedRun) -> None:
    required = {
        (rank, step, microbatch)
        for rank in range(expected.world_size)
        for step in range(expected.start_step, expected.start_step + expected.steps)
        for microbatch in range(expected.accumulation)
    }
    observed: dict[tuple[int, int, int], Observation] = {}
    windows: dict[tuple[int, int], tuple[int, int]] = {}
    for before, _ in calls:
        sample = before.observation
        assert sample is not None
        if (
            sample.world_size != expected.world_size
            or sample.accumulation != expected.accumulation
        ):
            raise ValueError("Observation configuration differs from expected run")
        key = sample.rank, sample.step, sample.microbatch
        if key in observed:
            raise ValueError(f"Duplicate observation for rank/step/microbatch: {key}")
        observed[key] = sample
        window = sample.rank, sample.step
        counts = sample.local_tokens, sample.global_tokens
        if window in windows and windows[window] != counts:
            raise ValueError("Token counts differ within an accumulation window")
        windows[window] = counts
    if set(observed) != required:
        raise ValueError(
            f"Incomplete or unexpected observation coverage: "
            f"missing={sorted(required - set(observed))}, extra={sorted(set(observed) - required)}"
        )
    for step in range(expected.start_step, expected.start_step + expected.steps):
        rank_counts = [windows[(rank, step)] for rank in range(expected.world_size)]
        globals_seen = {global_count for _, global_count in rank_counts}
        if (
            len(globals_seen) != 1
            or sum(local for local, _ in rank_counts) != rank_counts[0][1]
        ):
            raise ValueError(
                "Cross-rank token counts do not sum to the shared global count"
            )


def observation_dict(sample: Observation) -> dict[str, int | float]:
    return {
        "rank": sample.rank,
        "world_size": sample.world_size,
        "step": sample.step,
        "microbatch": sample.microbatch,
        "accumulation": sample.accumulation,
        "local_tokens": sample.local_tokens,
        "global_tokens": sample.global_tokens,
        "applied_scale": sample.applied_scale,
    }


def check_run(root: Path, expected: ExpectedRun) -> CheckReport:
    expected.validate()
    events, trace_files = read_events(root, expected)
    calls = complete_calls(events)
    validate_coverage(calls, expected)
    decisions: list[DecisionRecord] = []
    for before, after in calls:
        sample = before.observation
        assert sample is not None
        decision = authored_rule.evaluate(sample)
        decisions.append(
            {
                "rule_id": decision.rule_id,
                "passed": decision.passed,
                "observed": decision.observed,
                "expected": decision.expected,
                "explanation": decision.explanation,
                "observation": observation_dict(sample),
                "pre": {
                    "path": before.source,
                    "line": before.line,
                    "record": before.raw,
                },
                "post": {"path": after.source, "line": after.line, "record": after.raw},
            }
        )
    violations = sum(not decision["passed"] for decision in decisions)
    return {
        "status": "violated" if violations else "passed",
        "scope": "custom agent-authored predicate on TrainCheck API traces; not built-in inferred invariants",
        "expected_run": expected.to_dict(),
        "expected_observations": expected.world_size
        * expected.steps
        * expected.accumulation,
        "completed_observations": len(decisions),
        "violations": violations,
        "rule_source": file_identity(Path(authored_rule.__file__)),
        "trace_files": trace_files,
        "decisions": decisions,
    }


class Options(argparse.Namespace):
    root: Path
    world_size: int
    start_step: int
    steps: int
    accumulation: int
    output: Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--start-step", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--accumulation", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args(namespace=Options())
    expected = ExpectedRun(
        options.world_size, options.start_step, options.steps, options.accumulation
    )
    try:
        report = check_run(options.root, expected)
    except (OSError, UnicodeError, ValueError, TypeError, KeyError) as error:
        failure: dict[str, object] = {
            "status": "error",
            "scope": "custom agent-authored predicate on TrainCheck API traces",
            "expected_run": expected.to_dict(),
            "completed_observations": None,
            "violations": None,
            "error_type": type(error).__name__,
            "error": str(error),
        }
        options.output.write_text(json.dumps(failure, indent=2, allow_nan=False) + "\n")
        return 2
    options.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return 1 if report["violations"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
