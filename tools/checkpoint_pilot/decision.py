"""Validate structured continuation decisions at the operator boundary."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal, TypeAlias

Action: TypeAlias = Literal["run", "finish"]
RequestKind: TypeAlias = Literal["resume", "diagnostic"]
Backend: TypeAlias = Literal["reference", "current"]


@dataclass(frozen=True)
class Observation:
    claim: str
    evidence: tuple[str, ...]


@dataclass(frozen=True)
class Request:
    kind: RequestKind
    backend: Backend
    steps: int
    script: str | None
    purpose: str


@dataclass(frozen=True)
class Decision:
    action: Action
    summary: str
    observations: tuple[Observation, ...]
    hypotheses: tuple[str, ...]
    next_action: str
    request: Request | None


def _object(value: object, expected: set[str], label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be an object")
    fields: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise TypeError(f"{label} fields must have string names")
        fields[key] = item
    if set(fields) != expected:
        missing = sorted(expected - set(fields))
        extra = sorted(set(fields) - expected)
        raise ValueError(f"{label} fields differ: missing={missing}, extra={extra}")
    return fields


def _text(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string")
    if not value.strip():
        raise ValueError(f"{label} must be nonempty")
    return value


def _texts(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise TypeError(f"{label} must be an array")
    return tuple(_text(item, f"{label}[{index}]") for index, item in enumerate(value))


def _backend(value: object) -> Backend:
    text = _text(value, "request.backend")
    if text == "reference":
        return "reference"
    if text == "current":
        return "current"
    raise ValueError("request.backend must be reference or current")


def parse_request(value: object) -> Request:
    fields = _object(
        value, {"kind", "backend", "steps", "script", "purpose"}, "request"
    )
    raw_kind = _text(fields["kind"], "request.kind")
    kind: RequestKind
    if raw_kind == "resume":
        kind = "resume"
    elif raw_kind == "diagnostic":
        kind = "diagnostic"
    else:
        raise ValueError("request.kind must be resume or diagnostic")
    steps = fields["steps"]
    # JSON counts require a native int; bool and application-defined subclasses
    # are outside this boundary contract, so an exact type comparison is needed.
    if type(steps) is not int:  # noqa: E721
        raise TypeError("request.steps must be an integer, excluding booleans")
    assert isinstance(steps, int)
    if not 1 <= steps <= 16:
        raise ValueError("request.steps must be in [1,16]")
    script: str | None
    if kind == "resume":
        if fields["script"] is not None:
            raise ValueError("A resume request must have script=null")
        script = None
    else:
        script = _text(fields["script"], "request.script")
        path = PurePosixPath(script)
        if (
            "\x00" in script
            or path.is_absolute()
            or ".." in path.parts
            or path.suffix != ".py"
        ):
            raise ValueError(
                "A diagnostic script must be a relative .py path without '..'"
            )
    return Request(
        kind=kind,
        backend=_backend(fields["backend"]),
        steps=steps,
        script=script,
        purpose=_text(fields["purpose"], "request.purpose"),
    )


def parse_decision(value: object) -> Decision:
    """Reject malformed decisions while retaining empty evidence/hypothesis sets.

    Empty arrays mean the solver has not supplied observations, evidence, or
    hypotheses yet. They are not converted into verified findings or success.
    Diagnostic paths are lexical POSIX packet paths; execution must additionally
    resolve them and verify containment/existence in the snapshotted workspace.
    """
    fields = _object(
        value,
        {"action", "summary", "observations", "hypotheses", "next_action", "request"},
        "decision",
    )
    raw_action = _text(fields["action"], "action")
    action: Action
    if raw_action == "run":
        action = "run"
    elif raw_action == "finish":
        action = "finish"
    else:
        raise ValueError("action must be run or finish")
    raw_observations = fields["observations"]
    if not isinstance(raw_observations, list):
        raise TypeError("observations must be an array")
    observations: list[Observation] = []
    for index, raw in enumerate(raw_observations):
        observation = _object(raw, {"claim", "evidence"}, f"observations[{index}]")
        observations.append(
            Observation(
                claim=_text(observation["claim"], f"observations[{index}].claim"),
                evidence=_texts(
                    observation["evidence"], f"observations[{index}].evidence"
                ),
            )
        )
    request: Request | None
    if action == "run":
        if fields["request"] is None:
            raise ValueError("action=run requires a request")
        request = parse_request(fields["request"])
    else:
        if fields["request"] is not None:
            raise ValueError("action=finish requires request=null")
        request = None
    return Decision(
        action=action,
        summary=_text(fields["summary"], "summary"),
        observations=tuple(observations),
        hypotheses=_texts(fields["hypotheses"], "hypotheses"),
        next_action=_text(fields["next_action"], "next_action"),
        request=request,
    )


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON object field: {key}")
        result[key] = value
    return result


def read_decision(path: Path) -> Decision:
    value: object = json.loads(path.read_text(), object_pairs_hook=_unique_object)
    return parse_decision(value)
