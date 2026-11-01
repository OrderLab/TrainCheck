"""Malformed or incomplete solver decisions must not become executable requests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.checkpoint_pilot.decision import parse_decision, parse_request, read_decision


def valid_request() -> dict[str, object]:
    return {
        "kind": "resume",
        "backend": "current",
        "steps": 8,
        "script": None,
        "purpose": "Measure the revised continuation.",
    }


def valid_decision() -> dict[str, object]:
    return {
        "action": "run",
        "summary": "Collect evidence before selecting the next revision.",
        "observations": [],
        "hypotheses": [],
        "next_action": "Run the current continuation.",
        "request": valid_request(),
    }


def test_empty_observations_and_hypotheses_remain_empty_not_success() -> None:
    decision = parse_decision(valid_decision())
    assert decision.observations == ()
    assert decision.hypotheses == ()
    assert decision.request is not None
    assert decision.request.kind == "resume"
    assert decision.request.script is None


def test_valid_observations_preserve_text_order_and_duplicate_evidence() -> None:
    payload = valid_decision()
    payload["observations"] = [
        {"claim": " Measured state is available. ", "evidence": ["a.json", "a.json"]},
        {"claim": "No supporting artifact is available yet.", "evidence": []},
    ]
    payload["hypotheses"] = ["A possible explanation.", "An alternative explanation."]
    decision = parse_decision(payload)
    assert decision.observations[0].claim == " Measured state is available. "
    assert decision.observations[0].evidence == ("a.json", "a.json")
    assert decision.observations[1].evidence == ()
    assert decision.hypotheses == (
        "A possible explanation.",
        "An alternative explanation.",
    )


def test_finish_can_retain_implementation_without_request(tmp_path: Path) -> None:
    payload = valid_decision()
    payload["action"] = "finish"
    payload["request"] = None
    payload["next_action"] = (
        "Retain the current implementation pending further evidence."
    )
    path = tmp_path / "decision.json"
    path.write_text(json.dumps(payload))
    decision = read_decision(path)
    assert decision.action == "finish"
    assert decision.request is None


def test_valid_diagnostic_can_select_reference_and_nested_script() -> None:
    payload = valid_request()
    payload.update(
        kind="diagnostic", backend="reference", steps=16, script="checks/inspect.py"
    )
    request = parse_request(payload)
    assert request.kind == "diagnostic"
    assert request.backend == "reference"
    assert request.steps == 16
    assert request.script == "checks/inspect.py"


@pytest.mark.parametrize("field", list(valid_decision()))
def test_missing_decision_fields_are_rejected(field: str) -> None:
    payload = valid_decision()
    del payload[field]
    with pytest.raises(ValueError, match="missing="):
        parse_decision(payload)


@pytest.mark.parametrize("field", list(valid_request()))
def test_missing_request_fields_are_rejected(field: str) -> None:
    payload = valid_request()
    del payload[field]
    with pytest.raises(ValueError, match="missing="):
        parse_request(payload)


@pytest.mark.parametrize("where", ["decision", "request", "observation"])
def test_unknown_fields_are_rejected(where: str) -> None:
    payload = valid_decision()
    if where == "decision":
        payload["success"] = True
    elif where == "request":
        request = valid_request()
        request["timeout"] = 9000
        payload["request"] = request
    else:
        payload["observations"] = [
            {"claim": "A claim.", "evidence": [], "verified": True}
        ]
    with pytest.raises(ValueError, match="extra="):
        parse_decision(payload)


@pytest.mark.parametrize(
    "action,requested", [("run", None), ("finish", valid_request())]
)
def test_action_request_combinations_are_required(
    action: str, requested: object
) -> None:
    payload = valid_decision()
    payload.update(action=action, request=requested)
    with pytest.raises(ValueError, match="requires"):
        parse_decision(payload)


@pytest.mark.parametrize("steps", [True, False, 1.0, "8", None, 0, -1, 17])
def test_invalid_update_counts_are_rejected(steps: object) -> None:
    payload = valid_request()
    payload["steps"] = steps
    with pytest.raises((TypeError, ValueError), match="steps"):
        parse_request(payload)


def test_update_count_subclasses_are_not_native_json_integers() -> None:
    class Count(int):
        pass

    payload = valid_request()
    payload["steps"] = Count(8)
    with pytest.raises(TypeError, match="steps"):
        parse_request(payload)


@pytest.mark.parametrize(
    "script",
    [
        None,
        "",
        "  ",
        "/tmp/check.py",
        "../check.py",
        "checks/../check.py",
        "check.txt",
        "check.py\x00",
    ],
)
def test_invalid_diagnostic_paths_are_rejected(script: object) -> None:
    payload = valid_request()
    payload.update(kind="diagnostic", script=script)
    with pytest.raises((TypeError, ValueError)):
        parse_request(payload)


def test_resume_cannot_execute_a_diagnostic_script() -> None:
    payload = valid_request()
    payload["script"] = "check.py"
    with pytest.raises(ValueError, match="script=null"):
        parse_request(payload)


@pytest.mark.parametrize(
    "field,value",
    [("kind", "train"), ("backend", "other"), ("purpose", "  "), ("purpose", False)],
)
def test_invalid_request_text_is_rejected(field: str, value: object) -> None:
    payload = valid_request()
    payload[field] = value
    with pytest.raises((TypeError, ValueError)):
        parse_request(payload)


@pytest.mark.parametrize(
    "field,value",
    [
        ("action", "inspect"),
        ("summary", ""),
        ("summary", 4),
        ("next_action", "\n"),
        ("observations", None),
        ("observations", ["a claim"]),
        ("observations", [{"claim": "", "evidence": []}]),
        ("observations", [{"claim": "A claim", "evidence": "file.py"}]),
        ("hypotheses", [""]),
        ("hypotheses", [False]),
    ],
)
def test_malformed_decision_content_is_rejected(field: str, value: object) -> None:
    payload = valid_decision()
    payload[field] = value
    with pytest.raises((TypeError, ValueError)):
        parse_decision(payload)


def test_duplicate_json_fields_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "decision.json"
    payload = json.dumps(valid_decision())
    path.write_text(payload[:-1] + ', "action": "finish"}')
    with pytest.raises(ValueError, match="Duplicate JSON object field"):
        read_decision(path)


def test_invalid_json_and_missing_files_propagate(tmp_path: Path) -> None:
    path = tmp_path / "decision.json"
    with pytest.raises(FileNotFoundError):
        read_decision(path)
    path.write_text("{")
    with pytest.raises(json.JSONDecodeError):
        read_decision(path)
