"""Regression tests for evidence collection and frozen solver inputs."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import pytest

from tools.token_budget_study.operator.agent_trials import (
    decision,
    filemap,
    freeze_submission,
    verify_packet,
)
from tools.token_budget_study.operator.agent_watch import watch_source


def test_frozen_training_and_immutable_inputs(tmp_path: Path) -> None:
    packet = tmp_path / "packet"
    packet.mkdir()
    for name in ("train.py", "training.py", "workload.py", "TASK.md"):
        (packet / name).write_text("original")
    frozen = filemap(packet)
    verify_packet(packet, frozen, initial=True)
    (packet / "training.py").write_text("first")
    with pytest.raises(ValueError, match="training.py"):
        verify_packet(packet, frozen, initial=True)
    verify_packet(packet, frozen, initial=False)
    hashes = freeze_submission(packet, tmp_path / "submission")
    (packet / "training.py").write_text("second")
    assert hashes["training.py"] == hashlib.sha256(b"first").hexdigest()
    assert (tmp_path / "submission" / "training.py").read_text() == "first"
    (packet / "workload.py").write_text("override")
    with pytest.raises(ValueError, match="workload.py"):
        verify_packet(packet, frozen, initial=False)


def test_decision_does_not_allow_finish_plus_request(tmp_path: Path) -> None:
    data = tmp_path / "choice.json"
    data.write_text(
        json.dumps(
            {"action": "finish", "entry": "train.py", "args": [], "summary": "done"}
        )
    )
    with pytest.raises(ValueError, match="Inconsistent"):
        decision(data)


def test_hidden_earlier_submission_before_finish(tmp_path: Path) -> None:
    source = tmp_path / "training.py"
    source.write_text("base")
    turn = tmp_path / "turn-1"
    with watch_source(source, turn):
        turn.mkdir()
        (turn / "received.jsonl").write_text("received\n")
        source.write_text("intermediate")
        index = tmp_path / "turn-1.watch" / "index.jsonl"
        intermediate = hashlib.sha256(b"intermediate").hexdigest()
        deadline = time.monotonic() + 3
        while intermediate not in index.read_text() and time.monotonic() < deadline:
            time.sleep(0.02)
        source.write_text("submitted")
    hashes = [
        (
            tmp_path
            / "turn-1.watch"
            / "blobs"
            / f"{hashlib.sha256(v.encode()).hexdigest()}.py"
        ).read_text()
        for v in ("base", "intermediate", "submitted")
    ]
    assert hashes == ["base", "intermediate", "submitted"]
    assert json.loads(index.read_text().splitlines()[-1])["final"] is True
