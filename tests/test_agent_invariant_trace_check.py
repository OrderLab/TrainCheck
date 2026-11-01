"""Coverage errors must remain distinct from completed rule evaluations."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from tools.agent_invariants.check import (
    EXCEPTION,
    FUNCTION,
    POST,
    PRE,
    ExpectedRun,
    check_run,
    main,
)


def records(rank: int, scale: float) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    for step in (3, 4):
        for microbatch in (0, 1):
            index = 2 * (step - 3) + microbatch
            keywords: dict[str, object] = {
                "rank": {"int": rank},
                "world_size": {"int": 2},
                "step": {"int": step},
                "microbatch": {"int": microbatch},
                "accumulation": {"int": 2},
                "local_tokens": {"int": 10 if rank == 0 else 22},
                "global_tokens": {"int": 32},
                "applied_scale": {"float": scale},
            }
            base: dict[str, object] = {
                "function": FUNCTION,
                "func_call_id": f"call-{rank}-{index}",
                "process_id": 100 + rank,
                "thread_id": 42,
                "meta_vars": {"step": step, "epoch": 0},
                "is_bound_method": False,
                "obj_id": None,
            }
            events.append(
                {
                    **base,
                    "time": 1000 + 2 * index,
                    "type": PRE,
                    "args": {},
                    "kwargs": keywords,
                }
            )
            events.append({**base, "time": 1001 + 2 * index, "type": POST})
    return events


def write_run(
    root: Path, rank0: list[dict[str, object]], rank1: list[dict[str, object]]
) -> None:
    for rank, events in enumerate((rank0, rank1)):
        directory = root / f"rank{rank}" / "trace"
        directory.mkdir(parents=True)
        (directory / "trace_API_100_42.log").write_text(
            "".join(json.dumps(event, allow_nan=False) + "\n" for event in events)
        )


def expected() -> ExpectedRun:
    return ExpectedRun(world_size=2, start_step=3, steps=2, accumulation=2)


def keywords(event: dict[str, object]) -> dict[str, object]:
    raw = event["kwargs"]
    assert isinstance(raw, dict)
    return raw


def test_complete_check_preserves_identity_and_provenance(tmp_path: Path) -> None:
    rank0, rank1 = records(0, 0.0625), records(1, 0.0625)
    unrelated: dict[str, object] = {
        "function": "torch.example",
        "arbitrary_metadata": [2, 2, 1],
    }
    rank0.insert(0, unrelated)
    write_run(tmp_path, rank0, rank1)
    report = check_run(tmp_path, expected())
    assert report["status"] == "passed"
    assert report["expected_observations"] == report["completed_observations"] == 8
    assert report["violations"] == 0
    assert report["decisions"][0]["pre"]["record"] == rank0[1]
    assert report["decisions"][0]["post"]["record"] == rank0[2]
    assert report["decisions"][0]["pre"]["line"] == 2
    assert report["decisions"][0]["observation"]["step"] == 3
    for source in (*report["trace_files"], report["rule_source"]):
        data = Path(source["path"]).read_bytes()
        assert source["bytes"] == len(data)
        assert source["sha256"] == hashlib.sha256(data).hexdigest()


def test_completed_bad_scale_is_a_violation_not_a_coverage_error(
    tmp_path: Path,
) -> None:
    write_run(tmp_path, records(0, 0.1), records(1, 1.0 / 22))
    report = check_run(tmp_path, expected())
    assert report["status"] == "violated"
    assert report["completed_observations"] == report["violations"] == 8
    assert all(not decision["passed"] for decision in report["decisions"])


def test_different_valid_window_size_does_not_memorize_scale(tmp_path: Path) -> None:
    rows = (records(0, 0.05), records(1, 0.05))
    for rank, events in enumerate(rows):
        for event in events:
            if event["type"] == PRE:
                keywords(event)["local_tokens"] = {"int": 13 if rank == 0 else 27}
                keywords(event)["global_tokens"] = {"int": 40}
    write_run(tmp_path, *rows)
    assert check_run(tmp_path, expected())["status"] == "passed"


@pytest.mark.parametrize("rank", [0, 1])
def test_missing_rank_trace_is_not_success(tmp_path: Path, rank: int) -> None:
    write_run(tmp_path, records(0, 0.0625), records(1, 0.0625))
    (tmp_path / f"rank{rank}" / "trace" / "trace_API_100_42.log").unlink()
    with pytest.raises(ValueError, match="No API trace"):
        check_run(tmp_path, expected())


def test_missing_entire_run_is_not_success(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="rank directories"):
        check_run(tmp_path, expected())


def test_empty_native_logs_are_not_success(tmp_path: Path) -> None:
    write_run(tmp_path, [], [])
    with pytest.raises(ValueError, match="coverage"):
        check_run(tmp_path, expected())


def test_missing_complete_call_is_not_success(tmp_path: Path) -> None:
    write_run(tmp_path, records(0, 0.0625)[2:], records(1, 0.0625))
    with pytest.raises(ValueError, match="coverage"):
        check_run(tmp_path, expected())


@pytest.mark.parametrize("drop", [0, 1])
def test_unpaired_events_are_not_success(tmp_path: Path, drop: int) -> None:
    rank0 = records(0, 0.0625)
    del rank0[drop]
    write_run(tmp_path, rank0, records(1, 0.0625))
    with pytest.raises(ValueError, match="Incomplete observation"):
        check_run(tmp_path, expected())


@pytest.mark.parametrize("index", [0, 1])
def test_duplicate_native_events_are_not_success(tmp_path: Path, index: int) -> None:
    rank0 = records(0, 0.0625)
    rank0.append(copy.deepcopy(rank0[index]))
    write_run(tmp_path, rank0, records(1, 0.0625))
    with pytest.raises(ValueError, match="Duplicate observation event"):
        check_run(tmp_path, expected())


def test_different_call_id_does_not_hide_duplicate_observation(tmp_path: Path) -> None:
    rank0 = records(0, 0.0625)
    duplicated = copy.deepcopy(rank0[:2])
    for event in duplicated:
        event["func_call_id"] = "extra-call"
    rank0.extend(duplicated)
    write_run(tmp_path, rank0, records(1, 0.0625))
    with pytest.raises(ValueError, match="Duplicate observation for"):
        check_run(tmp_path, expected())


def test_exception_post_is_not_a_completed_evaluation(tmp_path: Path) -> None:
    rank0 = records(0, 0.0625)
    rank0[1]["type"] = EXCEPTION
    rank0[1]["exception"] = "RuntimeError"
    rank0[1]["exception_msg"] = "failure"
    write_run(tmp_path, rank0, records(1, 0.0625))
    with pytest.raises(ValueError, match="raised an exception"):
        check_run(tmp_path, expected())


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("process_id", 400, "Incomplete observation"),
        ("thread_id", 9, "Incomplete observation"),
        ("func_call_id", "changed", "Incomplete observation"),
        ("meta_vars", {"step": 8}, "metadata differs"),
        ("time", 1, "precedes"),
        ("kwargs", {"applied_scale": {"float": 1.0}}, "must not repeat"),
    ],
)
def test_changed_post_contract_is_rejected(
    tmp_path: Path, field: str, value: object, match: str
) -> None:
    rank0 = records(0, 0.0625)
    rank0[1][field] = value
    write_run(tmp_path, rank0, records(1, 0.0625))
    with pytest.raises(ValueError, match=match):
        check_run(tmp_path, expected())


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("world_size", {"int": True}, TypeError),
        ("local_tokens", {"int": "10"}, TypeError),
        ("step", {"float": 3.0}, TypeError),
        ("rank", {"int": 1}, ValueError),
        ("accumulation", {"int": 3}, ValueError),
        ("global_tokens", {"int": 0}, ValueError),
        ("applied_scale", {"float": 1}, TypeError),
        ("applied_scale", {"float": True}, TypeError),
        ("applied_scale", {"float": "0.0625"}, TypeError),
        ("applied_scale", {"float": 0.0625, "extra": 1}, TypeError),
    ],
)
def test_malformed_required_scalar_is_rejected(
    tmp_path: Path, field: str, value: object, error: type[Exception]
) -> None:
    rank0 = records(0, 0.0625)
    keywords(rank0[0])[field] = value
    write_run(tmp_path, rank0, records(1, 0.0625))
    with pytest.raises(error):
        check_run(tmp_path, expected())


@pytest.mark.parametrize(
    "field", ["applied_scale", "global_tokens", "local_tokens", "world_size"]
)
def test_missing_required_scalar_is_rejected(tmp_path: Path, field: str) -> None:
    rank0 = records(0, 0.0625)
    del keywords(rank0[0])[field]
    write_run(tmp_path, rank0, records(1, 0.0625))
    with pytest.raises(ValueError, match="required fields"):
        check_run(tmp_path, expected())


def test_inconsistent_microbatch_window_counts_are_rejected(tmp_path: Path) -> None:
    rank0 = records(0, 0.0625)
    keywords(rank0[0])["local_tokens"] = {"int": 11}
    write_run(tmp_path, rank0, records(1, 0.0625))
    with pytest.raises(ValueError, match="within an accumulation window"):
        check_run(tmp_path, expected())


@pytest.mark.parametrize("both_ranks", [False, True])
def test_wrong_cross_rank_totals_are_rejected(tmp_path: Path, both_ranks: bool) -> None:
    rows = records(0, 0.0625), records(1, 0.0625)
    for events in rows if both_ranks else rows[:1]:
        for event in events:
            if event["type"] == PRE:
                keywords(event)["global_tokens"] = {"int": 33}
    write_run(tmp_path, *rows)
    with pytest.raises(ValueError, match="Cross-rank token counts"):
        check_run(tmp_path, expected())


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity"])
def test_nonfinite_json_is_rejected(tmp_path: Path, value: str) -> None:
    write_run(tmp_path, records(0, 0.0625), records(1, 0.0625))
    path = tmp_path / "rank0" / "trace" / "trace_API_100_42.log"
    path.write_text(path.read_text().replace('"float": 0.0625', f'"float": {value}', 1))
    with pytest.raises(ValueError, match="Nonfinite JSON"):
        check_run(tmp_path, expected())


def test_duplicate_json_fields_are_rejected(tmp_path: Path) -> None:
    write_run(tmp_path, records(0, 0.0625), records(1, 0.0625))
    path = tmp_path / "rank0" / "trace" / "trace_API_100_42.log"
    path.write_text(path.read_text().replace('"int": 10', '"int": 10, "int": 11', 1))
    with pytest.raises(ValueError, match="Duplicate JSON field"):
        check_run(tmp_path, expected())


@pytest.mark.parametrize(
    "config",
    [ExpectedRun(0, 1, 1, 1), ExpectedRun(1, 0, 1, 1), ExpectedRun(1, 1, 0, 1)],
)
def test_nonpositive_expected_dimensions_are_rejected(
    tmp_path: Path, config: ExpectedRun
) -> None:
    with pytest.raises(ValueError, match="positive"):
        check_run(tmp_path, config)


@pytest.mark.parametrize(
    "scenario,code,status",
    [("healthy", 0, "passed"), ("fault", 1, "violated"), ("missing", 2, "error")],
)
def test_cli_exit_status_keeps_error_separate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scenario: str,
    code: int,
    status: str,
) -> None:
    root = tmp_path / "run"
    scale = 0.0625 if scenario == "healthy" else 0.1
    write_run(root, records(0, scale), records(1, scale))
    if scenario == "missing":
        (root / "rank0" / "trace" / "trace_API_100_42.log").unlink()
    output = tmp_path / "report.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "check",
            "--root",
            str(root),
            "--world-size",
            "2",
            "--start-step",
            "3",
            "--steps",
            "2",
            "--accumulation",
            "2",
            "--output",
            str(output),
        ],
    )
    assert main() == code
    result: object = json.loads(output.read_text())
    assert isinstance(result, dict)
    assert result["status"] == status
    if scenario == "missing":
        assert result["violations"] is None
        assert result["completed_observations"] is None
