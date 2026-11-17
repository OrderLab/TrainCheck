"""Request boundary and non-allocation contracts for the Ring18 service."""

import hashlib
from pathlib import Path

import pytest

from tools.token_budget_study.operator import ring_service


def request() -> dict[str, object]:
    return {
        "id": "control_1-request1",
        "kind": "public",
        "entry": "train.py",
        "argv": [],
        "files": {"training.py": "print(1)\n"},
        "with_traincheck": False,
        "worker_sha256": hashlib.sha256(
            Path(ring_service.__file__).read_bytes()
        ).hexdigest(),
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("id", "../../outside"),
        ("entry", "private/grade.py"),
        ("files", {"training.py": "x", "private.py": "y"}),
        ("entry", "diagnostic.py"),
        ("argv", ["a\x00b"]),
        ("worker_sha256", "deadbeef"),
    ],
)
def test_reject_malformed_request(field: str, value: object) -> None:
    data = request()
    data[field] = value
    with pytest.raises(ValueError):
        ring_service.parse_request(data)


def test_busy_devices_do_not_incur_worker_cost_or_reuse_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for folder in ("public", "runs"):
        (tmp_path / folder).mkdir()
    for name in ("train.py", "workload.py"):
        (tmp_path / "public" / name).write_text("")
    monkeypatch.setattr(ring_service, "REMOTE", tmp_path)
    monkeypatch.setattr(ring_service, "gpu_state", lambda: {4: 0, 5: 20})
    parsed = ring_service.parse_request(request())
    result = ring_service.serve(parsed)
    assert (result["status"], result["worker_seconds"], result["gpu_seconds"]) == (
        "busy",
        0,
        0,
    )
    assert not (tmp_path / "runs" / parsed["id"] / "started.json").exists()
    with pytest.raises(FileExistsError):
        ring_service.serve(parsed)
