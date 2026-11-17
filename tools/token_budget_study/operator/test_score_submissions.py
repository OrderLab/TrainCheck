"""Evidence joins reject cross-arm, source and coverage mix-ups."""

import hashlib
import json
from pathlib import Path

import pytest

from tools.token_budget_study.operator.score_submissions import SCENARIOS, grade_status


def receipt(root: Path, rows: list[dict[str, object]], status: str) -> tuple[Path, str]:
    source = "def train(): pass\n"
    sha = hashlib.sha256(source.encode()).hexdigest()
    path = root / "receipt.json"
    (root / "request.json").write_text(
        json.dumps(
            {
                "id": "trial",
                "kind": "grade",
                "with_traincheck": False,
                "files": {"training.py": source},
            }
        )
    )
    path.write_text(
        json.dumps(
            {
                "id": "trial",
                "status": "ok",
                "grade": {"status": status, "source_sha256": sha, "rank_results": rows},
            }
        )
    )
    return path, sha


def complete_rows() -> list[dict[str, object]]:
    return [
        {"rank": rank, "name": name, "result": {"status": "pass"}}
        for rank in (0, 1)
        for name in sorted(SCENARIOS)
    ]


def test_numerical_failure_on_only_one_rank_is_retained(tmp_path: Path) -> None:
    rows = complete_rows()
    rows[-1]["result"] = {"status": "silent_wrong"}
    path, sha = receipt(tmp_path, rows, "silent_wrong")
    assert grade_status(path, sha, False) == "silent_wrong"
    with pytest.raises(ValueError, match="conditions"):
        grade_status(path, sha, True)
    with pytest.raises(ValueError, match="source"):
        grade_status(path, "0" * 64, False)


def test_duplicate_rank_cannot_complete_coverage(tmp_path: Path) -> None:
    rows = complete_rows()
    rows[-1] = rows[0]
    path, sha = receipt(tmp_path, rows, "pass")
    with pytest.raises(ValueError, match="Duplicate"):
        grade_status(path, sha, False)


def test_interrupted_phase_retains_unknown(tmp_path: Path) -> None:
    path, sha = receipt(tmp_path, [], "incomplete_execution")
    assert grade_status(path, sha, False) == "unknown"
