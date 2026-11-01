"""Exercise the real worker bootstrap without importing a training framework."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tools.checkpoint_pilot import run_worker

PROBE = """import json
import os
from pathlib import Path
import probe_helper

trace = Path(os.environ['TRAINCHECK_OUTPUT_DIR'])
assert trace.is_dir(), 'trace sink must exist before module execution'
result = {'trace': str(trace), 'cwd': str(Path.cwd()),
          'rank': os.environ['RANK'], 'helper': probe_helper.VALUE,
          'route': ROUTE}
(Path(os.environ['EXPERIMENT_OUTPUT']) / ('seen-' + os.environ['RANK'] + '.json')).write_text(json.dumps(result))
"""


def packet(tmp_path: Path) -> tuple[Path, Path]:
    workspace = tmp_path / "packet"
    workspace.mkdir()
    (workspace / "task").mkdir()
    (workspace / "task/__init__.py").write_text("")
    (workspace / "task/engine.py").write_text("ROUTE = 'engine'\n" + PROBE)
    (workspace / "diagnostic.py").write_text("ROUTE = 'diagnostic'\n" + PROBE)
    (workspace / "probe_helper.py").write_text("VALUE = 'workspace-imported'\n")
    return workspace, tmp_path / "output"


def execute(
    workspace: Path,
    output: Path,
    rank: str | None = "0",
    script: str | None = None,
) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    environment.update(
        CHECKPOINT_WORKSPACE=str(workspace),
        EXPERIMENT_OUTPUT=str(output),
        TRAINCHECK_OUTPUT_DIR="/stale/shared/trace",
        PYTHONDONTWRITEBYTECODE="1",
    )
    environment.pop("RANK", None)
    environment.pop("EXPERIMENT_SCRIPT", None)
    if rank is not None:
        environment["RANK"] = rank
    if script is not None:
        environment["EXPERIMENT_SCRIPT"] = script
    return subprocess.run(
        [sys.executable, str(Path(run_worker.__file__).resolve())],
        cwd=workspace.parent,
        env=environment,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )


@pytest.mark.parametrize(
    "script,route", [(None, "engine"), ("diagnostic.py", "diagnostic")]
)
def test_routes_set_unique_rank_sink_and_workspace_before_import(
    tmp_path: Path, script: str | None, route: str
) -> None:
    workspace, output = packet(tmp_path)
    for rank in ("0", "1"):
        result = execute(workspace, output, rank=rank, script=script)
        assert result.returncode == 0, result.stderr
        seen: object = json.loads((output / f"seen-{rank}.json").read_text())
        assert seen == {
            "trace": str(output / "traces" / f"rank{rank}"),
            "cwd": str(workspace),
            "rank": rank,
            "helper": "workspace-imported",
            "route": route,
        }


@pytest.mark.parametrize("rank", [None, "-1", "not-a-rank"])
def test_missing_or_invalid_rank_fails_before_import(
    tmp_path: Path, rank: str | None
) -> None:
    workspace, output = packet(tmp_path)
    result = execute(workspace, output, rank=rank)
    assert result.returncode != 0
    assert not list(output.glob("seen-*.json"))
    assert not (output / "traces").exists()


@pytest.mark.parametrize("route", ["parent", "symlink", "extension", "directory"])
def test_diagnostic_must_resolve_to_a_packet_python_file(
    tmp_path: Path, route: str
) -> None:
    workspace, output = packet(tmp_path)
    outside = tmp_path / "outside.py"
    outside.write_text("raise RuntimeError('escaped-code-executed')\n")
    if route == "parent":
        script = "../outside.py"
    elif route == "symlink":
        (workspace / "link.py").symlink_to(outside)
        script = "link.py"
    elif route == "extension":
        (workspace / "diagnostic.txt").write_text(
            "raise RuntimeError('escaped-code-executed')\n"
        )
        script = "diagnostic.txt"
    else:
        (workspace / "directory.py").mkdir()
        (workspace / "directory.py/__main__.py").write_text(
            "raise RuntimeError('escaped-code-executed')\n"
        )
        script = "directory.py"
    result = execute(workspace, output, script=script)
    assert result.returncode != 0
    assert "Diagnostic script must be a Python file in the packet" in result.stderr
    assert "escaped-code-executed" not in result.stderr
    assert not list(output.glob("seen-*.json"))


def test_existing_rank_trace_directory_cannot_be_reused(tmp_path: Path) -> None:
    workspace, output = packet(tmp_path)
    (output / "traces/rank0").mkdir(parents=True)
    result = execute(workspace, output)
    assert result.returncode != 0
    assert "FileExistsError" in result.stderr
    assert not list(output.glob("seen-*.json"))
