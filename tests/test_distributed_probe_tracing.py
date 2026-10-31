"""The probe must delimit real events and preserve its application computation."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tools.distributed_probe.tracing import TraceSession

ROOT = Path(__file__).resolve().parents[1]

WORKER = """
import copy
import json
import sys
from pathlib import Path

import torch
from tools.distributed_probe.tracing import start_tracing

torch.set_num_threads(1)
torch.manual_seed(731)
model = torch.nn.Linear(2, 1)
reference = copy.deepcopy(model)
values = torch.tensor([[1., 2.], [3., 4.]])
ref_opt = torch.optim.SGD(reference.parameters(), lr=0.03)
ref_loss = reference(values).square().mean()
ref_loss.backward()
ref_opt.step()
identities = [id(p) for p in model.parameters()]
session = start_tracing(sys.argv[1], Path(sys.argv[2]), model)
assert identities == [id(p) for p in model.parameters()]
optimizer = torch.optim.SGD(model.parameters(), lr=0.03)
with session.suppress():
    torch.sin(values)
with session.step(4):
    torch.cos(values)
    with session.suppress():
        torch.sin(values)
    loss = model(values).square().mean()
    loss.backward()
    optimizer.step()
with session.suppress():
    assert torch.equal(loss, ref_loss)
    for key, value in model.state_dict().items():
        assert torch.equal(value, reference.state_dict()[key]), key
try:
    with session.step(5):
        with session.suppress():
            raise RuntimeError('expected application failure')
except RuntimeError as exc:
    assert str(exc) == 'expected application failure'
else:
    raise AssertionError('application failure was swallowed')
report = session.close()
assert report['completed_trace_steps'] == [4]
Path(sys.argv[3]).write_text(json.dumps(report))
"""


def json_object(path: Path) -> dict[str, object]:
    value: object = json.loads(path.read_text())
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise TypeError("Expected JSON object with string keys")
    return {key: item for key, item in value.items()}


def trace_records(root: Path, kind: str) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for path in sorted(root.glob(f"trace_{kind}_*.log")):
        for line in path.read_text().splitlines():
            value: object = json.loads(line)
            if not isinstance(value, dict) or not all(
                isinstance(key, str) for key in value
            ):
                raise TypeError("Expected trace record with string keys")
            records.append({key: item for key, item in value.items()})
    return records


@pytest.mark.parametrize("mode", ["api", "subclass"])
def test_actual_events_and_bookkeeping_suppression(tmp_path: Path, mode: str) -> None:
    output = tmp_path / "traces"
    report_path = tmp_path / "report.json"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [sys.executable, "-c", WORKER, mode, str(output), str(report_path)],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json_object(report_path)
    assert report["mode"] == mode
    assert report["scan_proxy_in_args"] is (mode == "subclass")
    assert report["completed_trace_steps"] == [4]
    assert report["optimizer_state_registered"] is False
    assert report["cross_rank_checks_executed"] is False
    assert report["trace_record_counts"] is None
    records = trace_records(output, "API")
    assert records
    functions = [record["function"] for record in records]
    assert any(isinstance(name, str) and name.endswith(".cos") for name in functions)
    assert not any(
        isinstance(name, str) and name.endswith(".sin") for name in functions
    )
    variables = trace_records(output, "VAR")
    if mode == "subclass":
        assert any("proxy_obj_names" in record for record in records)
        assert report["registered_parameter_names"] == ["model.weight", "model.bias"]
        assert {record["var_name"] for record in variables} == {
            "model.weight",
            "model.bias",
        }
        assert any(
            record["dump_loc"] == "distributed_probe.update.after"
            for record in variables
        )
    else:
        assert report["registered_parameter_names"] == []
        assert variables == []


def test_scope_contracts_and_failure_not_completed(tmp_path: Path) -> None:
    session = TraceSession("plain", tmp_path, 0, ())
    for bad_step in (0, -1, True):
        with pytest.raises(ValueError, match="positive integer"):
            with session.step(bad_step):
                pass
    with session.step(3):
        with pytest.raises(RuntimeError, match="overlap"):
            with session.step(4):
                pass
        with pytest.raises(RuntimeError, match="inside an update"):
            session.close()
    with pytest.raises(ValueError, match="consecutive"):
        with session.step(5):
            pass
    with pytest.raises(LookupError):
        with session.step(4):
            raise LookupError("application")
    assert session.close()["completed_trace_steps"] == [3]
    with pytest.raises(RuntimeError, match="already closed"):
        with session.suppress():
            pass
