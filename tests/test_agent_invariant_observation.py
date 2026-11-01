"""Native scalar observations preserve the operation they describe."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tools.distributed_probe.workload import Options, parse_options, run

ROOT = Path(__file__).resolve().parents[1]
FUNCTION = "tools.agent_invariants.observation.apply_loss_scale"
SCALES = (0.0625, 0.1, 0.0, -0.25)

WORKER = """
import json
import sys
from pathlib import Path

import torch

from tools.agent_invariants.observation import wrapped_loss_scale
from tools.distributed_probe.tracing import start_tracing

torch.set_num_threads(1)
torch.manual_seed(733)
model = torch.nn.Linear(2, 1)
session = start_tracing('api', Path(sys.argv[1]), model)
operation = wrapped_loss_scale()
for step, scale in enumerate((0.0625, 0.1, 0.0, -0.25), start=1):
    with session.suppress():
        source = torch.tensor(3.75, requires_grad=True)
        reference = source.detach().clone().requires_grad_(True)
        expected = reference * scale
        expected.backward()
        rng_before = torch.get_rng_state().clone()
    with session.step(step):
        actual = operation(
            source, rank=0, world_size=2, step=step, microbatch=0,
            accumulation=2, local_tokens=10, global_tokens=32,
            applied_scale=scale,
        )
        assert isinstance(actual, torch.Tensor)
        actual.backward()
        with session.suppress():
            assert torch.equal(actual, expected)
            assert source.grad is not None and reference.grad is not None
            assert torch.equal(source.grad, reference.grad)
            assert torch.equal(torch.get_rng_state(), rng_before)

with session.step(5):
    with session.suppress():
        skipped = operation(
            torch.tensor(2.0), rank=0, world_size=2, step=5, microbatch=0,
            accumulation=2, local_tokens=10, global_tokens=32,
            applied_scale=0.0625,
        )
        assert isinstance(skipped, torch.Tensor) and skipped.item() == 0.125

try:
    with session.step(6):
        # Exercise malformed runtime input across an untyped external boundary.
        malformed_operation: object = operation
        assert callable(malformed_operation)
        malformed_operation(
            3.75, rank=0, world_size=2, step=6, microbatch=0,
            accumulation=2, local_tokens=10, global_tokens=32,
            applied_scale=0.0625,
        )
except TypeError as error:
    assert str(error) == 'Loss multiplication must return a Tensor'
else:
    raise AssertionError('Operation exception was swallowed')

report = session.close()
assert report['completed_trace_steps'] == [1, 2, 3, 4, 5]
Path(sys.argv[2]).write_text(json.dumps(report))
"""

LIFETIME_WORKER = """
import importlib.util
import sys
import weakref
from pathlib import Path

import torch

from tools.distributed_probe import tracing

spec = importlib.util.spec_from_file_location(
    'tools.distributed_probe.workload', sys.argv[1]
)
assert spec is not None and spec.loader is not None
workload = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = workload
spec.loader.exec_module(workload)

pending: list[weakref.ReferenceType[torch.Tensor]] = []
active = False
checked = 0
original_token_loss = workload.token_loss
original_start_tracing = tracing.start_tracing


def tracked_token_loss(logits: torch.Tensor, batch: workload.Batch) -> torch.Tensor:
    value = original_token_loss(logits, batch)
    if active:
        pending.append(weakref.ref(value))
    return value


def install_lifetime_check(
    mode: tracing.Mode, output: Path, model: torch.nn.Module
) -> tracing.TraceSession:
    global active
    session = original_start_tracing(mode, output, model)
    original_backward = torch.Tensor.backward

    def checked_backward(
        tensor: torch.Tensor, *args: object, **kwargs: object
    ) -> None:
        global checked
        assert len(pending) == 1, 'Expected one unscaled training loss per backward'
        reference = pending.pop()
        assert reference() is None, 'Unscaled loss remains live at backward entry'
        checked += 1
        original_backward(tensor, *args, **kwargs)

    torch.Tensor.backward = checked_backward
    active = True
    return session


workload.token_loss = tracked_token_loss
tracing.start_tracing = install_lifetime_check
mode = sys.argv[2]
options = workload.Options(
    Path(sys.argv[3]), None, 2, 2, 'fp32', 'none', mode, 'cpu', 2,
    observe_normalization=mode != 'plain',
)
workload.run(options)
assert checked == 4 and not pending
print('Four temporary loss lifetimes verified')
"""


def mapping(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError("Expected a JSON object")
    result: dict[str, object] = {}
    for key, child in value.items():
        if not isinstance(key, str):
            raise TypeError("Expected string keys")
        result[key] = child
    return result


def integer(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError("Expected an integer, excluding bool")
    return value


def trace_records(root: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for path in sorted(root.glob("trace_API_*.log")):
        for line in path.read_text().splitlines():
            record = mapping(json.loads(line))
            if record["function"] == FUNCTION:
                records.append(record)
    return records


def test_actual_native_records_preserve_scalars_gradients_rng_and_exceptions(
    tmp_path: Path,
) -> None:
    trace_root = tmp_path / "traces"
    report_path = tmp_path / "report.json"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [sys.executable, "-c", WORKER, str(trace_root), str(report_path)],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = mapping(json.loads(report_path.read_text()))
    assert report["completed_trace_steps"] == [1, 2, 3, 4, 5]
    records = trace_records(trace_root)
    assert len(records) == 10
    observed_steps = [
        integer(mapping(record["meta_vars"])["step"]) for record in records
    ]
    assert observed_steps == [1, 1, 2, 2, 3, 3, 4, 4, 6, 6]
    identifiers: set[tuple[int, int, str]] = set()
    for offset, step in enumerate((1, 2, 3, 4, 6)):
        pre, post = records[2 * offset : 2 * offset + 2]
        assert pre["type"] == "function_call (pre)"
        assert post["type"] == (
            "function_call (post) (exception)" if step == 6 else "function_call (post)"
        )
        assert integer(pre["time"]) <= integer(post["time"])
        for field in ("func_call_id", "thread_id", "process_id", "meta_vars"):
            assert pre[field] == post[field]
        call_id = pre["func_call_id"]
        assert isinstance(call_id, str)
        identity = (integer(pre["process_id"]), integer(pre["thread_id"]), call_id)
        assert identity not in identifiers
        identifiers.add(identity)
        assert pre["is_bound_method"] is False
        assert pre["obj_id"] is None
        assert pre["args"] == {}
        scalar = SCALES[step - 1] if step < 5 else SCALES[0]
        expected_kwargs: dict[str, object] = {
            "rank": {"int": 0},
            "world_size": {"int": 2},
            "step": {"int": step},
            "microbatch": {"int": 0},
            "accumulation": {"int": 2},
            "local_tokens": {"int": 10},
            "global_tokens": {"int": 32},
            "applied_scale": {"float": scalar},
        }
        assert pre["kwargs"] == expected_kwargs
        expected_post_fields = {
            "func_call_id",
            "thread_id",
            "process_id",
            "meta_vars",
            "type",
            "function",
            "is_bound_method",
            "obj_id",
            "time",
        }
        if step == 6:
            expected_post_fields.update(("exception", "exception_msg"))
            assert post["exception_msg"] == "Loss multiplication must return a Tensor"
        assert post.keys() == expected_post_fields


@pytest.mark.parametrize("mode", ["api", "subclass"])
def test_observation_option_requires_traced_mode(mode: str) -> None:
    options = parse_options(
        ["--output", "/unused", "--instrumentation", mode, "--observe-normalization"]
    )
    assert options.observe_normalization is True
    assert options.instrumentation == mode


def test_plain_observation_cli_is_rejected() -> None:
    with pytest.raises(SystemExit):
        parse_options(["--output", "/unused", "--observe-normalization"])


def test_plain_observation_programmatic_run_is_rejected_before_output(
    tmp_path: Path,
) -> None:
    output = tmp_path / "must-not-exist"
    options = Options(output, None, 1, 1, "fp32", "none", "plain", "cpu", 2, True)
    with pytest.raises(ValueError, match="require API or subclass tracing"):
        run(options)
    assert not output.exists()


@pytest.mark.parametrize("mode", ["plain", "api"])
def test_workload_releases_unscaled_loss_before_backward(
    tmp_path: Path, mode: str
) -> None:
    """Observe Python lifetime during actual training without keeping tensors alive."""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            LIFETIME_WORKER,
            str(ROOT / "tools/distributed_probe/workload.py"),
            mode,
            str(tmp_path / "run"),
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Four temporary loss lifetimes verified" in result.stdout
