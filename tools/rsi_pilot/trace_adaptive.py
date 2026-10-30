"""Trace only training updates in the pinned RSI adaptive-reasoning workload.

Run in a fresh interpreter. The upstream trainer is imported unchanged; a line
hook gates API tracing from optimizer.zero_grad through the optimizer update.
Initialization, data sampling, evaluation, and evidence capture stay untraced.
This CPU development adapter does not produce official benchmark receipts.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import json
import os
import pickle
import random
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import CodeType, FrameType, FunctionType
from typing import Literal, TypedDict

import numpy as np
import torch

Mode = Literal["plain", "reference", "selective"]
TRAIN_SHA256 = "809b62dfe68b4e9feb8bc38f24c85aad93df1f58ba0e9967f42e6560d8a6c08f"


@dataclass(frozen=True)
class WindowLines:
    main: int
    start: int
    stop: int


def locate_window(path: Path, expected_sha256: str) -> WindowLines:
    """Reject source drift and ambiguous/moved anchors instead of guessing."""
    source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != expected_sha256:
        raise ValueError("Upstream train.py differs from the pinned source hash")
    tree = ast.parse(source, filename=str(path))
    mains = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    ]
    if len(mains) != 1:
        raise ValueError("Expected exactly one upstream main function")
    main = mains[0]
    starts = [
        node
        for node in ast.walk(main)
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and isinstance(node.value.func.value, ast.Name)
        and node.value.func.value.id == "optimizer"
        and node.value.func.attr == "zero_grad"
    ]
    stops = [
        node
        for node in ast.walk(main)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "completed"
        and isinstance(node.value, ast.Name)
        and node.value.id == "step"
    ]
    if len(starts) != 1 or len(stops) != 1:
        raise ValueError("Expected unique zero_grad and completed = step anchors")
    start, stop = starts[0], stops[0]
    loops = [
        node
        for node in ast.walk(main)
        if isinstance(node, ast.For)
        and start in tuple(ast.walk(node))
        and stop in tuple(ast.walk(node))
    ]
    if len(loops) != 1 or start.lineno >= stop.lineno:
        raise ValueError("Training anchors must belong to one ordered for loop")
    return WindowLines(main.lineno, start.lineno, stop.lineno)


def tensor_evidence(value: torch.Tensor) -> dict[str, object]:
    if value.device.type != "cpu" or value.layout != torch.strided:
        raise ValueError("This pilot snapshots only strided CPU tensors")
    raw = value.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    return {
        "dtype": str(value.dtype),
        "shape": list(value.shape),
        "requires_grad": value.requires_grad,
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def state_value(value: object) -> object:
    """Snapshot optimizer containers without retaining live mutable references."""
    if isinstance(value, torch.Tensor):
        return {"kind": "tensor", "value": tensor_evidence(value)}
    if value is None or type(value) in (bool, int, float, str):
        return value
    if isinstance(value, (list, tuple)):
        return {
            "kind": "tuple" if isinstance(value, tuple) else "list",
            "items": [state_value(item) for item in value],
        }
    if isinstance(value, dict):
        items: list[list[object]] = []
        for key, item in value.items():
            if type(key) not in (int, str):
                raise TypeError("Optimizer state keys must be integers or strings")
            items.append([key, state_value(item)])
        return {"kind": "dict", "items": items}
    raise TypeError(f"Unsupported optimizer state value: {type(value).__name__}")


def state_snapshot(frame: FrameType) -> dict[str, object]:
    model: object = frame.f_locals["model"]
    optimizer: object = frame.f_locals["optimizer"]
    rng: object = frame.f_locals["rng"]
    if not isinstance(model, torch.nn.Module):
        raise TypeError("Upstream model must be a torch.nn.Module")
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("Upstream optimizer must be a torch.optim.Optimizer")
    if not isinstance(rng, random.Random):
        raise TypeError("Upstream rng must be random.Random")
    parameter_names = {id(value): name for name, value in model.named_parameters()}
    optimizer_parameters: list[list[str]] = []
    for group in optimizer.param_groups:
        names: list[str] = []
        for parameter in group["params"]:
            if id(parameter) not in parameter_names:
                raise ValueError("Optimizer references a tensor outside the model")
            names.append(parameter_names[id(parameter)])
        optimizer_parameters.append(names)
    result: dict[str, object] = {
        "parameters": {
            name: tensor_evidence(value) for name, value in model.named_parameters()
        },
        "gradients": {
            name: None if value.grad is None else tensor_evidence(value.grad)
            for name, value in model.named_parameters()
        },
        "buffers": {
            name: tensor_evidence(value) for name, value in model.named_buffers()
        },
        "optimizer": state_value(optimizer.state_dict()),
        "optimizer_parameter_names": optimizer_parameters,
        "custom_optimizer_states": state_value(frame.f_locals["optimizer_states"]),
        "torch_rng": tensor_evidence(torch.get_rng_state()),
        # These are trusted local RNG snapshots, serialized only for a byte digest.
        "python_rng": hashlib.sha256(pickle.dumps(random.getstate())).hexdigest(),
        "numpy_rng": hashlib.sha256(pickle.dumps(np.random.get_state())).hexdigest(),
        "sampler_rng": hashlib.sha256(pickle.dumps(rng.getstate())).hexdigest(),
    }
    if "sampler" in frame.f_locals:
        sampler: object = frame.f_locals["sampler"]
        if not isinstance(sampler, torch.Generator):
            raise TypeError("Upstream sampler must be a torch.Generator")
        result["pool_rng"] = tensor_evidence(sampler.get_state())
    return result


class StepRecord(TypedDict):
    step: int
    seconds: float
    before: dict[str, object]
    after: dict[str, object]
    loss: float


class StepWindow:
    """A trace callback which returns itself only for the original main frame."""

    def __init__(
        self,
        code: CodeType,
        lines: WindowLines,
        enable: Callable[[], None],
        disable: Callable[[], None],
        snapshot: Callable[[FrameType], dict[str, object]] = state_snapshot,
    ) -> None:
        self.code = code
        self.lines = lines
        self.enable = enable
        self.disable = disable
        self.snapshot = snapshot
        self.active = False
        self.records: list[StepRecord] = []
        self.before: dict[str, object] | None = None
        self.started: float | None = None
        self.step: int | None = None
        self.failed = False

    def __call__(self, frame: FrameType, event: str, arg: object) -> StepWindow | None:
        if frame.f_code is not self.code:
            return None
        if event in ("exception", "return") and self.active:
            self.disable()
            self.active = False
            self.failed = True
        if event != "line":
            return self
        if frame.f_lineno == self.lines.start:
            if self.active or self.failed:
                raise RuntimeError("Training window was reentered or previously failed")
            step: object = frame.f_locals["step"]
            if not isinstance(step, int) or isinstance(step, bool) or step < 1:
                raise TypeError("Upstream step must be a positive integer")
            if step != len(self.records) + 1:
                raise ValueError("Training steps must be consecutive from one")
            self.step = step
            self.before = self.snapshot(frame)
            self.started = time.perf_counter()
            self.active = True
            self.enable()
        elif frame.f_lineno == self.lines.stop:
            if not self.active:
                raise RuntimeError("Training end reached without an active window")
            self.disable()
            assert self.started is not None
            elapsed = time.perf_counter() - self.started
            self.active = False
            assert self.before is not None and self.step is not None
            after = self.snapshot(frame)
            loss: object = frame.f_locals["loss"]
            if not isinstance(loss, torch.Tensor) or loss.ndim != 0:
                raise TypeError("Upstream loss must be a scalar tensor")
            self.records.append(
                {
                    "step": self.step,
                    "seconds": elapsed,
                    "before": self.before,
                    "after": after,
                    "loss": float(loss.detach()),
                }
            )
            self.before = None
        return self

    def close(self) -> None:
        self.disable()
        if self.active:
            self.failed = True
        self.active = False


@dataclass(frozen=True)
class Instrumentation:
    enable: Callable[[], None]
    disable: Callable[[], None]
    drain: Callable[[], None]
    instrumented_apis: int


def noop() -> None:
    pass


def install_instrumentation(
    mode: Mode, trace_dir: Path, invariants: Path | None
) -> Instrumentation:
    if mode == "plain":
        return Instrumentation(noop, noop, noop, 0)
    from traincheck.config import config
    from traincheck.config.config import InstrOpt
    from traincheck.instrumentor.caches import META_VARS
    from traincheck.instrumentor.control import start_step
    from traincheck.instrumentor.proxy_wrapper.proxy_config import tensor_dump_format

    config.DISABLE_WRAPPER = True
    config.INSTRUMENTATION_POLICY = {"interval": 1, "warm_up": 0}
    config.INSTR_OPTS = None
    config.MODEL_TRACKER_STYLE = None
    config.INSTR_OPTS_FILE = str(trace_dir / "instr_opts.json")
    if META_VARS["step"] != 0:
        raise RuntimeError("Trace adapter requires a fresh interpreter")
    for key in ("dump_tensor_hash", "dump_tensor_stats", "dump_tensor_full"):
        tensor_dump_format[key] = False
    if mode == "selective":
        from traincheck.collect_trace import (
            get_model_tracker_instr_opts,
            get_per_func_instr_opts,
        )
        from traincheck.invariant.base_cls import read_inv_file

        assert invariants is not None
        rules = read_inv_file(str(invariants))
        if not rules:
            raise ValueError("Selective mode requires nonempty invariants")
        if get_model_tracker_instr_opts(rules, "subclass") is not None:
            raise ValueError("API-only pilot cannot satisfy variable-tracker rules")
        options = InstrOpt(get_per_func_instr_opts(rules), None, True)
        Path(config.INSTR_OPTS_FILE).write_text(options.to_json() + "\n")
        loaded: object = config.load_instr_opts()
        if not isinstance(loaded, InstrOpt) or loaded.model_tracker_style is not None:
            raise TypeError("Saved API-only instrumentation policy failed validation")
    from traincheck.instrumentor.tracer import Instrumentor

    count = Instrumentor(
        torch,
        scan_proxy_in_args=False,
        use_full_instr=False,
        API_dump_stack_trace=False,
        sampling_interval=1,
        warm_up_steps=0,
    ).instrument()

    def enable() -> None:
        start_step()

    def disable() -> None:
        config.DISABLE_WRAPPER = True

    def drain() -> None:
        from traincheck.instrumentor import dumper

        for name in ("api_trace_buffer", "var_trace_buffer"):
            # Absence means no event was emitted for that stream.
            buffer: object = getattr(dumper.THREAD_DATA, name, None)
            if buffer is not None:
                if not isinstance(buffer, dumper.TraceBuffer):
                    raise TypeError(f"Unexpected trace buffer: {name}")
                buffer._flush()
                buffer.queue.join()

    return Instrumentation(enable, disable, drain, count)


def run(
    workspace: Path,
    output: Path,
    mode: Mode,
    invariants: Path | None,
    training_args: Sequence[str],
) -> None:
    if mode not in ("plain", "reference", "selective"):
        raise ValueError("Unknown tracing mode")
    if (mode == "selective") != (invariants is not None):
        raise ValueError("--invariants is required exactly for selective mode")
    if sys.gettrace() is not None:
        raise RuntimeError("Cannot replace an existing Python trace hook")
    if any(arg.split("=", 1)[0] in ("--output", "--device") for arg in training_args):
        raise ValueError("Adapter owns upstream --output and --device")
    workspace, output = workspace.resolve(), output.resolve()
    source = workspace / "latent_reasoning" / "train.py"
    lines = locate_window(source, TRAIN_SHA256)
    output.mkdir(parents=True, exist_ok=False)
    trace_dir = output / "traces"
    trace_dir.mkdir()
    os.environ["TRAINCHECK_OUTPUT_DIR"] = str(trace_dir)
    # research.py captures Torch callables in its interpreter registry at import.
    # Install wrappers first so those calls participate in reference/selective traces.
    instrumentation = install_instrumentation(mode, trace_dir, invariants)
    sys.path.insert(0, str(workspace))
    module = importlib.import_module("latent_reasoning.train")
    entry: object = getattr(module, "main")
    if not isinstance(entry, FunctionType):
        raise TypeError("Upstream main must be a Python function")
    if (
        Path(entry.__code__.co_filename).resolve() != source
        or entry.__code__.co_firstlineno != lines.main
    ):
        raise ValueError("Imported trainer does not match the verified source")
    window = StepWindow(
        entry.__code__, lines, instrumentation.enable, instrumentation.disable
    )
    manifest: dict[str, object] = {
        "schema_version": 1,
        "mode": mode,
        "source_sha256": TRAIN_SHA256,
        "adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "invariants_sha256": (
            None
            if invariants is None
            else hashlib.sha256(invariants.read_bytes()).hexdigest()
        ),
        "policy_sha256": (
            hashlib.sha256((trace_dir / "instr_opts.json").read_bytes()).hexdigest()
            if mode == "selective"
            else None
        ),
        "window_start_line": lines.start,
        "window_stop_line": lines.stop,
        "model_tracker": None,
        "tensor_value_capture": False,
        "instrumented_apis": instrumentation.instrumented_apis,
        "training_args": list(training_args),
        "scope": "zero_grad through update; CPU; unsigned reduced-compute pilot",
        "status": "running",
    }
    old_argv, old_cwd = sys.argv, Path.cwd()
    sys.argv = [
        str(source),
        "--output",
        str(output / "training"),
        "--device",
        "cpu",
        *training_args,
    ]
    try:
        os.chdir(workspace)
        sys.settrace(window)
        entry()
        if window.active or window.failed or not window.records:
            raise RuntimeError("Training did not complete a valid traced update")
        manifest["status"] = "complete"
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["error_type"] = type(error).__name__
        raise
    finally:
        sys.settrace(None)
        window.close()
        sys.argv = old_argv
        os.chdir(old_cwd)
        instrumentation.drain()
        manifest["steps"] = window.records
        (output / "trace_manifest.json").write_text(
            json.dumps(manifest, indent=2, allow_nan=False) + "\n"
        )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("plain", "reference", "selective"), required=True
    )
    parser.add_argument("--invariants", type=Path)
    parser.add_argument("training_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    mode: Mode
    if args.mode == "plain":
        mode = "plain"
    elif args.mode == "reference":
        mode = "reference"
    else:
        mode = "selective"
    forwarded: list[str] = args.training_args
    if forwarded[:1] == ["--"]:
        forwarded = forwarded[1:]
    run(args.workspace, args.output, mode, args.invariants, forwarded)


if __name__ == "__main__":
    main()
