"""One-process, one-GPU TorchTitan checking-overhead pilot.

Run with torchrun --standalone --nproc-per-node=1 --module
tools.checking_overhead.workload. Each invocation owns a fresh interpreter and
output directory; installing instrumentation is intentionally process-global.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import random
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypedDict

import numpy as np
import torch

import traincheck
from traincheck.config import config as tc_config

Mode = Literal["plain", "reference", "selective"]
TrackerStyle = Literal["subclass", "sampler"] | None
EXPECTED_PARAMETERS = 6_163_712


@dataclass(frozen=True)
class Options:
    mode: Mode
    output_dir: Path
    torchtitan_root: Path
    steps: int = 10
    seq_len: int = 256
    batch_size: int = 1
    seed: int = 42
    invariants: Path | None = None

    def __post_init__(self) -> None:
        if self.mode not in ("plain", "reference", "selective"):
            raise ValueError("mode must be plain, reference, or selective")
        for name in ("steps", "seq_len", "batch_size"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            not isinstance(self.seed, int)
            or isinstance(self.seed, bool)
            or not 0 <= self.seed < 2**32
        ):
            raise ValueError("seed must be an integer in [0, 2**32)")
        if (self.mode == "selective") != (self.invariants is not None):
            raise ValueError("--invariants is required exactly for selective mode")


class TensorEvidence(TypedDict):
    dtype: str
    shape: list[int]
    sha256: str


class ModelEvidence(TypedDict):
    parameters: dict[str, TensorEvidence]
    gradients: dict[str, TensorEvidence | None]
    buffers: dict[str, TensorEvidence]


class LossRecord(TypedDict):
    step: int
    average: float
    maximum: float
    gradient_norm: float


class StepRecord(TypedDict):
    step: int
    seconds: float


@dataclass(frozen=True)
class TrainerHandle:
    """Validated boundary to the pinned TorchTitan trainer, imported lazily."""

    trainer: object
    model: torch.nn.Module
    optimizers: tuple[torch.optim.Optimizer, ...]
    metrics: object
    train: Callable[..., object]
    train_step: Callable[..., object]
    metrics_log: Callable[..., object]
    close: Callable[..., object]

    @classmethod
    def from_trainer(cls, trainer: object) -> TrainerHandle:
        parts: object = getattr(trainer, "model_parts")
        if not isinstance(parts, list) or len(parts) != 1:
            raise ValueError("This single-GPU pilot requires exactly one model part")
        model: object = parts[0]
        if not isinstance(model, torch.nn.Module):
            raise TypeError("model_parts[0] must be a torch.nn.Module")
        container: object = getattr(trainer, "optimizers")
        optimizers: object = getattr(container, "optimizers")
        if not isinstance(optimizers, list) or not optimizers:
            raise TypeError("TorchTitan optimizers.optimizers must be a nonempty list")
        checked: list[torch.optim.Optimizer] = []
        for optimizer in optimizers:
            if not isinstance(optimizer, torch.optim.Optimizer):
                raise TypeError("Every optimizer must be a torch.optim.Optimizer")
            checked.append(optimizer)
        metrics: object = getattr(trainer, "metrics_processor")
        return cls(
            trainer,
            model,
            tuple(checked),
            metrics,
            required_callable(trainer, "train"),
            required_callable(trainer, "train_step"),
            required_callable(metrics, "log"),
            required_callable(trainer, "close"),
        )


def required_callable(owner: object, name: str) -> Callable[..., object]:
    value: object = getattr(owner, name)
    if not callable(value):
        raise TypeError(f"{name} must be callable")
    return value


def tensor_evidence(tensor: torch.Tensor) -> TensorEvidence:
    """Exact storage bytes, retaining dtype and shape; never a sampled hash."""
    raw = tensor.detach().contiguous().cpu().reshape(-1).view(torch.uint8)
    return {
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "sha256": hashlib.sha256(raw.numpy().tobytes()).hexdigest(),
    }


def model_evidence(model: torch.nn.Module) -> ModelEvidence:
    return {
        "parameters": {
            name: tensor_evidence(parameter)
            for name, parameter in model.named_parameters()
        },
        "gradients": {
            name: None if parameter.grad is None else tensor_evidence(parameter.grad)
            for name, parameter in model.named_parameters()
        },
        "buffers": {
            name: tensor_evidence(buffer) for name, buffer in model.named_buffers()
        },
    }


def rng_evidence() -> dict[str, object]:
    numpy_state = np.random.get_state(legacy=True)
    if not isinstance(numpy_state, tuple) or len(numpy_state) != 5:
        raise TypeError("NumPy legacy RNG state must be a five-field tuple")
    return {
        "python_sha256": hashlib.sha256(repr(random.getstate()).encode()).hexdigest(),
        "numpy": {
            "algorithm": numpy_state[0],
            "state_sha256": hashlib.sha256(numpy_state[1].tobytes()).hexdigest(),
            "position": numpy_state[2],
            "has_gaussian": numpy_state[3],
            "cached_gaussian": numpy_state[4],
        },
        "torch_cpu": tensor_evidence(torch.get_rng_state()),
        "torch_cuda": [
            tensor_evidence(state) for state in torch.cuda.get_rng_state_all()
        ],
    }


def optimizer_parameters(
    optimizers: tuple[torch.optim.Optimizer, ...],
) -> tuple[torch.Tensor, ...]:
    parameters: list[torch.Tensor] = []
    for optimizer in optimizers:
        for group in optimizer.param_groups:
            values: object = group["params"]
            if not isinstance(values, list):
                raise TypeError("Optimizer param group params must be a list")
            for value in values:
                if not isinstance(value, torch.Tensor):
                    raise TypeError("Optimizer parameters must be tensors")
                parameters.append(value)
    return tuple(parameters)


def attach_model_tracker(handle: TrainerHandle, style: TrackerStyle) -> object:
    """Use the ordinary tracker implementation without adding sampling points."""
    before = tuple(handle.model.parameters())
    optimizer_before = optimizer_parameters(handle.optimizers)
    if not all(
        any(value is parameter for value in before) for parameter in optimizer_before
    ):
        raise ValueError("Optimizer has a parameter outside the single model part")
    if style == "subclass":
        # Custom subclasses (including DTensor parameters) take an out-of-place
        # path in proxy_parameter. Reject that unsupported boundary before mutation.
        if any(type(parameter) is not torch.nn.Parameter for parameter in before):
            types = sorted({str(type(parameter)) for parameter in before})
            raise TypeError(
                f"Subclass pilot requires exact nn.Parameter objects: {types}"
            )
        from traincheck.instrumentor.proxy_wrapper.subclass import proxy_parameter

        proxy_parameter(handle.model, parent_name="model")
        tracker: object = None
    elif style == "sampler":
        from traincheck.instrumentor.tracer import VarSampler

        sampler = VarSampler(handle.model, "model")
        for optimizer in handle.optimizers:
            sampler.register_hook(optimizer)
        tracker = sampler
    elif style is None:
        tracker = None
    else:
        raise ValueError(f"Unsupported model tracker style: {style}")
    after = tuple(handle.model.parameters())
    optimizer_after = optimizer_parameters(handle.optimizers)
    if len(before) != len(after) or any(a is not b for a, b in zip(before, after)):
        raise RuntimeError("Model tracking changed parameter identity")
    if len(optimizer_before) != len(optimizer_after) or any(
        a is not b for a, b in zip(optimizer_before, optimizer_after)
    ):
        raise RuntimeError("Model tracking changed optimizer parameter references")
    return tracker


def install_instrumentation(options: Options) -> TrackerStyle:
    from traincheck.config.config import InstrOpt
    from traincheck.instrumentor.caches import META_VARS

    tc_config.DISABLE_WRAPPER = True
    tc_config.INSTRUMENTATION_POLICY = {"interval": 1, "warm_up": 0}
    if META_VARS["step"] != 0:
        raise RuntimeError(
            "Workload requires a fresh interpreter with TrainCheck step zero"
        )
    tc_config.INSTR_OPTS = None
    # Never load an unrelated instr_opts.json from the working directory.
    tc_config.INSTR_OPTS_FILE = str(options.output_dir / "traces" / "instr_opts.json")
    style: TrackerStyle = None
    if options.mode == "plain":
        tc_config.MODEL_TRACKER_STYLE = None
        return style
    if options.mode == "reference":
        style = "subclass"
    else:
        from traincheck.collect_trace import (
            get_model_tracker_instr_opts,
            get_per_func_instr_opts,
        )
        from traincheck.invariant.base_cls import read_inv_file

        assert options.invariants is not None
        invariants = read_inv_file(str(options.invariants))
        if not invariants:
            raise ValueError("Selective instrumentation requires nonempty invariants")
        selected = get_model_tracker_instr_opts(invariants, "subclass")
        if selected not in (None, "subclass", "sampler"):
            raise ValueError(f"Unsupported selected tracker style: {selected}")
        # Equality checks above validate content; branches narrow the literal type.
        style = (
            "subclass"
            if selected == "subclass"
            else "sampler" if selected == "sampler" else None
        )
        instr_opts = InstrOpt(
            get_per_func_instr_opts(invariants), style, disable_proxy_dumping=True
        )
        # The legacy config global is inferred as None by its initializer,
        # although Instrumentor explicitly accepts an InstrOpt in this slot.
        setattr(tc_config, "INSTR_OPTS", instr_opts)
        Path(tc_config.INSTR_OPTS_FILE).write_text(instr_opts.to_json() + "\n")
    tc_config.MODEL_TRACKER_STYLE = style
    from traincheck.instrumentor.tracer import Instrumentor

    Instrumentor(
        torch,
        scan_proxy_in_args=style is not None,
        use_full_instr=False,
        API_dump_stack_trace=False,
        sampling_interval=1,
        warm_up_steps=0,
    ).instrument()
    if style == "subclass":
        install_standard_observers()
    return style


def install_standard_observers() -> None:
    from traincheck.instrumentor.proxy_wrapper.proxy_config import auto_observer_config
    from traincheck.static_analyzer.graph_generator.call_graph_parser import (
        add_observer_given_call_graph,
    )

    if auto_observer_config["enable_auto_observer"] is not True:
        raise ValueError("Reference pilot requires standard automatic model observers")
    graph_dir = Path(traincheck.__file__).parent / "static_analyzer" / "func_level"
    graphs = sorted(graph_dir.glob("*.log"))
    if not graphs:
        raise FileNotFoundError(f"No standard observer call graphs in {graph_dir}")
    for graph in graphs:
        add_observer_given_call_graph(
            str(graph),
            depth=auto_observer_config["enable_auto_observer_depth"],
            observe_up_to_depth=auto_observer_config["observe_up_to_depth"],
            neglect_hidden_func=auto_observer_config["neglect_hidden_func"],
            neglect_hidden_module=auto_observer_config["neglect_hidden_module"],
            observe_then_unproxy=auto_observer_config["observe_then_unproxy"],
        )


def run_training(
    handle: TrainerHandle,
    steps: int,
    instrumented: bool,
    synchronize: Callable[[], None],
) -> tuple[list[StepRecord], list[LossRecord]]:
    from traincheck.instrumentor.control import start_step

    timings: list[StepRecord] = []
    losses: list[LossRecord] = []

    def capture_metrics(*args: object, **kwargs: object) -> object:
        if len(args) != 4:
            raise TypeError("Pinned metrics.log must receive four positional arguments")
        step, average, maximum, gradient_norm = args
        if not isinstance(step, int) or isinstance(step, bool):
            raise TypeError("Metrics step must be an integer")
        if (
            not isinstance(average, float)
            or not isinstance(maximum, float)
            or not isinstance(gradient_norm, float)
        ):
            raise TypeError(
                "Pinned metrics loss and gradient norm must be Python floats"
            )
        result = handle.metrics_log(*args, **kwargs)
        losses.append(
            {
                "step": step,
                "average": average,
                "maximum": maximum,
                "gradient_norm": gradient_norm,
            }
        )
        return result

    def measured_step(*args: object, **kwargs: object) -> object:
        step: object = getattr(handle.trainer, "step")
        if (
            not isinstance(step, int)
            or isinstance(step, bool)
            or step != len(timings) + 1
        ):
            raise ValueError(
                "Expected consecutive native train_step calls starting at one"
            )
        synchronize()
        started = time.perf_counter()
        try:
            if instrumented:
                start_step()
            result = handle.train_step(*args, **kwargs)
        finally:
            tc_config.DISABLE_WRAPPER = True
            synchronize()
        timings.append({"step": step, "seconds": time.perf_counter() - started})
        return result

    tc_config.DISABLE_WRAPPER = True
    setattr(handle.trainer, "train_step", measured_step)
    setattr(handle.metrics, "log", capture_metrics)
    try:
        handle.train()
    finally:
        tc_config.DISABLE_WRAPPER = True
        setattr(handle.trainer, "train_step", handle.train_step)
        setattr(handle.metrics, "log", handle.metrics_log)
    if len(timings) != steps or [row["step"] for row in losses] != list(
        range(1, steps + 1)
    ):
        raise RuntimeError(
            "Native training did not produce exactly one timing and loss per requested step"
        )
    return timings, losses


def drain_trace_queues() -> float:
    """Flush producer buffers and wait for writes; file closure occurs at exit.

    This workload traces only the main Python training thread. TrainCheck's
    production writer monitor owns shutdown; joining it here would deadlock.
    """
    from traincheck.instrumentor import dumper

    started = time.perf_counter()
    for name in ("api_trace_buffer", "var_trace_buffer"):
        # Absence means that stream produced no events; it is not an error.
        buffer: object = getattr(dumper.THREAD_DATA, name, None)
        if buffer is not None:
            if not isinstance(buffer, dumper.TraceBuffer):
                raise TypeError(f"Unexpected trace buffer: {name}")
            buffer._flush()
            buffer.queue.join()
    return time.perf_counter() - started


def configure_trainer(options: Options) -> tuple[TrainerHandle, dict[str, object], str]:
    registry = importlib.import_module("torchtitan.models.llama3.config_registry")
    logging_module = importlib.import_module("torchtitan.tools.logging")
    required_callable(logging_module, "init_logger")()
    config = required_callable(registry, "llama3_debugmodel")()
    overrides: dict[str, object] = {
        "training.steps": options.steps,
        "training.seq_len": options.seq_len,
        "training.local_batch_size": options.batch_size,
        "debug.seed": options.seed,
        "checkpoint.enable": False,
        "validator.enable": False,
        "compile.enable": False,
        "profiler.enable_profiling": False,
        "metrics.log_freq": 1,
        "metrics.enable_tensorboard": False,
        "dump_folder": str(options.output_dir / "titan"),
    }
    for key, value in overrides.items():
        owner = config
        components = key.split(".")
        for component in components[:-1]:
            owner = getattr(owner, component)
        # Required pinned fields must exist; missing fields cannot become overrides.
        getattr(owner, components[-1])
        setattr(owner, components[-1], value)
        if getattr(owner, components[-1]) != value:
            raise ValueError(f"TorchTitan did not retain requested override {key}")
    representation = repr(config)
    return (
        TrainerHandle.from_trainer(required_callable(config, "build")()),
        overrides,
        representation,
    )


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def main(argv: Sequence[str] | None = None) -> None:
    options = parse_options(argv)
    if os.environ["WORLD_SIZE"] != "1" or os.environ["RANK"] != "0":
        raise ValueError("Run this pilot under torchrun with exactly one rank")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Pilot requires exactly one visible CUDA GPU")
    if not options.torchtitan_root.is_dir():
        raise FileNotFoundError(options.torchtitan_root)
    options.output_dir.mkdir(parents=True, exist_ok=False)
    (options.output_dir / "traces").mkdir()
    os.environ["TRAINCHECK_OUTPUT_DIR"] = str(options.output_dir / "traces")
    os.chdir(options.torchtitan_root)
    sys.path.insert(0, str(options.torchtitan_root))
    setup_started = time.perf_counter()
    tc_config.DISABLE_WRAPPER = True
    random.seed(options.seed)
    np.random.seed(options.seed)
    torch.manual_seed(options.seed)
    torch.cuda.manual_seed_all(options.seed)
    style = install_instrumentation(options)
    handle, overrides, representation = configure_trainer(options)
    try:
        parameter_count = sum(
            parameter.numel() for parameter in handle.model.parameters()
        )
        if parameter_count != EXPECTED_PARAMETERS:
            raise ValueError(
                f"Expected {EXPECTED_PARAMETERS} parameters, found {parameter_count}"
            )
        torch.cuda.synchronize()
        initial = model_evidence(handle.model)
        initial_rng = rng_evidence()
        parameter_types = sorted(
            {str(type(parameter)) for parameter in handle.model.parameters()}
        )
        tracker = attach_model_tracker(handle, style)
        if initial != model_evidence(handle.model) or initial_rng != rng_evidence():
            raise RuntimeError(
                "Model tracker installation changed application state or RNG"
            )
        setup_seconds = time.perf_counter() - setup_started
        write_json(
            options.output_dir / "initial.json", {"model": initial, "rng": initial_rng}
        )
        timings, losses = run_training(
            handle, options.steps, options.mode != "plain", torch.cuda.synchronize
        )
        torch.cuda.synchronize()
        final = model_evidence(handle.model)
        final_rng = rng_evidence()
        drain_seconds = drain_trace_queues()
        write_json(
            options.output_dir / "result.json",
            {
                "schema_version": 1,
                "status": "completed",
                "mode": options.mode,
                "model_config": "llama3_debugmodel",
                "parameter_count": parameter_count,
                "parameter_types_before_tracking": parameter_types,
                "selected_model_tracker": style,
                "sampler_initial_setup_snapshot": style == "sampler",
                "ordinary_full_collection": {
                    "use_full_instr": False,
                    "sampling_interval": 1,
                    "warm_up_steps": 0,
                },
                "seed_policy": "Seed Python, NumPy, Torch CPU, and every visible CUDA generator before instrumentation/configure; retain TorchTitan debug.seed override during trainer initialization.",
                "overrides": overrides,
                "effective_config_repr": representation,
                "invariants_sha256": (
                    None
                    if options.invariants is None
                    else hashlib.sha256(options.invariants.read_bytes()).hexdigest()
                ),
                "python_version": sys.version,
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(),
                "initial": {"model": initial, "rng": initial_rng},
                "final": {"model": final, "rng": final_rng},
                "losses": losses,
                "steps": timings,
                "training_seconds": sum(row["seconds"] for row in timings),
                "setup_seconds": setup_seconds,
                "trace_queue_drain_seconds": drain_seconds,
                "timing_scope": "Synchronized native train_step including start_step; excludes initialization, state evidence, outer train-loop housekeeping, queue drain, and process shutdown/file closure.",
            },
        )
        # Keep native sampler and its hook alive until all training has completed.
        del tracker
    finally:
        tc_config.DISABLE_WRAPPER = True
        handle.close()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def parse_options(argv: Sequence[str] | None = None) -> Options:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("plain", "reference", "selective"), required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--torchtitan-root", type=Path, required=True)
    parser.add_argument("--invariants", type=Path)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    return Options(
        mode=args.mode,
        output_dir=args.output_dir.resolve(),
        torchtitan_root=args.torchtitan_root.resolve(),
        steps=args.steps,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        seed=args.seed,
        invariants=None if args.invariants is None else args.invariants.resolve(),
    )


if __name__ == "__main__":
    main()
