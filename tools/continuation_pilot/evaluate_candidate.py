"""Operator-only accumulation contract evaluation for a trusted candidate.

This CPU evaluator is separate from held-out two-GPU training quality. It checks
one SGD update against the frozen full-batch implementation, with matched initial
state, data and learning rate. It does not certify distributed communication or
isolate arbitrary candidate code. Run each selected candidate in a fresh process.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import struct
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Literal, Protocol, TypeAlias, TypedDict

import torch
import torch.nn as nn

from tools.continuation_pilot.training_task import engine, reference

HELD_OUT_SEEDS = (20261117, 20261129)
ACCUMULATIONS = (1, 2, 4)
ATOL = 2e-6
RTOL = 2e-4

State: TypeAlias = (
    torch.Tensor
    | None
    | bool
    | int
    | float
    | str
    | list["State"]
    | tuple["State", ...]
    | dict[str | int, "State"]
)


class Update(Protocol):
    def __call__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        batches: list[engine.Batch],
    ) -> float: ...


class TensorComparison(TypedDict):
    kind: Literal["tensor"]
    path: str
    passed: bool
    reference_shape: list[int]
    candidate_shape: list[int]
    reference_dtype: str
    candidate_dtype: str
    reference_sha256: str
    candidate_sha256: str
    max_abs_error: float | None


class ScalarComparison(TypedDict):
    kind: Literal["scalar"]
    path: str
    passed: bool
    reference: str
    candidate: str


Comparison: TypeAlias = TensorComparison | ScalarComparison


class Scenario(TypedDict):
    seed: int
    accumulation: int
    full_batch_sha256: dict[str, str]
    reference_loss: float
    candidate_loss: float
    mean_loss_matches: bool
    optimizer_steps: int
    forward_batch_sizes: list[int]
    permitted_microbatch_size: int
    microbatch_bound_satisfied: bool | None
    state_matches: bool
    contract_passed: bool
    comparisons: list[Comparison]


class Evaluation(TypedDict):
    status: Literal["passed", "violated"]
    contract_passed: bool
    scope: str
    learning_rate: float
    atol: float
    rtol: float
    candidate_path: str
    candidate_sha256: str
    reference_sha256: str
    diagnostic_trace_directory: str
    scenarios: list[Scenario]
    limitations: list[str]


@dataclass
class Observation:
    loss: float
    state: State
    forward_batch_sizes: list[int]
    optimizer_steps: int


def clone_state(value: object, path: str) -> State:
    """Snapshot every accepted leaf; reject unsupported or nonfinite values."""
    if isinstance(value, torch.Tensor):
        if value.layout != torch.strided or value.is_quantized:
            raise TypeError(f"{path}: only dense nonquantized tensors are supported")
        if value.is_floating_point() or value.is_complex():
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"{path}: nonfinite tensor")
        return value.detach().cpu().clone()
    if value is None:
        return None
    if type(value) in (bool, int, float, str):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"{path}: nonfinite scalar")
        if isinstance(value, (bool, int, float, str)):
            return value
    if isinstance(value, list):
        return [
            clone_state(item, f"{path}[{index}]") for index, item in enumerate(value)
        ]
    if isinstance(value, tuple):
        return tuple(
            clone_state(item, f"{path}[{index}]") for index, item in enumerate(value)
        )
    if isinstance(value, dict):
        result: dict[str | int, State] = {}
        for key, item in value.items():
            if type(key) not in (str, int) or not isinstance(key, (str, int)):
                raise TypeError(f"{path}: invalid state mapping key")
            result[key] = clone_state(item, f"{path}[{key!r}]")
        return result
    raise TypeError(f"{path}: unsupported state value {type(value).__name__}")


def tensor_sha256(value: torch.Tensor) -> str:
    data = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    return hashlib.sha256(data.numpy().tobytes()).hexdigest()


def scalar_comparison(path: str, left: object, right: object) -> ScalarComparison:
    same_type = type(left) is type(right)
    equal = left == right
    if isinstance(left, float) and isinstance(right, float):
        equal = struct.pack("!d", left) == struct.pack("!d", right)
    return {
        "kind": "scalar",
        "path": path,
        "passed": same_type and equal is True,
        "reference": repr(left),
        "candidate": repr(right),
    }


def compare_states(left: State, right: State, path: str) -> list[Comparison]:
    """Compare all leaves, retaining mapping/sequence order and tensor metadata."""
    reports: list[Comparison] = []
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        comparable = left.shape == right.shape and left.dtype == right.dtype
        maximum: float | None = None
        if comparable and (left.is_floating_point() or left.is_complex()):
            passed = bool(torch.allclose(left, right, atol=ATOL, rtol=RTOL))
            maximum = float((left - right).abs().max().item()) if left.numel() else 0.0
        else:
            passed = comparable and torch.equal(left, right)
        reports.append(
            {
                "kind": "tensor",
                "path": path,
                "passed": passed,
                "reference_shape": list(left.shape),
                "candidate_shape": list(right.shape),
                "reference_dtype": str(left.dtype),
                "candidate_dtype": str(right.dtype),
                "reference_sha256": tensor_sha256(left),
                "candidate_sha256": tensor_sha256(right),
                "max_abs_error": maximum,
            }
        )
    elif isinstance(left, dict) and isinstance(right, dict):
        left_keys = [(type(key).__name__, key) for key in left]
        right_keys = [(type(key).__name__, key) for key in right]
        reports.append(scalar_comparison(path + ".keys", left_keys, right_keys))
        for key in left:
            if key in right and (type(key).__name__, key) in right_keys:
                reports.extend(
                    compare_states(left[key], right[key], f"{path}[{key!r}]")
                )
    elif isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        reports.append(
            scalar_comparison(path + ".type", type(left).__name__, type(right).__name__)
        )
        reports.append(scalar_comparison(path + ".length", len(left), len(right)))
        for index, (a, b) in enumerate(zip(left, right)):
            reports.extend(compare_states(a, b, f"{path}[{index}]"))
    elif isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        reports.append(
            scalar_comparison(path + ".type", type(left).__name__, type(right).__name__)
        )
    else:
        reports.append(scalar_comparison(path, left, right))
    return reports


def validated_update(value: object) -> Update:
    if not callable(value):
        raise TypeError("Candidate run_update must be callable")

    def invoke(
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        batches: list[engine.Batch],
    ) -> float:
        result: object = value(model, optimizer, batches)
        if not isinstance(result, float):
            raise TypeError("Candidate run_update must return a float")
        if not math.isfinite(result):
            raise ValueError("Candidate returned nonfinite mean loss")
        return result

    return invoke


@contextmanager
def load_candidate(path: Path) -> Iterator[Update]:
    """Load trusted packet code with relative imports, restoring import bindings.

    The packet's engine is verified immutable by the operator. This loader is an
    import adapter, not a security sandbox or rollback of arbitrary side effects.
    """
    candidate = path.resolve(strict=True)
    if not candidate.is_file() or candidate.suffix != ".py":
        raise ValueError("Candidate must be a Python source file")
    package_name = "training_task"
    module_name = package_name + ".current"
    previous_path = list(sys.path)
    previous_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == package_name or name.startswith(package_name + ".")
    }
    package = ModuleType(package_name)
    package.__path__ = [str(candidate.parent)]
    spec = importlib.util.spec_from_file_location(module_name, candidate)
    if spec is None or spec.loader is None:
        raise ImportError("Cannot construct candidate module loader")
    module = importlib.util.module_from_spec(spec)
    try:
        for name in previous_modules:
            del sys.modules[name]
        sys.modules[package_name] = package
        sys.modules[module_name] = module
        sys.path[:0] = [str(candidate.parent.parent), str(candidate.parent)]
        spec.loader.exec_module(module)
        if "run_update" not in vars(module):
            raise ValueError("Candidate does not define run_update")
        yield validated_update(vars(module)["run_update"])
    finally:
        sys.path[:] = previous_path
        for name in list(sys.modules):
            if name == package_name or name.startswith(package_name + "."):
                del sys.modules[name]
        sys.modules.update(previous_modules)


def run_observed_update(
    update: Update,
    model: nn.Module,
    batch: engine.Batch,
    accumulation: int,
    learning_rate: float,
) -> Observation:
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    applied_gradients: list[dict[str, torch.Tensor]] = []
    forward_sizes: list[int] = []

    def observe_forward(module: nn.Module, args: tuple[object, ...]) -> None:
        if len(args) != 1 or not isinstance(args[0], torch.Tensor) or args[0].ndim != 2:
            raise ValueError("Expected a single two-dimensional token batch")
        forward_sizes.append(args[0].shape[0])

    def observe_step(
        observed_optimizer: torch.optim.Optimizer,
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> None:
        if observed_optimizer is not optimizer:
            raise ValueError("Unexpected optimizer observed")
        gradients: dict[str, torch.Tensor] = {}
        for name, parameter in model.named_parameters():
            if parameter.grad is None:
                raise ValueError(f"Missing applied gradient for {name}")
            snapshot = clone_state(parameter.grad, f"gradients.{name}")
            assert isinstance(snapshot, torch.Tensor)
            gradients[name] = snapshot
        if not gradients:
            raise ValueError("No model parameter gradients observed")
        applied_gradients.append(gradients)

    forward_hook = model.register_forward_pre_hook(observe_forward)
    step_hook = optimizer.register_step_pre_hook(observe_step)
    try:
        loss = validated_update(update)(
            model, optimizer, engine.split_batch(batch, accumulation)
        )
    finally:
        forward_hook.remove()
        step_hook.remove()
    if len(applied_gradients) != 1:
        raise ValueError(
            f"Expected one optimizer step, observed {len(applied_gradients)}"
        )
    state = clone_state(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "applied_gradients": applied_gradients,
            "cpu_rng": torch.get_rng_state(),
        },
        "update",
    )
    return Observation(loss, state, forward_sizes, len(applied_gradients))


@dataclass(frozen=True)
class ReferenceCase:
    seed: int
    accumulation: int
    initial_model: nn.Module
    rng: torch.Tensor
    batch: engine.Batch
    observation: Observation


def prepare_reference_cases(learning_rate: float) -> list[ReferenceCase]:
    """Freeze reference outcomes before candidate import or instrumentation."""
    if isinstance(learning_rate, bool) or not isinstance(learning_rate, (float, int)):
        raise TypeError("Learning rate must be numeric")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Learning rate must be finite and positive")
    cases: list[ReferenceCase] = []
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.random.fork_rng(devices=[]):
            for seed in HELD_OUT_SEEDS:
                torch.manual_seed(seed)
                initial = engine.TinyTransformer().train()
                rng = torch.get_rng_state().clone()
                batch = engine.make_batch(seed, 1, 0, torch.device("cpu"))
                for accumulation in ACCUMULATIONS:
                    torch.set_rng_state(rng)
                    expected = run_observed_update(
                        reference.run_update,
                        copy.deepcopy(initial),
                        engine.Batch(batch.tokens.clone(), batch.labels.clone()),
                        accumulation,
                        learning_rate,
                    )
                    cases.append(
                        ReferenceCase(seed, accumulation, initial, rng, batch, expected)
                    )
    finally:
        torch.set_num_threads(previous_threads)
    return cases


def compare_candidate_cases(
    update: Update, learning_rate: float, cases: list[ReferenceCase]
) -> list[Scenario]:
    if not cases:
        raise ValueError("No reference cases supplied")
    scenarios: list[Scenario] = []
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.random.fork_rng(devices=[]):
            for case in cases:
                expected = case.observation
                torch.set_rng_state(case.rng)
                actual = run_observed_update(
                    update,
                    copy.deepcopy(case.initial_model),
                    engine.Batch(case.batch.tokens.clone(), case.batch.labels.clone()),
                    case.accumulation,
                    learning_rate,
                )
                comparisons = compare_states(expected.state, actual.state, "update")
                state_matches = all(item["passed"] for item in comparisons)
                loss_matches = math.isclose(
                    expected.loss, actual.loss, abs_tol=ATOL, rel_tol=RTOL
                )
                maximum_batch = engine.EXAMPLES_PER_RANK_UPDATE // case.accumulation
                batch_bound = (
                    all(
                        0 < size <= maximum_batch for size in actual.forward_batch_sizes
                    )
                    if actual.forward_batch_sizes
                    else None
                )
                scenarios.append(
                    {
                        "seed": case.seed,
                        "accumulation": case.accumulation,
                        "full_batch_sha256": {
                            "tokens": tensor_sha256(case.batch.tokens),
                            "labels": tensor_sha256(case.batch.labels),
                        },
                        "reference_loss": expected.loss,
                        "candidate_loss": actual.loss,
                        "mean_loss_matches": loss_matches,
                        "optimizer_steps": actual.optimizer_steps,
                        "forward_batch_sizes": actual.forward_batch_sizes,
                        "permitted_microbatch_size": maximum_batch,
                        "microbatch_bound_satisfied": batch_bound,
                        "state_matches": state_matches,
                        "contract_passed": state_matches
                        and loss_matches
                        and batch_bound is True,
                        "comparisons": comparisons,
                    }
                )
    finally:
        torch.set_num_threads(previous_threads)
    return scenarios


def evaluate_update(update: Update, learning_rate: float) -> list[Scenario]:
    """Evaluate an already-loaded trusted callable; imports are the caller's scope."""
    return compare_candidate_cases(
        update, learning_rate, prepare_reference_cases(learning_rate)
    )


def evaluate_candidate(
    path: Path, learning_rate: float, trace_parent: Path | None = None
) -> Evaluation:
    candidate_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    reference_path = Path(reference.__file__)
    reference_sha = hashlib.sha256(reference_path.read_bytes()).hexdigest()
    cases = prepare_reference_cases(learning_rate)
    trace_directory = Path(
        tempfile.mkdtemp(prefix="continuation-check-", dir=trace_parent)
    )
    previous_trace_directory = os.environ.get("TRAINCHECK_OUTPUT_DIR")
    os.environ["TRAINCHECK_OUTPUT_DIR"] = str(trace_directory)
    try:
        with load_candidate(path) as update:
            scenarios = compare_candidate_cases(update, learning_rate, cases)
    finally:
        if previous_trace_directory is None:
            os.environ.pop("TRAINCHECK_OUTPUT_DIR", None)
        else:
            os.environ["TRAINCHECK_OUTPUT_DIR"] = previous_trace_directory
    if hashlib.sha256(path.read_bytes()).hexdigest() != candidate_sha:
        raise ValueError("Candidate source changed during evaluation")
    if hashlib.sha256(reference_path.read_bytes()).hexdigest() != reference_sha:
        raise ValueError("Reference source changed during evaluation")
    passed = all(item["contract_passed"] for item in scenarios)
    return {
        "status": "passed" if passed else "violated",
        "contract_passed": passed,
        "scope": "CPU one-update accumulation arithmetic and microbatch bound; training quality evaluated separately",
        "learning_rate": learning_rate,
        "atol": ATOL,
        "rtol": RTOL,
        "candidate_path": str(path.resolve()),
        "candidate_sha256": candidate_sha,
        "reference_sha256": reference_sha,
        "diagnostic_trace_directory": str(trace_directory),
        "scenarios": scenarios,
        "limitations": [
            "single-process CPU check does not certify DDP collective behavior",
            "one update per seed/configuration does not certify long-run optimizer or checkpoint behavior",
            "floating tensor comparisons allow stated FP32 tolerance; every tensor retains exact content hashes",
            "stored state is compared without reconstructing storage aliasing or device placement",
            "trusted candidate code runs in-process; use a fresh process per candidate",
            "optional trace writer artifacts may finish at process exit; this report does not certify their coverage",
            "CPU numerical evaluation does not establish CPU-only compatibility of candidate instrumentation dependencies",
        ],
    }


class Options(argparse.Namespace):
    candidate: Path
    lr: float
    output: Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args(namespace=Options())
    try:
        report = evaluate_candidate(
            options.candidate, options.lr, options.output.parent
        )
    except Exception as error:
        # Trusted candidate code can raise arbitrary exception classes. This is
        # an explicit failed evaluation, with unknown outcome and nonzero exit.
        failure: dict[str, object] = {
            "status": "error",
            "contract_passed": None,
            "error_type": type(error).__name__,
            "error": str(error),
        }
        options.output.write_text(json.dumps(failure, indent=2, allow_nan=False) + "\n")
        return 2
    options.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return 0 if report["contract_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
