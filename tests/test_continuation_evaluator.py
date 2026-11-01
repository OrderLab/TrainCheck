"""Operator-only checks must separate numerical violations from invalid evidence."""

from __future__ import annotations

import importlib
import json
import os
import shutil
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from tools.continuation_pilot import evaluate_candidate as evaluator
from tools.continuation_pilot.training_task import current, engine, reference


def candidate_packet(root: Path, *, corrected: bool) -> Path:
    package = root / "training_task"
    package.mkdir()
    source = Path(engine.__file__).parent
    for name in ("__init__.py", "engine.py", "reference.py", "current.py"):
        shutil.copyfile(source / name, package / name)
    path = package / "current.py"
    if corrected:
        text = path.read_text()
        original = "backward(loss / accumulation, accumulation)"
        assert text.count(original) == 1
        path.write_text(text.replace(original, "backward(loss, accumulation)"))
    return path


def test_corrected_candidate_matches_full_batch_update_and_preserves_rng(
    tmp_path: Path,
) -> None:
    path = candidate_packet(tmp_path, corrected=True)
    rng = torch.get_rng_state().clone()
    threads = torch.get_num_threads()
    report = evaluator.evaluate_candidate(path, 0.15)
    assert report["status"] == "passed"
    assert report["contract_passed"] is True
    assert len(report["scenarios"]) == 6
    assert torch.equal(torch.get_rng_state(), rng)
    assert torch.get_num_threads() == threads
    for case in report["scenarios"]:
        assert case["mean_loss_matches"] is True
        assert case["state_matches"] is True
        assert case["optimizer_steps"] == 1
        assert (
            case["forward_batch_sizes"]
            == [8 // case["accumulation"]] * case["accumulation"]
        )
        assert case["microbatch_bound_satisfied"] is True
        tensors = [item for item in case["comparisons"] if item["kind"] == "tensor"]
        assert (
            len(tensors) == 82
        )  # 27 parameters, gradients, momentum buffers + CPU RNG.
        assert any("applied_gradients" in item["path"] for item in tensors)
        assert any("optimizer" in item["path"] for item in tensors)
        assert any("model" in item["path"] for item in tensors)
        assert any("cpu_rng" in item["path"] for item in tensors)
        assert all(len(item["candidate_sha256"]) == 64 for item in tensors)
    for seed in evaluator.HELD_OUT_SEEDS:
        cases = [item for item in report["scenarios"] if item["seed"] == seed]
        assert all(
            item["full_batch_sha256"] == cases[0]["full_batch_sha256"] for item in cases
        )


def test_original_candidate_has_real_gradient_and_update_discrepancies(
    tmp_path: Path,
) -> None:
    path = candidate_packet(tmp_path, corrected=False)
    report = evaluator.evaluate_candidate(path, 0.15)
    assert report["status"] == "violated"
    for case in report["scenarios"]:
        assert case["mean_loss_matches"] is True
        assert case["microbatch_bound_satisfied"] is True
        assert case["contract_passed"] is (case["accumulation"] == 1)
        if case["accumulation"] > 1:
            failures = [item for item in case["comparisons"] if not item["passed"]]
            for component in ("model", "optimizer", "applied_gradients"):
                assert any(component in item["path"] for item in failures)


def test_reverting_to_full_batch_does_not_pass_microbatch_contract() -> None:
    cases = evaluator.evaluate_update(reference.run_update, 0.15)
    for case in cases:
        assert case["state_matches"] is True
        assert case["forward_batch_sizes"] == [8]
        assert case["microbatch_bound_satisfied"] is (case["accumulation"] == 1)


def test_post_step_gradient_cleanup_does_not_hide_applied_gradient_evidence() -> None:
    def update(
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        batches: list[engine.Batch],
    ) -> float:
        value = reference.run_update(model, optimizer, batches)
        optimizer.zero_grad(set_to_none=True)
        return value

    model = engine.TinyTransformer()
    batch = engine.make_batch(3, 1, 0, torch.device("cpu"))
    result = evaluator.run_observed_update(update, model, batch, 1, 0.15)
    assert all(parameter.grad is None for parameter in model.parameters())
    assert isinstance(result.state, dict)
    gradients = result.state["applied_gradients"]
    assert isinstance(gradients, list) and len(gradients) == 1
    assert isinstance(gradients[0], dict) and len(gradients[0]) == 27


def test_missing_gradient_is_an_error_not_zero_discrepancy() -> None:
    def update(
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        batches: list[engine.Batch],
    ) -> float:
        optimizer.step()
        return 0.0

    model = engine.TinyTransformer()
    batch = engine.make_batch(3, 1, 0, torch.device("cpu"))
    with pytest.raises(ValueError, match="Missing applied gradient"):
        evaluator.run_observed_update(update, model, batch, 1, 0.15)


def test_nonfinite_gradient_is_an_error() -> None:
    def update(
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        batches: list[engine.Batch],
    ) -> float:
        for parameter in model.parameters():
            parameter.grad = torch.full_like(parameter, float("nan"))
        optimizer.step()
        return 0.0

    model = engine.TinyTransformer()
    batch = engine.make_batch(3, 1, 0, torch.device("cpu"))
    with pytest.raises(ValueError, match="nonfinite tensor"):
        evaluator.run_observed_update(update, model, batch, 1, 0.15)


def test_no_optimizer_step_is_an_error() -> None:
    def update(
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        batches: list[engine.Batch],
    ) -> float:
        return 0.0

    with pytest.raises(ValueError, match="Expected one optimizer step"):
        evaluator.evaluate_update(update, 0.15)


@pytest.mark.parametrize("value", [None, True, 1, torch.tensor(1.0), float("nan")])
def test_invalid_return_is_rejected(value: object) -> None:
    def invalid(
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        batches: list[engine.Batch],
    ) -> object:
        return value

    with pytest.raises((TypeError, ValueError)):
        evaluator.validated_update(invalid)(
            engine.TinyTransformer(),
            torch.optim.SGD([nn.Parameter(torch.ones(1))], lr=0.1),
            [],
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 0.0, -1.0])
def test_invalid_learning_rate_is_rejected(value: float) -> None:
    with pytest.raises(ValueError, match="Learning rate"):
        evaluator.evaluate_update(current.run_update, value)


def test_state_comparison_retains_dtype_shape_order_and_scalar_bits() -> None:
    left = evaluator.clone_state({"x": torch.tensor([1.0]), "y": [0.0]}, "state")
    right = evaluator.clone_state(
        {"y": [-0.0], "x": torch.tensor([1.0], dtype=torch.float64)}, "state"
    )
    records = evaluator.compare_states(left, right, "state")
    failures = [item for item in records if not item["passed"]]
    assert {item["path"] for item in failures} == {
        "state.keys",
        "state['x']",
        "state['y'][0]",
    }
    tensor = next(item for item in records if item["kind"] == "tensor")
    assert tensor["max_abs_error"] is None
    assert tensor["reference_dtype"] == "torch.float32"
    assert tensor["candidate_dtype"] == "torch.float64"


def test_loader_restores_existing_package_binding_even_after_error(
    tmp_path: Path,
) -> None:
    package = importlib.import_module("tools.continuation_pilot.training_task")
    previous = sys.modules.get("training_task")
    original_path = list(sys.path)
    sys.modules["training_task"] = package
    path = tmp_path / "current.py"
    path.write_text("run_update = 3\n")
    try:
        with pytest.raises(TypeError, match="callable"):
            with evaluator.load_candidate(path):
                raise AssertionError("Malformed callable unexpectedly loaded")
        assert sys.modules["training_task"] is package
        assert sys.path == original_path
    finally:
        if previous is None:
            del sys.modules["training_task"]
        else:
            sys.modules["training_task"] = previous


def test_cli_records_unknown_outcome_for_candidate_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "current.py"
    path.write_text("raise ZeroDivisionError('candidate import failed')\n")
    output = tmp_path / "report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["check", "--candidate", str(path), "--lr", "0.15", "--output", str(output)],
    )
    assert evaluator.main() == 2
    report: object = json.loads(output.read_text())
    assert isinstance(report, dict)
    assert report["status"] == "error"
    assert report["contract_passed"] is None
    assert report["error_type"] == "ZeroDivisionError"


def test_reference_is_frozen_before_candidate_installs_global_wrapper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = candidate_packet(tmp_path, corrected=True)
    # Keep pytest's restoration registered before packet import changes F.
    monkeypatch.setattr(F, "cross_entropy", F.cross_entropy)
    path.write_text(
        path.read_text()
        + """
import torch.nn.functional as _functional
_original_cross_entropy = _functional.cross_entropy
def _scaled_cross_entropy(*args: object, **kwargs: object) -> torch.Tensor:
    result: object = _original_cross_entropy(*args, **kwargs)
    if not isinstance(result, torch.Tensor):
        raise TypeError("Cross entropy must return a tensor")
    return result * 0.5
_functional.cross_entropy = _scaled_cross_entropy
"""
    )
    report = evaluator.evaluate_candidate(path, 0.15, tmp_path)
    assert report["contract_passed"] is False
    assert all(not case["mean_loss_matches"] for case in report["scenarios"])


def test_changed_source_is_an_error_instead_of_wrong_provenance(tmp_path: Path) -> None:
    path = candidate_packet(tmp_path, corrected=True)
    path.write_text(
        path.read_text()
        + """
from pathlib import Path as _Path
_source_path = _Path(__file__)
_source_path.write_text(_source_path.read_text() + "\\n")
"""
    )
    with pytest.raises(ValueError, match="source changed"):
        evaluator.evaluate_candidate(path, 0.15, tmp_path)


def test_optional_diagnostics_get_unique_output_before_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = candidate_packet(tmp_path, corrected=True)
    path.write_text(
        path.read_text()
        + """
import os as _os
from pathlib import Path as _Path
_trace_directory = _Path(_os.environ["TRAINCHECK_OUTPUT_DIR"])
(_trace_directory / "imported.txt").write_text("native diagnostics may initialize here")
"""
    )
    monkeypatch.setenv("TRAINCHECK_OUTPUT_DIR", "/previous-value")
    first = evaluator.evaluate_candidate(path, 0.15, tmp_path)
    second = evaluator.evaluate_candidate(path, 0.15, tmp_path)
    assert first["contract_passed"] and second["contract_passed"]
    assert first["diagnostic_trace_directory"] != second["diagnostic_trace_directory"]
    assert Path(first["diagnostic_trace_directory"], "imported.txt").is_file()
    assert os.environ["TRAINCHECK_OUTPUT_DIR"] == "/previous-value"
