"""CPU-only contracts for the operator's private clipping qualification probe."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import pytest
import torch
import torch.nn as nn

from tools.runtime_pilot import clipping as audit


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.rep_a = nn.Parameter(torch.ones(2, 2))
        self.rep_b = nn.Parameter(torch.ones(2, 2))
        self.shard = nn.Parameter(torch.ones(2, 2))
        self.frozen = nn.Parameter(torch.ones(2, 2), requires_grad=False)
        setattr(self.shard, "tensor_model_parallel", True)


@dataclass
class Optimizer:
    bf16_groups: list[list[torch.Tensor]]
    fp32_groups_gradients: list[list[torch.Tensor]]


class MPU:
    @staticmethod
    def get_slice_parallel_rank() -> int:
        return 1

    @staticmethod
    def get_model_parallel_rank() -> int:
        # The native grid combines PP and TP here. Do not label this as TP.
        return 5

    @staticmethod
    def get_data_parallel_rank() -> int:
        return 0


class Engine:
    def __init__(self, native: ModuleType) -> None:
        self.module = Model()
        self.optimizer = Optimizer(
            [
                [
                    self.module.rep_a,
                    self.module.rep_b,
                    self.module.shard,
                    self.module.frozen,
                ]
            ],
            [[torch.full((4,), 2.0) for _ in range(4)]],
        )
        self.global_steps = 0
        self.native = native
        self.mpu = MPU()
        gradients = self.optimizer.fp32_groups_gradients[0]
        self.clip_kwargs: dict[str, object] = {
            "input_tensors": [gradients[0], gradients[2]],
            "max_norm": 1.0,
            "global_norm": 4.0,
            "mpu": self.mpu,
            "eps": 0.5,
        }
        self.step_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
        self.clip_return = object()
        self.step_return = object()
        self.observed_clip_return: object = None

    def _take_model_step(self, *args: object, **kwargs: object) -> object:
        # This method stands in for an already installed state observer.
        self.step_calls.append((args, kwargs))
        clip: object = getattr(self.native, "clip_tensors_by_global_norm")
        assert callable(clip)
        self.observed_clip_return = clip(**self.clip_kwargs)
        self.global_steps += 1
        return self.step_return


@dataclass
class Harness:
    engine: Engine
    native: ModuleType
    clip_calls: list[dict[str, object]]
    output: Path

    def snapshot(self, step: int = 1) -> dict[object, object]:
        value: object = torch.load(
            self.output / "rank1" / f"step{step}.pt", weights_only=True
        )
        assert isinstance(value, dict)
        return value


@pytest.fixture
def harness(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Harness:
    native = ModuleType("deepspeed.runtime.bf16_optimizer")
    engine = Engine(native)
    clip_calls: list[dict[str, object]] = []

    def native_clip(*args: object, **kwargs: object) -> object:
        assert not args
        clip_calls.append(kwargs)
        tensors = kwargs["input_tensors"]
        assert isinstance(tensors, list)
        for tensor in tensors:
            assert isinstance(tensor, torch.Tensor)
            tensor.mul_(1.0 / 4.5)
        return engine.clip_return

    engine_module = ModuleType("deepspeed.runtime.engine")
    setattr(engine_module, "DeepSpeedEngine", Engine)
    setattr(native, "clip_tensors_by_global_norm", native_clip)
    monkeypatch.setitem(sys.modules, native.__name__, native)
    monkeypatch.setitem(sys.modules, engine_module.__name__, engine_module)
    monkeypatch.setattr(audit, "_installed", False)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 1)
    monkeypatch.setenv("RUNTIME_CLIPPING_OUTPUT", str(tmp_path))
    # The fixture restores the class method after install replaces it.
    monkeypatch.setattr(Engine, "_take_model_step", Engine._take_model_step)
    return Harness(engine, native, clip_calls, tmp_path)


def tensor_map(value: object) -> dict[str, torch.Tensor]:
    assert isinstance(value, dict)
    result: dict[str, torch.Tensor] = {}
    for name, tensor in value.items():
        assert isinstance(name, str)
        assert isinstance(tensor, torch.Tensor)
        result[name] = tensor
    return result


def test_native_call_once_and_correct_independent_snapshots(harness: Harness) -> None:
    audit.install()
    marker = object()
    rng_before = torch.get_rng_state().clone()
    parameters_before = {
        name: parameter.detach().clone()
        for name, parameter in harness.engine.module.named_parameters()
    }
    result = harness.engine._take_model_step(marker, future_option=marker)
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert all(
        torch.equal(parameter, parameters_before[name])
        for name, parameter in harness.engine.module.named_parameters()
    )
    assert result is harness.engine.step_return
    assert harness.engine.observed_clip_return is harness.engine.clip_return
    assert len(harness.engine.step_calls) == len(harness.clip_calls) == 1
    args, kwargs = harness.engine.step_calls[0]
    assert len(args) == 1 and args[0] is marker
    assert kwargs["future_option"] is marker
    for name, value in harness.engine.clip_kwargs.items():
        assert harness.clip_calls[0][name] is value
    snapshot = harness.snapshot()
    assert snapshot["rank"] == snapshot["tp_rank"] == snapshot["step"] == 1
    assert snapshot["dp_rank"] == 0
    assert snapshot["max_norm"] == 1.0
    assert snapshot["global_norm"] == 4.0
    assert snapshot["selected"] == {"rep_a": True, "rep_b": False}
    before = tensor_map(snapshot["before"])
    after = tensor_map(snapshot["after"])
    assert list(before) == list(after) == ["rep_a", "rep_b"]
    assert torch.equal(before["rep_a"], torch.full((4,), 2.0))
    assert torch.equal(before["rep_b"], before["rep_a"])
    assert torch.equal(after["rep_a"], before["rep_a"] * (1.0 / 4.5))
    assert torch.equal(after["rep_b"], before["rep_b"])
    for gradient in harness.engine.optimizer.fp32_groups_gradients[0]:
        gradient.zero_()
    persisted = tensor_map(harness.snapshot()["after"])
    assert torch.equal(persisted["rep_a"], after["rep_a"])
    assert audit._current_engine.get() is None


def test_successive_steps_have_distinct_artifacts(harness: Harness) -> None:
    audit.install()
    harness.engine._take_model_step()
    harness.engine._take_model_step()
    assert harness.snapshot(1)["step"] == 1
    assert harness.snapshot(2)["step"] == 2
    assert len(harness.clip_calls) == 2


def test_native_grid_tensor_slice_rank_is_used(harness: Harness) -> None:
    assert not hasattr(harness.engine.mpu, "get_tensor_model_parallel_rank")
    assert harness.engine.mpu.get_model_parallel_rank() != 1
    audit.install()
    harness.engine._take_model_step()
    assert harness.snapshot()["tp_rank"] == 1


def test_original_error_is_unchanged_and_context_restored(harness: Harness) -> None:
    error = RuntimeError("native clipping failed")
    calls: list[None] = []

    def failing_clip(*args: object, **kwargs: object) -> object:
        calls.append(None)
        raise error

    setattr(harness.native, "clip_tensors_by_global_norm", failing_clip)
    audit.install()
    outer = object()
    token = audit._current_engine.set(outer)
    try:
        with pytest.raises(RuntimeError) as caught:
            harness.engine._take_model_step()
        assert caught.value is error
        assert calls == [None]
        assert audit._current_engine.get() is outer
        assert not list(harness.output.rglob("*.pt"))
    finally:
        audit._current_engine.reset(token)


@pytest.mark.parametrize("field", ["input_tensors", "max_norm", "global_norm", "mpu"])
def test_missing_required_argument_is_not_defaulted(
    harness: Harness, field: str
) -> None:
    del harness.engine.clip_kwargs[field]
    audit.install()
    with pytest.raises(TypeError, match="Missing required clipping argument"):
        harness.engine._take_model_step()
    assert not harness.clip_calls
    assert audit._current_engine.get() is None


@pytest.mark.parametrize("flag", [1, None, "false"])
def test_malformed_partition_flag_rejected(harness: Harness, flag: object) -> None:
    setattr(harness.engine.module.rep_a, "tensor_model_parallel", flag)
    audit.install()
    with pytest.raises(TypeError, match="tensor_model_parallel"):
        harness.engine._take_model_step()
    assert not harness.clip_calls


def test_missing_replicated_gradient_rejected(harness: Harness) -> None:
    harness.engine.optimizer.bf16_groups[0].pop(1)
    harness.engine.optimizer.fp32_groups_gradients[0].pop(1)
    audit.install()
    with pytest.raises(ValueError, match="Missing replicated gradients"):
        harness.engine._take_model_step()
    assert not harness.clip_calls


def test_missing_optimizer_field_rejected(harness: Harness) -> None:
    delattr(harness.engine.optimizer, "fp32_groups_gradients")
    audit.install()
    with pytest.raises(AttributeError, match="fp32_groups_gradients"):
        harness.engine._take_model_step()
    assert not harness.clip_calls


def test_mismatched_gradient_group_rejected(harness: Harness) -> None:
    harness.engine.optimizer.fp32_groups_gradients[0].pop()
    audit.install()
    with pytest.raises(ValueError, match="lengths differ"):
        harness.engine._take_model_step()
    assert not harness.clip_calls


@pytest.mark.parametrize("value", [True, None, float("inf"), float("nan"), 0.0])
def test_malformed_norm_rejected(harness: Harness, value: object) -> None:
    harness.engine.clip_kwargs["global_norm"] = value
    audit.install()
    with pytest.raises((TypeError, ValueError), match="global_norm"):
        harness.engine._take_model_step()
    assert not harness.clip_calls


def test_duplicate_identity_rejected(harness: Harness) -> None:
    optimizer = harness.engine.optimizer
    optimizer.bf16_groups[0].append(optimizer.bf16_groups[0][0])
    optimizer.fp32_groups_gradients[0].append(torch.ones(4))
    audit.install()
    with pytest.raises(ValueError, match="Duplicate optimizer"):
        harness.engine._take_model_step()
    assert not harness.clip_calls


@pytest.mark.parametrize("dtype", [torch.float16, torch.float64])
def test_non_fp32_gradient_rejected(harness: Harness, dtype: torch.dtype) -> None:
    harness.engine.optimizer.fp32_groups_gradients[0][1] = torch.ones(4, dtype=dtype)
    audit.install()
    with pytest.raises(TypeError, match="FP32"):
        harness.engine._take_model_step()
    assert not harness.clip_calls


def test_selected_view_is_not_confused_with_selected_object(harness: Harness) -> None:
    gradient = harness.engine.optimizer.fp32_groups_gradients[0][0]
    harness.engine.clip_kwargs["input_tensors"] = [gradient.view_as(gradient)]
    audit.install()
    with pytest.raises(ValueError, match="Selected gradient object"):
        harness.engine._take_model_step()
    assert not harness.clip_calls


def test_empty_selection_records_omissions(harness: Harness) -> None:
    harness.engine.clip_kwargs["input_tensors"] = []
    audit.install()
    harness.engine._take_model_step()
    snapshot = harness.snapshot()
    assert snapshot["selected"] == {"rep_a": False, "rep_b": False}
    before = tensor_map(snapshot["before"])
    after = tensor_map(snapshot["after"])
    assert all(torch.equal(before[name], after[name]) for name in before)


def test_duplicate_artifact_rejected_before_clipping(harness: Harness) -> None:
    audit.install()
    harness.engine._take_model_step()
    original = (harness.output / "rank1" / "step1.pt").read_bytes()
    harness.engine.global_steps = 0
    with pytest.raises(FileExistsError):
        harness.engine._take_model_step()
    assert len(harness.clip_calls) == 1
    assert (harness.output / "rank1" / "step1.pt").read_bytes() == original


def test_outside_step_and_repeat_install_rejected(harness: Harness) -> None:
    audit.install()
    clip: object = getattr(harness.native, "clip_tensors_by_global_norm")
    assert callable(clip)
    with pytest.raises(RuntimeError, match="outside a model step"):
        clip(**harness.engine.clip_kwargs)
    with pytest.raises(RuntimeError, match="already installed"):
        audit.install()
    assert not harness.clip_calls
