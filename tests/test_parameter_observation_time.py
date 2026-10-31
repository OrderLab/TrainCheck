"""A sampled parameter describes its observation time, not its last assignment."""

from __future__ import annotations

import gc
import weakref
from dataclasses import dataclass
from types import ModuleType
from typing import Literal

import pytest
import torch

import traincheck.config.config as config
from traincheck.instrumentor.proxy_wrapper import proxy, subclass
from traincheck.instrumentor.proxy_wrapper.proxy_config import tensor_dump_format
from traincheck.instrumentor.proxy_wrapper.proxy_registry import VarRegistry


@dataclass
class Clock:
    now: int = 100
    calls: int = 0

    def __call__(self) -> int:
        self.calls += 1
        return self.now


@dataclass
class TrackedParameter:
    wrapper: proxy.Proxy | subclass.ProxyParameter
    tensor: torch.Tensor
    module: ModuleType
    registry: VarRegistry
    clock: Clock
    records: list[dict[str, object]]


@pytest.fixture(params=("subclass", "proxy"))
def tracked(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> TrackedParameter:
    style: object = request.param
    assert style in ("subclass", "proxy")
    registry = VarRegistry()
    records: list[dict[str, object]] = []
    clock = Clock()
    module = subclass if style == "subclass" else proxy

    def isolated_registry() -> VarRegistry:
        return registry

    monkeypatch.setattr(module, "get_global_registry", isolated_registry)
    monkeypatch.setattr(module, "get_timestamp_ns", clock)
    monkeypatch.setattr(config, "DISABLE_WRAPPER", True)
    monkeypatch.setattr(proxy.Proxy, "var_dict", {})
    monkeypatch.setitem(tensor_dump_format, "dump_tensor_hash", True)
    monkeypatch.setitem(tensor_dump_format, "dump_tensor_stats", False)
    monkeypatch.setitem(tensor_dump_format, "dump_tensor_full", False)

    def capture_record(record: dict[str, object]) -> None:
        records.append(record)

    def capture_legacy_record(**fields: object) -> None:
        records.append(
            {
                "time": fields["time"],
                "mode": fields["change_type"],
                "attributes": fields["var_attributes"],
                "dump_loc": fields["dump_loc"],
            }
        )

    monkeypatch.setattr(subclass, "dump_trace_VAR", capture_record)
    monkeypatch.setattr(proxy.Proxy.jsondumper, "dump_json", capture_legacy_record)
    source = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    wrapper: proxy.Proxy | subclass.ProxyParameter
    if style == "subclass":
        wrapper = subclass.ProxyParameter(
            source,
            var_name="model.weight",
            var_type="torch.nn.Parameter",
            should_dump_trace=False,
        )
        tensor = wrapper
    else:
        wrapper = proxy.Proxy(source, var_name="model.weight", should_dump_trace=False)
        tensor = source
    registry.add_var(wrapper, "model.weight", "torch.nn.Parameter")
    assert wrapper.__dict__["last_update_timestamp"] == 100
    clock.calls = 0
    monkeypatch.setattr(config, "DISABLE_WRAPPER", False)
    return TrackedParameter(wrapper, tensor, module, registry, clock, records)


def sample(
    tracked: TrackedParameter, phase: Literal["sample", "selective-sample"]
) -> None:
    if phase == "sample":
        tracked.registry.dump_sample(
            dump_loc="test", dump_config={"torch.nn.Parameter": {}}
        )
    else:
        tracked.registry.dump_modified(
            dump_loc="test", dump_config={"torch.nn.Parameter": {}}
        )


def attributes(record: dict[str, object]) -> dict[str, object]:
    value = record["attributes"]
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise TypeError("Parameter attributes must be a string-keyed dictionary")
    return {key: item for key, item in value.items()}


@pytest.mark.parametrize("phase", ["sample", "selective-sample"])
def test_samples_of_in_place_mutations_have_fresh_times(
    tracked: TrackedParameter, phase: Literal["sample", "selective-sample"]
) -> None:
    rng_before = torch.get_rng_state().clone()
    tracked.clock.now = 200
    sample(tracked, phase)
    with torch.no_grad():
        tracked.tensor.add_(3)
    # In-place tensor mutation does not notify the Python assignment hook. A
    # selective sample requires the caller to mark the object eligible again.
    tracked.registry.add_var(tracked.wrapper, "model.weight", "torch.nn.Parameter")
    tracked.clock.now = 300
    sample(tracked, phase)

    assert len(tracked.records) == 2
    before, after = tracked.records
    assert before["time"] == 200
    assert after["time"] == 300
    assert before["mode"] == after["mode"] == phase
    assert attributes(before)["data"] != attributes(after)["data"]
    assert tracked.wrapper.__dict__["last_update_timestamp"] == 100
    if tracked.module is proxy:
        assert proxy.Proxy.var_dict["model.weight"].last_update_timestamp == 100
    assert tracked.registry.registry["model.weight"].stale is True
    assert tracked.clock.calls == 2
    if phase == "selective-sample":
        # Observing a variable must not make it eligible for another modified-only
        # sample. The registry's normal stale-entry skip reads no new timestamp.
        sample(tracked, phase)
        assert len(tracked.records) == 2
        assert tracked.clock.calls == 2
    torch.testing.assert_close(
        tracked.tensor.detach(), torch.tensor([4.0, 5.0]), rtol=0, atol=0
    )
    torch.testing.assert_close(torch.get_rng_state(), rng_before, rtol=0, atol=0)


@pytest.mark.parametrize("phase", ["sample", "selective-sample"])
def test_time_is_captured_before_metadata_conversion(
    tracked: TrackedParameter,
    monkeypatch: pytest.MonkeyPatch,
    phase: Literal["sample", "selective-sample"],
) -> None:
    def metadata_after_clock_advance(owner: object) -> dict[str, int]:
        assert owner is tracked.wrapper
        tracked.clock.now = 900
        return {"step": 1}

    monkeypatch.setattr(tracked.module, "get_meta_vars", metadata_after_clock_advance)
    tracked.clock.now = 200
    sample(tracked, phase)
    assert len(tracked.records) == 1
    assert tracked.records[0]["time"] == 200
    assert tracked.clock.now == 900
    assert tracked.clock.calls == 1


@pytest.mark.parametrize("phase", ["update", "call", "iter"])
def test_non_sample_records_keep_assignment_time(
    tracked: TrackedParameter, phase: str
) -> None:
    tracked.clock.now = 200
    tracked.wrapper.dump_trace(phase=phase, dump_loc="test")
    assert len(tracked.records) == 1
    assert tracked.records[0]["time"] == 100
    assert tracked.clock.calls == 0
    assert tracked.wrapper.__dict__["last_update_timestamp"] == 100
    assert tracked.registry.registry["model.weight"].stale is False


def test_disabled_dump_does_not_read_clock_or_emit(
    tracked: TrackedParameter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config, "DISABLE_WRAPPER", True)
    tracked.wrapper.dump_trace(phase="sample", dump_loc="test")
    assert tracked.records == []
    assert tracked.clock.calls == 0
    assert tracked.wrapper.__dict__["last_update_timestamp"] == 100
    assert tracked.registry.registry["model.weight"].stale is False


@pytest.mark.parametrize("tracked", ["subclass"], indirect=True)
def test_retired_parameter_does_not_read_clock_or_emit(
    tracked: TrackedParameter,
) -> None:
    model = torch.nn.Linear(2, 1)
    subclass.proxy_parameter(model, parent_name="owned", should_dump_trace=False)
    parameter = model.weight
    assert isinstance(parameter, subclass.ProxyParameter)
    reference = weakref.ref(model)
    del model
    gc.collect()
    assert reference() is None
    assert list(tracked.registry.registry) == ["model.weight"]
    tracked.clock.calls = 0
    parameter.dump_trace(phase="sample", dump_loc="retired")
    assert tracked.clock.calls == 0
    assert tracked.records == []


@pytest.mark.parametrize("phase", ["sample", "selective-sample"])
def test_failed_sample_propagates_without_changing_assignment_time(
    tracked: TrackedParameter,
    monkeypatch: pytest.MonkeyPatch,
    phase: Literal["sample", "selective-sample"],
) -> None:
    failure = RuntimeError("metadata conversion failed")

    def fail_conversion(owner: object, value: object) -> dict[str, object]:
        raise failure

    monkeypatch.setattr(tracked.module, "dump_attributes", fail_conversion)
    with pytest.raises(RuntimeError) as caught:
        sample(tracked, phase)
    assert caught.value is failure
    assert tracked.records == []
    assert tracked.wrapper.__dict__["last_update_timestamp"] == 100
    assert tracked.registry.registry["model.weight"].stale is True
