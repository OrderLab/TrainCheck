"""Complete, non-invasive model observations and strict offline comparisons."""

from __future__ import annotations

import copy
import gc
import json
import subprocess
import sys
import weakref
from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch
from torch import nn

from traincheck._model_state import iter_model_tensors
from traincheck.config import config
from traincheck.instrumentor import tracer
from traincheck.instrumentor.proxy_wrapper.subclass import proxy_parameter
from traincheck.observation import (
    ModelSnapshot,
    compare_model_snapshots,
    compare_ranks,
    read_snapshot,
    snapshot_model,
    write_snapshot,
)


def capture(model: nn.Module, rank: int = 0, phase: str = "selected") -> ModelSnapshot:
    return snapshot_model(model, run_id="test", rank=rank, phase=phase)


def model_with_state() -> nn.Module:
    model = nn.Linear(3, 2)
    model.register_parameter("tied_weight", model.weight)
    model.register_buffer("running", torch.tensor([0.0, -0.0, float("nan")]))
    model.register_buffer("temporary", torch.tensor(True), persistent=False)
    model.register_buffer("empty", torch.empty(2, 0))
    model.register_buffer("absent", None)
    return model


def test_snapshot_keeps_all_names_buffers_aliases_and_registration_order() -> None:
    model = model_with_state()
    snapshot = capture(model)
    assert [tensor.name for tensor in snapshot.tensors] == [
        "weight",
        "bias",
        "tied_weight",
        "running",
        "temporary",
        "empty",
    ]
    assert snapshot.tensors[2].object_alias_of == "weight"
    assert snapshot.tensors[2].sha256 == snapshot.tensors[0].sha256
    assert [tensor.name for tensor in snapshot.tensors if tensor.kind == "buffer"] == [
        "running",
        "temporary",
        "empty",
    ]
    assert [name for _, name, _ in iter_model_tensors(model)] == ["weight", "bias"]


@pytest.mark.parametrize(
    "value",
    [
        torch.tensor(4.0),
        torch.tensor([True, False]),
        torch.arange(6, dtype=torch.bfloat16).reshape(2, 3),
        torch.tensor([1 + 2j, 3 - 4j]).conj(),
        torch.arange(6.0).reshape(2, 3).T,
        torch.empty(0, 2),
    ],
)
def test_supported_representations_hash_exact_logical_bytes(
    value: torch.Tensor,
) -> None:
    model = nn.Module()
    model.register_buffer("value", value)
    first = capture(model)
    model.value = value.resolve_conj().resolve_neg().contiguous().clone()
    assert compare_model_snapshots(first, capture(model)).equal


def test_every_row_sub_quantization_changes_and_cached_hash_are_observed() -> None:
    model = nn.Module()
    value = torch.tensor([[0.0, 0.0], [0.0, 0.0]], dtype=torch.float64)
    value.__dict__["_traincheck_tensor_hash"] = 123
    model.register_buffer("value", value)
    first = capture(model)
    value[1, 1] = 1e-12
    second = capture(model)
    comparison = compare_model_snapshots(first, second)
    assert not comparison.equal
    assert comparison.differences[0].fields == ("sha256",)
    assert value.__dict__["_traincheck_tensor_hash"] == 123


def test_signed_zero_and_distinct_nan_payloads_are_not_normalized() -> None:
    model = nn.Module()
    model.register_buffer("value", torch.tensor([0.0]))
    before = capture(model)
    model.value[0] = -0.0
    assert not compare_model_snapshots(before, capture(model)).equal
    model.value = torch.tensor([0x7FC00001], dtype=torch.int32).view(torch.float32)
    before_nan = capture(model)
    model.value = torch.tensor([0x7FC00002], dtype=torch.int32).view(torch.float32)
    assert not compare_model_snapshots(before_nan, capture(model)).equal


def test_snapshot_preserves_rng_gradients_optimizer_identity_and_model_state() -> None:
    model = nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    tensor_ids = [id(parameter) for parameter in model.parameters()]
    gradients = [parameter.grad.clone() for parameter in model.parameters()]
    state = copy.deepcopy(model.state_dict())
    parameter_metadata = [dict(parameter.__dict__) for parameter in model.parameters()]
    rng = torch.get_rng_state().clone()
    captured = capture(model)
    assert torch.equal(rng, torch.get_rng_state())
    assert tensor_ids == [id(parameter) for parameter in model.parameters()]
    assert tensor_ids == [
        id(parameter) for parameter in optimizer.param_groups[0]["params"]
    ]
    assert parameter_metadata == [
        dict(parameter.__dict__) for parameter in model.parameters()
    ]
    for parameter, gradient in zip(model.parameters(), gradients):
        assert torch.equal(parameter.grad, gradient)
    for name, value in state.items():
        assert torch.equal(model.state_dict()[name], value)
    with torch.no_grad():
        model.weight.add_(1)
    assert not compare_model_snapshots(captured, capture(model)).equal


def test_snapshot_does_not_retain_model_or_tensors() -> None:
    model = nn.Linear(2, 2)
    model_reference = weakref.ref(model)
    tensor_reference = weakref.ref(model.weight)
    snapshot = capture(model)
    del model
    gc.collect()
    assert model_reference() is None
    assert tensor_reference() is None
    assert len(snapshot.tensors) == 2


def test_existing_subclass_tracker_works_without_promoting_new_parameters() -> None:
    model = nn.Linear(2, 2)
    before = capture(model)
    proxy_parameter(model, parent_name="observed", should_dump_trace=False)
    tracked = capture(model)
    assert compare_model_snapshots(before, tracked).equal
    ordinary = nn.Linear(2, 2)
    capture(ordinary)
    assert type(ordinary.weight) is nn.Parameter


def test_recorder_operations_do_not_emit_api_records(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = nn.Linear(2, 2)
    records: list[object] = []
    monkeypatch.setattr(config, "DISABLE_WRAPPER", False)
    monkeypatch.setattr(tracer, "dump_trace_API", records.append)
    monkeypatch.setattr(
        torch.Tensor,
        "detach",
        tracer.wrapper(
            torch.Tensor.detach,
            is_bound_method=True,
            scan_proxy_in_args=False,
            dump_stack_trace=False,
            dump_ret=False,
        ),
    )
    capture(model)
    assert records == []
    assert not tracer.tracing_suppressed()


def test_active_parameter_tracker_and_api_wrapper_do_not_observe_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from traincheck.instrumentor.proxy_wrapper import subclass
    from traincheck.instrumentor.proxy_wrapper.proxy_registry import get_global_registry

    model = nn.Linear(2, 2)
    proxy_parameter(model, parent_name="combined", should_dump_trace=False)
    metadata = [dict(parameter.__dict__) for parameter in model.parameters()]
    registry = get_global_registry()
    entries = {
        name: (id(entry), entry.stale) for name, entry in registry.registry.items()
    }
    records: list[object] = []
    monkeypatch.setattr(config, "DISABLE_WRAPPER", False)
    monkeypatch.setattr(tracer, "dump_trace_API", records.append)
    monkeypatch.setattr(subclass, "dump_trace_VAR", records.append)
    monkeypatch.setattr(
        torch.Tensor,
        "detach",
        tracer.wrapper(
            torch.Tensor.detach,
            is_bound_method=True,
            scan_proxy_in_args=True,
            dump_stack_trace=False,
            dump_ret=False,
        ),
    )
    capture(model)
    assert records == []
    assert metadata == [dict(parameter.__dict__) for parameter in model.parameters()]
    assert entries == {
        name: (id(entry), entry.stale) for name, entry in registry.registry.items()
    }


def test_legacy_observer_does_not_change_parameters_during_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from traincheck.instrumentor.proxy_wrapper import subclass
    from traincheck.instrumentor.proxy_wrapper.proxy_observer import (
        add_observer_to_func,
    )

    model = nn.Linear(2, 2)
    proxy_parameter(model, parent_name="legacy", should_dump_trace=False)
    metadata = [dict(parameter.__dict__) for parameter in model.parameters()]
    records: list[object] = []
    monkeypatch.setattr(config, "DISABLE_WRAPPER", False)
    monkeypatch.setattr(subclass, "dump_trace_VAR", records.append)
    monkeypatch.setattr(
        torch.Tensor, "detach", add_observer_to_func(torch.Tensor.detach)
    )
    capture(model)
    assert records == []
    assert metadata == [dict(parameter.__dict__) for parameter in model.parameters()]


def test_observation_import_and_capture_have_no_framework_patch_side_effects() -> None:
    script = """
import sys
import tokenize
import torch
import torch.optim.optimizer as optimizer
from torch._C._distributed_c10d import ProcessGroup
before = (torch.nn.Module.__setattr__, ProcessGroup.broadcast, ProcessGroup.allreduce,
          ProcessGroup.allgather, tokenize._tokenize, optimizer._default_to_fused_or_foreach)
from traincheck.observation import snapshot_model
model = torch.nn.Linear(2, 2)
snapshot_model(model, run_id='isolated', rank=0, phase='chosen')
after = (torch.nn.Module.__setattr__, ProcessGroup.broadcast, ProcessGroup.allreduce,
         ProcessGroup.allgather, tokenize._tokenize, optimizer._default_to_fused_or_foreach)
assert all(left is right for left,right in zip(before,after))
assert not any(name.startswith('traincheck.instrumentor') for name in sys.modules)
assert all(not key.startswith('_tc_') for key in vars(model))
assert type(model.weight) is torch.nn.Parameter
"""
    subprocess.run(
        [sys.executable, "-c", script], check=True, capture_output=True, text=True
    )


def test_lazy_public_exports_keep_existing_objects_and_errors() -> None:
    script = """
import traincheck
expected = {'Instrumentor','get_all_subclasses','annotate_stage','annotate_answer_start_token_ids'}
assert set(traincheck.__all__) == expected
assert expected <= set(dir(traincheck))
try:
    traincheck.nonexistent_export
except AttributeError:
    pass
else:
    raise AssertionError('unknown export must fail')
from traincheck import Instrumentor, get_all_subclasses
from traincheck.instrumentor.tracer import Instrumentor as implementation
assert Instrumentor is implementation
namespace = {}
exec('from traincheck import *', namespace)
assert all(namespace[name] is getattr(traincheck,name) for name in expected)
"""
    subprocess.run(
        [sys.executable, "-c", script], check=True, capture_output=True, text=True
    )


@pytest.mark.parametrize("kind", ["sparse", "meta", "quantized", "subclass", "flat"])
def test_unsupported_tensors_fail_explicitly_without_partial_capture(kind: str) -> None:
    model = nn.Module()
    model.register_buffer("good", torch.ones(2))
    if kind == "sparse":
        bad = torch.eye(2).to_sparse()
    elif kind == "meta":
        bad = torch.empty(2, device="meta")
    elif kind == "quantized":
        bad = torch.quantize_per_tensor(torch.ones(2), 0.1, 0, torch.qint8)
    elif kind == "subclass":

        class CustomTensor(torch.Tensor):
            pass

        bad = torch.ones(2).as_subclass(CustomTensor)
    else:
        bad = torch.ones(2)
        bad.__dict__["_is_flat_param"] = True
    model.register_buffer("bad", bad)
    with pytest.raises((TypeError, ValueError)):
        capture(model)
    assert not tracer.tracing_suppressed()


def test_empty_model_is_not_reported_as_successful_observation() -> None:
    with pytest.raises(ValueError, match="nonempty"):
        capture(nn.Module())


def test_custom_enumeration_and_cycles_fail_before_silently_omitting_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = nn.Linear(2, 2)
    monkeypatch.setattr(model, "named_parameters", lambda **kwargs: iter(()))
    with pytest.raises(TypeError, match="custom named_parameters"):
        capture(model)
    parent = nn.Sequential(model)
    with pytest.raises(TypeError, match="custom named_parameters"):
        capture(parent)
    cyclic = nn.Module()
    cyclic.child = cyclic
    with pytest.raises(ValueError, match="cyclic"):
        capture(cyclic)


def test_capture_failure_propagates_and_restores_tracing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from traincheck import observation

    failure = RuntimeError("snapshot hash failure")

    def fail_hash(tensor: torch.Tensor) -> int:
        raise failure

    monkeypatch.setattr(observation, "exact_tensor_hash", fail_hash)
    with pytest.raises(RuntimeError) as caught:
        capture(nn.Linear(2, 2))
    assert caught.value is failure
    assert not tracer.tracing_suppressed()


@pytest.mark.parametrize(
    "module_name",
    [
        "traincheck.instrumentor.tracer",
        "traincheck.instrumentor.proxy_wrapper.subclass",
    ],
)
def test_present_blocked_tracker_module_is_not_treated_as_absent(
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
) -> None:
    model = nn.Linear(2, 2)
    proxy_parameter(model, parent_name="blocked", should_dump_trace=False)
    monkeypatch.setitem(sys.modules, module_name, None)
    with pytest.raises(TypeError, match="Python module"):
        capture(model)


def test_existing_exact_tensor_hash_keeps_digest_and_cached_hash_behavior() -> None:
    from traincheck.instrumentor.proxy_wrapper.hash import tensor_hash

    value = torch.tensor([0, 1, 255], dtype=torch.uint8)
    expected = int(
        "26a66b061e8f48f39927c312f25293959729eee95978e2892d49d3512a5cc092", 16
    )
    assert tensor_hash(value, with_parallel=False) == expected
    value.__dict__["_traincheck_tensor_hash"] = 99
    assert tensor_hash(value, with_parallel=False) == 99
    with pytest.raises(TypeError):
        tensor_hash(torch.ones(2, dtype=torch.bfloat16), with_parallel=False)


def test_sampler_default_retains_custom_no_argument_parameter_iterator() -> None:
    from collections.abc import Iterator

    class CustomModule(nn.Linear):
        def named_parameters(self) -> Iterator[tuple[str, nn.Parameter]]:
            yield "custom.weight", self.weight

    model = CustomModule(2, 2)
    entries = tuple(iter_model_tensors(model))
    assert len(entries) == 1
    assert entries[0][1] == "custom.weight"
    assert entries[0][2] is model.weight


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_snapshots_preserve_rng_and_match_cpu() -> None:
    model = nn.Linear(2, 2, device="cuda")
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state().clone()
    gpu_snapshot = capture(model)
    assert torch.equal(torch.get_rng_state(), cpu_rng)
    assert torch.equal(torch.cuda.get_rng_state(), cuda_rng)
    assert compare_model_snapshots(
        gpu_snapshot, capture(copy.deepcopy(model).cpu())
    ).equal


def test_round_trip_overwrite_guard_and_required_fields(tmp_path: Path) -> None:
    snapshot = capture(model_with_state())
    path = tmp_path / "snapshot.json"
    write_snapshot(snapshot, path)
    assert read_snapshot(path) == snapshot
    with pytest.raises(FileExistsError):
        write_snapshot(snapshot, path)
    value = asdict(snapshot)
    del value["phase"]
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="fields"):
        read_snapshot(path)


@pytest.mark.parametrize(
    "field,value",
    [
        ("rank", True),
        ("step", -1),
        ("phase", ""),
        ("schema_version", 2),
        ("tensors", []),
    ],
)
def test_malformed_top_level_values_fail(
    tmp_path: Path, field: str, value: object
) -> None:
    data = asdict(capture(nn.Linear(2, 2)))
    data[field] = value
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        read_snapshot(path)


@pytest.mark.parametrize(
    "field,value",
    [
        ("shape", [True]),
        ("dtype", "float"),
        ("sha256", ""),
        ("requires_grad", 1),
        ("object_alias_of", "absent"),
    ],
)
def test_malformed_tensor_values_fail(
    tmp_path: Path, field: str, value: object
) -> None:
    data = asdict(capture(nn.Linear(2, 2)))
    data["tensors"][0][field] = value
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        read_snapshot(path)


def test_duplicate_json_keys_and_tensor_names_fail(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text('{"rank":0,"rank":1}')
    with pytest.raises(ValueError, match="duplicate JSON"):
        read_snapshot(path)
    snapshot = capture(nn.Linear(2, 2))
    with pytest.raises(ValueError, match="duplicate tensor"):
        replace(snapshot, tensors=(snapshot.tensors[0], snapshot.tensors[0]))


def test_comparison_preserves_order_names_dtype_aliases_and_provenance() -> None:
    original = capture(model_with_state())
    peer = replace(
        original,
        rank=1,
        phase="after",
        step=1,
        time_ns=0,
        tensors=tuple(replace(tensor, device="cuda:1") for tensor in original.tensors),
    )
    assert compare_model_snapshots(original, peer).equal
    assert not compare_model_snapshots(
        original, replace(original, tensors=original.tensors[:-1])
    ).equal
    # Untying an alias matters even when every element remains identical.
    untied = replace(
        original,
        tensors=tuple(
            replace(tensor, object_alias_of=None) for tensor in original.tensors
        ),
    )
    assert not compare_model_snapshots(original, untied).equal
    renamed = replace(
        original,
        tensors=(*original.tensors[:-1], replace(original.tensors[-1], name="other")),
    )
    assert not compare_model_snapshots(original, renamed).equal
    with pytest.raises(ValueError, match="same run"):
        compare_model_snapshots(original, replace(original, run_id="different"))


def test_rank_comparison_requires_exact_coverage_and_matching_observation() -> None:
    first = capture(nn.Linear(2, 2))
    second = replace(first, rank=1)
    assert compare_ranks([second, first], expected_ranks=[0, 1]).equal
    with pytest.raises(ValueError, match="coverage"):
        compare_ranks([first], expected_ranks=[0, 1])
    with pytest.raises(ValueError, match="duplicate"):
        compare_ranks([first, first], expected_ranks=[0, 1])
    with pytest.raises(ValueError, match="phase and step"):
        compare_ranks([first, replace(second, phase="later")], expected_ranks=[0, 1])
    with pytest.raises(ValueError, match="distinct"):
        compare_ranks([first], expected_ranks=[0])
    with pytest.raises(ValueError, match="same run"):
        compare_ranks(
            [first, replace(second, model_name="other")], expected_ranks=[0, 1]
        )
