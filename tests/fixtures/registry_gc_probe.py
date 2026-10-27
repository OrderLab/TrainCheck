"""Bounded subprocess probe for finalization while registry dumping holds its lock."""

import gc
import sys
import weakref
from collections.abc import Callable
from unittest.mock import patch

from torch import nn

import traincheck.config.config as config
from traincheck.instrumentor.proxy_wrapper import subclass
from traincheck.instrumentor.proxy_wrapper.proxy_registry import VarRegistry
from traincheck.instrumentor.proxy_wrapper.subclass import proxy_parameter


class CallbackVariable:
    def __init__(self, name: str, calls: list[str]) -> None:
        self.name = name
        self.calls = calls
        self.callback: Callable[[], None] | None = None

    def dump_trace(self, phase: str, dump_loc: str | None) -> None:
        self.calls.append(self.name)
        callback = self.callback
        self.callback = None
        if callback is not None:
            callback()


def run_reentrant() -> None:
    registry = VarRegistry()
    calls: list[str] = []
    first = CallbackVariable("first", calls)
    second = CallbackVariable("second", calls)
    registry.add_var(first, "first", "kind")
    registry.add_var(second, "second", "kind")

    def nested_sample() -> None:
        registry.dump_modified(dump_config={"kind": {}})

    first.callback = nested_sample
    registry.dump_modified(dump_config={"kind": {}})
    assert calls == ["first", "second"]
    calls.clear()
    registry.dump_modified(dump_config={"kind": {}})
    assert calls == []

    def mark_second_modified() -> None:
        registry.add_var(second, "second", "kind")

    first.callback = mark_second_modified
    registry.add_var(first, "first", "kind")
    registry.dump_modified(dump_config={"kind": {}})
    assert calls == ["first", "second"]
    calls.clear()

    replacement = CallbackVariable("replacement", calls)
    added = CallbackVariable("added", calls)

    def change_membership() -> None:
        registry.add_var(replacement, "second", "kind")
        registry.add_var(added, "added", "kind")

    first.callback = change_membership
    registry.dump_sample(dump_config={"kind": {}})
    assert calls == ["first"]
    calls.clear()
    registry.dump_modified(dump_config={"kind": {}})
    assert calls == ["replacement", "added"]
    calls.clear()
    registry.dump_modified(dump_config={"kind": {}})
    assert calls == []


def run(mode: str) -> None:
    if mode == "reentrant":
        run_reentrant()
        print("OK reentrant")
        return
    if mode not in ("sample", "modified"):
        raise ValueError(f"Unknown sampling mode: {mode}")
    registry = VarRegistry()
    records: list[dict[str, object]] = []

    def collect_during_conversion(obj: object, value: object) -> dict[str, object]:
        gc.collect()
        return {}

    with (
        patch.object(subclass, "get_global_registry", return_value=registry),
        patch.object(config, "DISABLE_WRAPPER", False),
    ):
        shared = nn.Linear(2, 2)
        first = nn.Sequential(shared)
        second = nn.Sequential(shared)
        setattr(first, "saved_forward", first.forward)
        proxy_parameter(first, parent_name="first", should_dump_trace=False)
        proxy_parameter(second, parent_name="second", should_dump_trace=False)
        first_reference = weakref.ref(first)
        gc_was_enabled = gc.isenabled()
        gc.disable()
        try:
            del first
            assert first_reference() is not None
            with (
                patch.object(
                    subclass, "dump_attributes", side_effect=collect_during_conversion
                ),
                patch.object(subclass, "dump_trace_VAR", side_effect=records.append),
            ):
                if mode == "sample":
                    registry.dump_sample(dump_config={"torch.nn.Parameter": {}})
                else:
                    registry.dump_modified(dump_config={"torch.nn.Parameter": {}})
                assert first_reference() is None
                # The current sample completes under its original canonical paths;
                # deferred retirement publishes surviving-root paths afterwards.
                assert [record["var_name"] for record in records] == [
                    "first.0.weight",
                    "first.0.bias",
                ]
                assert list(registry.registry) == ["second.0.weight", "second.0.bias"]
                records.clear()
                registry.dump_modified(dump_config={"torch.nn.Parameter": {}})
                assert [record["var_name"] for record in records] == [
                    "second.0.weight",
                    "second.0.bias",
                ]
                records.clear()
                registry.dump_modified(dump_config={"torch.nn.Parameter": {}})
                assert records == []
        finally:
            if gc_was_enabled:
                gc.enable()
    print(f"OK {mode}")


if __name__ == "__main__":
    run(sys.argv[1])
