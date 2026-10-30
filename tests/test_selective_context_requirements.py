"""Context-dependent selective policies must preserve real checker activation."""

from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from collections.abc import Callable, Mapping
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace, TracebackType
from typing import Literal
from unittest.mock import patch

from tools.fresh_validation.check_coverage import ObservationReport, observe
from traincheck.collect_trace import get_per_func_instr_opts
from traincheck.config.config import InstrOpt
from traincheck.instrumentor import dumper, tracer
from traincheck.invariant.base_cls import (
    PT,
    APIParam,
    Arguments,
    GroupedPreconditions,
    Invariant,
    Precondition,
    PreconditionClause,
    Preconditions,
    UnconditionalPrecondition,
)
from traincheck.invariant.contain_relation import (
    APIContainRelation,
    cache_events_scanner,
)
from traincheck.trace.trace_pandas import read_trace_file_Pandas

CONTEXT = f"{__name__}.DeviceContext"
PROPERTY = f"meta_vars.context_managers.{CONTEXT}"
PARENT = f"{__name__}.context_parent"
CHILD = f"{__name__}.context_child"


class DeviceContext:
    def __init__(self, device: int) -> None:
        self.device = device

    def __enter__(self) -> DeviceContext:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        return False


def context_child() -> None:
    return None


def context_parent(include_child: bool) -> None:
    if include_child:
        context_child()


def context_rule(*, inverted: bool = False, prop_name: str = PROPERTY) -> Invariant:
    return Invariant(
        APIContainRelation,
        [APIParam(PARENT), APIParam(CHILD)],
        GroupedPreconditions(
            {
                "parent_func_call_pre": Preconditions(
                    [
                        Precondition(
                            [
                                PreconditionClause(
                                    prop_name, None, PT.EXIST, None, None
                                ),
                                PreconditionClause(
                                    prop_name, int, PT.CONSTANT, ("device",), {0}
                                ),
                            ]
                        )
                    ],
                    inverted=inverted,
                )
            }
        ),
        "parent calls child while device context is zero",
    )


def string_map(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError("Expected string-keyed object")
    result: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise TypeError("Expected string-keyed object")
        result[key] = item
    return result


def policy_wrapper(
    original: Callable[..., object],
    name: str,
    options: Mapping[str, object],
    *,
    bound: bool = False,
    constructor: bool = False,
) -> Callable[..., object]:
    dump_args = options["dump_args"]
    dump_ret = options["dump_ret"]
    if not isinstance(dump_args, bool) or not isinstance(dump_ret, bool):
        raise TypeError("Dump options must be boolean")
    # Absent selector is the documented request for complete inputs.
    selector = (
        dict(string_map(options["dump_args_config"]))
        if "dump_args_config" in options
        else None
    )

    def wrapped(*args: object, **kwargs: object) -> object:
        return tracer.function_wrapper(
            original,
            name,
            bound,
            False,
            False,
            dump_args,
            selector,
            dump_ret,
            None,
            False,
            {},
            False,
            constructor,
            "self" if constructor else None,
            *args,
            **kwargs,
        )

    return wrapped


class JsonTraceDestination:
    """Capture the production dumper's serialized bytes without background IO."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def add_trace(self, record: dict[str, object]) -> None:
        self.lines.append(dumper.serialize(record))


def collect_and_check(
    rule: Invariant, *, include_child: bool, device: int = 0
) -> tuple[ObservationReport, list[dict[str, object]]]:
    saved: object = InstrOpt.from_json(
        InstrOpt(get_per_func_instr_opts([rule]), None, True).to_json()
    )
    assert isinstance(saved, InstrOpt)
    policy = string_map(saved.funcs_instr_opts)
    destination = JsonTraceDestination()
    module = sys.modules[__name__]
    with ExitStack() as stack:
        stack.enter_context(patch.object(tracer.config, "DISABLE_WRAPPER", False))
        stack.enter_context(
            patch.object(
                dumper, "THREAD_DATA", SimpleNamespace(api_trace_buffer=destination)
            )
        )
        stack.enter_context(
            patch.object(
                tracer,
                "get_meta_vars",
                return_value={"step": 1, "stage": "training"},
            )
        )
        for method_name in ("__init__", "__enter__", "__exit__"):
            name = f"{CONTEXT}.{method_name}"
            if name in policy:
                original = getattr(DeviceContext, method_name)
                stack.enter_context(
                    patch.object(
                        DeviceContext,
                        method_name,
                        policy_wrapper(
                            original,
                            name,
                            string_map(policy[name]),
                            bound=True,
                            constructor=method_name == "__init__",
                        ),
                    )
                )
        stack.enter_context(
            patch.object(
                module,
                "context_child",
                policy_wrapper(context_child, CHILD, string_map(policy[CHILD])),
            )
        )
        stack.enter_context(
            patch.object(
                module,
                "context_parent",
                policy_wrapper(context_parent, PARENT, string_map(policy[PARENT])),
            )
        )
        with DeviceContext(device):
            context_parent(include_child)
    # No timestamp, identity, payload, or row-order rewriting: use exactly the
    # production dumper output through the ordinary JSONL parser.
    records = [dict(string_map(json.loads(line))) for line in destination.lines]
    with tempfile.TemporaryDirectory() as directory:
        trace_file = Path(directory) / "trace_API_context.jsonl"
        trace_file.write_text("\n".join(destination.lines) + "\n")
        trace = read_trace_file_Pandas(
            str(trace_file), truncate_incomplete_func_calls=False
        )
        cache_events_scanner.clear()
        try:
            report = observe(trace, rule)
        finally:
            cache_events_scanner.clear()
    return report, records


class SelectiveContextRequirementsTests(unittest.TestCase):
    def test_real_context_trace_activates_healthy_rule(self) -> None:
        rule = context_rule()
        original = copy.deepcopy(rule.to_dict())
        report, records = collect_and_check(rule, include_child=True)
        self.assertEqual(report["status"], "completed", report)
        self.assertIs(report["active"], True, report)
        self.assertEqual(report["decisions_completed"], 1)
        self.assertIs(report["check_passed"], True)
        constructor = [
            record
            for record in records
            if record["function"] == f"{CONTEXT}.__init__"
            and record["type"] == tracer.TraceLineType.FUNC_CALL_PRE
        ]
        self.assertEqual(len(constructor), 1)
        arguments = string_map(constructor[0]["args"])
        self.assertEqual(arguments["1"], {"int": 0})
        self.assertEqual(rule.to_dict(), original)

    def test_real_context_trace_detects_missing_child(self) -> None:
        report, _ = collect_and_check(context_rule(), include_child=False)
        self.assertEqual(report["status"], "completed", report)
        self.assertIs(report["active"], True, report)
        self.assertEqual(report["decisions_completed"], 1)
        self.assertIs(report["check_passed"], False)

    def test_inverted_context_requires_same_inputs_and_truth_is_preserved(self) -> None:
        report, _ = collect_and_check(context_rule(inverted=True), include_child=False)
        self.assertEqual(report["status"], "completed", report)
        self.assertIs(report["dormant"], True, report)
        self.assertEqual(report["decisions_completed"], 0)
        active, _ = collect_and_check(
            context_rule(inverted=True), include_child=False, device=1
        )
        self.assertIs(active["active"], True, active)
        self.assertIs(active["check_passed"], False)

    def test_literal_thread_condition_remains_dormant(self) -> None:
        rule = context_rule()
        assert rule.precondition is not None
        rule.precondition.grouped_preconditions["parent_func_call_pre"].preconditions[
            0
        ].clauses.append(PreconditionClause("thread_id", int, PT.CONSTANT, None, {-1}))
        original = copy.deepcopy(rule.to_dict())
        report, _ = collect_and_check(rule, include_child=False)
        self.assertEqual(report["status"], "completed", report)
        self.assertIs(report["dormant"], True, report)
        self.assertEqual(report["decisions_completed"], 0)
        self.assertEqual(rule.to_dict(), original)

    def test_complete_constructor_inputs_override_existing_subset(self) -> None:
        constructor = f"{CONTEXT}.__init__"
        subset = Invariant(
            APIContainRelation,
            [
                APIParam(PARENT),
                APIParam(
                    constructor,
                    arguments=Arguments({"0": {CONTEXT: {}}}, {}, constructor),
                ),
            ],
            GroupedPreconditions(
                {"parent_func_call_pre": Preconditions([UnconditionalPrecondition()])}
            ),
        )
        for rules in ([context_rule(), subset], [subset, context_rule()]):
            with self.subTest(order=[rule.text_description for rule in rules]):
                options = get_per_func_instr_opts(rules)[constructor]
                self.assertIs(options["dump_args"], True)
                self.assertNotIn("dump_args_config", options)
                for method in ("__enter__", "__exit__"):
                    self.assertIn(f"{CONTEXT}.{method}", get_per_func_instr_opts(rules))

    def test_non_context_preconditions_do_not_add_instrumentation(self) -> None:
        rule = context_rule(prop_name="meta_vars.other_metadata")
        self.assertEqual(set(get_per_func_instr_opts([rule])), {PARENT, CHILD})

    def test_missing_preconditions_fail_explicitly(self) -> None:
        rule = context_rule()
        rule.precondition = None
        with self.assertRaisesRegex(ValueError, "precondition"):
            get_per_func_instr_opts([rule])

    def test_malformed_context_names_fail_explicitly(self) -> None:
        for prop in (
            "meta_vars.context_managers",
            "meta_vars.context_managers.",
            "meta_vars.context_managers.torch..device",
        ):
            with self.subTest(property=prop), self.assertRaises(ValueError):
                get_per_func_instr_opts([context_rule(prop_name=prop)])

    def test_malformed_property_type_fails_explicitly(self) -> None:
        rule = context_rule()
        assert rule.precondition is not None
        clause = (
            rule.precondition.grouped_preconditions["parent_func_call_pre"]
            .preconditions[0]
            .clauses[0]
        )
        setattr(clause, "prop_name", 7)
        with self.assertRaisesRegex(TypeError, "property"):
            get_per_func_instr_opts([rule])


if __name__ == "__main__":
    unittest.main()
