"""Unsupported argument selectors must retain complete checker evidence."""

from __future__ import annotations

import copy
import inspect
import json
import tempfile
import unittest
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tests.test_selective_context_requirements import (
    JsonTraceDestination,
    policy_wrapper,
    string_map,
)
from tools.fresh_validation.check_coverage import ObservationReport, observe
from traincheck.collect_trace import get_per_func_instr_opts
from traincheck.config.config import InstrOpt
from traincheck.instrumentor import dumper, tracer
from traincheck.invariant.base_cls import (
    APIParam,
    Arguments,
    GroupedPreconditions,
    Invariant,
    Preconditions,
    UnconditionalPrecondition,
    load_function_signature,
)
from traincheck.invariant.contain_relation import (
    APIContainRelation,
    cache_events_scanner,
)
from traincheck.trace.trace_pandas import read_trace_file_Pandas
from traincheck.utils import typename


def positional_child(value: int, /, other: int = 0) -> None:
    pass


def mixed_child(value: int, /, other: int = 0, *, flag: bool = False) -> None:
    pass


def variadic_child(receiver: str, /, *values: int, **options: int) -> None:
    pass


def keyword_child(**options: int) -> None:
    pass


def unknown_child(value: int) -> None:
    pass


def parent_call(
    child: Callable[..., object],
    arguments: tuple[object, ...],
    keywords: dict[str, object],
) -> None:
    child(*arguments, **keywords)


# Discovery with `-s tests` imports this module without the `tests.` prefix.
# TrainCheck resolves fixture signatures from their package-qualified API names,
# so use the canonical module's actual callables in both unittest invocations.
# This leaves the production loader and the functions' metadata unchanged.
if __name__ != "tests.test_selective_argument_kinds":
    from tests import test_selective_argument_kinds as _fixtures

    positional_child = _fixtures.positional_child
    mixed_child = _fixtures.mixed_child
    variadic_child = _fixtures.variadic_child
    keyword_child = _fixtures.keyword_child
    unknown_child = _fixtures.unknown_child
    parent_call = _fixtures.parent_call


def containment_rule(child: Callable[..., object], arguments: Arguments) -> Invariant:
    return Invariant(
        APIContainRelation,
        [
            APIParam(typename(parent_call)),
            APIParam(typename(child), arguments=arguments),
        ],
        GroupedPreconditions(
            {"parent_func_call_pre": Preconditions([UnconditionalPrecondition()])}
        ),
        "parent calls child with the required argument metadata",
    )


def required_arguments(rule: Invariant) -> Arguments:
    child = rule.params[1]
    assert isinstance(child, APIParam)
    arguments = child.arguments
    assert isinstance(arguments, Arguments)
    return arguments


def collect_and_check(
    rule: Invariant,
    child: Callable[..., object],
    arguments: tuple[object, ...],
    keywords: dict[str, object],
) -> tuple[ObservationReport, list[dict[str, object]]]:
    loaded: object = InstrOpt.from_json(
        InstrOpt(get_per_func_instr_opts([rule]), None, True).to_json()
    )
    assert isinstance(loaded, InstrOpt)
    policy = string_map(loaded.funcs_instr_opts)
    destination = JsonTraceDestination()
    with (
        patch.object(tracer.config, "DISABLE_WRAPPER", False),
        patch.object(
            dumper, "THREAD_DATA", SimpleNamespace(api_trace_buffer=destination)
        ),
        patch.object(
            tracer, "get_meta_vars", return_value={"step": 1, "stage": "training"}
        ),
    ):
        wrapped_child = policy_wrapper(
            child, typename(child), string_map(policy[typename(child)])
        )
        wrapped_parent = policy_wrapper(
            parent_call,
            typename(parent_call),
            string_map(policy[typename(parent_call)]),
        )
        wrapped_parent(wrapped_child, arguments, keywords)
    records = [string_map(json.loads(line)) for line in destination.lines]
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "trace_API_arguments.jsonl"
        path.write_text("\n".join(destination.lines) + "\n")
        trace = read_trace_file_Pandas(str(path), truncate_incomplete_func_calls=False)
        cache_events_scanner.clear()
        try:
            report = observe(trace, rule)
        finally:
            cache_events_scanner.clear()
    return report, records


class SelectiveArgumentKindsTests(unittest.TestCase):
    def test_fixture_signatures_resolve_under_both_unittest_invocations(self) -> None:
        fixtures: tuple[Callable[..., object], ...] = (
            positional_child,
            mixed_child,
            variadic_child,
            keyword_child,
            unknown_child,
            parent_call,
        )
        for function in fixtures:
            with self.subTest(function=typename(function)):
                name = typename(function)
                self.assertTrue(name.startswith("tests.test_selective_argument_kinds."))
                self.assertEqual(
                    load_function_signature(name), inspect.signature(function)
                )

    def test_positional_only_trace_passes_and_detects_changed_argument(self) -> None:
        child = typename(positional_child)
        rule = containment_rule(
            positional_child, Arguments({"0": {"int": 7}}, {}, child)
        )
        original = copy.deepcopy(rule.to_dict())
        for value, expected in ((7, True), (8, False)):
            with self.subTest(value=value):
                report, records = collect_and_check(
                    rule, positional_child, (value,), {}
                )
                self.assertEqual(report["status"], "completed", report)
                self.assertIs(report["active"], True, report)
                self.assertEqual(report["decisions_completed"], 1)
                self.assertIs(report["check_passed"], expected)
                before = [
                    record
                    for record in records
                    if record["function"] == child
                    and record["type"] == tracer.TraceLineType.FUNC_CALL_PRE
                ]
                self.assertEqual(len(before), 1)
                self.assertEqual(string_map(before[0]["args"])["0"], {"int": value})
        self.assertEqual(rule.to_dict(), original)

    def test_variadic_positional_trace_passes_and_detects_changed_argument(
        self,
    ) -> None:
        child = typename(variadic_child)
        rule = containment_rule(variadic_child, Arguments({"1": {"int": 7}}, {}, child))
        for value, expected in ((7, True), (8, False)):
            with self.subTest(value=value):
                report, records = collect_and_check(
                    rule, variadic_child, ("receiver", value), {}
                )
                self.assertIs(report["active"], True, report)
                self.assertIs(report["check_passed"], expected, report)
                before = [
                    record
                    for record in records
                    if record["function"] == child
                    and record["type"] == tracer.TraceLineType.FUNC_CALL_PRE
                ]
                self.assertEqual(string_map(before[0]["args"])["1"], {"int": value})

    def test_variadic_keyword_trace_passes_and_detects_changed_argument(self) -> None:
        child = typename(keyword_child)
        rule = containment_rule(
            keyword_child, Arguments({}, {"options": {"int": 7}}, child)
        )
        for value, expected in ((7, True), (8, False)):
            with self.subTest(value=value):
                report, records = collect_and_check(
                    rule, keyword_child, (), {"options": value}
                )
                self.assertIs(report["active"], True, report)
                self.assertIs(report["check_passed"], expected, report)
                before = [
                    record
                    for record in records
                    if record["function"] == child
                    and record["type"] == tracer.TraceLineType.FUNC_CALL_PRE
                ]
                self.assertEqual(
                    string_map(before[0]["kwargs"])["options"], {"int": value}
                )

    def test_full_fallback_dominates_supported_subset_in_either_rule_order(
        self,
    ) -> None:
        name = typename(positional_child)
        full = containment_rule(
            positional_child, Arguments({"0": {"int": 7}}, {}, name)
        )
        narrow = containment_rule(
            positional_child, Arguments({}, {"other": {"int": 9}}, name)
        )
        expected = get_per_func_instr_opts([full, narrow])
        self.assertNotIn("dump_args_config", expected[name])
        self.assertIs(expected[name]["dump_args"], True)
        self.assertEqual(get_per_func_instr_opts([narrow, full]), expected)

    def test_required_supported_kinds_remain_selective(self) -> None:
        name = typename(mixed_child)
        arguments = Arguments({}, {"other": {"int": 9}, "flag": {"bool": True}}, name)
        rule = containment_rule(mixed_child, arguments)
        options = get_per_func_instr_opts([rule])[name]
        self.assertEqual(
            options["dump_args_config"], {"other": None, 1: None, "flag": None}
        )
        for flag, expected in ((True, True), (False, False)):
            report, _ = collect_and_check(rule, mixed_child, (7, 9), {"flag": flag})
            self.assertIs(report["active"], True, report)
            self.assertIs(report["check_passed"], expected, report)

    def test_unknown_signature_requests_complete_arguments(self) -> None:
        name = typename(unknown_child)
        arguments = Arguments({"0": {"int": 7}}, {}, name)
        arguments.signature = None
        options = get_per_func_instr_opts([containment_rule(unknown_child, arguments)])[
            name
        ]
        self.assertTrue(options["dump_args"])
        self.assertNotIn("dump_args_config", options)

    def test_unresolved_required_name_requests_complete_arguments(self) -> None:
        name = typename(unknown_child)
        arguments = Arguments({}, {"unexpected": {"int": 7}}, name)
        options = get_per_func_instr_opts([containment_rule(unknown_child, arguments)])[
            name
        ]
        self.assertTrue(options["dump_args"])
        self.assertNotIn("dump_args_config", options)

    def test_malformed_signature_or_argument_metadata_is_rejected(self) -> None:
        name = typename(positional_child)
        cases: list[tuple[str, object]] = [
            ("signature", "not a signature"),
            ("arguments", []),
            ("arguments", {7: {"int": 7}}),
            ("arguments", {"": {"int": 7}}),
            ("arguments", {"value": []}),
            ("arguments", {"value": {}}),
            ("arguments", {"value": {"int": 7, "str": "7"}}),
            ("func_name", ""),
            ("func_name", typename(unknown_child)),
        ]
        for field, malformed in cases:
            with self.subTest(field=field, value=malformed):
                arguments = Arguments({"0": {"int": 7}}, {}, name)
                setattr(arguments, field, malformed)
                with self.assertRaises((TypeError, ValueError)):
                    get_per_func_instr_opts(
                        [containment_rule(positional_child, arguments)]
                    )

    def test_malformed_later_requirement_is_not_hidden_by_fallback(self) -> None:
        name = typename(positional_child)
        arguments = Arguments({"0": {"int": 7}}, {}, name)
        setattr(arguments, "arguments", {"value": {"int": 7}, "other": []})
        with self.assertRaises(TypeError):
            get_per_func_instr_opts([containment_rule(positional_child, arguments)])

    def test_no_required_arguments_does_not_force_full_collection(self) -> None:
        name = typename(unknown_child)
        arguments = Arguments({}, {}, name)
        arguments.signature = None
        options = get_per_func_instr_opts([containment_rule(unknown_child, arguments)])[
            name
        ]
        self.assertEqual(options["dump_args_config"], {})


if __name__ == "__main__":
    unittest.main()
