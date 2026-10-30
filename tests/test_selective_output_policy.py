"""Selective collection retains the evidence required by API invariants."""

import itertools
import json
import logging
import sys
import tempfile
import unittest
from collections.abc import Callable, Sequence
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

import traincheck.collect_trace as collect_trace
import traincheck.config.config as config
from traincheck.config.config import InstrOpt
from traincheck.instrumentor import dumper, tracer
from traincheck.invariant.base_cls import (
    APIParam,
    Arguments,
    GroupedPreconditions,
    InputOutputParam,
    Invariant,
    Param,
    Preconditions,
    UnconditionalPrecondition,
    VarTypeParam,
    read_inv_file,
)
from traincheck.invariant.consistency_transient_vars import (
    ConsistentInputOutputRelation,
    ConsistentOutputRelation,
)
from traincheck.invariant.contain_relation import APIContainRelation
from traincheck.invariant.DistinctArgumentRelation import (
    EXP_GROUP_NAME,
    DistinctArgumentRelation,
)
from traincheck.trace.trace_pandas import TracePandas, read_trace_file_Pandas
from traincheck.utils import typename


def tensor_result(value: torch.Tensor) -> torch.Tensor:
    return value + 1


def reference_tensor(value: torch.Tensor) -> torch.Tensor:
    return value.clone()


def distinct_input(shared: str, key: int) -> int:
    return key


def unconditional(group: str = "pre_event") -> GroupedPreconditions:
    return GroupedPreconditions({group: Preconditions([UnconditionalPrecondition()])})


def output_rule(expected: object = "torch.float32") -> Invariant:
    return Invariant(
        ConsistentOutputRelation,
        [
            APIParam(typename(tensor_result)),
            VarTypeParam("torch.Tensor", "dtype", const_value=expected),
        ],
        unconditional(),
        "returned tensors have the expected dtype",
    )


def io_rule(function: str) -> Invariant:
    return Invariant(
        ConsistentInputOutputRelation,
        [
            InputOutputParam(None, 0, "torch.Tensor", ("dtype",), function, True),
            APIParam(function),
            InputOutputParam(None, 0, "torch.Tensor", ("dtype",), function, False),
        ],
        unconditional(),
        "input and output dtypes match",
    )


def distinct_rule() -> Invariant:
    return Invariant(
        DistinctArgumentRelation,
        [APIParam(typename(distinct_input))],
        unconditional(EXP_GROUP_NAME),
        "calls in one step have distinct arguments",
    )


def argument_rule(function: str) -> Invariant:
    return Invariant(
        APIContainRelation,
        [
            APIParam(
                function,
                arguments=Arguments({"0": {"str": "shared"}}, {}, function),
            ),
            APIParam(function),
        ],
        unconditional("parent_func_call_pre"),
        "an independent rule requests only the first argument",
    )


class JsonTraceDestination:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def add_trace(self, record: dict[str, object]) -> None:
        # Replace only the final buffer destination. Keep real metadata
        # conversion, timestamping, and production JSON serialization.
        self.lines.append(dumper.serialize(record))


def collect_calls(
    root: Path,
    invariants: list[Invariant],
    function: Callable[..., object],
    calls: Sequence[tuple[object, ...]],
) -> TracePandas:
    reference = io_rule(typename(reference_tensor))
    invariant_file = root / "invariants.jsonl"
    invariant_file.write_text(
        "".join(json.dumps(inv.to_dict()) + "\n" for inv in [*invariants, reference])
    )
    loaded_invariants = read_inv_file(str(invariant_file))
    options = collect_trace.get_per_func_instr_opts(loaded_invariants)
    saved: object = InstrOpt.from_json(InstrOpt(options, None, True).to_json())
    assert isinstance(saved, InstrOpt)
    destination = JsonTraceDestination()
    with (
        patch.object(
            dumper, "THREAD_DATA", SimpleNamespace(api_trace_buffer=destination)
        ),
        patch.object(config, "DISABLE_WRAPPER", False),
        patch.object(config, "INSTR_OPTS", saved),
        patch.object(
            tracer,
            "get_instrumentation_logger_for_process",
            return_value=logging.getLogger(__name__),
        ),
        patch.object(
            dumper,
            "get_instrumentation_logger_for_process",
            return_value=logging.getLogger(__name__),
        ),
        patch.dict(config.INSTRUMENTATION_POLICY, dict(config.INSTRUMENTATION_POLICY)),
        patch.dict(tracer.META_VARS, {"step": 0}, clear=True),
        patch.dict(
            dumper.tensor_dump_format,
            dump_tensor_hash=True,
            dump_tensor_stats=False,
            dump_tensor_full=False,
        ),
    ):
        instrumentor = tracer.Instrumentor(
            sys.modules[__name__],
            scan_proxy_in_args=False,
            use_full_instr=False,
            funcs_to_instr=list(options),
        )
        # An ordinary IO-instrumented API ensures missing output values take
        # the same normal-loader path as a real trace containing other returns.
        instrumentor.get_wrapped_function(reference_tensor)(torch.ones(2))
        wrapped = instrumentor.get_wrapped_function(function)
        for arguments in calls:
            wrapped(*arguments)

    trace_file = root / "trace_API_fixture.jsonl"
    trace_file.write_text("\n".join(destination.lines) + "\n")
    return read_trace_file_Pandas(str(trace_file), truncate_incomplete_func_calls=False)


class SelectiveOutputPolicyTests(unittest.TestCase):
    def test_output_only_rule_requests_full_returns_without_model_tracking(
        self,
    ) -> None:
        invariant = output_rule()
        options = collect_trace.get_per_func_instr_opts([invariant])[
            typename(tensor_result)
        ]
        self.assertTrue(options["dump_ret"])
        self.assertFalse(options["dump_args"])
        self.assertFalse(options["scan_proxy_in_args"])
        self.assertEqual(options["var_types_to_track"], {})
        self.assertNotIn("dump_ret_config", options)
        self.assertIsNone(
            collect_trace.get_model_tracker_instr_opts([invariant], "subclass")
        )

    def test_return_requirement_is_an_order_independent_union(self) -> None:
        function = typename(tensor_result)
        rules = [output_rule(), output_rule("torch.float64"), argument_rule(function)]
        expected = collect_trace.get_per_func_instr_opts(rules)
        for order in itertools.permutations(rules):
            with self.subTest(order=[rule.relation.__name__ for rule in order]):
                self.assertEqual(
                    collect_trace.get_per_func_instr_opts(list(order)), expected
                )
        self.assertTrue(expected[function]["dump_args"])
        self.assertTrue(expected[function]["dump_ret"])
        self.assertIn("dump_args_config", expected[function])
        self.assertNotIn("dump_ret_config", expected[function])

    def test_full_arguments_dominate_narrow_selectors_in_both_orders(self) -> None:
        function = typename(distinct_input)
        for full_rule in (distinct_rule(), io_rule(function)):
            for rules in (
                [full_rule, argument_rule(function)],
                [argument_rule(function), full_rule],
            ):
                with self.subTest(relation=full_rule.relation.__name__, order=rules):
                    options = collect_trace.get_per_func_instr_opts(rules)[function]
                    self.assertTrue(options["dump_args"])
                    self.assertNotIn("dump_args_config", options)
                    self.assertEqual(
                        options["dump_ret"],
                        full_rule.relation == ConsistentInputOutputRelation,
                    )

    def test_output_rule_rejects_malformed_selector_schema(self) -> None:
        cases: list[list[Param]] = [
            [],
            [APIParam("torch.fake")],
            [VarTypeParam("torch.Tensor", "dtype"), APIParam("torch.fake")],
            [APIParam("torch.fake"), APIParam("torch.other")],
            [
                APIParam("torch.fake"),
                VarTypeParam("torch.Tensor", "dtype"),
                APIParam("torch.extra"),
            ],
        ]
        for params in cases:
            with self.subTest(params=params):
                invariant = Invariant(ConsistentOutputRelation, params, unconditional())
                with self.assertRaises((TypeError, ValueError)):
                    collect_trace.get_per_func_instr_opts([invariant])

    def test_output_rule_rejects_invalid_names_and_missing_constant(self) -> None:
        for target, field in ((0, "api_full_name"), (1, "var_type"), (1, "attr_name")):
            for value in ("", None, 7):
                with self.subTest(field=field, value=value):
                    invariant = output_rule()
                    setattr(invariant.params[target], field, value)
                    with self.assertRaisesRegex(ValueError, field):
                        collect_trace.get_per_func_instr_opts([invariant])
        invariant = Invariant(
            ConsistentOutputRelation,
            [APIParam("torch.fake"), VarTypeParam("torch.Tensor", "dtype")],
            unconditional(),
        )
        with self.assertRaisesRegex(ValueError, "const_value"):
            collect_trace.get_per_func_instr_opts([invariant])
        # None is a legitimate expected value, distinct from an absent value.
        self.assertTrue(
            collect_trace.get_per_func_instr_opts([output_rule(None)])[
                typename(tensor_result)
            ]["dump_ret"]
        )

    def test_distinct_rule_rejects_malformed_selectors(self) -> None:
        cases: list[list[Param]] = [
            [],
            [VarTypeParam("torch.Tensor", "dtype")],
            [APIParam("one"), APIParam("two")],
        ]
        for params in cases:
            invariant = Invariant(
                DistinctArgumentRelation, params, unconditional(EXP_GROUP_NAME)
            )
            with (
                self.subTest(params=params),
                self.assertRaises((TypeError, ValueError)),
            ):
                collect_trace.get_per_func_instr_opts([invariant])
        for name in ("", None, 7):
            invariant = distinct_rule()
            setattr(invariant.params[0], "api_full_name", name)
            with (
                self.subTest(name=name),
                self.assertRaisesRegex(ValueError, "api_full_name"),
            ):
                collect_trace.get_per_func_instr_opts([invariant])

    def test_real_return_trace_passes_healthy_rule_and_rejects_wrong_dtype(
        self,
    ) -> None:
        invariant = output_rule()
        with tempfile.TemporaryDirectory() as directory:
            trace = collect_calls(
                Path(directory),
                [invariant],
                tensor_result,
                [(torch.ones(2),), (torch.ones(3),)],
            )
            self.assertEqual(len(trace.get_func_call_ids(typename(tensor_result))), 2)
            healthy = invariant.check(trace, check_relation_first=False)
            wrong = output_rule("torch.float64").check(
                trace, check_relation_first=False
            )
        self.assertTrue(healthy.triggered)
        self.assertTrue(healthy.check_passed)
        self.assertTrue(wrong.triggered)
        self.assertFalse(wrong.check_passed)

    def test_real_distinct_trace_preserves_key_and_detects_actual_duplicates(
        self,
    ) -> None:
        invariant = distinct_rule()
        rules = [invariant, argument_rule(typename(distinct_input))]
        for keys, should_pass in (((1, 2), True), ((1, 1), False)):
            with self.subTest(keys=keys), tempfile.TemporaryDirectory() as directory:
                trace = collect_calls(
                    Path(directory),
                    rules,
                    distinct_input,
                    [("shared", key) for key in keys],
                )
                self.assertEqual(
                    len(trace.get_func_call_ids(typename(distinct_input))), 2
                )
                result = invariant.check(trace, check_relation_first=False)
                self.assertTrue(result.triggered)
                self.assertEqual(result.check_passed, should_pass)


if __name__ == "__main__":
    unittest.main()
