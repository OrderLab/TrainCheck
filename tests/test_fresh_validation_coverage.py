"""Small real-checker counter oracles, including dormant and early-exit cases."""

import copy
import logging
import unittest

import pandas as pd

from tools.fresh_validation.check_coverage import observe
from traincheck.instrumentor.tracer import TraceLineType
from traincheck.invariant.base_cls import (
    PT,
    APIParam,
    GroupedPreconditions,
    InputOutputParam,
    Invariant,
    Precondition,
    PreconditionClause,
    Preconditions,
    UnconditionalPrecondition,
    VarTypeParam,
)
from traincheck.invariant.consistency_transient_vars import (
    ConsistentInputOutputRelation,
    ConsistentOutputRelation,
    ThresholdRelation,
)
from traincheck.invariant.contain_relation import (
    APIContainRelation,
    cache_events_scanner,
)
from traincheck.invariant.DistinctArgumentRelation import (
    EXP_GROUP_NAME,
    DistinctArgumentRelation,
)
from traincheck.trace.trace_pandas import TracePandas


def frame(outputs, *, steps=None, inputs=None):
    rows = []
    for i, output in enumerate(outputs):
        tensor = {"torch.FloatTensor": {"shape": [2], "dtype": "torch.float32"}}
        common = {
            "func_call_id": f"call-{i}",
            "function": "torch.fake",
            "process_id": 1,
            "thread_id": 2,
            "is_bound_method": False,
            "obj_id": None,
            "meta_vars.step": steps[i] if steps else i,
            "args": {"0": copy.deepcopy(tensor if inputs is None else inputs[i])},
            "kwargs": {},
            "return_values": copy.deepcopy(output),
        }
        rows += [
            {**common, "type": TraceLineType.FUNC_CALL_PRE, "time": i * 3},
            {**common, "type": TraceLineType.FUNC_CALL_POST, "time": i * 3 + 1},
        ]
    return TracePandas(pd.DataFrame(rows), truncate_incomplete_func_calls=False)


def precondition(group="pre_event", enabled=True):
    return GroupedPreconditions(
        {group: Preconditions([UnconditionalPrecondition()] if enabled else [])}
    )


def output_rule(enabled=True):
    return Invariant(
        ConsistentOutputRelation,
        [
            APIParam("torch.fake"),
            VarTypeParam("torch.Tensor", "dtype", const_value="torch.float32"),
        ],
        precondition(enabled=enabled),
        "fixture output dtype",
    )


class CoverageTests(unittest.TestCase):
    def setUp(self):
        logging.disable(logging.CRITICAL)
        self.good = {"torch.FloatTensor": {"shape": [2], "dtype": "torch.float32"}}

    def assert_same_result(self, make_trace, inv):
        cache_events_scanner.clear()
        observed = observe(make_trace(), inv)
        cache_events_scanner.clear()
        plain = observe(make_trace(), inv, enabled=False)
        self.assertEqual(observed.get("checker_result"), plain.get("checker_result"))
        self.assertIsNone(plain["active"])
        self.assertIsNone(plain["dormant"])
        self.assertEqual(observed["status"], "completed", observed)
        return observed

    def test_healthy_real_decisions(self):
        got = self.assert_same_result(lambda: frame([self.good] * 3), output_rule())
        self.assertEqual(
            (
                got["opportunities"],
                got["precondition_group_passes"],
                got["decisions_completed"],
            ),
            (3, 3, 3),
        )
        self.assertTrue(got["active"])

    def test_all_preconditions_false_are_dormant(self):
        got = self.assert_same_result(
            lambda: frame([self.good] * 3), output_rule(False)
        )
        self.assertEqual(
            (
                got["opportunities"],
                got["precondition_group_failures"],
                got["decisions_completed"],
            ),
            (3, 3, 0),
        )
        self.assertTrue(got["dormant"])

    def test_missing_io_value_passes_precondition_but_runs_no_predicate(self):
        inv = Invariant(
            ConsistentInputOutputRelation,
            [
                InputOutputParam(None, 0, "torch.Tensor", ("shape",), None, True),
                APIParam("torch.fake"),
                InputOutputParam(None, 0, "torch.Tensor", ("missing",), None, False),
            ],
            precondition(),
            "missing value",
        )
        got = self.assert_same_result(lambda: frame([self.good]), inv)
        self.assertEqual(got["precondition_group_passes"], 1)
        self.assertEqual(got["decisions_completed"], 0)
        self.assertTrue(got["dormant"])

    def test_distinct_single_call_has_zero_pairs_despite_triggered(self):
        inv = Invariant(
            DistinctArgumentRelation,
            [APIParam("torch.fake")],
            precondition(EXP_GROUP_NAME),
            "distinct",
        )
        got = self.assert_same_result(lambda: frame([self.good]), inv)
        self.assertEqual(
            (
                got["opportunities"],
                got["precondition_group_passes"],
                got["decisions_completed"],
            ),
            (0, 1, 0),
        )
        self.assertTrue(got["checker_triggered"])
        self.assertTrue(got["dormant"])

    def test_first_failure_leaves_unvisited_remainder(self):
        wrong = {"torch.FloatTensor": {"shape": [2], "dtype": "torch.float64"}}
        got = self.assert_same_result(
            lambda: frame([wrong, self.good, self.good]), output_rule()
        )
        self.assertEqual(
            (
                got["opportunities"],
                got["decisions_completed"],
                got["unvisited_calls_after_stop"],
            ),
            (3, 1, 2),
        )
        self.assertFalse(got["check_passed"])

    def test_missing_tensor_is_completed_output_violation(self):
        got = self.assert_same_result(lambda: frame([{"int": 1}]), output_rule())
        self.assertEqual(got["decisions_completed"], 1)
        self.assertFalse(got["check_passed"])

    def test_distinct_pair_count_and_first_violation(self):
        inv = Invariant(
            DistinctArgumentRelation,
            [APIParam("torch.fake")],
            precondition(EXP_GROUP_NAME),
            "distinct",
        )
        got = self.assert_same_result(
            lambda: frame([self.good] * 3, steps=[1, 1, 1]), inv
        )
        self.assertEqual((got["opportunities"], got["decisions_completed"]), (3, 1))
        self.assertFalse(got["check_passed"])
        self.assertIsNone(got["unvisited_calls_after_stop"])
        self.assertEqual(got["unvisited_pairs_after_stop"], 2)

    def test_io_actual_comparison_and_failure(self):
        inv = Invariant(
            ConsistentInputOutputRelation,
            [
                InputOutputParam(None, 0, "torch.Tensor", ("shape",), None, True),
                APIParam("torch.fake"),
                InputOutputParam(None, 0, "torch.Tensor", ("shape",), None, False),
            ],
            precondition(),
            "shape agreement",
        )
        for shape, passed in (([2], True), ([9], False)):
            got = self.assert_same_result(
                lambda: frame([{"torch.FloatTensor": {"shape": shape}}]), inv
            )
            self.assertEqual(
                (got["decisions_completed"], got["check_passed"]), (1, passed)
            )

    def test_containment_counts_parent_decisions_even_when_empty(self):
        cache_events_scanner.clear()
        inv = Invariant(
            APIContainRelation,
            [APIParam("torch.fake"), APIParam("torch.child")],
            precondition("parent_func_call_pre"),
            "requires child",
        )
        got = self.assert_same_result(lambda: frame([self.good] * 3), inv)
        self.assertEqual((got["opportunities"], got["decisions_completed"]), (3, 1))
        self.assertFalse(got["check_passed"])

    def test_requested_variable_precondition_skipped_without_proxy_data(self):
        conditions = precondition("parent_func_call_pre")
        conditions.grouped_preconditions["var_events"] = Preconditions(
            [
                Precondition(
                    [
                        PreconditionClause(
                            "meta_vars.phase", str, PT.CONSTANT, None, {"train"}
                        )
                    ]
                )
            ]
        )
        inv = Invariant(
            APIContainRelation,
            [APIParam("torch.fake"), VarTypeParam("torch.Tensor", "shape")],
            conditions,
            "requires variable checking",
        )

        def bound_trace():
            trace = frame([self.good])
            trace.events["is_bound_method"] = True
            trace.events["obj_id"] = 123
            return trace

        got = self.assert_same_result(bound_trace, inv)
        self.assertTrue(got["variable_precondition_requested"])
        self.assertTrue(got["variable_check_skipped"])
        self.assertFalse(got["fully_validated_active"])

    def test_threshold_comparison_after_successful_extraction(self):
        # torch.clamp has a real signature; max=3 is the checker threshold.
        def make_trace():
            trace = frame([{"torch.FloatTensor": {"max": 2}}])
            trace.events["function"] = "torch.clamp"
            trace.events["kwargs"] = [{"max": {"int": 3}}] * 2
            return trace

        inv = Invariant(
            ThresholdRelation,
            [
                InputOutputParam("max", None, "int", None, "torch.clamp", True),
                APIParam("torch.clamp"),
                InputOutputParam(None, 0, "torch.Tensor", ("max",), None, False),
            ],
            precondition(),
            "max threshold",
        )
        got = self.assert_same_result(make_trace, inv)
        self.assertEqual(got["decisions_completed"], 1)
        self.assertTrue(got["check_passed"])


if __name__ == "__main__":
    unittest.main()
