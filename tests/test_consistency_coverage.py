"""Real variable-checker coverage, including vacuous success and early stops."""

import logging
import unittest
from collections.abc import Callable
from unittest.mock import patch

import pandas as pd

from tools.fresh_validation.check_coverage import ObservationReport, observe
from tools.fresh_validation.contracts import mapping, required, sequence
from traincheck.instrumentor.tracer import TraceLineType
from traincheck.invariant.base_cls import (
    GroupedPreconditions,
    Invariant,
    Preconditions,
    UnconditionalPrecondition,
    VarTypeParam,
)
from traincheck.invariant.consistency_relation import ConsistencyRelation
from traincheck.trace.trace_pandas import TracePandas


def variable_trace(values: list[int], *, attribute: str = "value") -> TracePandas:
    rows: list[dict[str, object]] = []
    for index, value in enumerate(values):
        for timestamp in (0.0, 10.0):
            rows.append(
                {
                    "type": TraceLineType.STATE_CHANGE,
                    "time": timestamp,
                    "process_id": "worker-1",
                    "thread_id": 2.0,
                    "var_name": f"weight-{index}",
                    "var_type": "torch.nn.Parameter",
                    f"attributes.{attribute}": value,
                }
            )
    return TracePandas(pd.DataFrame(rows), truncate_incomplete_func_calls=False)


def consistency_rule(
    *, enabled: bool = True, var_type: str = "torch.nn.Parameter"
) -> Invariant:
    return Invariant(
        ConsistencyRelation,
        [VarTypeParam(var_type, "value"), VarTypeParam(var_type, "value")],
        GroupedPreconditions(
            {"var": Preconditions([UnconditionalPrecondition()] if enabled else [])}
        ),
        "fixture parameter value agreement",
    )


class ConsistencyCoverageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.previous_log_level = logging.root.manager.disable
        logging.disable(logging.CRITICAL)

    def tearDown(self) -> None:
        logging.disable(self.previous_log_level)

    def parity(
        self, make_trace: Callable[[], TracePandas], inv: Invariant
    ) -> ObservationReport:
        observed = observe(make_trace(), inv)
        direct = observe(make_trace(), inv, enabled=False)
        self.assertEqual(observed["status"], "completed", observed)
        self.assertEqual(observed["checker_result"], direct["checker_result"])
        self.assertIsNone(direct["active"])
        self.assertIsNone(direct["dormant"])
        return observed

    def test_actual_variable_pair_comparisons_are_active(self) -> None:
        report = self.parity(lambda: variable_trace([7, 7, 7]), consistency_rule())
        self.assertEqual(report["opportunity_unit"], "overlapping_variable_value_pair")
        self.assertIsNone(report["candidate_calls"])
        self.assertIsNone(report["visited_calls"])
        self.assertEqual(report["opportunities"], 3)
        self.assertEqual(report["visited_variable_pairs"], 3)
        self.assertEqual(report["decisions_completed"], 3)
        self.assertEqual(report["precondition_group_passes"], 3)
        self.assertTrue(report["fully_validated_active"])
        records = report["decision_records"]
        self.assertEqual(len(records), 3)
        decision = mapping(records[0], "decision")
        pair = sequence(required(decision, "variable_value_pair"), "variable pair")
        first = mapping(pair[0], "first value")
        self.assertEqual(first["process_id"], "worker-1")
        self.assertIsInstance(first["process_id"], str)
        self.assertEqual(first["thread_id"], 2.0)
        self.assertIsInstance(first["thread_id"], float)
        self.assertEqual(first["var_name"], "weight-0")
        self.assertEqual(first["attribute"], "value")
        self.assertEqual(first["event_time"], 10.0)
        self.assertEqual((first["liveness_start"], first["liveness_end"]), (0.0, 10.0))

    def test_violation_stops_before_remaining_pairs(self) -> None:
        report = self.parity(lambda: variable_trace([7, 8, 7]), consistency_rule())
        self.assertFalse(report["check_passed"])
        self.assertEqual(report["opportunities"], 3)
        self.assertEqual(report["visited_variable_pairs"], 1)
        self.assertEqual(report["decisions_completed"], 1)
        self.assertEqual(report["unvisited_pairs_after_stop"], 2)

    def test_failed_preconditions_do_not_count_comparisons(self) -> None:
        report = self.parity(
            lambda: variable_trace([7, 8]), consistency_rule(enabled=False)
        )
        self.assertTrue(report["check_passed"])
        self.assertEqual(report["opportunities"], 1)
        self.assertEqual(report["visited_variable_pairs"], 1)
        self.assertEqual(report["precondition_group_failures"], 1)
        self.assertEqual(report["decisions_completed"], 0)
        self.assertTrue(report["dormant"])
        self.assertFalse(report["fully_validated_active"])

    def test_single_variable_is_dormant(self) -> None:
        report = self.parity(lambda: variable_trace([7]), consistency_rule())
        self.assertEqual(report["opportunities"], 0)
        self.assertEqual(report["decisions_completed"], 0)
        self.assertTrue(report["dormant"])
        self.assertFalse(report["variable_check_skipped"])
        self.assertFalse(report["fully_validated_active"])

    def test_missing_variable_type_is_skipped_not_validated(self) -> None:
        report = self.parity(
            lambda: variable_trace([7, 7]), consistency_rule(var_type="MissingType")
        )
        self.assertTrue(report["check_passed"])
        self.assertFalse(report["checker_triggered"])
        self.assertEqual(report["decisions_completed"], 0)
        self.assertTrue(report["variable_check_skipped"])
        self.assertFalse(report["fully_validated_active"])

    def test_missing_attribute_is_skipped_not_validated(self) -> None:
        report = self.parity(
            lambda: variable_trace([7, 7], attribute="other"), consistency_rule()
        )
        self.assertTrue(report["check_passed"])
        self.assertFalse(report["checker_triggered"])
        self.assertEqual(report["decisions_completed"], 0)
        self.assertTrue(report["variable_check_skipped"])
        self.assertFalse(report["fully_validated_active"])

    def test_predicate_exception_does_not_complete_comparison(self) -> None:
        with patch.object(
            ConsistencyRelation, "evaluate", side_effect=ValueError("fixture failure")
        ):
            report = observe(variable_trace([7, 7]), consistency_rule())
        self.assertEqual(report["status"], "checker_error")
        self.assertIsNone(report["check_passed"])
        self.assertEqual(report["decisions_started"], 1)
        self.assertEqual(report["decisions_completed"], 0)
        self.assertEqual(report["decisions_not_completed"], 1)
        self.assertFalse(report["fully_validated_active"])

    def test_pair_collection_failure_has_unknown_opportunity_count(self) -> None:
        trace = variable_trace([7, 7])
        for attributes in trace.get_var_insts().values():
            attributes["value"][0].liveness.end_time = None
        report = observe(trace, consistency_rule())
        self.assertEqual(report["status"], "checker_error")
        self.assertIsNone(report["opportunities"])
        self.assertIsNone(report["unvisited_pairs_after_stop"])
        self.assertIsNone(report["check_passed"])
        self.assertEqual(report["decisions_completed"], 0)

    def test_nonoverlapping_liveness_is_not_a_check_opportunity(self) -> None:
        def make_trace() -> TracePandas:
            trace = variable_trace([7, 7])
            for identity, attributes in trace.get_var_insts().items():
                state = attributes["value"][0]
                if identity.var_name == "weight-0":
                    state.liveness.end_time = 5.0
                else:
                    state.liveness.start_time = 5.0
            return trace

        report = self.parity(make_trace, consistency_rule())
        self.assertEqual(report["opportunities"], 0)
        self.assertEqual(report["decisions_completed"], 0)
        self.assertFalse(report["fully_validated_active"])

    def test_precondition_failure_is_not_a_completed_comparison(self) -> None:
        with patch.object(
            GroupedPreconditions, "verify", side_effect=ValueError("bad condition")
        ):
            report = observe(variable_trace([7, 7]), consistency_rule())
        self.assertEqual(report["status"], "checker_error")
        self.assertEqual(report["precondition_group_errors"], 1)
        self.assertEqual(report["decisions_started"], 0)
        self.assertEqual(report["decisions_completed"], 0)
        self.assertFalse(report["fully_validated_active"])

    def test_missing_identity_propagates_checker_failure_without_fabricated_id(
        self,
    ) -> None:
        trace = variable_trace([7, 7])
        for attributes in trace.get_var_insts().values():
            del attributes["value"][0].traces[-1]["thread_id"]
        report = observe(trace, consistency_rule())
        self.assertEqual(report["status"], "checker_error")
        self.assertFalse(report["active"])
        self.assertIsNone(report["check_passed"])
        self.assertEqual(report["precondition_group_errors"], 1)
        self.assertFalse(report["fully_validated_active"])
        self.assertIn("thread_id", report["error"]["message"])

    def test_wrong_identity_type_fails_observation(self) -> None:
        trace = variable_trace([7, 7])
        for attributes in trace.get_var_insts().values():
            attributes["value"][0].traces[-1]["thread_id"] = True
        report = observe(trace, consistency_rule())
        self.assertEqual(report["status"], "observer_error")
        self.assertIsNone(report["active"])
        self.assertFalse(report["coverage_observed"])


if __name__ == "__main__":
    unittest.main()
