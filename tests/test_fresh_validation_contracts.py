"""Negative-path contracts for emitted reports and observed call identities."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tests.test_fresh_validation_coverage import frame, output_rule
from tools.fresh_validation.check_coverage import (
    ObserverContractError,
    decision_record,
    finalize_report,
    observe,
    summarize,
)
from tools.fresh_validation.compare_study import compare, losses
from tools.fresh_validation.contracts import boolean, integer, json_object, required
from tools.fresh_validation.fault_control import digest, mutate
from traincheck.invariant.base_cls import CheckerResult, Invariant
from traincheck.invariant.consistency_transient_vars import ConsistentOutputRelation
from traincheck.trace.trace import Trace


def renamed_local_checker(
    trace: Trace, inv: Invariant, check_relation_first: bool
) -> CheckerResult:
    # This still has the exact output-predicate AST but its local identity was
    # renamed. The previous observer silently emitted {func_call_id: None}.
    for renamed_call in trace.get_func_call_ids("torch.fake"):
        func_call_event = trace.query_func_call_event(renamed_call)
        returned_tensors = [func_call_event]
        if len(returned_tensors) == 0:
            raise AssertionError("Fixture unexpectedly empty")
    return CheckerResult(None, inv, True, True)


def malformed_local_checker(
    trace: Trace, inv: Invariant, check_relation_first: bool
) -> CheckerResult:
    for real_call in trace.get_func_call_ids("torch.fake"):
        func_call_event = trace.query_func_call_event(real_call)
        func_call_id = None
        returned_tensors = [func_call_event, func_call_id]
        if len(returned_tensors) == 0:
            raise AssertionError("Fixture unexpectedly empty")
    return CheckerResult(None, inv, True, True)


def catching_local_checker(
    trace: Trace, inv: Invariant, check_relation_first: bool
) -> CheckerResult:
    for renamed_call in trace.get_func_call_ids("torch.fake"):
        try:
            func_call_event = trace.query_func_call_event(renamed_call)
            returned_tensors = [func_call_event]
            if len(returned_tensors) == 0:
                raise AssertionError("Fixture unexpectedly empty")
        except ValueError:
            # The observer must retain failure even when the checker catches it.
            continue
    return CheckerResult(None, inv, True, True)


def catching_then_failing_checker(
    trace: Trace, inv: Invariant, check_relation_first: bool
) -> CheckerResult:
    for renamed_call in trace.get_func_call_ids("torch.fake"):
        try:
            func_call_event = trace.query_func_call_event(renamed_call)
            returned_tensors = [func_call_event]
            if len(returned_tensors) == 0:
                raise AssertionError("Fixture unexpectedly empty")
        except ValueError:
            pass
    raise RuntimeError("checker failed after catching observer error")


class UnsupportedOutput(ConsistentOutputRelation):
    pass


class ValidationContractTests(unittest.TestCase):
    def test_required_family_identity_does_not_use_another_family_fallback(
        self,
    ) -> None:
        with self.assertRaisesRegex(ObserverContractError, "func_call_id"):
            decision_record(
                "ConsistentOutputRelation", {"parent_func_call_id": "parent"}
            )
        with self.assertRaisesRegex(ObserverContractError, "parent_func_call_id"):
            decision_record("APIContainRelation", {"func_call_id": "child"})

    def test_pair_requires_both_record_identities(self) -> None:
        with self.assertRaisesRegex(ObserverContractError, "func_call_id"):
            decision_record(
                "DistinctArgumentRelation",
                {"event1": {"func_call_id": "a"}, "event2": {}},
            )

    def test_identity_wrong_types_fail_explicitly(self) -> None:
        for identity in (None, False, 42, ["call"]):
            with (
                self.subTest(identity=identity),
                self.assertRaises(ObserverContractError),
            ):
                decision_record("ConsistentOutputRelation", {"func_call_id": identity})

    def test_checker_source_drift_is_observer_error_not_successful_null_identity(
        self,
    ) -> None:
        for checker in (
            renamed_local_checker,
            malformed_local_checker,
            catching_local_checker,
        ):
            with (
                self.subTest(checker=checker.__name__),
                patch.object(ConsistentOutputRelation, "static_check_all", checker),
            ):
                report = observe(
                    frame([{"torch.FloatTensor": {"dtype": "torch.float32"}}]),
                    output_rule(),
                )
            self.assertEqual(report["status"], "observer_error")
            self.assertEqual(report["coverage_status"], "error")
            self.assertIsNone(report["check_passed"])
            self.assertIsNone(report["active"])
            self.assertFalse(report["coverage_observed"])
            self.assertEqual(report["decision_records"], [])

    def test_observer_failure_remains_primary_if_checker_then_raises(self) -> None:
        with patch.object(
            ConsistentOutputRelation, "static_check_all", catching_then_failing_checker
        ):
            report = observe(
                frame([{"torch.FloatTensor": {"dtype": "torch.float32"}}]),
                output_rule(),
            )
        self.assertEqual(report["status"], "observer_error")
        self.assertEqual(report["coverage_status"], "error")
        self.assertFalse(report["coverage_observed"])
        self.assertIsNone(report["active"])
        self.assertEqual(report["error"]["type"], "ObserverContractError")
        self.assertEqual(report["checker_error_after_observer"]["type"], "RuntimeError")

    def test_disabled_and_unsupported_counts_are_unknown_not_zero(self) -> None:
        trace = frame([{"torch.FloatTensor": {"dtype": "torch.float32"}}])
        direct = observe(trace, output_rule(), enabled=False)
        rule = output_rule()
        rule.relation = UnsupportedOutput
        unsupported = observe(trace, rule)
        for report in (direct, unsupported):
            summary = summarize([report])
            self.assertIsNone(summary["active_invariants"])
            self.assertIsNone(summary["dormant_invariants"])
            self.assertEqual(summary["coverage_unobserved_invariants"], 1)
            self.assertEqual(summary["observed_active_invariants"], 0)
        self.assertEqual(direct["coverage_status"], "disabled")
        self.assertEqual(unsupported["coverage_status"], "unsupported")
        self.assertIsNone(summarize([unsupported])["violated_invariants"])
        self.assertEqual(summarize([direct])["violated_invariants"], 0)

    def test_completed_report_requires_checker_payload(self) -> None:
        report = observe(
            frame([{"torch.FloatTensor": {"dtype": "torch.float32"}}]), output_rule()
        )
        del report["checker_result"]
        with self.assertRaisesRegex(ObserverContractError, "checker_result"):
            finalize_report(report)

    def test_observed_report_requires_decision_counters(self) -> None:
        report = observe(
            frame([{"torch.FloatTensor": {"dtype": "torch.float32"}}]), output_rule()
        )
        del report["decisions_completed"]
        with self.assertRaisesRegex(ObserverContractError, "decisions_completed"):
            finalize_report(report)

    def test_checker_error_does_not_imply_zero_violations(self) -> None:
        rule = output_rule()
        with patch.object(
            rule, "check", side_effect=ValueError("fixture checker error")
        ):
            report = observe(
                frame([{"torch.FloatTensor": {"dtype": "torch.float32"}}]),
                rule,
                enabled=False,
            )
        self.assertEqual(report["status"], "checker_error")
        self.assertIsNone(summarize([report])["violated_invariants"])
        self.assertEqual(summarize([report])["observed_violated_invariants"], 0)

    def test_cli_bootstrap_preserves_composed_traincheck_source(self) -> None:
        adapter = (
            Path(__file__).resolve().parents[1]
            / "tools/fresh_validation/benchmark_workload.py"
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            composed = root / "composed"
            (composed / "traincheck").mkdir(parents=True)
            module = composed / "traincheck/__init__.py"
            module.write_text("SOURCE = 'fixture'\n")
            code = (
                "import runpy; runpy.run_path("
                + repr(str(adapter))
                + "); import traincheck; print(traincheck.__file__)"
            )
            result = subprocess.run(
                [sys.executable, "-c", code],
                cwd=root,
                env={
                    **os.environ,
                    "PYTHONPATH": str(composed),
                    "PYTHONDONTWRITEBYTECODE": "1",
                },
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(Path(result.stdout.strip()), module)

    def test_json_required_fields_and_wrong_types_are_not_coerced(self) -> None:
        with self.assertRaisesRegex(ValueError, "required"):
            required({}, "required")
        with self.assertRaises(ValueError):
            boolean("false", "active")
        with self.assertRaises(ValueError):
            integer(True, "count")
        with self.assertRaises(ValueError):
            json_object({"invalid": object()})

    def test_empty_loss_records_do_not_establish_equality(self) -> None:
        for steps in ([], [{"step": 1}]):
            with self.subTest(steps=steps), self.assertRaisesRegex(ValueError, "loss"):
                losses({"steps": steps})

    def test_truncated_loss_rows_do_not_match_claimed_completed_steps(self) -> None:
        with self.assertRaisesRegex(ValueError, "completed_steps"):
            losses({"steps": [{"step": 1, "loss": 1.0}], "completed_steps": 2})

    def test_incomplete_phase_set_cannot_report_study_completed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "Missing completed phases"):
                compare(Path(temporary))

    def test_fault_index_cannot_silently_select_last_rule(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trace, rules, coverage = (
                root / name for name in ("trace.jsonl", "rules.jsonl", "coverage.json")
            )
            trace.write_text(
                json.dumps({"type": "function_call (post)", "func_call_id": "a"}) + "\n"
            )
            rules.write_text(json.dumps(output_rule().to_dict()) + "\n")
            coverage.write_text(
                json.dumps(
                    {
                        "inputs_unchanged": True,
                        "observer_enabled": True,
                        "input_sha256": {str(p): digest(p) for p in (trace, rules)},
                        "per_invariant": [{"invariant_index": -1}],
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, "out of range"):
                mutate([trace], rules, coverage, root / "fault")
            self.assertFalse((root / "fault").exists())


if __name__ == "__main__":
    unittest.main()
