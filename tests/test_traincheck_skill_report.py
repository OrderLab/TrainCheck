"""Agent summaries distinguish real checking from absent or malformed evidence."""

from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from skills.traincheck.scripts.summarize_check import main, read_artifact, summarize
from tests.test_fresh_validation_coverage import frame, output_rule
from tools.fresh_validation.check_coverage import ObservationReport, observe
from tools.fresh_validation.check_coverage import summarize as summarize_observations
from tools.fresh_validation.contracts import (
    JSONObject,
    JSONValue,
    json_object,
    mapping,
    sequence,
)


def observation(*, valid: bool = True, enabled: bool = True) -> ObservationReport:
    dtype = "torch.float32" if valid else "torch.float64"
    return observe(
        frame([{"torch.FloatTensor": {"shape": [2], "dtype": dtype}}]),
        output_rule(),
        enabled=enabled,
    )


def coverage(*reports: ObservationReport, rows: int = 2) -> JSONObject:
    copied = [copy.deepcopy(report) for report in reports]
    for index, record in enumerate(copied):
        record["invariant_index"] = index
    return json_object(
        {
            "schema_version": 3,
            "inputs_unchanged": True,
            "trace_rows": rows,
            "input_sha256": {"/source/trace.log": "a" * 64},
            "summary": summarize_observations(copied),
            "per_invariant": copied,
        }
    )


class SkillReportTests(unittest.TestCase):
    def test_real_active_checks_and_real_violations_remain_distinct(self) -> None:
        passed = summarize(coverage(observation()))
        failed = summarize(coverage(observation(valid=False)))
        self.assertEqual(passed["outcome"], "active_checks_passed")
        self.assertEqual(failed["outcome"], "violations")
        counts = mapping(failed["counts"], "counts")
        self.assertEqual(counts["observed_violations"], 1)
        self.assertEqual(counts["completed_decisions"], 1)

    def test_dormant_disabled_and_empty_are_not_active_success(self) -> None:
        dormant = observe(
            frame([{"torch.FloatTensor": {"shape": [2]}}]), output_rule(False)
        )
        self.assertEqual(summarize(coverage(dormant))["outcome"], "no_evidence")
        mixed = summarize(coverage(observation(), dormant))
        self.assertEqual(mixed["outcome"], "incomplete")
        disabled = summarize(coverage(observation(enabled=False)))
        self.assertEqual(disabled["outcome"], "no_evidence")
        self.assertTrue(disabled["coverage_incomplete"])
        self.assertEqual(summarize(coverage())["outcome"], "no_evidence")
        self.assertEqual(
            summarize(coverage(observation(), rows=0))["outcome"], "no_evidence"
        )

    def test_error_unknown_outcome_and_skipped_variable_checks_stay_visible(
        self,
    ) -> None:
        error = observation()
        error.update(
            {
                "status": "checker_error",
                "check_passed": None,
                "checker_triggered": None,
                "error": {"type": "ValueError", "message": "malformed tensor"},
                "fully_validated_active": False,
            }
        )
        report = summarize(coverage(error))
        self.assertEqual(report["outcome"], "incomplete")
        self.assertEqual(mapping(report["counts"], "counts")["unknown_outcomes"], 1)
        skipped = observation()
        skipped["variable_check_skipped"] = True
        skipped["fully_validated_active"] = False
        self.assertEqual(summarize(coverage(skipped))["outcome"], "incomplete")

    def test_unsupported_coverage_is_not_zero_violations_or_success(self) -> None:
        unsupported = observation(enabled=False)
        unsupported.update(
            {
                "status": "unsupported_coverage",
                "coverage_status": "unsupported",
                "check_passed": None,
                "checker_triggered": None,
            }
        )
        report = summarize(coverage(unsupported))
        self.assertEqual(report["outcome"], "no_evidence")
        counts = mapping(report["counts"], "counts")
        self.assertEqual(counts["unsupported_coverage"], 1)
        self.assertEqual(counts["unknown_outcomes"], 1)
        self.assertEqual(counts["unobserved_coverage"], 1)

    def test_required_fields_types_and_contradictory_evidence_are_rejected(
        self,
    ) -> None:
        good = coverage(observation())
        cases: list[JSONObject] = []
        for field in ("per_invariant", "trace_rows", "input_sha256", "summary"):
            missing = copy.deepcopy(good)
            del missing[field]
            cases.append(missing)
        top_level_changes: list[tuple[str, JSONValue]] = [
            ("trace_rows", True),
            ("trace_rows", -1),
            ("schema_version", 2),
            ("inputs_unchanged", False),
            ("input_sha256", {}),
        ]
        for field, value in top_level_changes:
            bad = copy.deepcopy(good)
            bad[field] = value
            cases.append(bad)
        for bad in cases:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                summarize(bad)
        rule_changes: list[tuple[str, JSONValue]] = [
            ("status", "success"),
            ("active", False),
            ("decisions_completed", True),
            ("check_passed", False),
            ("invariant_index", 3),
        ]
        for field, value in rule_changes:
            rule = json_object(observation())
            rule["invariant_index"] = 0
            rule[field] = value
            bad = copy.deepcopy(good)
            bad["per_invariant"] = [rule]
            with self.subTest(field=field), self.assertRaises(ValueError):
                summarize(bad)

    def test_identity_alarm_and_complete_source_are_preserved(self) -> None:
        rule = observation(valid=False)
        rule["identity_clauses"] = [{"object_id": "literal-007", "thread_id": 3.5}]
        source = coverage(rule)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "coverage.json"
            encoded = json.dumps(source).encode()
            path.write_bytes(encoded)
            loaded, provenance = read_artifact(path)
            result = summarize(loaded)
            self.assertEqual(loaded, source)
            self.assertEqual(path.read_bytes(), encoded)
            self.assertEqual(provenance["sha256"], hashlib.sha256(encoded).hexdigest())
            self.assertEqual(result["outcome"], "violations")
            self.assertIn('"identity_bearing": true', json.dumps(result))

    def test_compact_display_does_not_hide_total_problems_or_change_source(
        self,
    ) -> None:
        source = coverage(*([observation(valid=False)] * 12))
        before = copy.deepcopy(source)
        result = summarize(source)
        self.assertEqual(len(sequence(result["problem_rules"], "problem_rules")), 10)
        self.assertEqual(result["additional_problem_rules"], 2)
        self.assertEqual(mapping(result["counts"], "counts")["observed_violations"], 12)
        self.assertEqual(source, before)

    def test_aggregate_claim_cannot_override_individual_checker_evidence(self) -> None:
        source = coverage(observation(valid=False))
        aggregate = json_object(source["summary"])
        aggregate["observed_violated_invariants"] = 0
        source["summary"] = aggregate
        with self.assertRaisesRegex(ValueError, "summary disagrees"):
            summarize(source)

    def test_cli_records_actual_policy_count_check_and_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, policy, direct, output = (
                root / name
                for name in (
                    "coverage.json",
                    "policy.json",
                    "direct.json",
                    "agent.json",
                )
            )
            source.write_text(json.dumps(coverage(observation(valid=False))))
            policy.write_text(json.dumps({"model_tracker_style": None}))
            direct.write_text(
                json.dumps({"distinct_invariants_violated": 1, "violations": [{}]})
            )
            command = [
                "summarize_check.py",
                "--coverage",
                str(source),
                "--collection-policy",
                str(policy),
                "--violations-summary",
                str(direct),
                "--output",
                str(output),
            ]
            with patch("sys.argv", command):
                self.assertEqual(main(), 0)
            result, _ = read_artifact(output)
            self.assertEqual(result["outcome"], "violations")
            self.assertEqual(result["collection_mode"], "api-only")
            with patch("sys.argv", command), self.assertRaises(FileExistsError):
                main()

    def test_direct_mismatch_and_malformed_policy_fail_before_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, policy, direct, output = (
                root / name
                for name in (
                    "coverage.json",
                    "policy.json",
                    "direct.json",
                    "agent.json",
                )
            )
            source.write_text(json.dumps(coverage(observation())))
            policy.write_text(json.dumps({"model_tracker_style": []}))
            direct.write_text(
                json.dumps({"distinct_invariants_violated": 1, "violations": [{}]})
            )
            for flag, path in (
                ("--collection-policy", policy),
                ("--violations-summary", direct),
            ):
                with patch(
                    "sys.argv",
                    [
                        "report",
                        "--coverage",
                        str(source),
                        flag,
                        str(path),
                        "--output",
                        str(output),
                    ],
                ):
                    with self.assertRaises(ValueError):
                        main()
                self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
