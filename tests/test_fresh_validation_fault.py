"""The fallback must activate the real frozen checker, without other edits."""

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from tests.test_fresh_validation_coverage import frame, output_rule
from tools.fresh_validation.check_coverage import observe
from tools.fresh_validation.fault_control import differences, digest, mutate
from traincheck.trace.trace_pandas import TracePandas


class FaultControlTests(unittest.TestCase):
    def run_mutation(self, attribute, shape):
        rule = output_rule()
        if attribute == "shape":
            rule.params[1].attr_name = "shape"
            rule.params[1].const_value = tuple(shape)
        good = {"torch.FloatTensor": {"shape": shape, "dtype": "torch.float32"}}
        traced = frame([good, good])
        healthy = observe(traced, rule)
        healthy["invariant_index"] = 0
        before = traced.events.to_dict("records")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trace, invariants, coverage = (
                root / x for x in ("trace.log", "rules.jsonl", "coverage.json")
            )
            trace.write_text("".join(json.dumps(row) + "\n" for row in before))
            invariants.write_text(json.dumps(rule.to_dict()) + "\n")
            coverage.write_text(
                json.dumps(
                    {
                        "inputs_unchanged": True,
                        "observer_enabled": True,
                        "input_sha256": {
                            str(p): digest(p) for p in (trace, invariants)
                        },
                        "per_invariant": [healthy],
                    }
                )
            )
            manifest = mutate([trace], invariants, coverage, root / "fault")
            after = [
                json.loads(line)
                for line in (root / "fault/traces/trace.log").read_text().splitlines()
            ]
            self.assertEqual(len(differences(before, after)), 1)
            self.assertEqual(before[0], after[0])
            self.assertEqual(before[2:], after[2:])
            self.assertEqual(manifest["attribute"], attribute)
            self.assertEqual(
                manifest["edits"][0]["path"][-1], 0 if attribute == "shape" else "dtype"
            )
            checked = observe(
                TracePandas(pd.DataFrame(after), truncate_incomplete_func_calls=False),
                rule,
            )
            self.assertTrue(healthy["check_passed"])
            self.assertEqual(checked["status"], "completed")
            self.assertFalse(checked["check_passed"])
            self.assertEqual(checked["precondition_group_passes"], 1)
            self.assertEqual(checked["decisions_completed"], 1)
            self.assertEqual(
                checked["decision_records"][0]["func_call_id"], manifest["func_call_id"]
            )
            self.assertTrue(
                any(
                    row["func_call_id"] == manifest["func_call_id"]
                    for row in checked["checker_result"]["trace"]
                )
            )

    def test_no_shape_rule_uses_live_dtype_and_detects_exact_edited_call(self):
        self.run_mutation("dtype", [])

    def test_mutable_shape_keeps_original_one_dimension_control(self):
        self.run_mutation("shape", [2, 3])
