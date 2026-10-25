"""Keep pruning state local to one precondition solve, including its children."""

import copy
import json
import logging
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import traincheck.invariant.precondition as precondition
from traincheck.invariant.base_cls import PT, Precondition, PreconditionClause
from traincheck.trace.trace_pandas import TracePandas


def example(phase, signal=None):
    record = {
        "time": 0,
        "process_id": 1,
        "thread_id": 2,
        "function": "fixture.step",
        "meta_vars.phase": phase,
    }
    if signal is not None:
        record["meta_vars.signal"] = signal
    return [record]


def trace():
    return TracePandas(
        pd.DataFrame({"time": [0, 1], "process_id": [1, 1], "thread_id": [2, 2]}),
        truncate_incomplete_func_calls=False,
    )


def solve(name, **kwargs):
    fixtures = {
        "A": ([example("train", True)], [example("train", False)]),
        "B": ([example("train")], [example("eval")]),
    }
    positive, negative = copy.deepcopy(fixtures[name])
    return precondition.find_precondition_from_single_group(
        positive, negative, [trace()], **kwargs
    )


def constant(prop, dtype, value):
    return PreconditionClause(prop, dtype, PT.CONSTANT, None, {value})


class PreconditionStateIsolationTests(unittest.TestCase):
    def setUp(self):
        previous = logging.root.manager.disable
        logging.disable(logging.CRITICAL)
        self.addCleanup(logging.disable, previous)

    def assert_phase_condition(self, result):
        self.assertEqual(
            result, [Precondition([constant("meta_vars.phase", str, "train")])]
        )

    def test_prior_independent_solve_does_not_prune_required_clause(self):
        self.assertEqual(
            solve("A"),
            [Precondition([constant("meta_vars.signal", bool, True)])],
        )
        self.assert_phase_condition(solve("B"))

    def test_reversed_independent_order_preserves_later_result(self):
        self.assert_phase_condition(solve("B"))
        solve("A")
        self.assert_phase_condition(solve("B"))

    def test_repeated_independent_solve(self):
        for _ in range(3):
            self.assert_phase_condition(solve("B"))

    def test_explicit_empty_set_is_updated_and_reused(self):
        state = set()
        solve("A", _pruned_clauses=state)
        self.assertIn(constant("meta_vars.phase", str, "train"), state)
        # Explicit state still means to suppress these clauses in this solve.
        self.assertEqual(solve("B", _pruned_clauses=state), [])
        self.assert_phase_condition(solve("B"))

    def test_recursive_siblings_share_state_but_new_roots_do_not(self):
        original = precondition.find_precondition_from_single_group
        root_children = []
        for _ in range(2):
            children = []

            def observe(*args, **kwargs):
                children.append(
                    (
                        kwargs["_pruned_clauses"],
                        kwargs["_skip_pruning"],
                        kwargs["_current_depth"],
                    )
                )
                return original(*args, **kwargs)

            positive = [example("train"), example("eval")]
            negative = [example("test")]
            with patch.object(
                precondition, "find_precondition_from_single_group", side_effect=observe
            ):
                result = original(positive, negative, [trace()])
            self.assertEqual(len(result), 2)
            self.assertTrue(
                all(
                    any(condition.verify(item) for condition in result)
                    for item in positive
                )
            )
            self.assertTrue(
                all(not condition.verify(negative[0]) for condition in result)
            )
            self.assertEqual(len(children), 2)
            self.assertIs(children[0][0], children[1][0])
            self.assertTrue(all(skip and depth == 1 for _, skip, depth in children))
            self.assertIn(constant("process_id", int, 1), children[0][0])
            root_children.append(children)
        self.assertIsNot(root_children[0][0][0], root_children[1][0][0])

    def test_fresh_and_warm_processes_find_same_clause(self):
        outputs = []
        for order in (("B",), ("A", "B"), ("B", "A", "B")):
            script = (
                "import json\n"
                "from tests.test_precondition_state_isolation import solve\n"
                f"results = [solve(name) for name in {order!r}]\n"
                "print(json.dumps([item.to_dict() for item in results[-1]], sort_keys=True))\n"
            )
            child = subprocess.run(
                [sys.executable, "-c", script],
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                text=True,
                check=True,
            )
            outputs.append(json.loads(child.stdout))
        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(outputs[0], outputs[2])
        self.assertEqual(len(outputs[0]), 1)
        self.assertEqual(outputs[0][0]["clauses"][0]["prop_name"], "meta_vars.phase")


if __name__ == "__main__":
    unittest.main()
