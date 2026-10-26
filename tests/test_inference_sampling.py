"""Regression coverage for task-local sampling, including actual relation paths."""

import importlib
import json
import multiprocessing
import os
import random
import subprocess
import sys
import threading
import unittest
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

from traincheck.infer_engine import (
    InferEngine,
    _find_precondition_task,
    _generate_hypothesis_task,
    _genhypo_worker_task,
    _hypothesis_sampling_key,
    _precond_worker_task,
)
from traincheck.inference_sampling import sampling_rng, sampling_task
from traincheck.invariant import relation_pool
from traincheck.invariant.base_cls import (
    APIParam,
    Example,
    ExampleList,
    Hypothesis,
    Invariant,
    UnconditionalPrecondition,
)
from traincheck.invariant.contain_relation import (
    PARENT_GROUP_NAME,
    APIContainRelation,
    cache_events_scanner,
    prune_func_call,
)
from traincheck.parallel import config_snapshot, worker_init
from traincheck.trace.trace_pandas import TracePandas
from traincheck.trace.types import FuncCallEvent

PARENT = "torch.nn.modules.activation.ReLU.forward"
CHILD = "torch.nn.modules.dropout.Dropout.forward"


def nested_call_trace(prefix, calls=1005):
    """Complete synthetic events, not training: >1000 calls exercises both samplers."""
    rows = []
    for index in range(calls):
        events = [
            (PARENT, "parent", "pre"),
            (CHILD, "child", "pre"),
            (CHILD, "child", "post"),
            (PARENT, "parent", "post"),
        ]
        for offset, (function, name, kind) in enumerate(events):
            row = {
                "time": index * 10 + offset + 1,
                "func_call_id": f"{prefix}-{name}-{index}",
                "function": function,
                "type": f"function_call ({kind})",
                "process_id": 1,
                "thread_id": 1,
                "is_bound_method": True,
                "obj_id": 1,
                "meta_vars": {},
                "args": {"0": {"object": {}}, "1": {"float": 1.0}},
                "kwargs": {},
            }
            if kind == "post":
                row["return_values"] = {"float": 1.0}
            rows.append(row)
    return TracePandas(rows)


def hypothesis_evidence(hypotheses):
    """Compare exact retained call IDs/order as well as relation and parameters."""
    results = []
    for hypothesis in hypotheses:

        def examples(items):
            return [
                {
                    name: [row["func_call_id"] for row in rows]
                    for name, rows in sorted(example.trace_groups.items())
                }
                for example in items.examples
            ]

        results.append(
            {
                "key": _hypothesis_sampling_key(hypothesis),
                "positive": examples(hypothesis.positive_examples),
                "negative": examples(hypothesis.negative_examples),
            }
        )
    return sorted(results, key=lambda item: item["key"])


def simple_hypothesis(positive_count=3, negative_count=0):
    def examples(count, prefix):
        # This fixture deliberately reaches the >5000 precondition branch;
        # disable the independent reservoir cap while constructing its input.
        with patch("traincheck.config.config.MAX_EXAMPLE_LIST_SIZE", None):
            return ExampleList.from_iterable_of_examples(
                [
                    Example(
                        {
                            PARENT_GROUP_NAME: [
                                {"id": f"{prefix}-{i}", "function": PARENT}
                            ]
                        }
                    )
                    for i in range(count)
                ],
                {PARENT_GROUP_NAME},
            )

    return Hypothesis(
        Invariant(
            APIContainRelation, [APIParam(PARENT), APIParam(CHILD)], None, "fixture"
        ),
        examples(positive_count, "positive"),
        examples(negative_count, "negative"),
    )


class SamplingScopeTests(unittest.TestCase):
    def test_reproducible_keys_and_untouched_global_rng(self):
        before = random.getstate()
        self.assertIs(sampling_rng(), random)
        with sampling_task("generate", 0, "relation"):
            first = [sampling_rng().random() for _ in range(20)]
        self.assertEqual(random.getstate(), before)
        random.random()
        after_noise = random.getstate()
        with sampling_task("generate", 0, "relation"):
            second = [sampling_rng().random() for _ in range(20)]
        with sampling_task("generate", 1, "relation"):
            other = [sampling_rng().random() for _ in range(20)]
        self.assertEqual(first, second)
        self.assertNotEqual(first, other)
        self.assertEqual(random.getstate(), after_noise)

    def test_nested_scopes_restore_on_frozen_exception(self):
        @dataclass(frozen=True)
        class FrozenError(Exception):
            reason: str

        with sampling_task("outer") as expected_rng:
            expected = [expected_rng.random() for _ in range(2)]
        error = FrozenError("expected")
        before = random.getstate()
        with sampling_task("outer") as outer:
            first = sampling_rng().random()
            try:
                with sampling_task("inner"):
                    sampling_rng().random()
                    raise error
            except FrozenError as caught:
                self.assertIs(caught, error)
            self.assertIs(sampling_rng(), outer)
            second = sampling_rng().random()
        self.assertEqual([first, second], expected)
        self.assertEqual(random.getstate(), before)
        self.assertIs(sampling_rng(), random)

    def test_context_is_thread_local(self):
        errors = []
        values = []

        def worker():
            try:
                self.assertIs(sampling_rng(), random)
                with sampling_task("worker"):
                    values.append(sampling_rng().random())
            except BaseException as error:
                errors.append(error)

        with sampling_task("parent") as parent:
            thread = threading.Thread(target=worker)
            thread.start()
            thread.join(5)
            self.assertFalse(thread.is_alive())
            self.assertIs(sampling_rng(), parent)
        self.assertEqual(errors, [])
        self.assertEqual(len(values), 1)

    def test_seed_does_not_depend_on_python_hash_seed(self):
        script = (
            "import json; from traincheck.inference_sampling import sampling_task,sampling_rng; "
            "\nwith sampling_task('generate', 4, {'b': 2, 'a': 1}): "
            "print(json.dumps([sampling_rng().random() for _ in range(8)]))"
        )
        root = str(Path(__file__).resolve().parents[1])
        outputs = []
        for seed in ("1", "98765"):
            env = dict(os.environ, PYTHONHASHSEED=seed, PYTHONPATH=root)
            outputs.append(
                subprocess.check_output(
                    [sys.executable, "-c", script], env=env, text=True
                )
            )
        self.assertEqual(outputs[0], outputs[1])

    def test_actual_pruning_is_nontrivial_and_task_deterministic(self):
        def retained(identity):
            sizes, types, kept = [], [], []
            with sampling_task("pruning", identity):
                for index in range(1500):
                    if not prune_func_call(1500, FuncCallEvent, sizes, types):
                        kept.append(index)
                        sizes.append(1)
                        types.append(FuncCallEvent)
            return kept

        first = retained("same")
        for _ in range(101):
            random.random()
        self.assertEqual(first, retained("same"))
        self.assertNotEqual(first, retained("different"))
        self.assertGreater(len(first), 10)
        self.assertLess(len(first), 1500)


class InferenceTaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.traces = [nested_call_trace("trace0"), nested_call_trace("trace1")]

    def setUp(self):
        cache_events_scanner.clear()

    def test_hypothesis_key_ignores_mutable_results_but_tracks_semantics(self):
        hypothesis = simple_hypothesis()
        key = _hypothesis_sampling_key(hypothesis)
        hypothesis.invariant.num_positive_examples = 900
        hypothesis.invariant.num_negative_examples = 8
        hypothesis.invariant.text_description = "different display text"
        hypothesis.invariant.precondition = object()
        self.assertEqual(_hypothesis_sampling_key(hypothesis), key)
        hypothesis.valid_only_for_stages = True
        hypothesis.valid_stages = {"training", "evaluation"}
        staged = _hypothesis_sampling_key(hypothesis)
        self.assertNotEqual(staged, key)
        hypothesis.valid_stages = set(["evaluation", "training"])
        self.assertEqual(_hypothesis_sampling_key(hypothesis), staged)
        hypothesis.invariant.params.reverse()
        self.assertNotEqual(_hypothesis_sampling_key(hypothesis), staged)

    def test_real_relation_retained_examples_repeat_despite_ambient_rng(self):
        before = random.getstate()
        hypotheses = _generate_hypothesis_task(0, APIContainRelation, self.traces[0])
        self.assertEqual(random.getstate(), before)
        first = hypothesis_evidence(hypotheses)
        self.assertEqual(len(first), 1)
        self.assertGreater(len(first[0]["positive"]), 10)
        self.assertLess(len(first[0]["positive"]), 120)
        for _ in range(77):
            random.random()
        second = hypothesis_evidence(
            _generate_hypothesis_task(0, APIContainRelation, self.traces[0])
        )
        self.assertEqual(first, second)

    def test_engine_serial_dispatches_each_relation_through_shared_task(self):
        disabled = [
            relation for relation in relation_pool if relation is not APIContainRelation
        ]
        engine = InferEngine(self.traces, disabled_relations=disabled, num_workers=1)
        with patch(
            "traincheck.infer_engine._generate_hypothesis_task",
            wraps=_generate_hypothesis_task,
        ) as task:
            hypotheses = engine.generate_hypothesis()
        self.assertEqual(task.call_count, 2)
        self.assertEqual([call.args[0] for call in task.call_args_list], [0, 1])
        self.assertEqual(len(hypotheses), 1)
        self.assertEqual(next(iter(hypotheses.values())), [0, 1])

    def test_actual_relation_matches_process_counts_orders_and_start_methods(self):
        expected = {
            index: hypothesis_evidence(
                _generate_hypothesis_task(index, APIContainRelation, trace)
            )
            for index, trace in enumerate(self.traces)
        }
        methods = [
            method
            for method in ("spawn", "fork")
            if method in multiprocessing.get_all_start_methods()
        ]
        self.assertTrue(methods)
        for method in methods:
            for workers in (1, 2, 4):
                with self.subTest(start_method=method, workers=workers):
                    # Repeated tasks consume distinct worker histories. Reversing
                    # dispatch changes assignment without changing semantic keys.
                    tasks = [(index, APIContainRelation) for index in (1, 0, 1, 0)]
                    with ProcessPoolExecutor(
                        max_workers=workers,
                        mp_context=multiprocessing.get_context(method),
                        initializer=worker_init,
                        initargs=(self.traces, config_snapshot()),
                    ) as pool:
                        futures = [
                            pool.submit(_genhypo_worker_task, task) for task in tasks
                        ]
                        for future in futures:
                            index, relation, hypotheses = future.result(timeout=90)
                            self.assertIs(relation, APIContainRelation)
                            self.assertEqual(
                                hypothesis_evidence(hypotheses), expected[index]
                            )

    def test_precondition_over_5000_preserves_legacy_samples_without_global_seed(self):
        precondition = importlib.import_module("traincheck.invariant.precondition")
        hypothesis = simple_hypothesis(5007, 5011)
        captured = []

        def solve(positive, negative, traces, keys_to_skip):
            captured.append(
                (
                    [item[0]["id"] for item in positive],
                    [item[0]["id"] for item in negative],
                )
            )
            return [UnconditionalPrecondition()]

        legacy_rng = random.Random(42)
        expected = (
            legacy_rng.sample([f"positive-{i}" for i in range(5007)], 5000),
            legacy_rng.sample([f"negative-{i}" for i in range(5011)], 5000),
        )
        before = random.getstate()
        with patch.object(
            precondition, "find_precondition_from_single_group", side_effect=solve
        ):
            result = _find_precondition_task(
                hypothesis, [], _hypothesis_sampling_key(hypothesis)
            )
        self.assertIsNotNone(result)
        self.assertEqual(captured, [expected])
        self.assertEqual(random.getstate(), before)

    def test_precondition_worker_dispatch_matches_serial_key(self):
        hypothesis = simple_hypothesis()
        key = _hypothesis_sampling_key(hypothesis)
        worker_init([], config_snapshot())

        def result(*args):
            return [sampling_rng().random() for _ in range(6)]

        with patch("traincheck.infer_engine.find_precondition", side_effect=result):
            expected = _find_precondition_task(hypothesis, [], key)
            random.random()
            actual = _precond_worker_task((key, hypothesis))
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
