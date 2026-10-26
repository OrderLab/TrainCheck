"""Reservoir invariants, weighted population merges, and actual merge call sites."""

import multiprocessing
import random
import unittest
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from unittest.mock import patch

import traincheck.config.config as config
from traincheck.infer_engine import InferEngine
from traincheck.invariant.base_cls import (
    APIParam,
    Example,
    ExampleList,
    GroupedPreconditions,
    Hypothesis,
    Invariant,
    Preconditions,
    UnconditionalPrecondition,
)
from traincheck.invariant.contain_relation import PARENT_GROUP_NAME, APIContainRelation
from traincheck.parallel import config_snapshot, worker_init

GROUP = PARENT_GROUP_NAME


def example(value, group=GROUP):
    return Example({group: [{"id": value}]})


def values(reservoir):
    return [item.get_group(GROUP)[0]["id"] for item in reservoir.examples]


def reservoir(items, cap, seed=42, groups=None):
    with (
        patch.object(config, "MAX_EXAMPLE_LIST_SIZE", cap),
        patch.object(config, "EXAMPLE_SAMPLING_SEED", seed),
    ):
        result = ExampleList({GROUP} if groups is None else groups)
    for item in items:
        result.add_example(example(item))
    return result


def snapshot(reservoir):
    return (
        reservoir.num_seen,
        reservoir.max_size,
        set(reservoir.group_names),
        list(reservoir.examples),
        reservoir._rng.getstate(),
    )


def hypothesis(prefix, positive, negative, cap):
    precondition = GroupedPreconditions(
        {GROUP: Preconditions([UnconditionalPrecondition()])}
    )
    invariant = Invariant(
        APIContainRelation,
        [
            APIParam("torch.nn.modules.activation.ReLU.forward"),
            APIParam("torch.nn.modules.dropout.Dropout.forward"),
        ],
        precondition,
        "ReLU.forward() contains Dropout.forward()",
    )
    return Hypothesis(
        invariant,
        reservoir([f"{prefix}-positive-{i}" for i in range(positive)], cap),
        reservoir([f"{prefix}-negative-{i}" for i in range(negative)], cap),
    )


def worker_reservoir_result(count):
    result = ExampleList({GROUP})
    for index in range(count):
        result.add_example(example(index))
    return (
        result.max_size,
        config.EXAMPLE_SAMPLING_SEED,
        result.num_seen,
        values(result),
    )


@dataclass
class ReservoirFixtureTrace:
    """A picklable trace carrying populations, without any trace parser work."""

    prefix: str
    positive_count: int
    negative_count: int

    def is_stage_annotated(self):
        return False


class ReservoirFixtureRelation(APIContainRelation):
    @classmethod
    def generate_hypothesis(cls, trace):
        # Read config directly here: patching construction inside the fixture
        # would mask a missing config_snapshot/worker_init propagation field.
        precondition = GroupedPreconditions(
            {GROUP: Preconditions([UnconditionalPrecondition()])}
        )
        invariant = Invariant(
            cls,
            [APIParam("fixture.parent"), APIParam("fixture.child")],
            precondition,
            "fixture.parent contains fixture.child",
        )
        positive, negative = ExampleList({GROUP}), ExampleList({GROUP})
        for kind, count, examples in [
            ("positive", trace.positive_count, positive),
            ("negative", trace.negative_count, negative),
        ]:
            for index in range(count):
                examples.add_example(example(f"{trace.prefix}-{kind}-{index}"))
        return [Hypothesis(invariant, positive, negative)]


class ExampleReservoirConstructionTests(unittest.TestCase):
    def test_from_iterable_streams_through_capped_add_and_counts_every_item(self):
        applied = []
        original = ExampleList.add_example

        def observed_add(result, item):
            original(result, item)
            applied.append(result.num_seen)
            self.assertLessEqual(len(result), 3)

        def items():
            for index in range(37):
                self.assertEqual(
                    len(applied), index, "input must be consumed incrementally"
                )
                yield example(index)

        with (
            patch.object(config, "MAX_EXAMPLE_LIST_SIZE", 3),
            patch.object(ExampleList, "add_example", observed_add),
        ):
            result = ExampleList.from_iterable_of_examples(items(), {GROUP})
        self.assertEqual(len(result), 3)
        self.assertEqual(result.num_seen, 37)
        self.assertEqual(applied, list(range(1, 38)))

    def test_unlimited_zero_and_empty_iterables(self):
        for cap, expected in [(None, list(range(12))), (0, []), (20, list(range(12)))]:
            with (
                self.subTest(cap=cap),
                patch.object(config, "MAX_EXAMPLE_LIST_SIZE", cap),
            ):
                result = ExampleList.from_iterable_of_examples(
                    (example(i) for i in range(12)), {GROUP}
                )
                self.assertEqual(values(result), expected)
                self.assertEqual(result.num_seen, 12)
                empty = ExampleList.from_iterable_of_examples(iter(()), {GROUP})
                self.assertEqual(len(empty), 0)
                self.assertEqual(empty.num_seen, 0)

    def test_invalid_caps_fail_at_construction(self):
        for cap in [-1, True, False, 1.5, "3"]:
            with (
                self.subTest(cap=cap),
                patch.object(config, "MAX_EXAMPLE_LIST_SIZE", cap),
            ):
                with self.assertRaises(ValueError):
                    ExampleList({GROUP})

    def test_from_iterable_validates_group_membership(self):
        with self.assertRaises((AssertionError, ValueError)):
            ExampleList.from_iterable_of_examples([example(1, "different")], {GROUP})


class ExampleReservoirMergeTests(unittest.TestCase):
    def assert_rejected_without_mutation(self, left, right):
        prior_left, prior_right = snapshot(left), snapshot(right)
        rng = random.Random(761)
        rng_before = rng.getstate()
        with self.assertRaises(ValueError):
            left.merge(right, rng=rng)
        self.assertEqual(snapshot(left), prior_left)
        self.assertEqual(snapshot(right), prior_right)
        self.assertEqual(rng.getstate(), rng_before)

    def test_exact_unlimited_and_under_cap_order_without_rng_consumption(self):
        for cap in [None, 10]:
            with self.subTest(cap=cap):
                left = reservoir(["a", "a", "b"], cap)
                right = reservoir(["c", "a"], cap)
                before = snapshot(right)
                rng = random.Random(54)
                rng_before = rng.getstate()
                self.assertIsNone(left.merge(right, rng=rng))
                self.assertEqual(values(left), ["a", "a", "b", "c", "a"])
                self.assertEqual(left.num_seen, 5)
                self.assertEqual(snapshot(right), before)
                self.assertEqual(rng.getstate(), rng_before)

    def test_empty_destination_adopts_groups_and_empty_source_preserves_order(self):
        left = reservoir([], 3, groups=set())
        right = reservoir([1, 2], 3)
        left.merge(right, rng=random.Random(1))
        self.assertEqual(left.group_names, {GROUP})
        self.assertEqual(values(left), [1, 2])
        self.assertEqual(left.num_seen, 2)

        uninitialized = reservoir([], 3, groups=set())
        rng = random.Random(3)
        before = rng.getstate()
        uninitialized.merge(reservoir([], 3), rng=rng)
        self.assertEqual(uninitialized.group_names, {GROUP})
        self.assertEqual(uninitialized.num_seen, 0)
        self.assertEqual(rng.getstate(), before)
        left.merge(reservoir([], 3), rng=random.Random(2))
        self.assertEqual(values(left), [1, 2])
        self.assertEqual(left.num_seen, 2)

    def test_zero_capacity_accumulates_counts_and_preserves_no_items(self):
        left, right = reservoir(range(20), 0), reservoir(range(7), 0)
        left.merge(right, rng=random.Random(7))
        left.add_example(example("next"))
        self.assertEqual(left.num_seen, 28)
        self.assertEqual(len(left), 0)

    def test_unequal_populations_are_counted_and_source_remains_unchanged(self):
        left, right = reservoir(range(10), 4, 11), reservoir(range(10, 100), 4, 19)
        prior_right = snapshot(right)
        left.merge(right, rng=random.Random(37))
        self.assertEqual(left.num_seen, 100)
        self.assertEqual(len(left), 4)
        self.assertTrue(set(values(left)).issubset(range(100)))
        self.assertEqual(snapshot(right), prior_right)
        for value in range(100, 117):
            left.add_example(example(value))
        self.assertEqual(left.num_seen, 117)
        self.assertEqual(len(left), 4)

    def test_duplicate_examples_are_a_multiset_not_a_set(self):
        left, right = reservoir(["same"] * 7, 3), reservoir(["same"] * 11, 3)
        left.merge(right, rng=random.Random(42))
        self.assertEqual(left.num_seen, 18)
        self.assertEqual(values(left), ["same", "same", "same"])

    def test_different_caps_are_allowed_when_source_has_enough_information(self):
        left, right = reservoir(range(10), 5), reservoir([10, 11], 2)
        left.merge(right, rng=random.Random(8))
        self.assertEqual((len(left), left.num_seen), (5, 12))
        left, right = reservoir(range(7), 2), reservoir(range(7, 17), None)
        left.merge(right, rng=random.Random(9))
        self.assertEqual((len(left), left.num_seen), (2, 17))

    def test_rejects_self_group_mismatch_and_incompatible_sampled_sources(self):
        left = reservoir(range(3), 3)
        self.assert_rejected_without_mutation(left, left)
        right = reservoir([], 3, groups={"other"})
        self.assert_rejected_without_mutation(left, right)
        self.assert_rejected_without_mutation(
            reservoir(range(7), 5), reservoir(range(9), 2)
        )
        self.assert_rejected_without_mutation(
            reservoir(range(3), None), reservoir(range(9), 2)
        )

    def test_rejects_underfilled_and_malformed_reservoir_states(self):
        for corrupt_left in [False, True]:
            left, right = reservoir(range(9), 3), reservoir(range(12), 3)
            (left if corrupt_left else right).examples.pop()
            self.assert_rejected_without_mutation(left, right)
        for malformed in [-1, True, 2.5]:
            left, right = reservoir(range(3), 3), reservoir(range(3), 3)
            right.num_seen = malformed
            self.assert_rejected_without_mutation(left, right)

    def test_weighted_merge_and_continued_add_have_expected_marginals(self):
        # Fixed seed enumeration is deterministic, not a fresh statistical draw.
        # Concatenating retained samples gives 'left' ~75%, not the required30%.
        before_add = after_add = 0
        trials = 2000
        for seed in range(trials):
            left = reservoir(["left"], 3, 2 * seed + 1)
            right = reservoir(range(9), 3, 2 * seed + 2)
            left.merge(right, rng=random.Random(100000 + seed))
            before_add += "left" in values(left)
            for index in range(10):
                left.add_example(example(f"new-{index}"))
            after_add += "left" in values(left)
        self.assertAlmostEqual(before_add / trials, 3 / 10, delta=0.04)
        self.assertAlmostEqual(after_add / trials, 3 / 20, delta=0.04)

    def test_nested_merge_tree_preserves_marginals_with_shared_or_distinct_source_seeds(
        self,
    ):
        # An independent sequential merge RNG must be threaded through the entire
        # uneven tree. Replayed per-list merge streams can over-select 'e' (~27%).
        trials = 4000
        for same_source_seed in [True, False]:
            merged, continued = Counter(), Counter()
            for seed in range(trials):
                source_seeds = (
                    [seed] * 5
                    if same_source_seed
                    else [31 * seed + i for i in range(5)]
                )
                a, b, c, d, e = [
                    reservoir([name], 1, value)
                    for name, value in zip("abcde", source_seeds)
                ]
                rng = random.Random(200000 + seed)
                a.merge(b, rng=rng)
                a.merge(c, rng=rng)
                d.merge(e, rng=rng)
                a.merge(d, rng=rng)
                self.assertEqual(a.num_seen, 5)
                merged.update(values(a))
                a.add_example(example("f"))
                self.assertEqual(a.num_seen, 6)
                continued.update(values(a))
            with self.subTest(same_source_seed=same_source_seed):
                for name in "abcde":
                    self.assertAlmostEqual(merged[name] / trials, 1 / 5, delta=0.025)
                for name in "abcdef":
                    self.assertAlmostEqual(continued[name] / trials, 1 / 6, delta=0.025)

    def test_sampled_unequal_sources_then_add_preserve_every_occurrence_marginal(self):
        # Both sources have already sampled, using the same construction seed.
        # Continuing the destination's old RNG replays draws used by the longer
        # source and biases some occurrence marginals to ~14% or ~20%, versus
        # 2/12. Checking just aggregate source weights would hide this effect.
        merged, continued = Counter(), Counter()
        trials = 6000
        for seed in range(trials):
            left = reservoir(range(3), 2, seed)
            right = reservoir(range(3, 10), 2, seed)
            left.merge(right, rng=random.Random(300000 + seed))
            self.assertEqual((left.num_seen, len(left)), (10, 2))
            merged.update(values(left))
            for index in (10, 11):
                left.add_example(example(index))
            self.assertEqual((left.num_seen, len(left)), (12, 2))
            continued.update(values(left))
        for occurrence in range(10):
            self.assertAlmostEqual(merged[occurrence] / trials, 2 / 10, delta=0.02)
        for occurrence in range(12):
            self.assertAlmostEqual(continued[occurrence] / trials, 2 / 12, delta=0.02)


class ReservoirIntegrationTests(unittest.TestCase):
    def test_actual_generate_hypothesis_matches_workers_1_2_4_fork_and_spawn(self):
        traces = [
            ReservoirFixtureTrace("a", 17, 11),
            ReservoirFixtureTrace("b", 31, 19),
            ReservoirFixtureTrace("c", 53, 29),
        ]

        def run(workers):
            generated = InferEngine(traces, num_workers=workers).generate_hypothesis()
            self.assertEqual(len(generated), 1)
            item, trace_indexes = next(iter(generated.items()))
            self.assertEqual(trace_indexes, [0, 1, 2])
            result = []
            for examples, count in [
                (item.positive_examples, 101),
                (item.negative_examples, 59),
            ]:
                self.assertEqual((examples.max_size, len(examples)), (5, 5))
                self.assertEqual(examples.num_seen, count)
                # Include the continuation state: pickling must preserve it and
                # the parent must consume the same merge stream in both paths.
                result.append((values(examples), count, examples._rng.getstate()))
            return result

        with (
            patch("traincheck.infer_engine.relation_pool", [ReservoirFixtureRelation]),
            patch.object(config, "MAX_EXAMPLE_LIST_SIZE", 5),
            patch.object(config, "EXAMPLE_SAMPLING_SEED", 907),
        ):
            expected = run(1)
            for method in [
                value
                for value in ("spawn", "fork")
                if value in multiprocessing.get_all_start_methods()
            ]:
                executor = partial(
                    ProcessPoolExecutor, mp_context=multiprocessing.get_context(method)
                )
                for workers in (1, 2, 4):
                    with (
                        self.subTest(method=method, workers=workers),
                        patch("traincheck.infer_engine.ProcessPoolExecutor", executor),
                    ):
                        self.assertEqual(run(workers), expected)

    def test_actual_engine_cross_trace_merge_keeps_caps_and_total_counts(self):
        first, second = hypothesis("a", 17, 8, 3), hypothesis("b", 29, 11, 3)
        engine, trace_indexes, lookup = InferEngine([]), {}, {}
        rng = random.Random(712)
        engine._merge_hypotheses([first], 0, trace_indexes, lookup, rng)
        engine._merge_hypotheses([second], 1, trace_indexes, lookup, rng)
        self.assertEqual(len(trace_indexes), 1)
        self.assertEqual(next(iter(trace_indexes.values())), [0, 1])
        self.assertEqual(
            (len(first.positive_examples), first.positive_examples.num_seen), (3, 46)
        )
        self.assertEqual(
            (len(first.negative_examples), first.negative_examples.num_seen), (3, 19)
        )

    def test_actual_stage_merge_keeps_caps_and_total_counts(self):
        first, second = hypothesis("a", 17, 8, 3), hypothesis("b", 29, 11, 3)

        class Stages:
            def is_stage_annotated(self):
                return True

            def get_traces_for_stage(self):
                return {"training": "training", "evaluation": "evaluation"}

            def get_all_stages(self):
                return {"training", "evaluation"}

        by_stage = {"training": first, "evaluation": second}

        def inferred(stage):
            item = by_stage[stage]
            return [item.invariant], [], [item]

        with patch.object(APIContainRelation, "_infer", side_effect=inferred):
            invariants, failures, successful = APIContainRelation.infer(Stages(), True)
        self.assertEqual((len(invariants), len(failures), len(successful)), (1, 0, 1))
        self.assertEqual(
            (
                len(successful[0].positive_examples),
                successful[0].positive_examples.num_seen,
            ),
            (3, 46),
        )
        self.assertEqual(
            (
                len(successful[0].negative_examples),
                successful[0].negative_examples.num_seen,
            ),
            (3, 19),
        )

    def test_sampled_generalization_does_not_treat_retained_union_as_population(self):
        from traincheck.invariant.contain_relation import _merge_hypotheses

        hypotheses = [hypothesis("a", 20, 9, 3), hypothesis("b", 40, 15, 3)]
        with patch(
            "traincheck.invariant.contain_relation.calc_likelihood",
            side_effect=AssertionError("sampled union must not be generalized"),
        ):
            result = _merge_hypotheses(hypotheses)
        self.assertEqual(
            [id(item) for item in result], [id(item) for item in hypotheses]
        )

    def test_complete_generalization_deduplicates_known_union_then_applies_cap(self):
        from traincheck.invariant.contain_relation import _merge_hypotheses

        first, second = hypothesis("a", 0, 0, 10), hypothesis("b", 0, 0, 10)
        first.positive_examples = reservoir(range(10), 10)
        second.positive_examples = reservoir(range(10, 20), 10)
        first.negative_examples = reservoir(range(10, 20), 10)
        second.negative_examples = reservoir(range(10), 10)
        expected = values(reservoir(range(20), 10))
        # Repeating one complete input must not double-count the known union.
        for inputs in ([first, second], [first, second, first]):
            with (
                self.subTest(inputs=len(inputs)),
                patch.object(config, "MAX_EXAMPLE_LIST_SIZE", 10),
                patch.object(config, "EXAMPLE_SAMPLING_SEED", 42),
            ):
                merged = _merge_hypotheses(inputs)
            self.assertEqual(len(merged), 1)
            positive, negative = (
                merged[0].positive_examples,
                merged[0].negative_examples,
            )
            self.assertEqual((len(positive), positive.num_seen), (10, 20))
            self.assertEqual((len(negative), negative.num_seen), (0, 0))
            self.assertEqual(values(positive), expected)
            self.assertEqual(negative.group_names, {GROUP})
            self.assertEqual(first.positive_examples.num_seen, 10)
            self.assertEqual(second.positive_examples.num_seen, 10)

    def test_worker_snapshot_carries_reservoir_settings_under_fork_and_spawn(self):
        with (
            patch.object(config, "MAX_EXAMPLE_LIST_SIZE", 7),
            patch.object(config, "EXAMPLE_SAMPLING_SEED", 907),
        ):
            settings = config_snapshot()
            expected = worker_reservoir_result(29)
        self.assertEqual(settings["MAX_EXAMPLE_LIST_SIZE"], 7)
        self.assertEqual(settings["EXAMPLE_SAMPLING_SEED"], 907)
        for method in [
            value
            for value in ("spawn", "fork")
            if value in multiprocessing.get_all_start_methods()
        ]:
            with (
                self.subTest(method=method),
                ProcessPoolExecutor(
                    max_workers=2,
                    mp_context=multiprocessing.get_context(method),
                    initializer=worker_init,
                    initargs=([], settings),
                ) as executor,
            ):
                futures = [
                    executor.submit(worker_reservoir_result, 29) for _ in range(2)
                ]
                self.assertEqual(
                    [future.result(timeout=60) for future in futures],
                    [expected, expected],
                )


if __name__ == "__main__":
    unittest.main()
