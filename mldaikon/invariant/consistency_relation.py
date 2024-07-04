import logging
from itertools import combinations

from tqdm import tqdm

from mldaikon.config import config
from mldaikon.invariant.base_cls import (
    Example,
    ExampleList,
    Hypothesis,
    Invariant,
    Relation,
    VarTypeParam,
)
from mldaikon.invariant.precondition import find_precondition
from mldaikon.trace.trace import Liveness, Trace

tracker_var_field_prefix = "attributes."


def calc_liveness_overlap(liveness1: Liveness, liveness2: Liveness) -> float:
    assert (
        liveness1.start_time is not None
        and liveness1.end_time is not None
        and liveness2.start_time is not None
        and liveness2.end_time is not None
    ), "Liveness should have both start_time and end_time."

    if (
        liveness1.start_time >= liveness2.end_time
        or liveness1.end_time <= liveness2.start_time
    ):
        return 0
    return (
        min(liveness1.end_time, liveness2.end_time)
        - max(liveness1.start_time, liveness2.start_time)
    ) / (
        max(liveness1.end_time, liveness2.end_time)
        - min(liveness1.start_time, liveness2.start_time)
    )


def get_attr_name(col_name: str) -> str:
    if tracker_var_field_prefix not in col_name:
        raise ValueError(f"{col_name} does not contain the tracker_var_field_prefix.")
    return col_name[len(tracker_var_field_prefix) :]


def compare_with_fp_tolerance(value1, value2):
    if type(value1) != type(value2):
        return False
    if isinstance(value1, list):
        if len(value1) != len(value2):
            return False
        for idx, val in enumerate(value1):
            if not compare_with_fp_tolerance(val, value2[idx]):
                return False
        return True
    if isinstance(value1, dict):
        if len(value1) != len(value2):
            return False
        for key in value1:
            if key not in value2:
                return False
            if not compare_with_fp_tolerance(value1[key], value2[key]):
                return False
        return True
    if isinstance(value1, float):
        return abs(value1 - value2) < 1e-8
    return value1 == value2


def skip_init_values(var_type: str):
    for skip_init_type in config.SKIP_INIT_VALUE_TYPES_KEY_WORDS:
        if skip_init_type in var_type.lower():
            return True
    return False


class VariableValueSelector:
    def __init__(self, var_type1, attr1, var_type2, attr2, precondition):
        self.var_type1 = var_type1
        self.attr1 = attr1
        self.var_type2 = var_type2
        self.attr2 = attr2
        self.precondition = precondition

    def __call__(self, trace: Trace) -> list | None:
        # TODO: Implement this scanner

        # YOU CAN'T SIMPLY SCAN ON A PARTIAL TRACE, YOU NEED TO SCAN ON THE WHOLE TRACE TO ESTABLISH THE INVARIANTs

        return None


class ConsistencyRelation(Relation):

    @staticmethod
    def infer(trace: Trace) -> list[Invariant]:
        """Infer Invariants for the ConsistencyRelation."""

        logger = logging.getLogger(__name__)

        ## 1. Pre-scanning: Collecting variable instances and their values from the trace
        # get identifiers of the variables, those variables can be used to query the actual values
        var_insts = trace.get_var_insts()
        if len(var_insts) == 0:
            logger.warning("No variables found in the trace.")
            return []

        ## 2. Hypothesis Generation Based on Liveness Overlapping
        hypothesis = set()  # {(var_type1, attr1, var_type2, attr2)}
        for var_inst, other_var_inst in tqdm(
            combinations(var_insts, 2),
            desc="Generating Hypothesis",
            total=len(var_insts) * (len(var_insts) - 1) // 2,
        ):
            for attr in var_insts[var_inst]:
                for other_attr in var_insts[other_var_inst]:
                    if var_inst == other_var_inst and attr == other_attr:
                        # skipping the same variable instance's same attribute
                        continue

                    # if we already have such hypothesis, skipping
                    if (
                        var_inst.var_type,
                        attr,
                        other_var_inst.var_type,
                        other_attr,
                    ) in hypothesis:
                        continue

                    if (
                        other_var_inst.var_type,
                        other_attr,
                        var_inst.var_type,
                        attr,
                    ) in hypothesis:
                        continue

                    if (
                        trace.events[tracker_var_field_prefix + attr].dtype
                        != trace.events[tracker_var_field_prefix + other_attr].dtype
                    ):
                        continue

                    # for each pair of attributes, calculate the liveness overlapping
                    done_creating_hypothesis = False
                    for value in var_insts[var_inst][attr]:
                        saw_overlap = False
                        if done_creating_hypothesis:
                            break
                        for other_value in var_insts[other_var_inst][other_attr]:
                            overlap = calc_liveness_overlap(
                                value.liveness, other_value.liveness
                            )
                            if overlap > config.LIVENESS_OVERLAP_THRESHOLD:
                                saw_overlap = True
                                if compare_with_fp_tolerance(
                                    value.value, other_value.value
                                ):
                                    hypothesis.add(
                                        (
                                            var_inst.var_type,
                                            attr,
                                            other_var_inst.var_type,
                                            other_attr,
                                        )
                                    )
                                    logger.debug(
                                        f"Adding Hypothesis: ({var_inst.var_type}, {attr}, {other_var_inst.var_type}, {other_attr})"
                                    )
                                    done_creating_hypothesis = True
                                    break
                            else:
                                if saw_overlap:
                                    # there won't be any more overlap, so we can break
                                    break

        ## 3. Hypothesis Pruning
        logger.debug(f"Hypothesis: {hypothesis}")
        logger.debug(f"Number of Hypothesis: {len(hypothesis)}")

        # for each hypothesis, collect number of positive examples seen, if it is below a threshold, prune it
        filtered_hypothesis = []  # [(var_type1, attr1, var_type2, attr2)]
        for hypo in hypothesis:
            var_type1 = hypo[0]
            attr1 = hypo[1]
            var_type2 = hypo[2]
            attr2 = hypo[3]

            # collect all variables that have the same types as var_type1 and var_type2
            var_type1_vars = [
                var_inst for var_inst in var_insts if var_inst.var_type == var_type1
            ]
            var_type2_vars = [
                var_inst for var_inst in var_insts if var_inst.var_type == var_type2
            ]

            positive_examples = 0
            positive_examples_threshold = 0  # This number should be the total number of varInst pairs on which the hypothesis is applicable

            # HACK: if both types are torch types, let's skip the init values (we've seen in DS-1801 that many unrelated layers have the same value due to the initialization at step 0)
            is_skipping_init_values = False

            for skip_init_type in config.SKIP_INIT_VALUE_TYPES_KEY_WORDS:
                if (
                    skip_init_type in var_type1.lower()
                    and skip_init_type in var_type2.lower()
                ):
                    is_skipping_init_values = True
                    logger.debug(
                        f"Skipping init values for {var_type1} and {var_type2}"
                    )
                    break

            for idx1, var_inst1 in enumerate(
                tqdm(var_type1_vars, desc=f"Pruning Hypo {hypo}")
            ):
                for idx2, var_inst2 in enumerate(var_type2_vars):
                    if var_type1 == var_type2 and attr1 == attr2 and idx1 >= idx2:
                        continue
                    found_positive_example = False
                    if var_inst1 == var_inst2:
                        continue

                    for val_idx1, value1 in enumerate(var_insts[var_inst1][attr1]):
                        for val_idx2, value2 in enumerate(var_insts[var_inst2][attr2]):
                            if (
                                is_skipping_init_values
                                and val_idx1 == 0
                                and val_idx2 == 0
                            ):
                                # skipping the init values
                                continue

                            overlap = calc_liveness_overlap(
                                value1.liveness, value2.liveness
                            )
                            if overlap > config.LIVENESS_OVERLAP_THRESHOLD:
                                if compare_with_fp_tolerance(
                                    var_insts[var_inst1][attr1][val_idx1].value,
                                    var_insts[var_inst2][attr2][val_idx2].value,
                                ):
                                    positive_examples += 1
                                    found_positive_example = True
                    if found_positive_example:
                        positive_examples_threshold += 1

            if is_skipping_init_values and positive_examples > 0:
                filtered_hypothesis.append(hypo)
                logger.debug(
                    f"Keeping hypothesis (INIT VALUEs SKIPPED): {hypo} with num positive examples {positive_examples}, expected threshold: {positive_examples_threshold}"
                )
            elif positive_examples > positive_examples_threshold:
                filtered_hypothesis.append(hypo)
                logger.debug(
                    f"Keeping hypothesis: {hypo} with num positive examples {positive_examples}, expected threshold: {positive_examples_threshold}"
                )
            else:
                logger.debug(
                    f"Filtering out hypothesis: {hypo} with num positive examples: {positive_examples}, expected threshold: {positive_examples_threshold}"
                )

        logger.debug(f"Filtered Hypothesis: {filtered_hypothesis}")

        ## 4.  Positive Examples and Negative Examples Collection
        group_name = "var"  # TODO: hacky, need to fix this
        hypothesis_with_examples = {
            hypo: Hypothesis(
                invariant=Invariant(
                    relation=ConsistencyRelation,
                    params=[
                        VarTypeParam(var_type=hypo[0], attr_name=hypo[1]),
                        VarTypeParam(var_type=hypo[2], attr_name=hypo[3]),
                    ],
                    precondition=None,
                    text_description=f"Consistency Relation between {hypo[0]}.{hypo[1]} and {hypo[2]}.{hypo[3]}",
                ),
                positive_examples=ExampleList({group_name}),
                negative_examples=ExampleList({group_name}),
            )
            for hypo in filtered_hypothesis
        }
        for hypo in hypothesis_with_examples:
            var_type1 = hypo[0]
            attr1 = hypo[1]
            var_type2 = hypo[2]
            attr2 = hypo[3]

            # HACK: if both types are torch types, let's skip the init values (we've seen in DS-1801 that many unrelated layers have the same value due to the initialization at step 0)
            is_skipping_init_values = False
            if skip_init_values(var_type1) or skip_init_values(var_type2):
                is_skipping_init_values = True

            # collect all variables that have the same types as var_type1 and var_type2
            var_type1_vars = [
                var_inst for var_inst in var_insts if var_inst.var_type == var_type1
            ]
            var_type2_vars = [
                var_inst for var_inst in var_insts if var_inst.var_type == var_type2
            ]

            for var_inst1 in tqdm(
                var_type1_vars, desc=f"Collecting Examples for Hypo: {hypo}"
            ):
                for var_inst2 in var_type2_vars:
                    if var_inst1 == var_inst2:
                        continue
                    for val_idx1, value1 in enumerate(var_insts[var_inst1][attr1]):
                        for val_idx2, value2 in enumerate(var_insts[var_inst2][attr2]):
                            if is_skipping_init_values and (
                                val_idx1 == 0 or val_idx2 == 0
                            ):
                                # skipping the init values
                                continue

                            overlap = calc_liveness_overlap(
                                value1.liveness, value2.liveness
                            )
                            if overlap > config.LIVENESS_OVERLAP_THRESHOLD:
                                if compare_with_fp_tolerance(
                                    value1.value,
                                    value2.value,
                                ):
                                    hypothesis_with_examples[
                                        hypo
                                    ].positive_examples.add_example(
                                        Example(
                                            {
                                                group_name: [
                                                    value1.traces[0],
                                                    value2.traces[0],
                                                ]
                                            }  ## HACK to make preconditions inference work for `step`
                                        )
                                    )
                                else:
                                    hypothesis_with_examples[
                                        hypo
                                    ].negative_examples.add_example(
                                        Example(
                                            {
                                                group_name: [
                                                    value1.traces[0],
                                                    value2.traces[0],
                                                ]
                                            }  ## HACK to make preconditions inference work for `step`
                                        )
                                    )

        ## 5. Precondition Inference TODO: this can be abstracted into a separate function that takes a list of hypothesis and returns those with preconditions
        hypos_to_delete = []
        for hypo in hypothesis_with_examples:
            preconditions = find_precondition(
                hypothesis_with_examples[hypo],
                keys_to_skip=[f"attributes.{hypo[1]}", f"attributes.{hypo[3]}"],
            )
            logger.debug(f"Preconditions for {hypo}:\n{str(preconditions)}")

            if preconditions is not None:
                hypothesis_with_examples[hypo].invariant.precondition = preconditions
            else:
                logger.debug(f"Precondition not found for {hypo}")
                hypos_to_delete.append(hypo)

        for hypo in hypos_to_delete:
            del hypothesis_with_examples[hypo]

        return list([hypo.invariant for hypo in hypothesis_with_examples.values()])

    @staticmethod
    def evaluate(value_group: list) -> bool:
        """Evaluate the consistency relation between multiple values.

        Args:
            value_group: list
                - a list of values to be evaluated
                These values can be scalar values or a list of values.
                If the values are a list of values, it is essential that these lists
                will have the same length.
        """
        assert len(value_group) > 1, "The value_group must have at least two values."

        # simplified implementation
        return all(value == value_group[0] for value in value_group)
