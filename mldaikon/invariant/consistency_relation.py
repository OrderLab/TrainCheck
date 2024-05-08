import logging
from typing import NamedTuple
from tqdm import tqdm

import polars as pl

from mldaikon.invariant.base_cls import Hypothesis, Invariant, Relation
from mldaikon.ml_daikon_trace import Trace
from mldaikon.config import config

tracker_var_field_prefix = "attributes."


class VarInstId(NamedTuple):
    process_id: int
    var_name: str
    var_type: str


class Liveness:
    def __init__(self, start_time: int, end_time: int):
        self.start_time = start_time
        self.end_time = end_time


class AttrState:
    def __init__(self, value: type, liveness: Liveness):
        self.value: type = value
        self.liveness: Liveness = liveness


def calc_liveness_overlap(liveness1: Liveness, liveness2: Liveness) -> int:
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
            if not compare(val, value2[idx]):
                return False
        return True
    if isinstance(value1, dict):
        if len(value1) != len(value2):
            return False
        for key in value1:
            if key not in value2:
                return False
            if not compare(value1[key], value2[key]):
                return False
        return True
    if isinstance(value1, float):
        return abs(value1 - value2) < 1e-6
    return value1 == value2


class ConsistencyRelation(Relation):
    def __init__(self, parent_func_name: str, child_func_name: str):
        self.parent_func_name = parent_func_name
        self.child_func_name = child_func_name

    @staticmethod
    def infer(trace: Trace) -> list[Invariant]:
        """Infer Invariants for the ConsistencyRelation."""

        ## 1. Pre-scanning: Collecting variable instances and their values from the trace
        # get identifiers of the variables, those variables can be used to query the actual values
        var_insts = trace.get_variable_insts()
        var_inst_values = {}
        for var_inst in var_insts:
            var_inst_states = trace.filter(
                pl.col("process_id") == var_inst["process_id"],
                pl.col("var_name") == var_inst["var_name"],
                pl.col("var_type") == var_inst["var_type"],
            )

            state_init = var_inst_states.filter(
                pl.col("type") == "state_init"
            )  ## state_init is different in diff-based variable tracker, but not needed in proxyClass
            assert len(state_init) == 1, "There should be only one state_init event."
            state_init = state_init.row(0, named=True)
            state_changes = var_inst_states.filter(pl.col("type") == "state_change")

            # init attribute values for this variable
            attr_values = {}
            for col in state_init:
                if col.startswith(tracker_var_field_prefix):
                    attr_name = get_attr_name(col)
                    attr_values[attr_name] = [
                        AttrState(state_init[col], Liveness(state_init["time"], None))
                    ]

            for state_change in state_changes:
                for col in state_change:
                    if col.startswith(tracker_var_field_prefix):
                        attr_name = get_attr_name(col)
                        if not attr_name in attr_values:
                            attr_values[attr_name] = [
                                AttrState(
                                    state_change[col],
                                    Liveness(state_change["time"], None),
                                )
                            ]
                        else:
                            if attr_values[attr_name][-1].value != state_change[col]:
                                attr_values[attr_name][-1].liveness.end_time = (
                                    state_change["time"]
                                )
                                attr_values[attr_name].append(
                                    AttrState(
                                        state_change[col],
                                        Liveness(state_change["time"], None),
                                    )
                                )

            # set end time for the last state change
            for attr_name in attr_values:
                if attr_values[attr_name][-1].liveness.end_time is None:
                    attr_values[attr_name][-1].liveness.end_time = trace.get_end_time()

            var_inst_values[
                VarInstId(
                    var_inst["process_id"], var_inst["var_name"], var_inst["var_type"]
                )
            ] = attr_values

        ## 2. Hypothesis Generation Based on Liveness Overlapping
        hypothesis = (
            []
        )  # key: (var_type1, attr1, var_type2, attr2), var_type1 might be the same as var_type2
        for var_inst in tqdm(var_inst_values):
            for attr in var_inst_values[var_inst]:
                for other_var_inst in var_inst_values:
                    for other_attr in var_inst_values[other_var_inst]:
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

                        # if the types are different, skipping
                        if type(var_inst_values[var_inst][attr][0].value) != type(
                            var_inst_values[other_var_inst][other_attr][0].value
                        ):
                            continue

                        # for each pair of attributes, calculate the liveness overlapping
                        for value in var_inst_values[var_inst][attr]:
                            saw_overlap = False
                            for other_value in var_inst_values[other_var_inst][
                                other_attr
                            ]:
                                overlap = calc_liveness_overlap(
                                    value.liveness, other_value.liveness
                                )
                                if overlap > config.LIVENESS_OVERLAP_THRESHOLD:
                                    saw_overlap = True
                                    if compare_with_fp_tolerance(
                                        value.value, other_value.value
                                    ):
                                        break
                                    hypothesis.append(
                                        (
                                            var_inst.var_type,
                                            attr,
                                            other_var_inst.var_type,
                                            other_attr,
                                        )
                                    )
                                    break
                                else:
                                    if saw_overlap:
                                        # there won't be any more overlap, so we can break
                                        break

        ## 3. Hypothesis Pruning

        # for each hypothesis, collect number of positive examples seen, if it is below a threshold, prune it
        filtered_hypothesis = []
        for hypo in hypothesis:
            var_type1 = hypo[0]
            attr1 = hypo[1]
            var_type2 = hypo[2]
            attr2 = hypo[3]

            # collect all variables that have the same types as var_type1 and var_type2
            var_type1_vars = [
                var_inst
                for var_inst in var_inst_values
                if var_inst.var_type == var_type1
            ]
            var_type2_vars = [
                var_inst
                for var_inst in var_inst_values
                if var_inst.var_type == var_type2
            ]

            positive_examples = 0
            POSITIVE_EXAMPLE_THRESHOLD = None
            if var_type1 != var_type2:
                POSITIVE_EXAMPLE_THRESHOLD = len(var_type1_vars) * len(var_type2_vars)
            elif attr1 != attr2:
                POSITIVE_EXAMPLE_THRESHOLD = len(var_type1_vars) * (
                    len(var_type1_vars) - 1
                )
            else:
                POSITIVE_EXAMPLE_THRESHOLD = (
                    len(var_type1_vars) * (len(var_type1_vars) - 1) / 2
                )

            for var_inst1 in var_type1_vars:
                for var_inst2 in var_type2_vars:
                    if var_inst1 == var_inst2:
                        continue
                    for value1 in var_inst_values[var_inst1][attr1]:
                        for value2 in var_inst_values[var_inst2][attr2]:
                            overlap = calc_liveness_overlap(
                                value1.liveness, value2.liveness
                            )
                            if overlap > config.LIVENESS_OVERLAP_THRESHOLD:
                                if compare_with_fp_tolerance(
                                    var_inst_values[var_inst1][attr1][0].value,
                                    var_inst_values[var_inst2][attr2][0].value,
                                ):
                                    positive_examples += 1
            if positive_examples > POSITIVE_EXAMPLE_THRESHOLD:
                filtered_hypothesis.append(hypo)

        ## 4.  Positive Examples and Negative Examples Collection
        hypothesis_with_examples = {
            key: Hypothesis(None, [], []) for key in filtered_hypothesis
        }
        for hypo in hypothesis_with_examples:
            var_type1 = hypo[0]
            attr1 = hypo[1]
            var_type2 = hypo[2]
            attr2 = hypo[3]

            # collect all variables that have the same types as var_type1 and var_type2
            var_type1_vars = [
                var_inst
                for var_inst in var_inst_values
                if var_inst.var_type == var_type1
            ]
            var_type2_vars = [
                var_inst
                for var_inst in var_inst_values
                if var_inst.var_type == var_type2
            ]

            for var_inst1 in var_type1_vars:
                for var_inst2 in var_type2_vars:
                    if var_inst1 == var_inst2:
                        continue
                    for value1 in var_inst_values[var_inst1][attr1]:
                        for value2 in var_inst_values[var_inst2][attr2]:
                            overlap = calc_liveness_overlap(
                                value1.liveness, value2.liveness
                            )
                            if overlap > config.LIVENESS_OVERLAP_THRESHOLD:
                                if compare_with_fp_tolerance(
                                    var_inst_values[var_inst1][attr1][0].value,
                                    var_inst_values[var_inst2][attr2][0].value,
                                ):
                                    hypothesis_with_examples[
                                        hypo
                                    ].positive_examples.append((var_inst1, var_inst2))
                                else:
                                    hypothesis_with_examples[
                                        hypo
                                    ].negative_examples.append((var_inst1, var_inst2))

        ## 5. Precondition Inference

        # Q: How can we proactively prune out useless stuff?

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
