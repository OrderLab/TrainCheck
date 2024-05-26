import re
from typing import NamedTuple

import polars as pl
from tqdm import tqdm

from mldaikon.config import config
from mldaikon.invariant.base_cls import Hypothesis, Invariant, Relation
from mldaikon.invariant.precondition import find_precondition
from mldaikon.ml_daikon_trace import Trace

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
    def __init__(self, value: type, liveness: Liveness, traces: list[dict]):
        self.value: type = value
        self.liveness: Liveness = liveness
        self.traces = traces


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
        for var_inst in tqdm(var_insts, desc="Indexing Variable Instances"):
            var_inst_states = trace.events.filter(
                pl.col("process_id") == var_inst["process_id"],
                pl.col("var_name") == var_inst["var_name"],
                pl.col("var_type") == var_inst["var_type"],
            )

            state_changes = var_inst_states.filter(pl.col("type") == "state_change")

            # init attribute values for this variable
            attr_values = {}
            for state_change in state_changes.rows(named=True):
                for col in state_change:
                    if col.startswith(tracker_var_field_prefix):
                        attr_name = get_attr_name(col)
                        # pruning out the attributes that might be properties
                        if any(
                            [
                                re.match(pattern, attr_name) is not None
                                for pattern in config.PROP_ATTR_PATTERNS
                            ]
                        ) or any(
                            [
                                isinstance(state_change[col], _type)
                                for _type in config.PROP_ATTR_TYPES
                            ]
                        ):
                            continue

                        if attr_name not in attr_values:
                            attr_values[attr_name] = [
                                AttrState(
                                    state_change[col],
                                    Liveness(state_change["time"], None),
                                    [state_change],
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
                                        [state_change],
                                    )
                                )
                            else:
                                attr_values[attr_name][-1].traces.append(state_change)

            # set end time for the last state change
            for attr_name in attr_values:
                if attr_values[attr_name][-1].liveness.end_time is None:
                    attr_values[attr_name][-1].liveness.end_time = trace.get_end_time()

            var_inst_values[
                VarInstId(
                    var_inst["process_id"], var_inst["var_name"], var_inst["var_type"]
                )
            ] = attr_values

        ## CHECK EVERY VALUE SHOULD HAVE A NON-EMPTY TRACES FIELD
        for var_inst in var_inst_values:
            for attr in var_inst_values[var_inst]:
                for value in var_inst_values[var_inst][attr]:
                    if len(value.traces) == 0:
                        print(f"Warning: No traces found for {var_inst} {attr}")

        ## 2. Hypothesis Generation Based on Liveness Overlapping  ## TODO: this part can be made more efficient by iterating over the types instead of the variables
        hypothesis = set()  # (var_type1, attr1, var_type2, attr2)
        for var_inst in tqdm(var_inst_values, desc="Generating Hypothesis"):
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

                        if trace.events[tracker_var_field_prefix + attr].dtype != trace.events[tracker_var_field_prefix + other_attr].dtype:
                            continue

                        # for each pair of attributes, calculate the liveness overlapping
                        done_creating_hypothesis = False
                        for value in var_inst_values[var_inst][attr]:
                            saw_overlap = False
                            if done_creating_hypothesis:
                                break
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
                                        hypothesis.add(
                                            (
                                                var_inst.var_type,
                                                attr,
                                                other_var_inst.var_type,
                                                other_attr,
                                            )
                                        )
                                        done_creating_hypothesis = True
                                        break
                                else:
                                    if saw_overlap:
                                        # there won't be any more overlap, so we can break
                                        break

        ## 3. Hypothesis Pruning
        print(f"Hypothesis: {hypothesis}")
        print(f"Number of Hypothesis: {len(hypothesis)}")

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
            positive_examples_threshold = 0  # This number should be the total number of varInst pairs on which the hypothesis is applicable

            # HACK: if both types are torch types, let's skip the init values (we've seen in DS-1801 that many unrelated layers have the same value due to the initialization at step 0)
            SKIP_INIT_VALUES = False
            if "tensor" in var_type1.lower() and "tensor" in var_type2.lower():
                SKIP_INIT_VALUES = True

            for idx1, var_inst1 in enumerate(
                tqdm(var_type1_vars, desc=f"Pruning Hypo {hypo}")
            ):
                for idx2, var_inst2 in enumerate(var_type2_vars):
                    if var_type1 == var_type2 and attr1 == attr2 and idx1 >= idx2:
                        continue
                    found_positive_example = False
                    if var_inst1 == var_inst2:
                        continue

                    for val_idx1, value1 in enumerate(
                        var_inst_values[var_inst1][attr1]
                    ):
                        for val_idx2, value2 in enumerate(
                            var_inst_values[var_inst2][attr2]
                        ):
                            if SKIP_INIT_VALUES and val_idx1 == 0 and val_idx2 == 0:
                                # skipping the init values
                                continue

                            overlap = calc_liveness_overlap(
                                value1.liveness, value2.liveness
                            )
                            if overlap > config.LIVENESS_OVERLAP_THRESHOLD:
                                if compare_with_fp_tolerance(
                                    var_inst_values[var_inst1][attr1][0].value,
                                    var_inst_values[var_inst2][attr2][0].value,
                                ):
                                    positive_examples += 1
                                    found_positive_example = True
                    if found_positive_example:
                        positive_examples_threshold += 1

            if SKIP_INIT_VALUES and positive_examples > 0:
                filtered_hypothesis.append(hypo)
                print(
                    f"Keeping hypothesis (INIT VALUEs SKIPPED): {hypo} with num positive examples {positive_examples}, expected threshold: {positive_examples_threshold}"
                )
            elif positive_examples > positive_examples_threshold:
                filtered_hypothesis.append(hypo)
                print(
                    f"Keeping hypothesis: {hypo} with num positive examples {positive_examples}, expected threshold: {positive_examples_threshold}"
                )
            else:
                print(
                    f"Filtering out hypothesis: {hypo} with num positive examples: {positive_examples}, expected threshold: {positive_examples_threshold}"
                )

        print(f"Filtered Hypothesis: {filtered_hypothesis}")

        ## 4.  Positive Examples and Negative Examples Collection
        hypothesis_with_examples = {
            key: Hypothesis(Invariant(None, None, None), [], [])
            for key in filtered_hypothesis
        }
        for hypo in hypothesis_with_examples:
            var_type1 = hypo[0]
            attr1 = hypo[1]
            var_type2 = hypo[2]
            attr2 = hypo[3]

            # HACK: if both types are torch types, let's skip the init values (we've seen in DS-1801 that many unrelated layers have the same value due to the initialization at step 0)
            SKIP_INIT_VALUES = False
            if "tensor" in var_type1.lower() and "tensor" in var_type2.lower():
                SKIP_INIT_VALUES = True

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

            for var_inst1 in tqdm(
                var_type1_vars, desc=f"Collecting Examples for Hypo: {hypo}"
            ):
                for var_inst2 in var_type2_vars:
                    if var_inst1 == var_inst2:
                        continue
                    for val_idx1, value1 in enumerate(
                        var_inst_values[var_inst1][attr1]
                    ):
                        for val_idx2, value2 in enumerate(
                            var_inst_values[var_inst2][attr2]
                        ):
                            if SKIP_INIT_VALUES and val_idx1 == 0 and val_idx2 == 0:
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
                                    ].positive_examples.append(
                                        [
                                            value1.traces[0],
                                            value2.traces[0],
                                        ]  ## HACK to make preconditions inference work for `step`
                                    )
                                else:
                                    hypothesis_with_examples[
                                        hypo
                                    ].negative_examples.append(
                                        [
                                            value1.traces[0],
                                            value2.traces[0],
                                        ]  ## HACK to make preconditions inference work for `step`
                                    )

        # ## DEBUGGING:
        # all_modules_that_have_hypothesis = []
        # for hypo in hypothesis_with_examples:
        #     for trace_pair in hypothesis_with_examples[hypo].positive_examples:
        #         all_modules_that_have_hypothesis.append(
        #                 trace_pair[0]["var_name"] + f" {trace_pair[0]['attributes.tensor_model_parallel']}" + f" {trace_pair[0]['meta_vars.step']} TP: {trace_pair[0]['meta_vars._TENSOR_MODEL_PARALLEL_GROUP_YUXUAN_RANK']} PP: {trace_pair[0]['meta_vars._PIPELINE_MODEL_PARALLEL_GROUP_YUXUAN_RANK']} DP: {trace_pair[0]['meta_vars._DATA_PARALLEL_GROUP_YUXUAN_RANK']}" + \
        #                     " -> " + trace_pair[1]["var_name"] + f" {trace_pair[1]['attributes.tensor_model_parallel']}" + f" {trace_pair[1]['meta_vars.step']} TP: {trace_pair[1]['meta_vars._TENSOR_MODEL_PARALLEL_GROUP_YUXUAN_RANK']} PP: {trace_pair[1]['meta_vars._PIPELINE_MODEL_PARALLEL_GROUP_YUXUAN_RANK']} DP: {trace_pair[1]['meta_vars._DATA_PARALLEL_GROUP_YUXUAN_RANK']}" + f" same_process?:{trace_pair[0]['process_id'] == trace_pair[1]['process_id']}"
        #         )

        # print("Positive Examples: {}".format("\n".join(all_modules_that_have_hypothesis)))

        # all_negative_modules_that_have_hypothesis = []
        # for hypo in hypothesis_with_examples:
        #     for trace_pair in hypothesis_with_examples[hypo].negative_examples:
        #         all_negative_modules_that_have_hypothesis.append(
        #                 trace_pair[0]["var_name"] + f" {trace_pair[0]['attributes.tensor_model_parallel']}" + f" {trace_pair[0]['meta_vars.step']} TP: {trace_pair[0]['meta_vars._TENSOR_MODEL_PARALLEL_GROUP_YUXUAN_RANK']} PP: {trace_pair[0]['meta_vars._PIPELINE_MODEL_PARALLEL_GROUP_YUXUAN_RANK']} DP: {trace_pair[0]['meta_vars._DATA_PARALLEL_GROUP_YUXUAN_RANK']}" + \
        #                     " -> " + trace_pair[1]["var_name"] + f" {trace_pair[1]['attributes.tensor_model_parallel']}" + f" {trace_pair[1]['meta_vars.step']} TP: {trace_pair[1]['meta_vars._TENSOR_MODEL_PARALLEL_GROUP_YUXUAN_RANK']} PP: {trace_pair[1]['meta_vars._PIPELINE_MODEL_PARALLEL_GROUP_YUXUAN_RANK']} DP: {trace_pair[1]['meta_vars._DATA_PARALLEL_GROUP_YUXUAN_RANK']}" + f" same_process?:{trace_pair[0]['process_id'] == trace_pair[1]['process_id']}"
        #         )

        # print("Negative Examples: {}".format("\n".join(all_negative_modules_that_have_hypothesis)))

        ## 5. Precondition Inference
        hypos_to_delete = []
        for hypo in hypothesis_with_examples:
            preconditions = find_precondition(hypothesis_with_examples[hypo])
            print(f"Preconditions for {hypo}:")
            print(f"{str(preconditions)}")
            # if we cannot find any preconditions, and there are no negative examples, we can infer the invariant
            if (
                len(preconditions) == 0
                and len(hypothesis_with_examples[hypo].negative_examples) == 0
            ):
                hypothesis_with_examples[hypo].invariant.precondition = (
                    None  # Unconditional
                )
            elif (
                len(preconditions) == 0
                and len(hypothesis_with_examples[hypo].negative_examples) > 0
            ):
                # delete the hypothesis
                """TODO: even if we cannot find any precondition, it might be possible that the invariant holds, but its just that our tracer didn't capture the necessary information.
                Thus, we might still want to evaluate the invariant's statistical likelihood instead of just deleting it.
                """
                hypos_to_delete.append(hypo)
            else:
                hypothesis_with_examples[hypo].invariant.precondition = preconditions

            """NOTE: If a hypo have no negative examples, potentially there might be noises in the preconditions inferred."""

        for hypo in hypos_to_delete:
            del hypothesis_with_examples[hypo]

        ## 6. TODO: Invariant Construction
        ## NEED TO THINK ABOUT HOW TO EXPRESS THIS INVARIANT
        print(f"Hypothesis Passed: {hypothesis_with_examples.keys()}")
        return list(hypothesis_with_examples.values())

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
