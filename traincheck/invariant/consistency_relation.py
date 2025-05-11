import logging
import time
from itertools import combinations

from tqdm import tqdm

from traincheck.config import config
from traincheck.invariant.base_cls import (
    CheckerResult,
    Example,
    ExampleList,
    FailedHypothesis,
    Hypothesis,
    Invariant,
    Relation,
    VarTypeParam,
)
from traincheck.invariant.precondition import find_precondition
from traincheck.trace.trace import Trace
from traincheck.trace.types import Liveness

tracker_var_field_prefix = "attributes."

VAR_GROUP_NAME = "var"


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
    """Infer the Consistency Relation between two variables.

    the variables here are the instances of the same type, and should be long-lived variables like model parameters and optimizer states.

    This relation is mainly helpful to infer nuances in distributed training,
    where the consistency relationships of the variables across different nodes is crucial to maintain.
    """

    @staticmethod
    def infer(trace: Trace) -> tuple[list[Invariant], list[FailedHypothesis]]:
        """Infer Invariants for the ConsistencyRelation."""
        logger = logging.getLogger(__name__)

        hypotheses = ConsistencyRelation.generate_hypothesis(trace)

        failed_hypothesis = []
        passed_hypothesis = []
        for hypo in hypotheses:
            param1 = hypo.invariant.params[0]
            param2 = hypo.invariant.params[1]

            assert isinstance(param1, VarTypeParam) and isinstance(
                param2, VarTypeParam
            ), "Invariant parameters should be VarTypeParam."

            logger.debug(f"Finding Precondition for: {hypo.invariant.text_description}")
            preconditions = find_precondition(hypo, [trace])
            logger.debug(f"Preconditions for {hypo}:\n{str(preconditions)}")

            if preconditions is not None:
                hypo.invariant.precondition = preconditions
                hypo.invariant.num_positive_examples = len(hypo.positive_examples)
                hypo.invariant.num_negative_examples = len(hypo.negative_examples)
                passed_hypothesis.append(hypo)
            else:
                logger.debug(
                    f"Precondition not found for {hypo.invariant.text_description}"
                )
                failed_hypothesis.append(
                    FailedHypothesis(hypo, "Precondition not found")
                )

        return (
            list([hypothesis.invariant for hypothesis in passed_hypothesis]),
            failed_hypothesis,
        )

    @staticmethod
    def generate_hypothesis(trace: Trace) -> list[Hypothesis]:
        """Generate Hypothesis for the ConsistencyRelation."""
        logger = logging.getLogger(__name__)

        ## 1. Pre-scanning: Collecting variable instances and their values from the trace
        # get identifiers of the variables, those variables can be used to query the actual values
        var_insts = trace.get_var_insts()
        if len(var_insts) == 0:
            logger.warning("No variables found in the trace.")
            return []

        def is_hypo_already_in_hypothesis(hypo: tuple, hypothesis: set) -> bool:
            return (
                hypo in hypothesis or (hypo[2], hypo[3], hypo[0], hypo[1]) in hypothesis
            )

        def skip_attrs_with_different_dtypes(attr1, attr2):
            return trace.get_column_dtype(
                tracker_var_field_prefix + attr1
            ) != trace.get_column_dtype(tracker_var_field_prefix + attr2)

        ## 2. Hypothesis Generation Based on Liveness Overlapping
        hypothesis: set[tuple[str, str, str, str]] = (
            set()
        )  # {(var_type1, attr1, var_type2, attr2)}
        for var_inst, other_var_inst in tqdm(
            combinations(var_insts, 2),
            desc="Generating Hypothesis for Consistency Relation",
            total=len(var_insts) * (len(var_insts) - 1) // 2,
        ):
            for attr in var_insts[var_inst]:
                for other_attr in var_insts[other_var_inst]:
                    if var_inst == other_var_inst and attr == other_attr:
                        # skipping the same variable instance's same attribute
                        continue

                    hypo = (
                        var_inst.var_type,
                        attr,
                        other_var_inst.var_type,
                        other_attr,
                    )
                    if is_hypo_already_in_hypothesis(hypo, hypothesis):
                        continue

                    if skip_attrs_with_different_dtypes(attr, other_attr):
                        continue

                    is_skipping_init_values = False
                    if skip_init_values(var_inst.var_type) or skip_init_values(
                        other_var_inst.var_type
                    ):
                        is_skipping_init_values = True

                    # for each pair of attributes, calculate the liveness overlapping
                    done_creating_hypothesis = False
                    seen_positive_examples = 0
                    for value in var_insts[var_inst][attr][
                        int(is_skipping_init_values) :
                    ]:
                        saw_overlap = False
                        if done_creating_hypothesis:
                            break
                        for other_value in var_insts[other_var_inst][other_attr][
                            int(is_skipping_init_values) :
                        ]:
                            overlap = calc_liveness_overlap(
                                value.liveness, other_value.liveness
                            )
                            if overlap > config.LIVENESS_OVERLAP_THRESHOLD:
                                saw_overlap = True
                                if compare_with_fp_tolerance(
                                    value.value, other_value.value
                                ):
                                    seen_positive_examples += 1

                                if (
                                    seen_positive_examples
                                    >= config.POSITIVE_EXAMPLES_THRESHOLD
                                ):
                                    logger.debug(
                                        f"Adding Hypothesis: ({var_inst.var_type}, {attr}, {other_var_inst.var_type}, {other_attr})"
                                    )
                                    hypothesis.add(hypo)
                                    done_creating_hypothesis = True
                                    break
                            else:
                                if saw_overlap:
                                    # there won't be any more overlap, so we can break
                                    break

        filtered_hypothesis = hypothesis
        logger.debug(f"Filtered Hypothesis: {filtered_hypothesis}")

        ## 4.  Positive Examples and Negative Examples Collection
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
                positive_examples=ExampleList({VAR_GROUP_NAME}),
                negative_examples=ExampleList({VAR_GROUP_NAME}),
            )
            for hypo in filtered_hypothesis
        }

        for hypo in hypothesis_with_examples:
            ConsistencyRelation.collect_examples(trace, hypothesis_with_examples[hypo])

        return list(hypothesis_with_examples.values())

    @staticmethod
    def collect_examples(trace: Trace, hypothesis: Hypothesis):
        """Collect Examples for the ConsistencyRelation.
        The modification of the hypothesis is done in-place.
        """
        inv = hypothesis.invariant
        assert (
            inv.relation == ConsistencyRelation
        ), "Invariant should be ConsistencyRelation."
        assert len(inv.params) == 2, "Invariant should have exactly two parameters."

        param1 = inv.params[0]
        param2 = inv.params[1]

        assert isinstance(param1, VarTypeParam) and isinstance(
            param2, VarTypeParam
        ), "Invariant parameters should be VarTypeParam."
        var_type1, attr1 = param1.var_type, param1.attr_name
        var_type2, attr2 = param2.var_type, param2.attr_name

        is_skipping_init_values = False
        if skip_init_values(var_type1) or skip_init_values(var_type2):
            is_skipping_init_values = True

        var_insts = trace.get_var_insts()

        # collect all variables that have the same types as var_type1 and var_type2
        var_type1_vars = [
            var_inst for var_inst in var_insts if var_inst.var_type == var_type1
        ]
        var_type2_vars = [
            var_inst for var_inst in var_insts if var_inst.var_type == var_type2
        ]

        for idx1, var_inst1 in enumerate(
            tqdm(var_type1_vars, desc=f"Collecting Examples for Hypo: {hypothesis}")
        ):
            for idx2, var_inst2 in enumerate(var_type2_vars):
                if var_type1 == var_type2 and attr1 == attr2 and idx1 >= idx2:
                    continue
                if var_inst1 == var_inst2:
                    continue
                if (
                    attr1 not in var_insts[var_inst1]
                    or attr2 not in var_insts[var_inst2]
                ):
                    continue
                for _, value1 in enumerate(
                    var_insts[var_inst1][attr1][int(is_skipping_init_values) :]
                ):
                    for _, value2 in enumerate(
                        var_insts[var_inst2][attr2][int(is_skipping_init_values) :]
                    ):
                        saw_overlap = False
                        overlap = calc_liveness_overlap(
                            value1.liveness, value2.liveness
                        )
                        if overlap > config.LIVENESS_OVERLAP_THRESHOLD:
                            if compare_with_fp_tolerance(
                                value1.value,
                                value2.value,
                            ):
                                hypothesis.positive_examples.add_example(
                                    Example(
                                        {
                                            VAR_GROUP_NAME: [
                                                value1.traces[0],
                                                value2.traces[0],
                                            ]
                                        }  ## HACK to make preconditions inference work for `step`
                                    )
                                )
                            else:
                                hypothesis.negative_examples.add_example(
                                    Example(
                                        {
                                            VAR_GROUP_NAME: [
                                                value1.traces[0],
                                                value2.traces[0],
                                            ]
                                        }  ## HACK to make preconditions inference work for `step`
                                    )
                                )
                        else:
                            if saw_overlap:
                                # there won't be any more overlap, so we can break
                                break

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

    @staticmethod
    def static_check_all(
        trace: Trace, inv: Invariant, check_relation_first: bool
    ) -> CheckerResult:
        # 1. examine the invariant, and get relevant variables based on type and attribute
        assert len(inv.params) == 2, "Invariant should have exactly two parameters."
        assert inv.precondition is not None, "Invariant should have a precondition."

        logger = logging.getLogger(__name__)

        param1 = inv.params[0]
        param2 = inv.params[1]

        assert isinstance(param1, VarTypeParam) and isinstance(
            param2, VarTypeParam
        ), "Invariant parameters should be VarTypeParam."
        inv_triggered = False

        var_type1, attr1 = param1.var_type, param1.attr_name
        var_type2, attr2 = param2.var_type, param2.attr_name

        all_var_insts = trace.get_var_insts()
        try:
            type1_attr1 = {
                var_id: all_var_insts[var_id][attr1]
                for var_id in all_var_insts
                if var_id.var_type == var_type1
            }
            type2_attr2 = {
                var_id: all_var_insts[var_id][attr2]
                for var_id in all_var_insts
                if var_id.var_type == var_type2
            }
        except KeyError as e:
            logger.error(
                f"Variable Type or Attribute not found in the trace: {var_type1}.{attr1} or {var_type2}.{attr2}, original error: {str(e)}"
            )
            # TODO: add a type of result being "not checked due to missing variable in the trace"
            return CheckerResult(
                trace=None,
                invariant=inv,
                check_passed=True,
                triggered=False,
            )

        # collect value pairs to be checked
        start_time_collecting_pairs = time.time()
        num_collected_pairs = 0
        value_pairs_to_check: dict[float, list[tuple]] = {}
        for i, var1_id in enumerate(tqdm(type1_attr1, desc="Collecting Value Pairs")):
            for j, var2_id in enumerate(type2_attr2):
                if var_type1 == var_type2 and attr1 == attr2 and i >= j:
                    continue
                assert var1_id != var2_id, "Variable instances should be different."
                for attr1_val in type1_attr1[var1_id]:
                    for attr2_val in type2_attr2[var2_id]:
                        assert (
                            attr1_val.liveness.start_time is not None
                            and attr1_val.liveness.end_time is not None
                            and attr2_val.liveness.start_time is not None
                            and attr2_val.liveness.end_time is not None
                        ), "Liveness should have both start_time and end_time."
                        if attr2_val.liveness.start_time >= attr1_val.liveness.end_time:
                            # attr2 starts after attr1 ends, no need to check further
                            break
                        if attr1_val.liveness.end_time <= attr2_val.liveness.start_time:
                            # attr1 ends before attr2 starts, fast forward to the next attr2
                            continue
                        overlap = calc_liveness_overlap(
                            attr1_val.liveness, attr2_val.liveness
                        )
                        if overlap > config.LIVENESS_OVERLAP_THRESHOLD:
                            time_pair = (
                                attr1_val.liveness.end_time
                            )  # FIXME: adhoc solution, need to find a better way to determine the time
                            if time_pair not in value_pairs_to_check:
                                value_pairs_to_check[time_pair] = []
                            value_pairs_to_check[time_pair].append(
                                (attr1_val, attr2_val)
                            )
                            num_collected_pairs += 1
        end_time_collecting_pairs = time.time()
        logger.debug(
            f"Time to collect {num_collected_pairs} value pairs to check: {end_time_collecting_pairs - start_time_collecting_pairs}"
        )

        # sort the value_pairs_to_check by key in ascending order
        start_time_checking_pairs = time.time()
        num_checked_pairs = 0
        value_pairs_to_check = dict(sorted(value_pairs_to_check.items()))
        for time_pair in tqdm(value_pairs_to_check, desc="Checking Value Pairs"):
            for attr1_val, attr2_val in value_pairs_to_check[time_pair]:
                traces = [attr1_val.traces[-1], attr2_val.traces[-1]]
                num_checked_pairs += 1
                if check_relation_first:
                    compare_result = ConsistencyRelation.evaluate(
                        [attr1_val.value, attr2_val.value]
                    )
                    if not compare_result:
                        # check for precondition match, if yes, report alarm
                        if inv.precondition.verify(traces, VAR_GROUP_NAME, trace):
                            inv_triggered = True
                            logger.error(
                                f"Invariant {inv} violated near time {attr1_val.liveness.end_time}, precentage: {trace.get_time_precentage(attr1_val.liveness.end_time)}"  # type: ignore
                            )
                            end_time_collecting_pairs = time.time()
                            logger.info(
                                f"VIOLATION HAPPENED: Time to check {num_checked_pairs} value pairs: {end_time_collecting_pairs - start_time_checking_pairs}"
                            )
                            return CheckerResult(
                                trace=traces,
                                invariant=inv,
                                check_passed=False,
                                triggered=True,
                            )
                        else:
                            logger.debug(
                                f"Violation detected but Precondition not satisfied at liveness 1: {attr1_val.liveness.start_time}, {attr1_val.liveness.end_time}, liveness 2: {attr2_val.liveness.start_time}, {attr2_val.liveness.end_time}, overlap: {calc_liveness_overlap(attr1_val.liveness, attr2_val.liveness)}"  # type: ignore
                            )
                else:
                    if inv.precondition.verify(traces, VAR_GROUP_NAME, trace):
                        inv_triggered = True
                        compare_result = ConsistencyRelation.evaluate(
                            [attr1_val.value, attr2_val.value]
                        )
                        if not compare_result:
                            end_time_collecting_pairs = time.time()
                            logger.info(
                                f"VIOLATION HAPPENED: Time to check {num_checked_pairs} value pairs: {end_time_collecting_pairs - start_time_checking_pairs}"
                            )
                            logger.error(
                                f"Invariant {inv} violated for {var1_id} and {var2_id} near time {attr1_val.liveness.end_time}, precentage: {trace.get_time_precentage(attr1_val.liveness.end_time)}"  # type: ignore
                            )
                            return CheckerResult(
                                trace=traces,
                                invariant=inv,
                                check_passed=False,
                                triggered=True,
                            )
                    else:
                        logger.debug(
                            f"Precondition not satisfied at liveness 1: {attr1_val.liveness.start_time}, {attr1_val.liveness.end_time}, liveness 2: {attr2_val.liveness.start_time}, {attr2_val.liveness.end_time}, overlap: {calc_liveness_overlap(attr1_val.liveness, attr2_val.liveness)}, skipping the check"  # type: ignore
                        )
        end_time_collecting_pairs = time.time()
        logger.info(
            f"PASSED: Time to check {num_checked_pairs} value pairs: {end_time_collecting_pairs - start_time_checking_pairs}"
        )

        # TODO: implement the precondition improvement logic here (i.e. when compare_result is True, check if the precondition is satisfied, if not, improve the precondition)
        return CheckerResult(
            trace=None,
            invariant=inv,
            check_passed=True,
            triggered=inv_triggered,
        )

    @staticmethod
    def get_precondition_infer_keys_to_skip(hypothesis: Hypothesis) -> list[str]:
        assert (
            len(hypothesis.invariant.params) == 2
        ), "Invariant should have exactly two parameters."
        param1 = hypothesis.invariant.params[0]
        param2 = hypothesis.invariant.params[1]

        assert isinstance(param1, VarTypeParam) and isinstance(
            param2, VarTypeParam
        ), "Invariant parameters should be VarTypeParam."
        attr1 = param1.var_type, param1.attr_name
        attr2 = param2.var_type, param2.attr_name

        return [
            f"attributes.{attr1}",
            f"attributes.{attr2}",
            "attributes.data",
            "attributes.grad",
        ]
