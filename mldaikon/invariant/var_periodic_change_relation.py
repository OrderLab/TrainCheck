import logging
from collections import defaultdict

import numpy as np

import mldaikon.config.config as config
from mldaikon.invariant.base_cls import (
    CheckerResult,
    Example,
    ExampleList,
    FailedHypothesis,
    Hypothesis,
    Invariant,
    Relation,
    VarNameParam,
    VarTypeParam,
)
from mldaikon.invariant.precondition import find_precondition
from mldaikon.trace.trace import Trace, VarInstId


def count_num_justification(count: int):
    # TODO: discuss to find a better way to distinguish between changed values
    return count > 1


def calculate_hypo_value(value) -> str:
    if value is None:
        return "None"

    if isinstance(value, (int, float)):
        hypo_value = f"{value:.7f}"
    elif isinstance(value, (list)):
        hypo_value = f"{np.linalg.norm(value, ord=1):.7f}"  # l1-norm
    elif isinstance(value, (str, bool)):
        hypo_value = f"{value}"
    else:
        hypo_value = str(value)

    return hypo_value


class VarPeriodicChangeRelation(Relation):
    @staticmethod
    def infer(trace: Trace) -> tuple[list[Invariant], list[FailedHypothesis]]:
        """Infer Invariants for the VariableChangeRelation."""
        logger = logging.getLogger(__name__)
        ## 1. Pre-scanning: Collecting variable instances and their values from the trace
        # get identifiers of the variables, those variables can be used to query the actual values
        var_insts = trace.get_var_insts()
        if len(var_insts) == 0:
            logger.warning("No variables found in the trace.")
            return [], []
        ## 2.Counting: count the number of each value of every variable attribute
        # TODO: record the intervals between occurrencess
        # TODO: improve time and memory efficiency

        occur_count: dict[VarInstId, dict[str, dict[str, int]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(int))
        )  # var_id -> attr_name -> hypo_value -> count
        for var_id, attrs in var_insts.items():
            for attr_name, attr_insts in attrs.items():
                for attr_inst in attr_insts:
                    hypo_value = calculate_hypo_value(attr_inst.value)
                    occur_count[var_id][attr_name][hypo_value] += 1

        # 3. Hypothesis generation
        all_hypothesis: dict[tuple[VarTypeParam | VarNameParam, str], Hypothesis] = {}
        var_group_name = "var"
        for var_id in occur_count:
            for attr_name in occur_count[var_id]:
                param: VarTypeParam | VarNameParam = (
                    VarNameParam(var_id.var_type, var_id.var_name, attr_name)
                    if config.VAR_INV_TYPE == "name"
                    else VarTypeParam(var_id.var_type, attr_name)
                )
                for hypo_value in occur_count[var_id][attr_name]:
                    key: tuple[VarTypeParam | VarNameParam, str] = (param, hypo_value)
                    if key in all_hypothesis:
                        continue

                    if count_num_justification(
                        occur_count[var_id][attr_name][hypo_value],
                    ):
                        hypothesis = Hypothesis(  # TODO: encode information about the hypo value to the hypothesis
                            Invariant(
                                text_description=f"{var_id.var_name + '.' + attr_name} is periodicaly set to {hypo_value}",
                                relation=VarPeriodicChangeRelation,
                                params=[param],
                                precondition=None,
                            ),
                            positive_examples=ExampleList({var_group_name}),
                            negative_examples=ExampleList({var_group_name}),
                        )
                        all_hypothesis[key] = hypothesis
                    else:
                        logger.debug(
                            f"Skip the value {hypo_value} of {var_id.var_name}.{attr_name} because it didn't have enough number of occurrences."
                        )

        # 4. Positive and negative examples collection
        for param, hypo_value in all_hypothesis:
            attr_name = param.attr_name
            hypothesis = all_hypothesis[(param, hypo_value)]
            relevant_var_ids = [
                var_id for var_id in occur_count if param.check_var_id_match(var_id)
            ]
            for var_id in relevant_var_ids:
                if hypo_value not in occur_count[var_id][
                    attr_name
                ] or not count_num_justification(
                    occur_count[var_id][attr_name][hypo_value]
                ):
                    # negative example
                    for attr_inst in var_insts[var_id][attr_name][1:]:
                        hypothesis.negative_examples.add_example(
                            Example(
                                {
                                    var_group_name: [
                                        attr_inst.traces[-1]
                                    ]  # TODO: getting the first trace of every attribute instance is a bit arbitrary
                                }
                            )
                        )
                else:
                    # positive example
                    hypothesis.positive_examples.add_example(
                        Example(
                            {
                                var_group_name: [
                                    attr_inst.traces[-1]
                                    for attr_inst in var_insts[var_id][attr_name]
                                ]
                            }
                        )
                    )

        # 4. find preconditions
        valid_invariants = []
        failed_hypothesis = []
        for hypothesis in all_hypothesis.values():
            preconditions = find_precondition(hypothesis, trace)
            if preconditions:
                hypothesis.invariant.precondition = preconditions
                valid_invariants.append(hypothesis.invariant)
            else:
                logger.debug(
                    f"Skip the invariant {hypothesis.invariant.text_description} due to failed precondition inference."
                )
                failed_hypothesis.append(FailedHypothesis(hypothesis))

        return valid_invariants, failed_hypothesis

    @staticmethod
    def evaluate(value_group: list) -> bool:
        """Given a group of values, should return a boolean value
        indicating whether the relation holds or not.

        args:
            value_group: list
                A list of values to evaluate the relation on. The length of the list
                should be equal to the number of variables in the relation.
        """
        return True

    @staticmethod
    def static_check_all(
        trace: Trace, inv: Invariant, check_relation_first: bool
    ) -> CheckerResult:
        """Given a trace and an invariant, should return a boolean value
        indicating whether the invariant holds on the trace.

        args:
            trace: Trace
                A trace to check the invariant on.
            inv: Invariant
                The invariant to check on the trace.
        """

        return CheckerResult(None, inv, True, False)
