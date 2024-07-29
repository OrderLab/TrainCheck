import logging

import numpy as np

from mldaikon.invariant.base_cls import (
    CheckerResult,
    Example,
    ExampleList,
    Hypothesis,
    Invariant,
    Relation,
    VarTypeParam,
)
from mldaikon.invariant.precondition import find_precondition
from mldaikon.trace.trace import Trace


def count_num_justification(
    occurrences_num: dict[str, dict[str, dict[str, int]]], count: int
):
    # TODO: discuss to find a better way to distinguish between changed values
    return count > 1


def calculate_hypo_value(value) -> str:
    if isinstance(value, (int, float)):
        hypo_value = f"{value:.7f}"
    elif isinstance(value, (list)):
        hypo_value = f"{np.linalg.norm(value, ord=1):.7f}"  # l1-norm
    elif isinstance(value, (str, bool)):
        hypo_value = f"{value}"
    else:
        hypo_value = "None"  # TODO: how to represent None,
    return hypo_value


class VarPeriodicChangeRelation(Relation):

    @staticmethod
    def infer(trace: Trace) -> list[Invariant]:
        """Infer Invariants for the VariableChangeRelation."""
        logger = logging.getLogger(__name__)
        ## 1. Pre-scanning: Collecting variable instances and their values from the trace
        # get identifiers of the variables, those variables can be used to query the actual values
        var_insts = trace.get_var_insts()
        if len(var_insts) == 0:
            logger.warning("No variables found in the trace.")
            return []
        ## 2.Counting: count the number of each value of every variable attribute
        # TODO: record the intervals between occurrencess
        # TODO: improve time and memory efficiency
        # occurrences_num: dict[str, dict[str, dict[str, (int, list[float])]]] = {}
        occurrences_num: dict[str, dict[str, dict[str, int]]] = {}
        for var_id, attrs in var_insts.items():
            for attr_name, attr_insts in attrs.items():
                for attr_inst in attr_insts:
                    hypo_value = calculate_hypo_value(attr_inst.value)
                    var_key = var_id.var_name
                    if var_key not in occurrences_num:
                        occurrences_num[var_key] = {}
                    if attr_name not in occurrences_num[var_key]:
                        occurrences_num[var_key][attr_name] = {}
                    if hypo_value not in occurrences_num[var_key][attr_name]:
                        occurrences_num[var_key][attr_name][hypo_value] = 1
                    else:
                        occurrences_num[var_key][attr_name][hypo_value] += 1

        # 3. Hypothesis generation
        hypothesis: dict[str, dict[str, dict[str, Hypothesis]]] = {}
        for var_id, attrs in var_insts.items():
            for attr_name, attr_insts in attrs.items():
                for attr_inst in attr_insts:
                    hypo_value = calculate_hypo_value(attr_inst.value)
                    var_key = var_id.var_type
                    group_names = "var"
                    example = Example()
                    example.add_group(group_names, attr_inst.traces)
                    if var_key not in hypothesis:
                        hypothesis[var_key] = {}
                    if attr_name not in hypothesis[var_key]:
                        hypothesis[var_key][attr_name] = {}
                    if count_num_justification(
                        occurrences_num,
                        occurrences_num[var_id.var_name][attr_name][hypo_value],
                    ):
                        if hypo_value not in hypothesis[var_key][attr_name]:
                            hypo = Hypothesis(
                                Invariant(
                                    relation=VarPeriodicChangeRelation,
                                    params=[VarTypeParam(var_key, attr_name)],
                                    precondition=None,
                                ),
                                positive_examples=ExampleList({group_names}),
                                negative_examples=ExampleList({group_names}),
                            )
                            hypothesis[var_key][attr_name][hypo_value] = hypo

                        hypothesis[var_key][attr_name][
                            hypo_value
                        ].positive_examples.add_example(
                            example
                        )  # If a value occurs more than once, mark it as positive
                    # else:
                    #     # TODO: how to add negative examples so that preconditions inference works
                    #     hypothesis[var_key][attr_name][
                    #         hypo_value
                    #     ].negative_examples.add_example(
                    #         example
                    #     )  # If a value occurs only once, mark it as negative

        # 4. find preconditions
        for var_name in hypothesis:
            for attr_name in hypothesis[var_name]:
                for hypo_value in hypothesis[var_name][attr_name]:
                    hypo = hypothesis[var_name][attr_name][hypo_value]
                    hypo.invariant.precondition = find_precondition(hypo)
                    hypo.invariant.text_description = f"{var_name + '.' + attr_name} is periodicaly set to {hypo_value}"

        return list(
            [
                hypothesis[var_name][attr_name][hypo_value].invariant
                for var_name in hypothesis
                for attr_name in hypothesis[var_name]
                for hypo_value in hypothesis[var_name][attr_name]
                if hypothesis[var_name][attr_name][hypo_value].invariant.precondition
                is not None
            ]
        )

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

        return CheckerResult(None, inv, True)
