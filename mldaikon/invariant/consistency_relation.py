import logging

import polars as pl

from mldaikon.invariant.base_cls import Hypothesis, Invariant, Relation
from mldaikon.ml_daikon_trace import Trace


class ConsistencyRelation(Relation):
    def __init__(self, parent_func_name: str, child_func_name: str):
        self.parent_func_name = parent_func_name
        self.child_func_name = child_func_name

    @staticmethod
    def infer(trace: Trace) -> list[Invariant]:
        """Infer Invariants for the ConsistencyRelation."""
        # find all variables in the trace, we need time stamps, as we only want to look forward in time

        # for a specific variable's value, check for consistent values in other variables

        # if found a consistent value, start a hypothesis, start to collect the consistent values

        # invariants

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
