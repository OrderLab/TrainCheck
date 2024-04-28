import abc
import logging

import polars as pl

from mldaikon.ml_daikon_trace import Trace


class Invariant:
    def __init__(self, relation, param_selectors: list, precondition: list | None):
        # def __init__(self, relation: Relation, param_selectors: list[Predicate], precondition: Predicate):
        self.relation = relation
        self.param_selectors = param_selectors
        self.precondition = precondition

    def verify(self, trace) -> bool:
        """Given a trace, should return a boolean value indicating
        whether the invariant holds or not.

        args:
            trace: str
                A trace to verify the invariant on.
        """
        relevant_trace = trace.filter(self.precondition)
        groups = relevant_trace.group(self.param_selectors)
        for g in groups:
            if not self.relation.evaluate(g):
                return False
        return True

    def __str__(self) -> str:
        return f"""Relation: {self.relation}\nParam Selectors: {self.param_selectors}\nPrecondition: {self.precondition}"""


class Hypothesis:
    def __init__(
        self,
        invariant: Invariant,
        positive_examples: list[Trace],
        negative_examples: list[Trace],
    ):
        self.invariant = invariant
        self.positive_examples = positive_examples
        self.negative_examples = negative_examples

    @staticmethod
    def refine(trace: Trace, hypothesis_list: list) -> list:
        # TODO: think about refinement for hypothesis (e.g. across multiple traces) / invariants (e.g A > B --> A >= B) needs abstaction for this
        raise NotImplementedError("refine method is not implemented yet.")

        # hypothesis would be a major part of the inference process, as inferring & refining the invariants needs to be based on the positive and negative examples

    def _print_debug(self):
        return f"Hypothesized Invariant: {self.invariant}\n# Positive examples: {len(self.positive_examples)}\n# Negative examples: {len(self.negative_examples)}"


class Relation(abc.ABC):

    @abc.abstractmethod
    def __init__(self):
        # TODO: indentify common attributes of relations and initialize them here
        pass

    def __str__(self):
        return self.__class__.__name__

    @staticmethod
    @abc.abstractmethod
    def infer(trace) -> list[Invariant]:
        """Given a trace, should return a boolean value indicating
        whether the relation holds or not.

        args:
            trace: str
                A trace to infer the relation on.
        """
        pass

    @staticmethod
    def instantiate_invariant(param_selector: list, precondition: list) -> Invariant:
        """Given a list of parameter selectors and a precondition, should return an invariant
        instance.

        args:
            param_selector: list
                A list of parameter selectors to be used in the invariant.
            precondition: list
                A list of preconditions to be used in the invariant.
        """
        raise NotImplementedError(
            "instantiate_invariant method is not implemented yet."
        )

    @staticmethod
    @abc.abstractmethod
    def evaluate(value_group: list) -> bool:
        """Given a group of values, should return a boolean value
        indicating whether the relation holds or not.

        args:
            value_group: list
                A list of values to evaluate the relation on. The length of the list
                should be equal to the number of variables in the relation.
        """
        pass

    @staticmethod
    def find_precondition(hypothesis: Hypothesis) -> list | None:
        """Given a hypothesis, should return a list of preconditions
        that should be satisfied for the invariant to hold.

        The preconditions should be certain properties of the relevant events that
        should be satisfied for the invariant to hold.

        args:
            hypothesis: Hypothesis
                A hypothesis to find preconditions for.
        """

        logger = logging.getLogger(__name__)

        ## 1. Find consistent properties of the positive examples & negative examples
        # merge all events from positive examples
        all_pos_events: pl.DataFrame = pl.concat(
            [trace.events for trace in hypothesis.positive_examples]
        )
        all_neg_events: pl.DataFrame = pl.concat(
            [trace.events for trace in hypothesis.negative_examples]
        )
        # TODO: think about candidate properties for preconditions
        """
        eliminate at least vars used in the relation
        """
        # find consistent properties of the positive examples
        consistent_pos_properties = []
        for col in all_pos_events.columns:
            if all_pos_events.select(col).drop_nulls().n_unique() == 1:
                # get the value of the property
                value = all_pos_events[col].drop_nulls().first()
                consistent_pos_properties.append((col, value))
        # find consistent properties of the negative examples
        consistent_neg_properties = []
        for col in all_neg_events.columns:
            if all_neg_events[col].drop_nulls.n_unique() == 1:
                # get the value of the property
                value = all_neg_events[col].drop_nulls().first()
                consistent_neg_properties.append((col, value))

        # now, find the properties that are consistent in positive examples but not in negative examples (at least not having the same value)
        preconditions = set(consistent_pos_properties) - set(consistent_neg_properties)
        # TODO: how is a 'property' defined? Single value? Groups of values? Single value / Set() only work for && relationships for multiple preconditions.

        if len(preconditions) == 0:
            logger.info("No preconditions found for the hypothesis.")
            return None
        # TODO: implement the disjoint-set based algorithm to find the preconditions
        return list(preconditions)
