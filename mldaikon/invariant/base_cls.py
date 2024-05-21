import abc

from mldaikon.ml_daikon_trace import Trace


class Invariant:
    def __init__(self, relation, param_selectors: list, precondition: list | None):
        # def __init__(self, relation: Relation, param_selectors: list[Predicate], precondition: Predicate):
        self.relation = relation
        self.param_selectors = param_selectors  ## Param selector
        self.precondition = precondition  # stateful preconditions

    def param_selectors(self, trace: Trace) -> list:
        """Given a trace, should return the values of the parameters
        that the invariant should be evaluated on.

        args:
            trace: str
                A trace to get the parameter values from.
        """
        raise NotImplementedError("param_selectors method is not implemented yet.")

    def verify(self, trace) -> bool:
        """Given a trace, should return a boolean value indicating
        whether the invariant holds or not.

        args:
            trace: str
                A trace to verify the invariant on.
        """
        # relevant_trace = trace.filter(self.precondition)
        # the pre-condition should be incorporated into the param_selectors
        groups = trace.group(self.param_selectors)
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
