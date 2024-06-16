import abc
from typing import Iterator

from mldaikon.trace.trace import Trace


class Invariant:
    def __init__(
        self,
        relation,
        param_selectors: list,
        precondition: dict[str, list] | None = None,
        text_description: str | None = None,
    ):
        # def __init__(self, relation: Relation, param_selectors: list[Predicate], precondition: Predicate):
        self.relation = relation
        self.param_selectors = param_selectors  ## Param selector
        self.precondition = precondition  # stateful preconditions
        self.text_description = text_description

    def get_params(self, trace: Trace) -> list:
        """Given a trace, should return the values of the parameters
        that the invariant should be evaluated on.

        args:
            trace: str
                A trace to get the parameter values from.
        """
        raise NotImplementedError("get_params method is not implemented yet.")

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
        return f"""Relation: {self.relation}\nParam Selectors: {self.param_selectors}\nPrecondition: {self.precondition}\nText Description: {self.text_description}"""


class Example:
    def __init__(self, trace_groups: dict[str, list[dict]] | None = None):
        self.trace_groups: dict[str, list[dict]] = trace_groups or {}

    def add_group(self, group_name: str, trace: list):
        assert group_name not in self.trace_groups, f"Group {group_name} already exists"
        self.trace_groups[group_name] = trace

    def get_group(self, group_name: str) -> list[dict]:
        return self.trace_groups[group_name]

    def __iter__(self):
        return iter(self.trace_groups)

    def __str__(self):
        return f"Example with Groups: {self.trace_groups.keys()}"

    def __repr__(self):
        return f"Example with Groups: {self.trace_groups.keys()}"


class ExampleList:
    def __init__(self, group_names: set[str]):
        self.group_names = group_names
        self.examples: list[Example] = []

    def add_example(self, example: Example):
        assert (
            set(example.trace_groups.keys()) == self.group_names
        ), f"Example groups do not match the expected group names"
        self.examples.append(example)

    def get_group_from_examples(self, group_name: str) -> list[list[dict]]:
        return [example.get_group(group_name) for example in self.examples]

    def get_group_names(self) -> set[str]:
        return self.group_names

    def __len__(self):
        return len(self.examples)


class Hypothesis:
    def __init__(
        self,
        invariant: Invariant,
        positive_examples: ExampleList,
        negative_examples: ExampleList,
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
