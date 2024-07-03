from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import TYPE_CHECKING

from mldaikon.trace.trace import Trace

if TYPE_CHECKING:
    from mldaikon.invariant.precondition import (
        GroupedPreconditions,
        UnconditionalPrecondition,
    )


@dataclass
class Param:
    # param_type: str  # ["func", "var_type", "var_name"]
    pass

    def to_dict(self):
        return self.__dict__


@dataclass
class APIParam(Param):
    def __init__(self, api_full_name: str):
        self.param_type = "api_param"
        self.api_full_name = api_full_name


@dataclass
class VarTypeParam(Param):
    def __init__(self, var_type: str, attr_name: str):
        self.param_type = "var_type"
        self.var_type = var_type
        self.attr_name = attr_name


@dataclass
class VarNameParam(Param):
    param_type = "var_name"

    def __init__(self, var_name: str, attr_name: str):
        self.param_type = "var_name"
        self.var_name = var_name
        self.attr_name = attr_name


class Invariant:
    def __init__(
        self,
        relation: Relation,
        params: list[Param],
        precondition: "GroupedPreconditions" | "UnconditionalPrecondition",
        text_description: str | None = None,
    ):
        self.relation = relation
        self.params = params  ## params to be used in the check
        self.precondition = precondition
        self.text_description = text_description

    def __str__(self) -> str:
        return f"""Relation: {self.relation}\nParam Selectors: {self.params}\nPrecondition: {self.precondition}\nText Description: {self.text_description}"""

    def to_dict(self) -> dict:
        return {
            "text_description": self.text_description,
            "relation": self.relation.get_name(),
            "params": [param.to_dict() for param in self.params],
            "precondition": self.precondition.to_dict(),
        }


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
        ), "Example groups do not match the expected group names"
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

    def get_name(self):
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
