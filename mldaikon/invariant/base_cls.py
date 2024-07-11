from __future__ import annotations

import abc
import json
from dataclasses import dataclass
from enum import Enum
from typing import Hashable, Iterable, Optional, Type

from mldaikon.trace.trace import Trace
from mldaikon.trace.types import (
    FuncCallEvent,
    FuncCallExceptionEvent,
    HighLevelEvent,
    VarChangeEvent,
)


@dataclass
class Param:
    # param_type: str  # ["func", "var_type", "var_name"]
    def to_dict(self):
        self_dict = {
            "param_type": self.__class__.__name__,
        }
        self_dict.update(self.__dict__)
        return self_dict

    @staticmethod
    def from_dict(param_dict: dict) -> Param:
        for param_type in Param.__subclasses__():
            if param_type.__name__ == param_dict["param_type"]:
                args = {k: v for k, v in param_dict.items() if k != "param_type"}
                return param_type(**args)
        raise ValueError(f"Unknown param type: {param_dict['param_type']}")

    def check_trace_line_match(self, trace_line: dict) -> bool:
        "Check if the event contains the required information for the param."
        raise NotImplementedError(
            "check_trace_line_match method is not implemented yet."
        )

    def check_event_match(self, event: HighLevelEvent) -> bool:
        "Check if the high level event contains the required information for the param."
        raise NotImplementedError("check_event_match method is not implemented yet.")


@dataclass
class APIParam(Param):
    def __init__(self, api_full_name: str):
        self.api_full_name = api_full_name

    def check_trace_line_match(self, trace_line: dict) -> bool:
        if "function" not in trace_line:
            return False
        return trace_line["function"] == self.api_full_name

    def check_event_match(self, event: HighLevelEvent) -> bool:
        if not isinstance(event, (FuncCallEvent, FuncCallExceptionEvent)):
            return False
        return event.func_name == self.api_full_name


@dataclass
class VarTypeParam(Param):
    def __init__(self, var_type: str, attr_name: str):
        self.var_type = var_type
        self.attr_name = attr_name

    def check_trace_line_match(self, trace_line: dict) -> bool:
        if "var_type" not in trace_line:
            return False
        return trace_line["var_type"] == self.var_type

    def check_event_match(self, event: HighLevelEvent) -> bool:
        if not isinstance(event, VarChangeEvent):
            return False
        return (
            event.var_id.var_type == self.var_type and event.attr_name == self.attr_name
        )


@dataclass
class VarNameParam(Param):
    def __init__(self, var_name: str, attr_name: str):
        self.var_name = var_name
        self.attr_name = attr_name

    def check_trace_line_match(self, trace_line: dict) -> bool:
        if "var_type" not in trace_line:
            return False
        return trace_line["var_name"] == self.var_name

    def check_event_match(self, event: HighLevelEvent) -> bool:
        if not isinstance(event, VarChangeEvent):
            return False
        return (
            event.var_id.var_name == self.var_name and event.attr_name == self.attr_name
        )


class PreconditionClauseType(Enum):
    CONSTANT = "constant"
    CONSISTENT = "consistent"
    UNEQUAL = "unequal"


PT = PreconditionClauseType


class PreconditionClause:
    def __init__(self, prop_name: str, prop_dtype: type, _type: PT, values: set | None):
        assert _type in [
            PT.CONSISTENT,
            PT.CONSTANT,
            PT.UNEQUAL,
        ], f"Invalid Precondition type {_type}"

        if _type in [PT.CONSISTENT, PT.CONSTANT]:
            assert (
                values is not None and len(values) > 0
            ), "Values should not be empty for CONSTANT or CONSISTENT type"

        self.prop_name = prop_name
        self.prop_dtype = prop_dtype
        self.type = _type
        self.values = values if isinstance(values, set) else {values}

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return f"Prop: {self.prop_name}, Type: {self.type}, Values: {self.values}"

    def to_dict(self) -> dict:
        clause_dict: dict[str, str | list] = {
            "type": self.type.value,
            "prop_name": self.prop_name,
            "prop_dtype": self.prop_dtype.__name__,
        }
        if self.type in [PT.CONSTANT, PT.CONSISTENT]:
            clause_dict["values"] = list(self.values)
        return clause_dict

    @staticmethod
    def from_dict(clause_dict: dict) -> PreconditionClause:
        prop_name = clause_dict["prop_name"]
        _type = PT(clause_dict["type"])
        prop_dtype = eval(clause_dict["prop_dtype"])

        values = None
        if _type in [PT.CONSTANT, PT.CONSISTENT]:
            assert "values" in clause_dict, "Values not found in the clause"
            assert isinstance(clause_dict["values"], list), "Values should be a list"
            values = set(clause_dict["values"])
        return PreconditionClause(prop_name, prop_dtype, _type, values)

    def __eq__(self, other):
        if not isinstance(other, PreconditionClause):
            return False

        if self.type == PT.CONSISTENT and other.type == PT.CONSISTENT:
            return (
                self.prop_name == other.prop_name
                and self.prop_dtype == other.prop_dtype
                and self.type == other.type
            )

        return (
            self.prop_name == other.prop_name
            and self.prop_dtype == other.prop_dtype
            and self.type == other.type
            and self.values == other.values
        )

    def __hash__(self):
        if self.type == PT.CONSISTENT:
            return hash((self.prop_name, self.prop_dtype, self.type))
        return hash((self.prop_name, self.prop_dtype, self.type, tuple(self.values)))

    def verify(self, example: list) -> bool:
        assert isinstance(example, list)
        assert len(example) > 0

        prop_name = self.prop_name
        prop_values_seen = set()
        for i in range(len(example)):
            if prop_name not in example[i]:
                return False

            if not isinstance(example[i][prop_name], Hashable):
                # print(
                #     f"ERROR: Property {prop_name} is not hashable, skipping this property"
                # )
                return False

            prop_values_seen.add(example[i][prop_name])

        if self.type == PT.CONSTANT:
            if len(prop_values_seen) == 1 and tuple(prop_values_seen)[0] in self.values:
                return True
            return False

        if self.type == PT.CONSISTENT:
            if len(prop_values_seen) == 1:
                return True
            return False

        if self.type == PT.UNEQUAL:
            if len(prop_values_seen) == len(example):
                return True
            return False

        raise ValueError(f"Invalid Precondition type {self.type}")


class Precondition:
    """A class to represent a precondition for a hypothesis. A precondition is a set of `PreconditionClause` objects that should hold for the hypothesis to be valid.
    Currently the `Precondition` object is a conjunction of the `PreconditionClause` objects.
    """

    def __init__(self, clauses: Iterable[PreconditionClause]):
        self.clauses = clauses

    def verify(self, example: list) -> bool:
        and_result = True
        for clause in self.clauses:
            and_result = and_result and clause.verify(example)
            if not and_result:
                return False
        return True

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        output = "** Start of Precondition **\n"
        for clause in self.clauses:
            output += str(clause) + "\n"
        output += "** End of Preconditions **"
        return output

    def implies(self, other) -> bool:
        """When self is True, other should also be True."""

        ## all the clauses in other should be in self
        for clause in other.clauses:
            if clause not in self.clauses:
                return False

        return True

    def to_dict(self) -> dict:
        return {"clauses": [clause.to_dict() for clause in self.clauses]}


class UnconditionalPrecondition(Precondition):
    def __init__(self):
        super().__init__([])

    def verify(self, example: list) -> bool:
        return True

    def __repr__(self) -> str:
        return "Unconditional Precondition"

    def __str__(self) -> str:
        return "Unconditional Precondition"

    def implies(self, other) -> bool:
        # Unconditional Precondition cannot imply any other preconditions as it is always True
        return False

    def to_dict(self) -> dict:
        return {"clauses": "Unconditional"}


class GroupedPreconditions:
    def __init__(self, grouped_preconditions: dict[str, list[Precondition]]):
        self.grouped_preconditions = grouped_preconditions

    def verify(self, example: list, group_name: str) -> bool:
        assert group_name in self.grouped_preconditions, f"Group {group_name} not found"
        for precondition in self.grouped_preconditions[group_name]:
            if precondition.verify(example):
                return True
        return False

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        output = "====================== Start of Grouped Precondition ======================\n"
        for group_name, preconditions in self.grouped_preconditions.items():
            output += f"Group: {group_name}\n"
            for precondition in preconditions:
                output += str(precondition) + "\n"
        output += (
            "====================== End of Grouped Precondition ======================"
        )
        return output

    def to_dict(self) -> dict:
        return {
            group_name: [precond.to_dict() for precond in preconditions]
            for group_name, preconditions in self.grouped_preconditions.items()
        }

    def get_group(self, group_name: str) -> list[Precondition]:
        assert group_name in self.grouped_preconditions, f"Group {group_name} not found"
        return self.grouped_preconditions[group_name]

    def get_group_names(self) -> set[str]:
        return set(self.grouped_preconditions.keys())

    @staticmethod
    def from_dict(precondition_dict: dict) -> GroupedPreconditions:
        grouped_preconditions: dict[str, list[Precondition]] = {}
        for group_name, preconditions in precondition_dict.items():
            grouped_preconditions[group_name] = []
            if (
                len(preconditions) > 0
                and preconditions[0]["clauses"] == "Unconditional"
            ):
                assert (
                    len(preconditions) == 1
                ), "Unconditional precondition should be the only precondition"
                grouped_preconditions[group_name].append(UnconditionalPrecondition())
                continue

            for precondition in preconditions:
                clauses = []
                for clause_dict in precondition["clauses"]:
                    clauses.append(
                        PreconditionClause.from_dict(clause_dict=clause_dict)
                    )
                grouped_preconditions[group_name].append(Precondition(clauses))
        return GroupedPreconditions(grouped_preconditions)

    def is_group_unconditional(self, group_name: str) -> bool:
        assert group_name in self.grouped_preconditions, f"Group {group_name} not found"
        is_all_unconditional = all(
            [
                isinstance(precond, UnconditionalPrecondition)
                for precond in self.grouped_preconditions[group_name]
            ]
        )
        if is_all_unconditional:
            assert (
                len(self.grouped_preconditions[group_name]) == 1
            ), "Multiple unconditional preconditions found"
        return is_all_unconditional


class Invariant:
    def __init__(
        self,
        relation: Type[Relation],
        params: list[Param],
        precondition: GroupedPreconditions | None,
        text_description: str | None = None,
    ):
        self.relation = relation
        self.params = params  ## params to be used in the check
        self.precondition = precondition
        self.text_description = text_description

    def __str__(self) -> str:
        return f"""Relation: {self.relation}\nParam Selectors: {self.params}\nPrecondition: {self.precondition}\nText Description: {self.text_description}"""

    def to_dict(self) -> dict:
        assert (
            self.precondition is not None
        ), f"Invariant precondition is not set, check the infer function of {self.relation.__name__}"

        return {
            "text_description": self.text_description,
            "relation": self.relation.__name__,
            "params": [param.to_dict() for param in self.params],
            "precondition": self.precondition.to_dict(),
        }

    @staticmethod
    def from_dict(invariant_dict: dict) -> Invariant:
        relation = Relation.from_name(invariant_dict["relation"])
        text_description = invariant_dict["text_description"]
        params = [
            Param.from_dict(param_dict) for param_dict in invariant_dict["params"]
        ]
        precondition = GroupedPreconditions.from_dict(invariant_dict["precondition"])
        return Invariant(relation, params, precondition, text_description)

    def check(self, trace: Trace) -> CheckerResult:
        assert (
            self.precondition is not None
        ), "Invariant precondition is None. It should at least be 'Unconditional' or an empty list. Please check the invariant file and the inference process."
        return self.relation.static_check_all(trace, self)


class CheckerResult:
    def __init__(
        self, trace: Optional[list[dict]], invariant: Invariant, check_passed: bool
    ):
        if trace is None:
            assert check_passed, "Check passed should be True for None trace"
        else:
            assert len(trace) > 0, "Trace should not be empty"
        self.trace = trace
        self.invariant = invariant
        self.check_passed = check_passed

    def __str__(self) -> str:
        return f"Trace: {self.trace}\nInvariant: {self.invariant}\nResult: {self.check_passed}"

    def get_min_time(self):
        if not hasattr(self, "min_time"):
            self.min_time = min([x["time"] for x in self.trace])
        return self.min_time

    def get_max_time(self):
        if not hasattr(self, "max_time"):
            self.max_time = max([x["time"] for x in self.trace])
        return self.max_time

    def calc_and_set_time_precentage(self, min_time, max_time):
        if self.check_passed:
            # don't do anything if the check passed
            return 1.0

        detection_time = self.get_max_time()
        assert (
            min_time <= detection_time <= max_time
        ), f"Detection time {detection_time} not in range [{min_time}, {max_time}]"
        self.time_precentage = (detection_time - min_time) / (max_time - min_time)
        return self.time_precentage

    def to_dict(self):
        result_dict = {
            "invariant": self.invariant.to_dict(),
            "check_passed": self.check_passed,
        }

        if not self.check_passed:
            assert hasattr(
                self, "time_precentage"
            ), "Time percentage not set for failed check, please call calc_and_set_time_precentage before converting to dict"
            result_dict.update(
                {
                    "detection_time": self.get_max_time(),  # the time when the invariant was detected, using max_time as the invariant cannot be checked before the
                    "detection_time_percentage": self.time_precentage,
                    "trace": self.trace,
                }
            )

        return result_dict


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
    def from_name(relation_name: str) -> Type[Relation]:
        """Given a relation name, should return the relation class.

        args:
            relation_name: str
                The name of the relation.
        """
        for type_relation in Relation.__subclasses__():
            if type_relation.__name__ == relation_name:
                return type_relation

        raise ValueError(f"Relation {relation_name} not found")

    @staticmethod
    @abc.abstractmethod
    def static_check_all(trace: Trace, inv: Invariant) -> CheckerResult:
        """Given a trace and an invariant, should return a boolean value
        indicating whether the invariant holds on the trace.

        args:
            trace: Trace
                A trace to check the invariant on.
            inv: Invariant
                The invariant to check on the trace.
        """
        pass


def read_inv_file(file_path: str | list[str]):
    if isinstance(file_path, str):
        file_path = [file_path]
    invs = []
    for file in file_path:
        with open(file, "r") as f:
            for line in f:
                inv_dict = json.loads(line)
                inv = Invariant.from_dict(inv_dict)
                invs.append(inv)
    return invs
