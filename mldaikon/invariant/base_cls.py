import abc
import logging

import polars as pl

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


class Precondition:
    def __init__(self, prop_name: str, _type: str, values: list | type):
        self.prop_name = prop_name
        if _type not in ["constant", "consistent"]:
            raise ValueError(f"Invalid type {_type}")
        self.type = _type  # either "constant" or "consistent"
        self.values = values if isinstance(values, list) else [values]

    def verify(self, example) -> bool:
        if isinstance(example, list):
            example = pl.DataFrame(example)
        assert isinstance(
            example, pl.DataFrame
        ), f"Expected example to be a DataFrame, got {type(example)}"

        # prop_key = prop_prefix + self.prop_name if self.prop_name != "param_value" else value_prefix
        prop_key = self.prop_name
        if prop_key not in example.columns:
            return False
        prop_values = example[prop_key].drop_nulls().unique().to_list()
        if self.type == "constant":
            return len(prop_values) == 1 and prop_values[0] in self.values
        if self.type == "consistent":
            return len(prop_values) == 1

    def try_relax(self) -> bool:
        if self.type == "consistent":
            self.type = "constant"
            return True
        return False  # cannot relax further


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
    positive_properties = []

    def find_conditions(example: list, key_to_skip: str = "value"):
        """A list of traces to find common properties from. The property should hold locally within the example."""
        try:
            example_df = pl.DataFrame(example)
        except:
            import pprint

            pprint.pprint(example)
            raise
        const_conds = {}
        # find properties that have only one value in the example
        for col in example_df.columns:
            if key_to_skip is not None and key_to_skip in col:
                continue

            # let's also skip anything with .old
            if ".old" in col:
                continue

            try:
                values = example_df.get_column(col).drop_nulls().unique().to_list()
            except:
                # .unique() might fail due to column having dtype 'list[null]' or something similar, let's just continue
                continue
            if len(values) == 1:
                # get the value of the property
                value = values[0]
                const_conds[col] = value
        return const_conds

    for example in hypothesis.positive_examples:
        conds = find_conditions(example)
        # print(f"found #conds: {len(conds)}")

        found = False
        for cond_name in conds:
            if "tensor_model_parallel" in cond_name:
                found = True
                break
        if not found:
            import pprint

            print("example no tensor_model_parallel:")
            pprint.pprint(example)
            return []
        if len(conds) == 0:
            print("example: ", example)
            # stop
            return []

        positive_properties.append(conds)

    # exclude those also hold in the negative examples
    # for each negative example, we verify the conds in the positive examples

    # find the common properties
    precondition_targets = set(positive_properties[0].keys())
    precondition_target_values = {key: [] for key in precondition_targets}

    for pos_props in positive_properties:
        precondition_targets = precondition_targets.intersection(pos_props.keys())
        for key in pos_props:
            if key in precondition_targets:
                # precondition_target_values[key].append(pos_props[key])
                if pos_props[key] not in precondition_target_values[key]:
                    precondition_target_values[key].append(pos_props[key])

    preconditions = {
        key: (
            Precondition(key, "constant", precondition_target_values[key])
            if len(precondition_target_values[key]) == 1
            else Precondition(key, "consistent", precondition_target_values[key])
        )
        for key in precondition_targets
    }

    print(f"# Initial Precondition: {len(preconditions)}")

    """
    1. Only one value (assumes to be the prop == constant precondition)
    2. Multiple Values (first assumes to be prop == prop precondition, if it do not hold, relax to prop in [const1, const2, ...] but constant in one example precondition)
    """

    # TODO: implement precondition refinement logic here

    return preconditions
