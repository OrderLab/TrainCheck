import abc
import logging
from enum import Enum
from typing import Hashable

from tqdm import tqdm

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


class PreconditionType(Enum):
    CONSTANT = "constant"
    CONSISTENT = "consistent"
    UNEQUAL = "unequal"


PT = PreconditionType


class Precondition:
    def __init__(self, prop_name: str, _type: PT, values: set | type):
        self.prop_name = prop_name
        if _type not in [PT.CONSISTENT, PT.CONSTANT, PT.UNEQUAL]:
            raise ValueError(f"Invalid type {_type}")
        self.type = _type  # either "constant" or "consistent"
        self.values = values if isinstance(values, set) else {values}

    def __str__(self) -> str:
        return f"Prop: {self.prop_name}, Type: {self.type}, Values: {len(self.values)}"

    def verify(self, example: list) -> bool:
        assert isinstance(example, list)
        assert len(example) > 0

        prop_name = self.prop_name
        for i in range(len(example)):
            if prop_name not in example[i]:
                return False

        if self.type in [PT.CONSTANT, PT.CONSISTENT]:
            if self.type == PT.CONSTANT:
                if example[0][prop_name] not in self.values:
                    return False
            for i in range(1, len(example)):
                if example[i][prop_name] != example[0][prop_name]:
                    return False
        if self.type == PT.UNEQUAL:
            all_values = set()
            for i in range(len(example)):
                all_values.add(example[i][prop_name])
            if len(all_values) != len(example):
                return False
        return True

    def try_relax(self) -> bool:
        if self.type == PT.CONSISTENT:
            self.type = PT.CONSTANT
            return True
        return False  # cannot relax further


def pprint_preconds(preconditions: list):
    for precond in preconditions:
        print("values", preconditions[precond].values)
        print("type", preconditions[precond].type)
        print("target", preconditions[precond].prop_name)
        print("==============================")


def find_precondition(hypothesis: Hypothesis) -> list:
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

    def find_conditions(example: list, key_to_skip: str = "param_value"):
        """A list of traces to find common properties from. The property should hold locally within the example."""

        conds = {
            PT.CONSTANT: {},
            PT.UNEQUAL: set(),
        }
        # find properties that have only one value in the example
        for prop in example[0]:

            # skip tensor values as preconditions ## TODO: revisit this decision, we might not have data-dependent control-flow because of this.
            if key_to_skip in prop:
                continue

            is_constant = True

            if not isinstance(example[0][prop], Hashable):
                print(
                    f"Warning: Non-hashable property found in the example, skipping this property {prop} ({type(example[0][prop])})"
                )
                continue

            all_values = {example[0][prop]}  # TODO: support lists as well
            for i in range(1, len(example)):
                if prop not in example[i]:
                    # TODO: we might not want to skip this, as if this prop is a local attribute of a specific variable type, it might not be other traces
                    logger.error(
                        f"Property {prop} not found in example {example[i]}, precondition inference might not be correct if this prop is not a local attribute of the variable"
                    )
                    continue
                if example[i][prop] != example[0][prop]:
                    is_constant = False
                all_values.add(example[i][prop])
            if len(all_values) == 1 and is_constant:
                conds[PT.CONSTANT][prop] = example[0][prop]
            elif len(all_values) == len(example):
                conds[PT.UNEQUAL].add(prop)

        return conds

    for example in tqdm(hypothesis.positive_examples):
        if len(example) == 0:
            # raise ValueError("Empty example found in positive examples")
            print("Warning: empty examples found in positive examples")
            continue

        conds = find_conditions(example)

        if len(conds[PT.CONSTANT]) == 0 and len(conds[PT.UNEQUAL]) == 0:
            print("example: ", example)
            # stop
            return []

        positive_properties.append(conds)

    const_precond_targets = set(positive_properties[0][PT.CONSTANT].keys())
    print(
        f"# Initial Precondition Targets for {PT.CONSTANT} or {PT.CONSISTENT}: {const_precond_targets}"
    )

    const_precond_target_values = {key: set() for key in const_precond_targets}

    for props in positive_properties:
        const_conds = props[PT.CONSTANT]
        const_precond_targets = const_precond_targets.intersection(const_conds.keys())
        for key in const_conds:
            if key in const_precond_targets:
                const_precond_target_values[key].add(const_conds[key])

    preconditions = {
        key: (
            Precondition(
                key, PT.CONSTANT, const_precond_target_values[key]
            )  # FIXME: disabling the consistent preconditions for now as it is less strict than the constant preconditions and we don't have a good way to refine it
            if len(const_precond_target_values[key]) == 1
            else Precondition(key, PT.CONSISTENT, const_precond_target_values[key])
        )
        for key in const_precond_targets
    }

    unequal_precond_targets = set(positive_properties[0][PT.UNEQUAL])
    print(f"# Initial Precondition Targets for {PT.UNEQUAL}: {unequal_precond_targets}")
    for props in positive_properties:
        unequal_precond_targets = unequal_precond_targets.intersection(
            props[PT.UNEQUAL]
        )

    for key in unequal_precond_targets:
        if key in preconditions:
            # both consistent and unequal preconditions cannot co-exist
            print(
                f"Warning: inconsistent preconditions found {key} (UNEQUAL) and {preconditions[key]}"
            )
            # delete the existing precondition
            del preconditions[key]
        else:
            preconditions[key] = Precondition(key, PT.UNEQUAL, None)

    print(f"# Initial Precondition: {len(preconditions)}")
    pprint_preconds(preconditions)

    """
    1. Only one value (assumes to be the prop == constant precondition)
    2. Multiple Values (first assumes to be prop == prop precondition, if it do not hold, relax to prop in [const1, const2, ...] but constant in one example precondition)
    """

    # TODO: implement precondition refinement logic here

    """ Precondition Refinement Logic:

    # Background: The preconditions inferred from the positive examples might be over-constrained and contains a lot of noises. 
        The goal here is get rid of such noises, such as (is_cuda, constant, [True]). Note: The goal here is not to refine the preconditions that might be inaccurate, but to get rid of the ones that are not necessary.
    """
    pre_cond_num_false_in_neg = {}
    for key in preconditions:
        pre_cond_num_false_in_neg[key] = 0

    for neg_example in tqdm(hypothesis.negative_examples, desc="Refining Precondition"):
        whether_precondition_holds = True
        for key in preconditions:
            whether_key_holds = preconditions[key].verify(neg_example)
            whether_precondition_holds = (
                whether_precondition_holds and whether_key_holds
            )
            if not preconditions[key].verify(neg_example):
                pre_cond_num_false_in_neg[key] += 1
        if whether_precondition_holds:
            # print preconditions and the negative example
            print("Negative example satisfies the preconditions")
            import pprint

            pprint.pprint(neg_example)

            for key in preconditions:
                # print the precondition itself (target, type, values)
                print(preconditions[key], preconditions[key].verify(neg_example))
                print("values", preconditions[key].values)
                print("type", preconditions[key].type)
                print("target", preconditions[key].prop_name)
                print("==============================")
            raise ValueError("Negative example satisfies the preconditions")

    precond_keys_to_delete = []
    for key in preconditions:
        if pre_cond_num_false_in_neg[key] == 0:
            print(
                f"Precondition {key} is not necessary as it is not violated in any of the negative examples"
            )
            precond_keys_to_delete.append(key)

    for key in precond_keys_to_delete:
        del preconditions[key]

    return preconditions
