import logging
from enum import Enum
from typing import Hashable

from tqdm import tqdm

from mldaikon.invariant.base_cls import Hypothesis

logger = logging.getLogger("Precondition")


class PreconditionClauseType(Enum):
    CONSTANT = "constant"
    CONSISTENT = "consistent"
    UNEQUAL = "unequal"


PT = PreconditionClauseType


class PreconditionClause:
    def __init__(self, prop_name: str, _type: PT, values: set | type):
        assert _type in [
            PT.CONSISTENT,
            PT.CONSTANT,
            PT.UNEQUAL,
        ], f"Invalid Precondition type {_type}"
        self.prop_name = prop_name
        self.type = _type
        self.values = values if isinstance(values, set) else {values}

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return f"Prop: {self.prop_name}, Type: {self.type}, Values: {self.values}"

    def verify(self, example: list) -> bool:
        assert isinstance(example, list)
        assert len(example) > 0

        prop_name = self.prop_name
        prop_values_seen = set()
        for i in range(len(example)):
            if prop_name not in example[i]:
                return False
            prop_values_seen.add(example[i][prop_name])

        if self.type == PT.CONSTANT:
            if len(prop_values_seen) == 1 and prop_values_seen == self.values:
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

    def __init__(self, clauses: list[PreconditionClause]):
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
        output = "======================\n"
        for clause in self.clauses:
            output += str(clause) + "\n"
        output += "======================"
        return output


def pprint_preconds(clauses: dict):
    for clause in clauses:
        print("==============================")
        print("values", clauses[clause].values)
        print("type", clauses[clause].type)
        print("target", clauses[clause].prop_name)
    print("==============================")


def is_statistical_significant(positive_examples: list) -> bool:
    return len(positive_examples) > 100


def _find_local_clause_targets(example: list, key_to_skip: str = "param_value") -> dict:
    """A list of traces to find common properties from. The property should hold locally within the example."""

    clause_targets = {
        PT.CONSTANT: {},
        PT.UNEQUAL: set(),
    }
    # find properties that have only one value in the example
    for prop in example[0]:
        if prop in ['process_id", "thread_id', "time", "type"]:
            # skip meta_info about each event
            continue

        if key_to_skip in prop:
            # skip tensor values as preconditions ## TODO: revisit this decision, we might not have data-dependent control-flow because of this.
            continue

        if not isinstance(example[0][prop], Hashable):
            # we cannot use non-hashable properties as preconditions, due to limitations in the current implementation (set cannot contain non-hashable objects)
            print(
                f"Warning: Non-hashable property found in the example, skipping this property {prop} ({type(example[0][prop])})"
            )
            continue

        prop_values_seen = {example[0][prop]}
        for i in range(1, len(example)):
            if prop not in example[i]:
                # TODO: we might not want to skip this, as if this prop is a local attribute of a specific variable type, it might not be other traces
                logger.error(
                    f"Property {prop} not found in example {example[i]}, precondition inference might not be correct if this prop is not a local attribute of the variable"
                )
                continue
            prop_values_seen.add(example[i][prop])
        if len(prop_values_seen) == 1:
            clause_targets[PT.CONSTANT][prop] = tuple(prop_values_seen)[0]
        elif len(prop_values_seen) == len(example):
            clause_targets[PT.UNEQUAL].add(prop)
    return clause_targets

def verify_precondition_safety(precondition: Precondition, negative_examples: list) -> bool:
    """Given a precondition and a list of negative examples, should return True if the precondition is safe to use, False otherwise.

    args:
        precondition: Precondition
            A precondition to verify against the negative examples.
        negative_examples: list
            A list of negative examples to verify the precondition against.
    """
    for example in negative_examples:
        if precondition.verify(example):
            return False
    return True

def find_precondition(hypothesis: Hypothesis) -> list[Precondition]:
    """Given a hypothesis, should return a list of `Precondition` objects that invariants should hold if one of the `Precondition` is satisfied.

    args:
        hypothesis: Hypothesis
            A hypothesis to find preconditions for.

    This function will perform inference on the positive examples to find special properties that consistently show up in the positive examples.
    Then, the found properties will be scanned in the negative examples to prune out unnecessary properties that also hold for the negative examples.
    The pruning process is relaxing the precondition by just removing noises. Thus, if at anytime the precondition is verified in the negative examples, the function will abort.

    To implement the invariant split OP. We need to determine how this verification / pruning process should be done, because now all the `Precondition` objects have to be violated in the negative examples.
    """

    logger = logging.getLogger(__name__)

    ## 1. Find the properties (meta_vars and variable local attributes) that are consistently shows up positive examples
    clause_targets_values_seen = {}
    clause_targets_and_example_ids = (
        {}
    )  # prepared for later precondition split if verification fails

    for idx, example in enumerate(tqdm(hypothesis.positive_examples)):
        if len(example) == 0:
            # raise ValueError("Empty example found in positive examples")
            print("Warning: empty examples found in positive examples")
            continue

        clause_targets = _find_local_clause_targets(example)

        if (
            len(clause_targets[PT.CONSTANT]) == 0
            and len(clause_targets[PT.UNEQUAL]) == 0
        ):
            print("example: ", example)
            raise ValueError("No conditions found in the example")

        for clause_target in clause_targets[PT.CONSTANT]:
            key = (clause_target, PT.CONSTANT)
            if key not in clause_targets_and_example_ids:
                clause_targets_values_seen[key] = set()
                clause_targets_and_example_ids[key] = []
            clause_targets_values_seen[key].add(
                clause_targets[PT.CONSTANT][clause_target]
            )
            clause_targets_and_example_ids[key].append(idx)

        for clause_target in clause_targets[PT.UNEQUAL]:
            key = (clause_target, PT.UNEQUAL)
            if key not in clause_targets_and_example_ids:
                clause_targets_values_seen[key] = set()
                clause_targets_and_example_ids[key] = []
            clause_targets_and_example_ids[key].append(idx)

    ## generate clauses for all the properties found
    clauses_and_example_ids: dict[PreconditionClause, list[int]] = {}
    for key in clause_targets_and_example_ids:
        prop_name, _type = key
        values = clause_targets_values_seen[key]
        if _type == PT.CONSTANT and len(values) > 1:
            _type = PT.CONSISTENT
        clause = PreconditionClause(prop_name, _type, values)
        clauses_and_example_ids[clause] = clause_targets_and_example_ids[key]

    ## 2. Precondition Pruning & Verification: Remove the properties that are not necessary

    # use the clauses that are consistent in all the positive examples as the initial preconditions
    precond_clause_candidates = {
        clause
        for clause in clauses_and_example_ids
        if len(clauses_and_example_ids[clause]) == len(hypothesis.positive_examples)
    }
    clause_ever_false_in_neg = {clause: False for clause in clauses_and_example_ids}
    neg_examples_passing_preconditions = []

    for neg_example in tqdm(hypothesis.negative_examples, desc="Pruning Precondition"):
        whether_precondition_holds = True
        for clause in precond_clause_candidates:
            res = clause.verify(neg_example)
            if not res:
                clause_ever_false_in_neg[clause] = True
            whether_precondition_holds = whether_precondition_holds and res
        if whether_precondition_holds:
            neg_examples_passing_preconditions.append(neg_example)

    # delete the clauses that are never violated in the negative examples from both the candidates and the cluses_and_example_ids
    precond_clause_candidates = {
        clause
        for clause in precond_clause_candidates
        if clause_ever_false_in_neg[clause]
    }
    clauses_and_example_ids = {
        clause: clauses_and_example_ids[clause]
        for clause in clauses_and_example_ids
        if clause_ever_false_in_neg[clause]
    }

    # if no negative examples pass the preconditions, then the precondition is correct and we can return
    if len(neg_examples_passing_preconditions) == 0:
        return [Precondition(list(precond_clause_candidates))]

    # if we have violations, let's try to add constraints to the preconditions

    # print(Precondition(list(clauses_and_example_ids.keys())))
    # print(len(neg_examples_passing_preconditions))
    # raise NotImplementedError("Precondition Split is not implemented yet.")

    # let's first find the existing clauses whose target are bool type and values are {True, False}
    consistent_bool_clauses = {
        clause
        for clause in precond_clause_candidates
        if clause.type == PT.CONSISTENT
        and len(clause.values) == 2
        and True in clause.values
        and False in clause.values
    }

    split_bool_clauses = []

    for clause in consistent_bool_clauses:
        # we can split the clause into two clauses
        # one with True and one with False
        true_clause = PreconditionClause(clause.prop_name, PT.CONSTANT, {True})
        false_clause = PreconditionClause(clause.prop_name, PT.CONSTANT, {False})

        # construct the new preconditions by removing the old clause and adding the new clauses
        true_precondition = Precondition(
            list(precond_clause_candidates - {clause} | {true_clause})
        )

        false_precondition = Precondition(
            list(precond_clause_candidates - {clause} | {false_clause})
        )

        true_res = verify_precondition_safety(true_precondition, neg_examples_passing_preconditions)
        false_res = verify_precondition_safety(false_precondition, neg_examples_passing_preconditions)
        assert true_res or false_res, f"Both true and false preconditions are unsafe for clause {clause}"

        if true_res and false_res:
            # both are safe, splitting this clause is not necessary
            continue

        if not true_res:
            split_bool_clauses.append((clause, True))
        if not false_res:
            split_bool_clauses.append((clause, False))

    print("Split Bool Clauses")
    for clause, val in split_bool_clauses:
        print(clause, val)

    partial_clauses = {
        clause: clause_targets_and_example_ids[clause]
        for clause in clause_targets_and_example_ids if len(clause_targets_and_example_ids[clause]) < len(hypothesis.positive_examples)
    }

    # sort the partial clauses by the number of examples they are found in
    partial_clauses = dict(sorted(partial_clauses.items(), key=lambda x: len(x[1]), reverse=True))

    print("Partial Clauses")
    for clause in partial_clauses:
        print(clause, len(partial_clauses[clause]) / len(hypothesis.positive_examples))

    # print("Partial Clauses")

    
        
        