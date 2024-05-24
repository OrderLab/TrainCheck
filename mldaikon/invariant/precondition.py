import logging
from enum import Enum
from typing import Hashable

from tqdm import tqdm

from mldaikon.invariant.base_cls import Hypothesis
from mldaikon.config.config import CONST_CLAUSE_NUM_VALUES_THRESHOLD

logger = logging.getLogger("Precondition")


class PreconditionClauseType(Enum):
    CONSTANT = "constant"
    CONSISTENT = "consistent"
    UNEQUAL = "unequal"


PT = PreconditionClauseType


class PreconditionClause:
    def __init__(self, prop_name: str, prop_type: type, _type: PT, values: set | type):
        assert _type in [
            PT.CONSISTENT,
            PT.CONSTANT,
            PT.UNEQUAL,
        ], f"Invalid Precondition type {_type}"

        if _type in [PT.CONSISTENT, PT.CONSTANT]:
            assert len(values) > 0, "Values should not be empty for CONSTANT or CONSISTENT type"

        self.prop_name = prop_name
        self.prop_type = prop_type
        self.type = _type
        self.values = values if isinstance(values, set) else {values}

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return f"Prop: {self.prop_name}, Type: {self.type}, Values: {self.values}"
    
    def __eq__(self, other):
        if not isinstance(other, PreconditionClause):
            return False
        return self.prop_name == other.prop_name and self.prop_type == other.prop_type and self.type == other.type and self.values == other.values
    
    def __hash__(self):
        return hash((self.prop_name, self.prop_type, self.type, tuple(self.values)))

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


def _find_local_clauses(example: list, key_to_skip: str = "param_value") -> dict:
    """A list of traces to find common properties from. The property should hold locally within the example."""

    clauses = []
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

        prop_type = type(example[0][prop])

        if len(prop_values_seen) == 1:
            clauses.append(PreconditionClause(prop, prop_type, PT.CONSTANT, prop_values_seen))
        elif len(prop_values_seen) == len(example):
            clauses.append(PreconditionClause(prop, prop_type, PT.UNEQUAL, None))

    return clauses

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
            print("Precondition is not safe")
            print("Example", example)
            return False
    return True

def _merge_clauses(clauses_lists: list[list[PreconditionClause]]) -> dict[PreconditionClause, list[int]]:
    """Given a list of clauses, should merge the 'constant' clauses into 'consistent' clauses if the number of values seen is too large
    
    args:
        clauses: list[list[PreconditionClause]]
            A list of clauses to merge. **The index of the list should correspond to the example index.**

    returns:
        dict[PreconditionClause, list[int]]
            A dictionary where the key is the merged clause and the value is the list of example indices that the clause is found in.
    """

    # step 1: Grouping the clauses by the target
    clause_targets_and_example_ids = {}
    for exp_idx, clauses in enumerate(clauses_lists):
        for clause in clauses:
            clause_target = clause.prop_name
            if clause_target not in clause_targets_and_example_ids:
                clause_targets_and_example_ids[clause_target] = {clause: []}
            elif clause not in clause_targets_and_example_ids[clause_target]:
                clause_targets_and_example_ids[clause_target][clause] = []
            clause_targets_and_example_ids[clause_target][clause].append(exp_idx)

    # step 2: Merging the clauses
    clauses_and_example_ids = {}
    for target, clauses_exp_ids in clause_targets_and_example_ids.items():
        seen_constant_values = set()
        example_ids = set()
        prop_type = None
        for clause in clauses_exp_ids:
            if prop_type is None:
                prop_type = clause.prop_type
            if clause.type == PT.CONSTANT and prop_type is not bool:
                seen_constant_values.update(clause.values)
                example_ids.update(clauses_exp_ids[clause])
            if clause.type == PT.CONSISTENT:
                raise ValueError("Consistent clause found in the local clauses, this should not happen")

            if clause.type == PT.CONSTANT and prop_type is bool:
                # if the prop_type is bool, we should not merge the constant clauses
                clauses_and_example_ids[clause] = clauses_exp_ids[clause]
            if clause.type == PT.UNEQUAL:
                # if we see a unequal clause, just add it to the clauses_and_example_ids
                clauses_and_example_ids[clause] = clauses_exp_ids[clause]

        # merge the constant clauses into consistent clauses
        if len(seen_constant_values) == 0:
            continue

        if len(seen_constant_values) > CONST_CLAUSE_NUM_VALUES_THRESHOLD and prop_type is not bool:
            consistent_clause = PreconditionClause(target, prop_type, PT.CONSISTENT, seen_constant_values)
            clauses_and_example_ids[consistent_clause] = example_ids
        else:
            constant_clause = PreconditionClause(target, prop_type, PT.CONSTANT, seen_constant_values)
            clauses_and_example_ids[constant_clause] = example_ids

    return clauses_and_example_ids

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

    ## 1. Find the properties (meta_vars and variable local attributes) that are consistently shows up positive examples
    all_local_clauses = []

    for idx, example in enumerate(tqdm(hypothesis.positive_examples)):
        if len(example) == 0:
            # raise ValueError("Empty example found in positive examples")
            print("Warning: empty examples found in positive examples")
            continue

        local_cluases = _find_local_clauses(example)

        if len(local_cluases) == 0:
            print("example: ", example)
            raise ValueError("No clauses can be found in the example, precondition will be empty.")

        all_local_clauses.append(local_cluases)

    ## merge the local clauses: 1) group by the clause target and 2) merge into consistent if too many values are found
    clauses_and_example_ids = _merge_clauses(all_local_clauses)

    # use the clauses that are consistent in all the positive examples as the initial preconditions
    base_precond_clauses = {
        clause
        for clause in clauses_and_example_ids
        if len(clauses_and_example_ids[clause]) == len(hypothesis.positive_examples)
    }

    clause_ever_false_in_neg = {clause: False for clause in clauses_and_example_ids}
    passing_neg_exps = []

    print("Number of base preconditions", len(base_precond_clauses))
    print("Number of total preconditions", len(clauses_and_example_ids))
    print("Clauses")
    for clause in clauses_and_example_ids:
        print(clause, len(clauses_and_example_ids[clause]) / len(hypothesis.positive_examples))

    print("Base preconditions")
    for clause in base_precond_clauses:
        print(clause)

    for neg_example in tqdm(hypothesis.negative_examples, desc="Pruning Precondition"):
        whether_precondition_holds = True
        for clause in clauses_and_example_ids:
            res = clause.verify(neg_example)
            if not res:
                clause_ever_false_in_neg[clause] = True
            if clause in base_precond_clauses:
                whether_precondition_holds = whether_precondition_holds and res
        if whether_precondition_holds:
            passing_neg_exps.append(neg_example)

    # delete the clauses that are never violated in the negative examples from both the candidates and the cluses_and_example_ids
    base_precond_clauses = {
        clause
        for clause in base_precond_clauses
        if clause_ever_false_in_neg[clause]
    }
    clauses_and_example_ids = {
        clause: clauses_and_example_ids[clause]
        for clause in clauses_and_example_ids
        if clause_ever_false_in_neg[clause]
    }

    partial_clauses_and_example_ids = {
        clause: clauses_and_example_ids[clause]
        for clause in clauses_and_example_ids
        if clause not in base_precond_clauses
    }

    # group the clauses by the example indices
    grouped_clauses = {}
    for clause, exp_ids in partial_clauses_and_example_ids.items():
        if exp_ids not in grouped_clauses:
            grouped_clauses[exp_ids] = []
        grouped_clauses[exp_ids].append(clause)

    # find the top-level partial examples
    top_level_example_ids = []
    for exp_ids in grouped_clauses:
        found_relevant = False
        for ids in range(len(top_level_example_ids)):
            if exp_ids.issubset(top_level_example_ids[ids]):
                found_relevant = True
                break
            if top_level_example_ids[ids].issubset(exp_ids):
                top_level_example_ids[ids] = exp_ids
                found_relevant = True
                break
        if not found_relevant:
            top_level_example_ids.append(exp_ids)

    # construct the sub-hypothesis with the top-level partial examples
    sub_preconditions = []
    for exp_ids in top_level_example_ids:
        sub_hypothesis = Hypothesis(
            hypothesis.invariant,
            [hypothesis.positive_examples[i] for i in exp_ids],
            passing_neg_exps,
        )
        sub_preconditions.extend(find_precondition(sub_hypothesis))

    # verify that the sub-preconditions covers all the positive examples
    for exp in hypothesis.positive_examples:
        if not any(precond.verify(exp) for precond in sub_preconditions):
            print("Warning: sub-preconditions do not cover all the positive examples")
            print("Example", exp)        
            raise ValueError("Sub-preconditions do not cover all the positive examples")
        
    return sub_preconditions