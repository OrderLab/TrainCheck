import logging
from enum import Enum
from itertools import combinations
from typing import Hashable, Iterable

from tqdm import tqdm

from mldaikon.config.config import (
    CONST_CLAUSE_NUM_VALUES_THRESHOLD,
    NOT_USE_AS_CLAUSE_FIELDS,
)
from mldaikon.invariant.base_cls import Hypothesis

logger = logging.getLogger("Precondition")


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
        output = "====================== Start of Precondition ======================\n"
        for clause in self.clauses:
            output += str(clause) + "\n"
        output += "====================== End of Preconditions ======================\n"
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
        output += "====================== End of Grouped Precondition ======================\n"
        return output

    def to_dict(self) -> dict:
        return {
            group_name: [precond.to_dict() for precond in preconditions]
            for group_name, preconditions in self.grouped_preconditions.items()
        }


def is_statistical_significant(positive_examples: list) -> bool:
    return len(positive_examples) > 100


def _find_local_clauses(
    example: list, key_to_skip: str | list[str] = "param_value"
) -> list[PreconditionClause]:
    """A list of traces to find common properties from. The property should hold locally within the example."""

    clauses = []
    # find properties that have only one value in the example
    for prop in example[0]:
        if prop in NOT_USE_AS_CLAUSE_FIELDS:
            # skip meta_info about each event
            continue

        if isinstance(key_to_skip, list) and any(key in prop for key in key_to_skip):
            continue

        if isinstance(key_to_skip, str) and key_to_skip in prop:
            continue

        if not all(isinstance(example[i][prop], Hashable) for i in range(len(example))):
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

        # get the type of the property
        prop_dtype = None
        for value in prop_values_seen:
            if value is None:
                continue
            if prop_dtype is None:
                prop_dtype = type(value)
            if prop_dtype != type(value) and value is not None:
                raise ValueError(
                    f"Property {prop} has inconsistent types {prop_dtype, type(value)} in the example"
                )

        if prop_dtype is None:
            logger.warning(
                f"Property {prop} has no real values in the example, skipping this property as a clause."
            )
            continue

        if len(prop_values_seen) == 1 and prop_dtype is not None:
            clauses.append(
                PreconditionClause(prop, prop_dtype, PT.CONSTANT, prop_values_seen)
            )
        elif len(prop_values_seen) == len(example) and None not in prop_values_seen:
            clauses.append(PreconditionClause(prop, prop_dtype, PT.UNEQUAL, None))

    return clauses


def verify_precondition_safety(
    precondition: Precondition, negative_examples: list
) -> bool:
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


def _merge_clauses(
    clauses_lists: list[list[PreconditionClause]],
) -> dict[PreconditionClause, list[int]]:
    """Given a list of clauses, should merge the 'constant' clauses into 'consistent' clauses if the number of values seen is too large

    args:
        clauses: list[list[PreconditionClause]]
            A list of clauses to merge. **The index of the list should correspond to the example index.**

    returns:
        dict[PreconditionClause, list[int]]
            A dictionary where the key is the merged clause and the value is the list of example indices that the clause is found in.
    """

    # step 1: Grouping the clauses by the target
    clause_targets_and_exp_ids: dict[str, dict[PreconditionClause, list[int]]] = {}
    for exp_id, clauses in enumerate(clauses_lists):
        for clause in clauses:
            clause_target = clause.prop_name
            if clause_target not in clause_targets_and_exp_ids:
                clause_targets_and_exp_ids[clause_target] = {clause: []}
            elif clause not in clause_targets_and_exp_ids[clause_target]:
                clause_targets_and_exp_ids[clause_target][clause] = []
            clause_targets_and_exp_ids[clause_target][clause].append(exp_id)

    # step 2: Merging the clauses
    merged_clauses_and_exp_ids = {}
    for target, clauses_and_exp_ids in clause_targets_and_exp_ids.items():
        seen_constant_values = set()
        seen_constant_exp_ids = set()
        prop_dtype = None
        for clause in clauses_and_exp_ids:
            if prop_dtype is None:
                prop_dtype = clause.prop_dtype
            if clause.type == PT.CONSTANT and prop_dtype is not bool:
                seen_constant_values.update(clause.values)
                seen_constant_exp_ids.update(clauses_and_exp_ids[clause])
            if clause.type == PT.CONSISTENT:
                raise ValueError(
                    "Consistent clause found in the local clauses, this should not happen"
                )

            if clause.type == PT.CONSTANT and prop_dtype is bool:
                # if the prop_dtype is bool, we should not merge the constant clauses
                merged_clauses_and_exp_ids[clause] = clauses_and_exp_ids[clause]
            if clause.type == PT.UNEQUAL:
                # if we see a unequal clause, just add it to the merged_clauses_and_exp_ids
                merged_clauses_and_exp_ids[clause] = clauses_and_exp_ids[clause]

        assert prop_dtype is not None, "Property type should not be None"

        # merge the constant clauses into consistent clauses
        if len(seen_constant_values) == 0:
            continue

        if (
            len(seen_constant_values) > CONST_CLAUSE_NUM_VALUES_THRESHOLD
            and prop_dtype is not bool
        ):
            consistent_clause = PreconditionClause(
                target, prop_dtype, PT.CONSISTENT, seen_constant_values
            )
            merged_clauses_and_exp_ids[consistent_clause] = list(seen_constant_exp_ids)
        else:
            constant_clause = PreconditionClause(
                target, prop_dtype, PT.CONSTANT, seen_constant_values
            )
            merged_clauses_and_exp_ids[constant_clause] = list(seen_constant_exp_ids)

    return merged_clauses_and_exp_ids


def find_precondition(
    hypothesis: Hypothesis,
    keys_to_skip: list[str] = [],
) -> GroupedPreconditions | None:
    """When None is returned, it means that we cannot find a precondition that is safe to use for the hypothesis."""

    # postive examples and negative examples should have the same group names
    group_names = hypothesis.positive_examples.group_names
    # assert group_names == hypothesis.negative_examples.group_names
    if group_names != hypothesis.negative_examples.group_names:
        logger.warning(
            f"Group names in positive and negative examples do not match in the hypothesis. This might lead to unexpected results.\n Positive Examples: {hypothesis.positive_examples.group_names}\n Negative Examples: {hypothesis.negative_examples.group_names}"
        )

    grouped_preconditions = {}
    for group_name in group_names:
        positive_examples = hypothesis.positive_examples.get_group_from_examples(
            group_name
        )
        try:
            negative_examples = hypothesis.negative_examples.get_group_from_examples(
                group_name
            )
        except KeyError:
            logger.warning(
                f"Negative examples not found for group {group_name}, assigning this group an unconditional precondition."
            )
            # the negative examples are not found, assign an unconditional precondition (to be handled in find_precondition_from_single_group)
            negative_examples = []

        grouped_preconditions[group_name] = find_precondition_from_single_group(
            positive_examples, negative_examples, keys_to_skip
        )

    # if every group's precondition is of length 0, return None
    if all(
        len(grouped_preconditions[group_name]) == 0
        for group_name in grouped_preconditions
    ):
        return None

    return GroupedPreconditions(grouped_preconditions)


def find_precondition_from_single_group(
    positive_examples: list[list[dict]],
    negative_examples: list[list[dict]],
    keys_to_skip: list[str] = [],
    _pruned_clauses: set[PreconditionClause] = set(),
    _skip_pruning: bool = False,
) -> list[Precondition]:
    """Given a hypothesis, should return a list of `Precondition` objects that invariants should hold if one of the `Precondition` is satisfied.

    args:
        - hypothesis: A hypothesis to find preconditions for.
        - (private) _pruned_clauses: A set of clauses that should not be considered as a precondition
        - (private) _skip_pruning: Whether to skip the pruning process, should only be used when `_pruned_clauses` is provided
            and the hypothesis comes with a reduced set of negative examples



    This function will perform inference on the positive examples to find special properties that consistently show up in the positive examples.
    Then, the found properties will be scanned in the negative examples to prune out unnecessary properties that also hold for the negative examples.
    The pruning process is relaxing the precondition by just removing noises. Thus, if at anytime the precondition is verified in the negative examples, the function will abort.

    To implement the invariant split OP. We need to determine how this verification / pruning process should be done, because now all the `Precondition` objects have to be violated in the negative examples.
    """
    logger.debug(
        f"Calling precondition inference with \n# positive examples: {len(positive_examples)}, \n# negative examples: {len(negative_examples)}"
    )

    if len(negative_examples) == 0:
        logger.warning(
            "No negative examples found, assigning unconditional precondition"
        )
        return [UnconditionalPrecondition()]

    ## 1. Find the properties (meta_vars and variable local attributes) that consistently shows up positive examples
    all_local_clauses = []

    for example in tqdm(positive_examples):
        if len(example) == 0:
            raise ValueError("Empty example found in positive examples")

        # HACK: in ConsistencyRelation in order to avoid the field used in the invariant, we need to skip the field in the precondition. It is up to the caller to provide the keys to skip. We should try to refactor this to have a more generic solution.
        local_clauses = _find_local_clauses(example, key_to_skip=keys_to_skip)

        if len(local_clauses) == 0:
            # NOTE: this would also happen under the unconditional case, but since the unconditional case is handled separately, we should not reach here
            print("example: ", example)
            raise ValueError(
                "No clauses can be found in the example, precondition will be empty."
            )

        all_local_clauses.append(local_clauses)

    ## merge the local clauses: 1) group by the clause target and 2) merge into consistent if too many values are found
    merged_clauses_and_exp_ids = _merge_clauses(all_local_clauses)

    if _pruned_clauses:
        merged_clauses_and_exp_ids = {
            clause: merged_clauses_and_exp_ids[clause]
            for clause in merged_clauses_and_exp_ids
            if clause not in _pruned_clauses
        }

    # use the clauses that are consistent in all the positive examples as the initial preconditions
    base_precond_clauses = {
        clause
        for clause in merged_clauses_and_exp_ids
        if len(merged_clauses_and_exp_ids[clause]) == len(positive_examples)
    }

    clause_ever_false_in_neg = {clause: False for clause in merged_clauses_and_exp_ids}
    passing_neg_exps = []

    for neg_example in tqdm(
        negative_examples,
        desc="Scanning Base Precondition on All Negative Examples",
    ):
        whether_precondition_holds = True
        for clause in merged_clauses_and_exp_ids:
            res = clause.verify(neg_example)
            if not res:
                clause_ever_false_in_neg[clause] = True
            if clause in base_precond_clauses:
                whether_precondition_holds = whether_precondition_holds and res
        if whether_precondition_holds:
            passing_neg_exps.append(neg_example)

    if not _skip_pruning:
        # delete the clauses that are never violated in the negative examples from both the candidates and the cluses_and_exp_ids
        base_precond_clauses = {
            clause
            for clause in base_precond_clauses
            if clause_ever_false_in_neg[clause]
        }
        merged_clauses_and_exp_ids = {
            clause: merged_clauses_and_exp_ids[clause]
            for clause in merged_clauses_and_exp_ids
            if clause_ever_false_in_neg[clause]
        }
        # update _pruned_clauses
        _pruned_clauses.update(
            {
                clause
                for clause in clause_ever_false_in_neg
                if not clause_ever_false_in_neg[clause]
            }
        )
        print("Base Precondition Clauses After Pruning")
        print(str(Precondition(base_precond_clauses)))
    else:
        # skip pruning is necessary when we are inferring on a reduced set of negative examples as many clauses may not be violated and thus pruned unnecessarily
        assert (
            _pruned_clauses
        ), "_pruned_clauses must be provided if pruning process are to skipped"
        print("Skipping Pruning")

    if not passing_neg_exps:
        return [Precondition(list(base_precond_clauses))]

    partial_merged_clauses_and_exp_ids = {
        clause: tuple(
            merged_clauses_and_exp_ids[clause]
        )  # convert to tuple to make it hashable
        for clause in merged_clauses_and_exp_ids
        if clause not in base_precond_clauses
    }

    # group the clauses by the example indices
    grouped_clauses: dict[tuple[int, ...], list[PreconditionClause]] = {}
    for clause, exp_ids in partial_merged_clauses_and_exp_ids.items():
        if exp_ids not in grouped_clauses:
            grouped_clauses[exp_ids] = []
        grouped_clauses[exp_ids].append(clause)

    # find the top-level partial examples
    top_level_exp_ids: list[tuple[int, ...]] = []
    for exp_ids in grouped_clauses:
        set_exp_ids = set(exp_ids)  # convert to set for the subset operation
        found_relevant = False
        for ids in range(len(top_level_exp_ids)):
            set_top_level_ids = set(top_level_exp_ids[ids])
            if set_exp_ids.issubset(set_top_level_ids):
                found_relevant = True
                break
            if set_top_level_ids.issubset(set_exp_ids):
                print(
                    "Replace top-level example ids from group",
                    grouped_clauses[top_level_exp_ids[ids]],
                    "with",
                    grouped_clauses[exp_ids],
                )
                top_level_exp_ids[ids] = exp_ids
                found_relevant = True
                break
        if not found_relevant:
            print(
                "Adding new top-level example ids from group", grouped_clauses[exp_ids]
            )
            top_level_exp_ids.append(exp_ids)

    # construct the top-level preconditions
    print(f"Splitting into {len(top_level_exp_ids)} sub-hypotheses")
    print("Length of the top-level examples")
    for exp_ids in top_level_exp_ids:
        print(len(exp_ids), len(exp_ids) / len(positive_examples))

    print("Partial Clauses")
    for clause in partial_merged_clauses_and_exp_ids:
        print("==============================")
        print("values", clause.values)
        print("type", clause.type)
        print("target", clause.prop_name)
        print("Examples", len(partial_merged_clauses_and_exp_ids[clause]))
        print(
            "%examples",
            len(partial_merged_clauses_and_exp_ids[clause]) / len(positive_examples),
        )
    print("==============================")

    # construct the sub-hypothesis with the top-level partial examples
    preconditions: list[Precondition] = []
    for exp_ids in top_level_exp_ids:
        sub_positive_examples = [positive_examples[i] for i in exp_ids]
        sub_preconditions = find_precondition_from_single_group(
            sub_positive_examples,
            passing_neg_exps,
            keys_to_skip=keys_to_skip,
            _pruned_clauses=_pruned_clauses,
            _skip_pruning=True,
        )
        if len(sub_preconditions) == 0:
            print("Warning: empty preconditions found in the sub-hypothesis")

        preconditions.extend(sub_preconditions)

    # deduplicate the preconditions
    child_preconds = set()
    for precond1, precond2 in combinations(set(preconditions), 2):
        if precond1.implies(precond2):
            child_preconds.add(precond1)
        elif precond2.implies(precond1):
            child_preconds.add(precond2)
        else:
            continue

    # remove the child preconditions
    for child_precond in child_preconds:
        preconditions.remove(child_precond)

    # verify that the sub-preconditions covers all the positive examples
    for exp in positive_examples:
        if not any(precond.verify(exp) for precond in preconditions):
            print(
                "Warning: sub-preconditions do not cover all the positive examples",
                len(positive_examples),
            )
            print("No precondition found for this sub-hypothesis")
            print("Sub-preconditions")
            for precond in preconditions:
                print(precond)

            print("==============================")
            print("Example")
            print(exp)
            print("Example Clauses")
            print(_find_local_clauses(exp, key_to_skip=keys_to_skip))
            print("==============================")

            # raise ValueError("Sub-preconditions do not cover all the positive examples")
            return []

    return preconditions
