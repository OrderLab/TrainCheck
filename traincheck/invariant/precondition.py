import logging
from itertools import combinations
from typing import Hashable

from tqdm import tqdm

import traincheck.config.config as config
from traincheck.invariant.base_cls import (
    PT,
    GroupedPreconditions,
    Hypothesis,
    Precondition,
    PreconditionClause,
    Preconditions,
    UnconditionalPrecondition,
)
from traincheck.trace.trace import Trace
from traincheck.trace.types import MD_NONE
from traincheck.utils import safe_isnan

logger = logging.getLogger("Precondition")

STAGE_KEY = "meta_vars.stage"


def is_statistical_significant(positive_examples: list) -> bool:
    return len(positive_examples) > 100


def _find_local_clauses(
    example: list, key_to_skip: str | list[str] = "param_value"
) -> list[PreconditionClause]:
    """A list of traces to find common properties from. The property should hold locally within the example."""

    clauses = []

    # TODO: figure out how to extend this to args, kwargs and return values
    relevant_key_prefixes = {
        "process_id",
        "thread_id",
        "meta_vars",
        "var_name",
        "function",
        "exception",
        "attributes",
        "var_type",
    }

    fields_for_inference = []
    for field in example[0]:
        for prefix in relevant_key_prefixes:
            if field.startswith(prefix):
                break
        else:
            # we skip inference on properties that are not relevant
            continue

        if isinstance(key_to_skip, list) and any(key in field for key in key_to_skip):
            continue

        if isinstance(key_to_skip, str) and key_to_skip in field:
            continue

        if not all(
            isinstance(example[i][field], Hashable) for i in range(len(example))
        ):
            # we cannot use non-hashable properties as preconditions, due to limitations in the current implementation (set cannot contain non-hashable objects)
            continue

        all_record_has_field = True
        for record in example:
            if field not in record:
                all_record_has_field = False
                break
            if safe_isnan(record[field]):
                # we should not use NaN as a precondition
                all_record_has_field = False
                break

        if not all_record_has_field:
            continue

        fields_for_inference.append(field)

    # find properties that have only one value in the example
    for field in fields_for_inference:
        field_values_seen = {example[0][field]}
        for i in range(1, len(example)):
            field_values_seen.add(example[i][field])

        # get the type of the property
        field_dtype = None
        for value in field_values_seen:
            if value is None:
                continue
            if field_dtype is None:
                field_dtype = type(value)

        if field_dtype is None:
            # logger.warning(
            #     f"Property {prop} has no real values in the example, skipping this property as a clause."
            # )
            continue

        if len(field_values_seen) == 1 and field_dtype is not None:
            if field_dtype is MD_NONE:
                clauses.append(
                    PreconditionClause(field, None, PT.CONSTANT, None, {None})
                )
            clauses.append(
                PreconditionClause(
                    field, field_dtype, PT.CONSTANT, None, field_values_seen
                )
            )
        elif len(field_values_seen) == len(example) and None not in field_values_seen:
            clauses.append(
                PreconditionClause(field, field_dtype, PT.UNEQUAL, None, None)
            )

    # let's deal with meta_vars.context_managers separately
    all_context_managers = []
    for k in example[0]:
        if k.startswith("meta_vars.context_managers"):
            all_context_managers.append(k)

    for context_manager_key in all_context_managers:
        # emit the exist clause first
        clauses.append(
            PreconditionClause(context_manager_key, None, PT.EXIST, None, None)
        )
        # for each argument of the context manager, emit the CONSTANT clauses for their values
        for arg, value in example[0][context_manager_key].items():
            if not isinstance(value, Hashable):
                # we cannot use non-hashable objects as preconditions, that's so bad aint it?
                continue

            clauses.append(
                PreconditionClause(
                    f"{context_manager_key}",
                    type(value),
                    PT.CONSTANT,
                    (arg,),
                    {value},
                )
            )

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
    clause_targets_and_exp_ids: dict[
        tuple[str, tuple[str] | None], dict[PreconditionClause, list[int]]
    ] = {}
    for exp_id, clauses in enumerate(clauses_lists):
        for clause in clauses:
            clause_target = (clause.prop_name, clause.additional_path)
            if clause_target not in clause_targets_and_exp_ids:
                clause_targets_and_exp_ids[clause_target] = {clause: []}
            elif clause not in clause_targets_and_exp_ids[clause_target]:
                clause_targets_and_exp_ids[clause_target][clause] = []
            clause_targets_and_exp_ids[clause_target][clause].append(exp_id)

    # step 2: Merging the clauses
    merged_clauses_and_exp_ids = {}
    for target, clauses_and_exp_ids in clause_targets_and_exp_ids.items():
        seen_unique_constant_values = set()
        seen_unique_constant_exp_ids = set()
        constant_value_to_exp_ids: dict[object, set] = {}
        field_dtype = None
        for clause in clauses_and_exp_ids:
            if field_dtype is None:
                field_dtype = clause.prop_dtype
            if (
                clause.type == PT.CONSTANT and field_dtype is not bool
            ):  # tensor_model_parallel (bool) and meta_vars.stage (str) are not merged
                assert (
                    len(clause.values) == 1
                ), "Constant clause should only have one value prior to merging"
                seen_unique_constant_values.update(clause.values)
                seen_unique_constant_exp_ids.update(clauses_and_exp_ids[clause])

                for value in clause.values:
                    if value not in constant_value_to_exp_ids:
                        constant_value_to_exp_ids[value] = set()
                    constant_value_to_exp_ids[value].update(clauses_and_exp_ids[clause])

            if clause.type == PT.CONSISTENT:
                raise ValueError(
                    "Consistent clause found in the local clauses, this should not happen"
                )

            if clause.type == PT.CONSTANT and field_dtype is bool:
                # if the field_dtype is bool, we should not merge the constant clauses
                merged_clauses_and_exp_ids[clause] = clauses_and_exp_ids[clause]

            if clause.type == PT.UNEQUAL:
                # if we see a unequal clause, just add it to the merged_clauses_and_exp_ids
                merged_clauses_and_exp_ids[clause] = clauses_and_exp_ids[clause]
            if clause.type == PT.EXIST:
                # if we see an exist clause, just add it to the merged_clauses_and_exp_ids for now
                merged_clauses_and_exp_ids[clause] = clauses_and_exp_ids[clause]

        # assert field_dtype is not None, "Property type should not be None"

        # merge the constant clauses into consistent clauses
        if len(seen_unique_constant_values) == 0:
            continue

        if (
            field_dtype is str
            and len(seen_unique_constant_values)
            > config.CONST_CLAUSE_STR_NUM_VALUES_THRESHOLD
        ) or (
            field_dtype is not str
            and len(seen_unique_constant_values)
            > config.CONST_CLAUSE_NUM_VALUES_THRESHOLD
            and field_dtype is not bool
        ):
            consistent_clause = PreconditionClause(
                target[0],
                field_dtype,
                PT.CONSISTENT,
                target[1],
                seen_unique_constant_values,
            )
            merged_clauses_and_exp_ids[consistent_clause] = list(
                seen_unique_constant_exp_ids
            )
        else:
            # if the number of values seen is not too large, we should just keep the constant clauses
            for value in constant_value_to_exp_ids:
                constant_clause = PreconditionClause(
                    target[0], field_dtype, PT.CONSTANT, target[1], {value}
                )
                merged_clauses_and_exp_ids[constant_clause] = list(
                    constant_value_to_exp_ids[value]
                )

    return merged_clauses_and_exp_ids


def find_precondition(
    hypothesis: Hypothesis,
    traces: list[Trace],
) -> GroupedPreconditions | None:
    """When None is returned, it means that we cannot find a precondition that is safe to use for the hypothesis."""

    keys_to_skip = hypothesis.invariant.relation.get_precondition_infer_keys_to_skip(
        hypothesis
    )
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

        import random

        random.seed(42)
        if len(positive_examples) > 5000:
            logger.warning(
                f"Too many positive examples found for group {group_name}, downsampling to 100000"
            )
            positive_examples = random.sample(positive_examples, 5000)
        if len(negative_examples) > 5000:
            logger.warning(
                f"Too many negative examples found for group {group_name}, downsampling to 100000"
            )
            negative_examples = random.sample(negative_examples, 5000)

        grouped_preconditions[group_name] = Preconditions(
            find_precondition_from_single_group(
                positive_examples, negative_examples, traces, keys_to_skip
            )
        )

        if (
            len(grouped_preconditions[group_name]) == 0
            and len(positive_examples) > 0
            and len(negative_examples) > 0
        ):
            # try doing inverted precondition inference
            logger.debug(
                f"Empty preconditions found for group {group_name}, trying to infer the preconditions by inverting the negative examples."
            )
            # i.e. use negative examples to infer the preconditions, and then invert the final precondition
            # introducing this for inferring invariants related to context managers (e.g. input/output dtype should be the same when no autocast context is used)
            grouped_preconditions[group_name] = Preconditions(
                find_precondition_from_single_group(
                    negative_examples, positive_examples, traces, keys_to_skip
                ),
                inverted=True,
            )

    # if any group's precondition is of length 0, return None
    if all(
        len(grouped_preconditions[group_name]) == 0
        for group_name in grouped_preconditions
    ):
        return None

    return GroupedPreconditions(grouped_preconditions)


def _stage_grouping_eligible(
    examples: list[list[dict]],
) -> bool:
    """Check if the examples are eligible for stage grouping

    elgibility:
      1. stages should be consistent across all the examples
      2. `meta_vars.stage` should be present in all the examples
      3. the number of stages should be more than 1
    """

    if len(examples) == 0:
        return False

    stages = set()
    for example in examples:
        stage = None
        for record in example:
            if STAGE_KEY not in record:
                # rule 2
                return False

            if stage is None:
                stage = record[STAGE_KEY]
            else:
                if record[STAGE_KEY] != stage:
                    # rule 1
                    return False

        stages.add(example[0][STAGE_KEY])

    return len(stages) > 1  # rule 3


def _group_examples_by_stage(
    examples: list[list[dict]],
) -> dict[str, list[list[dict]]]:
    """Group the examples by the meta_vars.stage values

    Grouping elgibility has to be checked by the caller, as the caller might want to group the examples by other values.
        elgibility:
          1. stages should be consistent across all the examples
          2. `meta_vars.stage` should be present in all the examples
          3. the number of stages should be more than 1
    """
    stage_to_examples: dict[str, list[list[dict]]] = {}
    for example in examples:
        stage = None
        skip = False
        for record in example:
            if stage is None:
                stage = record[STAGE_KEY]
            else:
                if record[STAGE_KEY] != stage:
                    # HACK: unlike positive examples, negative examples in the relations are not naturally grouped by the stage values, so we should skip the negative examples that are not consistent with the stage values
                    skip = True
                    break
        if skip:
            continue

        assert isinstance(stage, str), "Stage should be a string"
        if stage not in stage_to_examples:
            stage_to_examples[stage] = []
        stage_to_examples[stage].append(example)
    return stage_to_examples


def find_precondition_from_single_group(
    positive_examples: list[list[dict]],
    negative_examples: list[list[dict]],
    traces: list[Trace],
    keys_to_skip: list[str] = [],
    _pruned_clauses: set[PreconditionClause] = set(),
    _skip_pruning: bool = False,
    _current_depth: int = 0,
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
        f"Calling precondition inference with \n# positive examples: {len(positive_examples)}, \n# negative examples: {len(negative_examples)}, at depth {_current_depth}"
    )

    preconditions: list[Precondition] = []

    if _current_depth > config.MAX_PRECOND_DEPTH:
        logger.debug(
            f"Max depth reached, returning empty preconditions, current depth: {_current_depth}"
        )
        return []

    if len(negative_examples) == 0:
        assert (
            len(positive_examples) > 0
        ), "No negative examples found, but no positive examples found either"
        logger.debug("No negative examples found, assigning unconditional precondition")
        return [UnconditionalPrecondition()]

    # if (
    #     _current_depth == 0
    #     and _stage_grouping_eligible(positive_examples)
    #     and _stage_grouping_eligible(negative_examples)
    # ):
    #     # NOTE: the purpose of this stage-based grouping is to relax the requirement of the preconditions, if the precondition can be established for at least
    #     # one stage, then the invariant should be considered as established. This is useful for invariants that are not always true, but true for some stages.
    #     # HOWEVER, this will cause problems when doing reverse inference (i.e. when the negative examples are used to infer the preconditions), as when a stage has
    #     # completely no positive examples which leads to an unconditional precondition, the invariant will be considered as established for that stage, which is not true.
    #     logger.info(
    #         "Stage grouping is eligible, splitting the hypothesis according to the stage values"
    #     )
    #     # if the examples are eligible for stage grouping, we should group the examples by the stage values
    #     grouped_positive_examples = _group_examples_by_stage(positive_examples)
    #     grouped_negative_examples = _group_examples_by_stage(negative_examples)

    #     if not set(grouped_negative_examples).issubset(set(grouped_positive_examples)):
    #         logger.warning(
    #             f"Negative examples {grouped_positive_examples.keys()} should be a subset of the positive examples {grouped_positive_examples.keys()}, but this is not the case, falling back to the normal precondition inference"
    #         )
    #     else:
    #         logger.debug(
    #             f"All stages found: pos {grouped_positive_examples.keys()}, neg {grouped_negative_examples.keys()}"
    #         )
    #         grouped_preconditions: dict[str, list[Precondition]] = {}
    #         for stage in grouped_positive_examples:
    #             logger.debug(f"Finding preconditions for stage {stage}")
    #             grouped_preconditions[stage] = find_precondition_from_single_group(
    #                 grouped_positive_examples[stage],
    #                 (
    #                     grouped_negative_examples[stage]
    #                     if stage in grouped_negative_examples
    #                     else []
    #                 ),
    #                 traces,
    #                 keys_to_skip,
    #                 _pruned_clauses,
    #                 _skip_pruning,
    #                 _current_depth + 1,
    #             )

    #             # if for a particular stage, no preconditions are found, we should return an empty list for now but drop a huge warning
    #             if len(grouped_preconditions[stage]) == 0:
    #                 logger.warning(
    #                     f"Exception purely for debugging of cases that we don't properly support now. FEEL FREE TO SUPRESS IT: No preconditions found for stage {stage}, dropping this stage for now!!!"
    #                 )

    #         # for the preconditions for each stage, adding the stage clause
    #         for stage in grouped_preconditions:
    #             stage_clause = PreconditionClause(
    #                 STAGE_KEY, str, PT.CONSTANT, None, {stage}
    #             )
    #             if len(grouped_preconditions[stage]) == 1 and isinstance(
    #                 grouped_preconditions[stage][0], UnconditionalPrecondition
    #             ):
    #                 grouped_preconditions[stage] = [Precondition([stage_clause])]
    #             else:
    #                 for precond in grouped_preconditions[stage]:
    #                     precond.add_clause(stage_clause)

    #         # flatten the grouped preconditions to a list
    #         for stage in grouped_preconditions:
    #             preconditions.extend(grouped_preconditions[stage])

    #         return preconditions

    ## 1. Find the properties (meta_vars and variable local attributes) that consistently shows up positive examples
    all_local_clauses = []

    for example in tqdm(positive_examples, desc="Scanning Positive Examples"):
        if len(example) == 0:
            raise ValueError("Empty example found in positive examples")

        # HACK: in ConsistencyRelation in order to avoid the field used in the invariant, we need to skip the field in the precondition. It is up to the caller to provide the keys to skip. We should try to refactor this to have a more generic solution.
        earliest_time = example[0]["time"]
        process_id = example[0]["process_id"]
        thread_id = example[0]["thread_id"]

        if _current_depth == 0:
            for trace in traces:
                meta_vars = trace.get_meta_vars(  # FIXME: add meta_vars to the examples prior find precondition
                    earliest_time, process_id=process_id, thread_id=thread_id
                )  # HACK: get the context at the earliest time, ideally we should find the context that coverred the entire example duration
                if meta_vars is not None:
                    break
            if meta_vars is None:
                logger.critical(
                    "Meta_vars not found for the positive examples, this should never happen but for the inference to continue we just skip the meta_vars update for this positive example"
                )
                meta_vars = {}

            # update every trace with the meta_vars
            for key in meta_vars:
                for i in range(len(example)):
                    example[i][f"meta_vars.{key}"] = meta_vars[key]

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
        # 1. add meta_vars to the negative examples as well (TODO: we only need do this if there are meta_vars related clauses in the base_precond_clauses)
        if _current_depth == 0:
            earliest_time = neg_example[0]["time"]
            process_id = neg_example[0]["process_id"]
            thread_id = neg_example[0]["thread_id"]
            for trace in traces:
                meta_vars = trace.get_meta_vars(
                    earliest_time, process_id=process_id, thread_id=thread_id
                )  # HACK: get the context at the earliest time, ideally we should find the context that coverred the entire example duration
                if meta_vars is not None:
                    break
            if meta_vars is None:
                logger.critical(
                    "Meta_vars not found for the negative examples, this should never happen but for the inference to continue we just skip the meta_vars update for this negative example"
                )
                meta_vars = {}

            # update every trace with the meta_vars
            for key in meta_vars:
                for i in range(len(neg_example)):
                    neg_example[i][f"meta_vars.{key}"] = meta_vars[key]

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
    else:
        # skip pruning is necessary when we are inferring on a reduced set of negative examples as many clauses may not be violated and thus pruned unnecessarily
        assert (
            _pruned_clauses
        ), "_pruned_clauses must be provided if pruning process are to skipped"
        # print("Skipping Pruning")

    # success if no negative examples are passing
    if not passing_neg_exps:
        return [Precondition(list(base_precond_clauses))]

    partial_merged_clauses_and_exp_ids = {
        clause: tuple(
            merged_clauses_and_exp_ids[clause]
        )  # convert to tuple to make it hashable
        for clause in merged_clauses_and_exp_ids
        if clause not in base_precond_clauses
    }

    # print(f"{_current_depth}: Base Precondition Clauses After Pruning")
    # print(str(Precondition(list(base_precond_clauses))))

    # print("Partial Merged Clauses")
    # for clause in partial_merged_clauses_and_exp_ids:
    #     print(f"{len(partial_merged_clauses_and_exp_ids[clause])}\t", clause)

    if len(partial_merged_clauses_and_exp_ids) == 0:
        logger.debug("No partial preconditions found, cannot infer further")
        return []

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
                # print(
                #     "Replace top-level example ids from group",
                #     grouped_clauses[top_level_exp_ids[ids]],
                #     "with",
                #     grouped_clauses[exp_ids],
                # )
                top_level_exp_ids[ids] = exp_ids
                found_relevant = True
                break
        if not found_relevant:
            # print(
            #     "Adding new top-level example ids from group", grouped_clauses[exp_ids]
            # )
            top_level_exp_ids.append(exp_ids)

    # # if detected meta_vars.stage in partial_merged_clauses_and_exp_ids, we should split the hypothesis according to the meta_vars.stage values
    # stage_related_clauses = {
    #     clause: partial_merged_clauses_and_exp_ids[clause]
    #     for clause in partial_merged_clauses_and_exp_ids
    #     if "meta_vars.stage" in clause.prop_name
    # }

    # # if multiple stages exist and all exp ids related to the stages add up to all the positive examples, we should split the hypothesis according to the stage values
    # if len(stage_related_clauses) > 1:
    #     all_stage_exp_ids = set()
    #     for clause in stage_related_clauses:
    #         all_stage_exp_ids.update(stage_related_clauses[clause])
    #     if len(all_stage_exp_ids) == len(positive_examples):
    #         print(f"Detected {len(stage_related_clauses)} stage-related clauses, splitting the hypothesis according to the stage values")
    #         logger.debug(
    #             f"Detected stage-related clauses, splitting the hypothesis according to the stage values"
    #         )
    #         # override the top_level_exp_ids
    #         top_level_exp_ids = [tuple(exp_ids) for exp_ids in stage_related_clauses.values()]

    # construct the sub-hypothesis with the top-level partial examples
    coverred_exp_ids: set[int] = set()
    logger.debug(
        f"Depth:{_current_depth}, Splitting into {len(top_level_exp_ids)} sub-hypotheses, with size {[len(exp_ids) for exp_ids in top_level_exp_ids]}"
    )
    for i, exp_ids in enumerate(top_level_exp_ids):
        # if coverred_exp_ids and the potentially coverred exp_ids does not cover all the positive examples, we should directly return inference failure (empty preconditions)
        all_future_exp_ids: set[int] = set()
        for j in range(i, len(top_level_exp_ids)):
            all_future_exp_ids.update(top_level_exp_ids[j])

        if len(positive_examples) != len(coverred_exp_ids.union(all_future_exp_ids)):
            logger.debug(
                f"Depth:{_current_depth}, Warning: no chances to cover all the positive examples already, early stopping (i={i}, len(coverred_exp_ids.union(all_future_exp_ids) = {len(coverred_exp_ids.union(all_future_exp_ids))}, len(positive_examples) = {len(positive_examples)}"
            )
            return []

        logger.debug(
            f"Depth:{_current_depth}, Current at the {i}-th sub-hypothesis, with size {len(exp_ids)}"
        )
        sub_positive_examples = [positive_examples[i] for i in exp_ids]
        sub_preconditions = find_precondition_from_single_group(
            sub_positive_examples,
            passing_neg_exps,
            traces=traces,
            keys_to_skip=keys_to_skip,
            _pruned_clauses=_pruned_clauses,
            _skip_pruning=True,
            _current_depth=_current_depth + 1,
        )
        if len(sub_preconditions) == 0:
            logger.warning(
                f"Warning: empty preconditions found in the sub-hypothesis at depth {_current_depth}"
            )
        else:
            # print(f"{i}-th Sub-preconditions at depth {_current_depth}")
            # print(str(sub_preconditions))
            coverred_exp_ids.update(exp_ids)

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
            # print(
            #     "Warning: sub-preconditions do not cover all the positive examples",
            #     len(positive_examples),
            # )
            # print("No precondition found for this sub-hypothesis")
            # print("Sub-preconditions")
            # for precond in preconditions:
            #     print(precond)

            # print("==============================")
            # print("Example")
            # print(exp)
            # print("Example Clauses")
            # print(_find_local_clauses(exp, key_to_skip=keys_to_skip))
            # print("==============================")

            # raise ValueError("Sub-preconditions do not cover all the positive examples")
            return []

    return preconditions
