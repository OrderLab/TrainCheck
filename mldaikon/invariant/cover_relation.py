"""TODOs @Boyu:
1. Implementation Clean-up for both Cover and Lead Relations
    1. add comments to certain variable names as they are a bit unclear.
"""

import logging
from itertools import permutations
from typing import Any, Dict, List, Set, Tuple

from tqdm import tqdm

from mldaikon.invariant.base_cls import (
    APIParam,
    CheckerResult,
    Example,
    ExampleList,
    FailedHypothesis,
    GroupedPreconditions,
    Hypothesis,
    Invariant,
    Relation,
)
from mldaikon.invariant.precondition import find_precondition
from mldaikon.trace.trace import Trace


def is_complete_subgraph(
    path: List[APIParam], new_node: APIParam, graph: Dict[APIParam, List[APIParam]]
) -> bool:
    """Check if adding new_node to path forms a complete (directed) graph."""
    for node in path:
        if new_node not in graph[node]:
            return False
    return True


def merge_relations(pairs: List[Tuple[APIParam, APIParam]]) -> List[List[APIParam]]:
    graph: Dict[APIParam, List[APIParam]] = {}
    indegree: Dict[APIParam, int] = {}

    for a, b in pairs:
        if a not in graph:
            graph[a] = []
        if b not in graph:
            graph[b] = []
        graph[a].append(b)

        if b in indegree:
            indegree[b] += 1
        else:
            indegree[b] = 1

        if a not in indegree:
            indegree[a] = 0

    start_nodes: List[APIParam] = [node for node in indegree if indegree[node] == 0]

    paths: List[List[APIParam]] = []

    def is_subset(path1: List[APIParam], path2: List[APIParam]) -> bool:
        return set(path1).issubset(set(path2))

    def add_path(new_path: List[APIParam]) -> None:
        nonlocal paths
        for existing_path in paths[:]:
            if is_subset(existing_path, new_path):
                paths.remove(existing_path)
            if is_subset(new_path, existing_path):
                return
        paths.append(new_path)

    def dfs(node: APIParam, path: List[APIParam], visited: Set[APIParam]) -> None:
        path.append(node)
        visited.add(node)
        if node in graph:
            for neighbor in graph[node]:
                if neighbor not in visited and is_complete_subgraph(
                    path, neighbor, graph
                ):
                    dfs(neighbor, path, visited)
        if not graph.get(node):
            add_path(path.copy())
        path.pop()
        visited.remove(node)

    for start_node in start_nodes:
        dfs(start_node, [], set())

    # print(paths)
    return paths


class FunctionCoverRelation(Relation):

    @staticmethod
    def infer(trace: Trace) -> Tuple[List[Invariant], List[FailedHypothesis]]:
        """Infer Invariants for the FunctionCoverRelation."""

        logger = logging.getLogger(__name__)

        # 1. Pre-process all the events
        print("Start preprocessing....")
        function_times: Dict[Tuple[str, str], Dict[str, Dict[str, Any]]] = {}
        function_id_map: Dict[Tuple[str, str], Dict[str, List[str]]] = {}

        # If the trace contains no function, safely exists infer process
        func_names = trace.get_func_names()
        if len(func_names) == 0:
            logger.warning(
                "No function calls found in the trace, skipping the analysis"
            )
            return [], []

        events = trace.get_filtered_function()
        function_pool = set(
            events["function"].unique().to_list()
        )  # All filtered function names

        with open("check_function_pool.txt", "w") as file:
            for function in function_pool:
                file.write(f"{function}\n")

        required_columns = {"function", "func_call_id", "type", "time"}
        if not required_columns.issubset(events.columns):
            raise ValueError(
                f"Missing column: {required_columns - set(events.columns)}"
            )

        group_by_events = events.group_by(["process_id", "thread_id"])

        for group_events in tqdm(group_by_events):
            (process_id, thread_id), evs = group_events
            sorted_group_events = evs.sort("time")
            if (process_id, thread_id) not in function_id_map:
                function_id_map[(process_id, thread_id)] = {}

            if (process_id, thread_id) not in function_times:
                function_times[(process_id, thread_id)] = {}

            for event in sorted_group_events.iter_rows(named=True):
                if event["function"] in function_pool:
                    if (
                        event["function"]
                        not in function_id_map[(process_id, thread_id)]
                    ):
                        function_id_map[(process_id, thread_id)][event["function"]] = []
                    func_id = event["func_call_id"]
                    function_id_map[(process_id, thread_id)][event["function"]].append(
                        func_id
                    )

                    if event["type"] == "function_call (pre)":
                        if func_id not in function_times[(process_id, thread_id)]:
                            function_times[(process_id, thread_id)][func_id] = {}
                        function_times[(process_id, thread_id)][func_id]["start"] = (
                            event["time"]
                        )
                        function_times[(process_id, thread_id)][func_id]["function"] = (
                            event["function"]
                        )
                    elif event["type"] in [
                        "function_call (post)",
                        "function_call (post) (exception)",
                    ]:
                        function_times[(process_id, thread_id)][func_id]["end"] = event[
                            "time"
                        ]
        print("End preprocessing")

        # 2. Check if two function on the same level for each thread and process
        def check_same_level(funcA: str, funcB: str, process_id: str, thread_id: str):
            if funcA == funcB:
                return False

            if funcB not in function_id_map[(process_id, thread_id)]:
                return False

            if funcA not in function_id_map[(process_id, thread_id)]:
                return True

            for idA in function_id_map[(process_id, thread_id)][funcA]:
                for idB in function_id_map[(process_id, thread_id)][funcB]:
                    preA = function_times[(process_id, thread_id)][idA]["start"]
                    postA = function_times[(process_id, thread_id)][idA]["end"]
                    preB = function_times[(process_id, thread_id)][idB]["start"]
                    postB = function_times[(process_id, thread_id)][idB]["end"]
                    if preB >= postA:
                        break
                    if postB <= preA:
                        continue
                    return False
            return True

        print("Start same level checking...")
        same_level_func: Dict[Tuple[str, str], Dict[str, Any]] = {}
        valid_relations: Dict[Tuple[str, str], bool] = {}

        for group_events in tqdm(
            group_by_events, ascii=True, leave=True, desc="Groups Processed"
        ):
            (process_id, thread_id), _ = group_events
            same_level_func[(process_id, thread_id)] = {}
            for funcA, funcB in tqdm(
                permutations(function_pool, 2),
                ascii=True,
                leave=True,
                desc="Combinations Checked",
            ):
                if check_same_level(funcA, funcB, process_id, thread_id):
                    if funcA not in same_level_func[(process_id, thread_id)]:
                        same_level_func[(process_id, thread_id)][funcA] = []
                    same_level_func[(process_id, thread_id)][funcA].append(funcB)
                    valid_relations[(funcA, funcB)] = True
        print("End same level checking")

        # 3. Generating hypothesis
        print("Start generating hypo...")
        group_name = "func_cover"
        hypothesis_with_examples = {
            (func_A, func_B): Hypothesis(
                invariant=Invariant(
                    relation=FunctionCoverRelation,
                    params=[
                        APIParam(func_A),
                        APIParam(func_B),
                    ],
                    precondition=None,
                    text_description=f"FunctionCoverRelation between {func_A} and {func_B}",
                ),
                positive_examples=ExampleList({group_name}),
                negative_examples=ExampleList({group_name}),
            )
            for (func_A, func_B), _ in valid_relations.items()
        }
        print("End generating hypo")

        # 4. Add positive and negative examples
        print("Start adding examples...")
        for group_events in tqdm(group_by_events, ascii=True, leave=True, desc="Group"):
            (process_id, thread_id), evs = group_events
            sorted_group_events = evs.sort("time")

            for (func_A, func_B), _ in tqdm(
                valid_relations.items(), ascii=True, leave=True, desc="Function Pair"
            ):

                if func_A not in same_level_func[(process_id, thread_id)]:
                    continue

                if func_B not in same_level_func[(process_id, thread_id)][func_A]:
                    continue

                flag_A = None
                flag_B = None
                pre_record_A = []
                pre_record_B = []

                for event in sorted_group_events.iter_rows(named=True):
                    if event["type"] != "function_call (pre)":
                        continue

                    if func_A == event["function"]:
                        flag_A = event["time"]
                        flag_B = None
                        pre_record_A = [event]

                    if func_B == event["function"]:
                        if flag_B is not None:
                            valid_relations[(func_A, func_B)] = False
                            neg = Example()
                            neg.add_group(group_name, pre_record_B)
                            hypothesis_with_examples[
                                (func_A, func_B)
                            ].negative_examples.add_example(neg)
                            pre_record_B = [event]
                            flag_B = event["time"]
                            continue

                        flag_B = event["time"]
                        if flag_A is None:
                            valid_relations[(func_A, func_B)] = False
                            neg = Example()
                            neg.add_group(group_name, [event])
                            hypothesis_with_examples[
                                (func_A, func_B)
                            ].negative_examples.add_example(neg)
                        else:
                            pos = Example()
                            pos.add_group(group_name, pre_record_A)
                            hypothesis_with_examples[
                                (func_A, func_B)
                            ].positive_examples.add_example(pos)

                        pre_record_B = [event]
        print("End adding examples")

        # 5. Precondition inference
        brief_moode = False
        if_merge = True

        failed_hypothesis = []
        if not brief_moode:
            # Do complete precondition inference
            print("Start precondition inference...")
            hypos_to_delete: list[tuple[str, str]] = []
            for hypo in hypothesis_with_examples:
                logger.debug(
                    f"Finding Precondition for {hypo}: {hypothesis_with_examples[hypo].invariant.text_description}"
                )
                preconditions = find_precondition(hypothesis_with_examples[hypo])
                logger.debug(f"Preconditions for {hypo}:\n{str(preconditions)}")

                if preconditions is not None:
                    hypothesis_with_examples[hypo].invariant.precondition = (
                        preconditions
                    )
                else:
                    logger.debug(f"Precondition not found for {hypo}")
                    failed_hypothesis.append(
                        FailedHypothesis(hypothesis_with_examples[hypo])
                    )
                    hypos_to_delete.append(hypo)

            for hypo in hypos_to_delete:
                # remove key from hypothesis_with_examples
                hypothesis_with_examples.pop(hypo)

            if not if_merge:
                return (
                    list(
                        [hypo.invariant for hypo in hypothesis_with_examples.values()]
                    ),
                    failed_hypothesis,
                )
            print("End precondition inference")

            # 6. Merge invariants
            print("Start merging invariants...")
            relation_pool: Dict[
                GroupedPreconditions | None, List[Tuple[APIParam, APIParam]]
            ] = (
                {}
            )  # relation_pool contains all binary relations classified by GroupedPreconditions (key)

            for hypo in hypothesis_with_examples:
                if (
                    hypothesis_with_examples[hypo].invariant.precondition
                    not in relation_pool
                ):
                    relation_pool[
                        hypothesis_with_examples[hypo].invariant.precondition
                    ] = []
                relation_pool[
                    hypothesis_with_examples[hypo].invariant.precondition
                ].append((APIParam(hypo[0]), APIParam(hypo[1])))

            merged_relations: Dict[
                GroupedPreconditions | None, List[List[APIParam]]
            ] = {}

            for key, values in relation_pool.items():
                merged_relations[key] = merge_relations(values)

            merged_ininvariants = []

            for key, merged_values in merged_relations.items():
                for merged_value in merged_values:
                    new_invariant = Invariant(
                        relation=FunctionCoverRelation,
                        params=[param for param in merged_value],
                        precondition=key,
                        text_description="Merged FunctionCoverRelation in Ordered List",
                    )
                    merged_ininvariants.append(new_invariant)
            print("End merging invariants")

            return merged_ininvariants, failed_hypothesis

        else:

            def dp_merge(
                pair: Tuple[APIParam, APIParam],
                pairs: List[Tuple[APIParam, APIParam]],
                precondition_cache: Dict[
                    Tuple[APIParam, APIParam], GroupedPreconditions | None
                ],
                sequence_cache: Dict[Tuple[APIParam, APIParam], Dict[str, Any]],
            ):
                a, b = pair

                if pair in sequence_cache:
                    return sequence_cache[pair]

                current_sequence = [a, b]

                if pair not in precondition_cache:
                    precondition_cache[pair] = find_precondition(
                        hypothesis_with_examples[(a.api_full_name, b.api_full_name)]
                    )

                current_precondition = precondition_cache[pair]

                if current_precondition is None:
                    pairs.remove(pair)
                    failed_hypothesis.append(
                        FailedHypothesis(
                            hypothesis_with_examples[(a.api_full_name, b.api_full_name)]
                        )
                    )
                    return None

                for next_pair in pairs[:]:
                    if next_pair[0] == b:
                        if next_pair not in precondition_cache:
                            precondition_cache[next_pair] = find_precondition(
                                hypothesis_with_examples[
                                    (
                                        next_pair[0].api_full_name,
                                        next_pair[1].api_full_name,
                                    )
                                ]
                            )

                        next_precondition = precondition_cache[next_pair]

                        if current_precondition == next_precondition:
                            result = dp_merge(
                                next_pair, pairs, precondition_cache, sequence_cache
                            )
                            merged_sequence = result["sequence"]
                            if merged_sequence is not None:
                                current_sequence.extend(merged_sequence[1:])

                sequence_cache[pair] = {}
                sequence_cache[pair]["sequence"] = current_sequence
                sequence_cache[pair]["precondition"] = current_precondition

                # Add pruning logic
                for i in range(len(current_sequence) - 1):
                    for j in range(i + 1, len(current_sequence)):
                        sub_pair = (current_sequence[i], current_sequence[j])
                        if sub_pair not in sequence_cache:
                            sub_sequence = []
                            sub_sequence.append(current_sequence[i])
                            sub_sequence.extend(current_sequence[j:])
                            sequence_cache[sub_pair] = {}
                            sequence_cache[sub_pair]["sequence"] = sub_sequence
                            sequence_cache[sub_pair][
                                "precondition"
                            ] = current_precondition

                return sequence_cache[pair]

            pairs: List[Tuple[APIParam, APIParam]] = [
                (APIParam(hypo[0]), APIParam(hypo[1]))
                for hypo in hypothesis_with_examples
            ]

            merged_sequences: Dict[
                GroupedPreconditions | None, List[List[APIParam]]
            ] = {}
            precondition_cache: Dict[
                Tuple[APIParam, APIParam], GroupedPreconditions | None
            ] = {}
            sequence_cache: Dict[Tuple[APIParam, APIParam], Dict[str, Any]] = {}

            for pair in pairs[:]:
                if pair not in sequence_cache:
                    result = dp_merge(pair, pairs, precondition_cache, sequence_cache)
                    if result is not None:
                        merged_sequence = result["sequence"]
                        precondition = result["precondition"]
                        if precondition not in merged_sequences:
                            merged_sequences[precondition] = []
                        merged_sequences[precondition].append(merged_sequence)

            merged_ininvariants = []

            for key, merged_values in merged_sequences.items():
                for merged_value in merged_values:
                    new_invariant = Invariant(
                        relation=FunctionCoverRelation,
                        params=[param for param in merged_value],
                        precondition=key,
                        text_description="Merged FunctionCoverRelation in Ordered List",
                    )
                    merged_ininvariants.append(new_invariant)

            return merged_ininvariants, failed_hypothesis

    @staticmethod
    def evaluate(value_group: list) -> bool:
        """Given a group of values, should return a boolean value
        indicating whether the relation holds or not.

        args:
            value_group: list
                A list of values to evaluate the relation on. The length of the list
                should be equal to the number of variables in the relation.
        """
        return True

    @staticmethod
    def static_check_all(
        trace: Trace, inv: Invariant, check_relation_first: bool
    ) -> CheckerResult:
        """Given a trace and an invariant, should return a boolean value
        indicating whether the invariant holds on the trace.

        args:
            trace: Trace
                A trace to check the invariant on.
            inv: Invariant
                The invariant to check on the trace.
        """
        assert inv.precondition is not None, "Invariant should have a precondition."

        # If the trace contains no function, return vacuous true result
        func_names = trace.get_func_names()
        if len(func_names) == 0:
            print("No function calls found in the trace, skipping the checking")
            return CheckerResult(
                trace=None,
                invariant=inv,
                check_passed=True,
            )

        events = trace.get_filtered_function()

        function_pool = (
            []
        )  # Here function_pool only contains functions existing in given invariant

        invariant_length = len(inv.params)
        for i in range(invariant_length):
            func = inv.params[i]
            assert isinstance(
                func, APIParam
            ), "Invariant parameters should be APIParam."
            function_pool.append(func.api_full_name)

        required_columns = {"function", "func_call_id", "type", "time"}
        if not required_columns.issubset(events.columns):
            raise ValueError(
                f"Missing column: {required_columns - set(events.columns)}"
            )

        function_times: Dict[Tuple[str, str], Dict[str, Dict[str, Any]]] = {}
        function_id_map: Dict[Tuple[str, str], Dict[str, List[str]]] = {}

        group_by_events = events.group_by(["process_id", "thread_id"])

        for group_events in group_by_events:
            (process_id, thread_id), evs = group_events
            assert isinstance(process_id, str) and isinstance(thread_id, str)
            sorted_group_events = evs.sort("time")
            if (process_id, thread_id) not in function_id_map:
                function_id_map[(process_id, thread_id)] = {}

            if (process_id, thread_id) not in function_times:
                function_times[(process_id, thread_id)] = {}

            for event in sorted_group_events.iter_rows(named=True):
                if event["function"] in function_pool:
                    if (
                        event["function"]
                        not in function_id_map[(process_id, thread_id)]
                    ):
                        function_id_map[(process_id, thread_id)][event["function"]] = []
                    func_id = event["func_call_id"]
                    function_id_map[(process_id, thread_id)][event["function"]].append(
                        func_id
                    )

                    if event["type"] == "function_call (pre)":
                        if func_id not in function_times[(process_id, thread_id)]:
                            function_times[(process_id, thread_id)][func_id] = {}
                        function_times[(process_id, thread_id)][func_id]["start"] = (
                            event["time"]
                        )
                        function_times[(process_id, thread_id)][func_id]["function"] = (
                            event["function"]
                        )
                    elif event["type"] in [
                        "function_call (post)",
                        "function_call (post) (exception)",
                    ]:
                        function_times[(process_id, thread_id)][func_id]["end"] = event[
                            "time"
                        ]

        def check_same_level(funcA: str, funcB: str, process_id: str, thread_id: str):
            if funcA == funcB:
                return False

            if funcB not in function_id_map[(process_id, thread_id)]:
                return False

            if funcA not in function_id_map[(process_id, thread_id)]:
                return True

            for idA in function_id_map[(process_id, thread_id)][funcA]:
                for idB in function_id_map[(process_id, thread_id)][funcB]:
                    preA = function_times[(process_id, thread_id)][idA]["start"]
                    postA = function_times[(process_id, thread_id)][idA]["end"]
                    preB = function_times[(process_id, thread_id)][idB]["start"]
                    postB = function_times[(process_id, thread_id)][idB]["end"]
                    if preB >= postA:
                        break
                    if postB <= preA:
                        continue
                    return False
            return True

        for i in range(invariant_length - 1):
            func_A = inv.params[i]
            func_B = inv.params[i + 1]

            assert isinstance(func_A, APIParam) and isinstance(
                func_B, APIParam
            ), "Invariant parameters should be string."

            for group_events in group_by_events:
                (process_id, thread_id), evs = group_events
                assert isinstance(process_id, str) and isinstance(thread_id, str)
                sorted_group_events = evs.sort("time")

                funcA = func_A.api_full_name
                funcB = func_B.api_full_name

                if not check_same_level(funcA, funcB, process_id, thread_id):
                    continue

                # check
                # flag_A = None
                flag_B = None
                pre_recordB = None
                for event in sorted_group_events.iter_rows(named=True):
                    if event["type"] != "function_call (pre)":
                        continue

                    if funcA == event["function"]:
                        # flag_A = event["time"]
                        flag_B = None
                        pre_recordB = None

                    if funcB == event["function"]:
                        if flag_B is not None:
                            if inv.precondition.verify([events], "func_cover"):
                                return CheckerResult(
                                    trace=[pre_recordB, event],
                                    invariant=inv,
                                    check_passed=False,
                                )

                        flag_B = event["time"]
                        pre_recordB = event

                        # if flag_A is None:
                        #     if inv.precondition.verify([events], "func_cover"):
                        #         return CheckerResult(
                        #             trace=[event],
                        #             invariant=inv,
                        #             check_passed=False,
                        #         )

        return CheckerResult(
            trace=None,
            invariant=inv,
            check_passed=True,
        )
