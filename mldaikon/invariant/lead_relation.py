import logging
from itertools import combinations
from typing import Any, Dict, List, Set, Tuple

from tqdm import tqdm

from mldaikon.invariant.base_cls import (
    APIParam,
    CheckerResult,
    Example,
    ExampleList,
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
            paths.append(path.copy())
        path.pop()
        visited.remove(node)

    for start_node in start_nodes:
        dfs(start_node, [], set())

    print(paths)
    return paths


class FunctionLeadRelation(Relation):

    @staticmethod
    def infer(trace: Trace) -> list[Invariant]:
        """Infer Invariants for the FunctionCoverRelation."""

        logger = logging.getLogger(__name__)

        # 1. Pre-process all the events
        function_times: Dict[Tuple[str, str], Dict[str, Dict[str, Any]]] = {}
        function_id_map: Dict[Tuple[str, str], Dict[str, List[str]]] = {}

        events = trace.events

        function_pool_df = events.filter(
            (
                (
                    (events["function"].str.starts_with("torch.optim"))
                    | (events["function"].str.starts_with("torch.nn"))
                    | (events["function"].str.starts_with("torch.autograd"))
                )
                & (~events["function"].str.contains("._"))
            )
            | (events["function"].str.contains("step"))
        )

        function_pool = set(function_pool_df["function"].unique().to_list())

        with open("check_function_pool.txt", "w") as file:
            for function in function_pool:
                file.write(f"{function}\n")

        required_columns = {"function", "func_call_id", "type", "time"}
        if not required_columns.issubset(events.columns):
            raise ValueError(
                f"Missing column: {required_columns - set(events.columns)}"
            )

        group_by_events = events.group_by(["process_id", "thread_id"])

        for group_events in group_by_events:
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

        # 2. Check if two function on the same level for each thread and process
        def check_same_level(funcA: str, funcB: str, process_id: str, thread_id: str):
            if funcA == funcB:
                return False

            if funcA not in function_id_map[(process_id, thread_id)]:
                return False

            if funcB not in function_id_map[(process_id, thread_id)]:
                return False

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

        same_level_func: Dict[Tuple[str, str], Dict[str, Any]] = {}
        valid_relations: Dict[Tuple[str, str], bool] = {}

        for group_events in group_by_events:
            (process_id, thread_id), _ = group_events
            same_level_func[(process_id, thread_id)] = {}
            for funcA, funcB in combinations(function_pool, 2):
                if check_same_level(funcA, funcB, process_id, thread_id):
                    if funcA not in same_level_func[(process_id, thread_id)]:
                        same_level_func[(process_id, thread_id)][funcA] = []
                    same_level_func[(process_id, thread_id)][funcA].append(funcB)
                    valid_relations[(funcA, funcB)] = True

        # 3. Generating hypothesis
        group_name = "func_lead"
        hypothesis_with_examples = {
            (func_A, func_B): Hypothesis(
                invariant=Invariant(
                    relation=FunctionLeadRelation,
                    params=[
                        APIParam(func_A),
                        APIParam(func_B),
                    ],
                    precondition=None,
                    text_description=f"FunctionLeadRelation between {func_A} and {func_B}",
                ),
                positive_examples=ExampleList({group_name}),
                negative_examples=ExampleList({group_name}),
            )
            for (func_A, func_B), _ in valid_relations.items()
        }

        # 4. Add positive and negative examples
        for group_events in group_by_events:
            (process_id, thread_id), evs = group_events
            sorted_group_events = evs.sort("time")

            for (func_A, func_B), _ in valid_relations.items():

                if func_A not in same_level_func[(process_id, thread_id)]:
                    continue

                if func_B not in same_level_func[(process_id, thread_id)][func_A]:
                    continue

                flag_A = None
                # flag_B = None
                pre_record_A = []
                # pre_record_B = []

                for event in tqdm(sorted_group_events.iter_rows(named=True)):
                    if event["type"] != "function_call (pre)":
                        continue

                    if func_A == event["function"]:
                        if flag_A is None:
                            flag_A = event["time"]
                            # flag_B = None
                            pre_record_A = [event]
                            continue

                        valid_relations[(func_A, func_B)] = False
                        neg = Example()
                        neg.add_group("func_lead", pre_record_A)
                        hypothesis_with_examples[
                            (func_A, func_B)
                        ].negative_examples.add_example(neg)
                        pre_record_A = [event]
                        continue

                    if func_B == event["function"]:
                        # pre_record_B = [event]
                        # flag_B = event["time"]
                        if flag_A is None:
                            continue

                        pos = Example()
                        pos.add_group("func_lead", pre_record_A)
                        hypothesis_with_examples[
                            (func_A, func_B)
                        ].positive_examples.add_example(pos)

                        flag_A = None
                        pre_record_A = []

                if flag_A is not None:
                    flag_A = None
                    neg = Example()
                    neg.add_group("func_lead", pre_record_A)
                    hypothesis_with_examples[
                        (func_A, func_B)
                    ].negative_examples.add_example(neg)
                    pre_record_A = []

        # 5. Precondition inference
        brief_moode = False
        if_merge = True

        if not brief_moode:
            # Do complete precondition inference
            hypos_to_delete = []
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
                    hypos_to_delete.append(hypo)

            for hypo in hypos_to_delete:
                del hypothesis_with_examples[hypo]

            if not if_merge:
                return list(
                    [hypo.invariant for hypo in hypothesis_with_examples.values()]
                )

            # 6. Merge invariants
            relation_pool: Dict[
                GroupedPreconditions | None, List[Tuple[APIParam, APIParam]]
            ] = {}
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
                        relation=FunctionLeadRelation,
                        params=[param for param in merged_value],
                        precondition=key,
                        text_description="Merged FunctionLeadRelation in Ordered List",
                    )
                    merged_ininvariants.append(new_invariant)

            return merged_ininvariants

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
                        relation=FunctionLeadRelation,
                        params=[param for param in merged_value],
                        precondition=key,
                        text_description="Merged FunctionLeadRelation in Ordered List",
                    )
                    merged_ininvariants.append(new_invariant)

            return merged_ininvariants

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
        # assert len(inv.params) == 2, "Invariant should have exactly two parameters."

        assert inv.precondition is not None, "Invariant should have a precondition."

        invariant_length = len(inv.params)

        for i in range(invariant_length - 1):
            funcA = inv.params[i]
            funcB = inv.params[i + 1]

            assert isinstance(funcA, APIParam) and isinstance(
                funcB, APIParam
            ), "Invariant parameters should be string."

            all_functions = trace.get_func_names()

            if funcB not in all_functions:
                continue

            # check
            events = trace.events
            flag_A = None
            for event in events.iter_rows(named=True):

                if funcA == event["function"]:
                    if flag_A is None:
                        flag_A = event["time"]
                        # flag_B = None
                        continue

                    if inv.precondition.verify([trace], "func_lead"):
                        return CheckerResult(
                            trace=[event],
                            invariant=inv,
                            check_passed=False,
                        )

                if funcB == event["function"]:
                    # pre_record_B = [event]
                    # flag_B = event["time"]
                    if flag_A is None:
                        continue

                    flag_A = None

            if flag_A is not None:
                flag_A = None
                if inv.precondition.verify([trace], "func_lead"):
                    return CheckerResult(
                        trace=[event],
                        invariant=inv,
                        check_passed=False,
                    )

        return CheckerResult(
            trace=None,
            invariant=inv,
            check_passed=True,
        )
