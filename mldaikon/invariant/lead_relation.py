import logging
import re
from itertools import permutations
from typing import Any, Dict, Iterable, List, Set, Tuple

from tqdm import tqdm

from mldaikon.invariant.base_cls import (
    APIParam,
    CheckerResult,
    Example,
    ExampleList,
    FailedHypothesis,
    GroupedPreconditions,
    Hypothesis,
    IncompleteFuncCallEvent,
    Invariant,
    Relation,
)
from mldaikon.invariant.precondition import find_precondition
from mldaikon.trace.trace import Trace

EXP_GROUP_NAME = "func_lead"
FUNC_CALL_FILTERING_THRESHOLD = 100  # ideally this should be proportional to the number of training and testing iterations in the trace


def get_func_names_to_deal_with(trace: Trace) -> List[str]:
    """Get all functions in the trace."""
    function_pool: Set[str] = set()

    # get all functions in the trace
    all_func_names = trace.get_func_names()

    # filtering 1: remove private functions
    private_function_patterns = ["_.*"]
    for func_name in all_func_names:
        for pattern in private_function_patterns:
            if re.match(pattern, func_name):
                continue
        function_pool.add(func_name)

    # filtering 2: remove functions that occurred too many times
    func_occur_num = {
        func_name: len(trace.get_func_call_ids(func_name))
        for func_name in function_pool
    }
    for func_name, occur_num in func_occur_num.items():
        if occur_num > FUNC_CALL_FILTERING_THRESHOLD:
            function_pool.remove(func_name)

    return list(function_pool)


def get_func_data_per_PT(trace: Trace, function_pool: Iterable[str]):
    """
    Get
        1. all function timestamps per process and thread.
        2. all function ids per process and thread.
        3. all events per process and thread.

    # see below code for the structure of the return values

    """
    function_times: Dict[Tuple[str, str], Dict[str, Dict[str, Any]]] = (
        {}
    )  # map from (process_id, thread_id) to function call id to start and end time and function name
    function_id_map: Dict[Tuple[str, str], Dict[str, List[str]]] = (
        {}
    )  # map from (process_id, thread_id) to function name to function call ids
    listed_events: Dict[Tuple[str, str], List[dict[str, Any]]] = (
        {}
    )  # map from (process_id, thread_id) to all events
    # for all func_ids, get their corresponding events
    for func_name in function_pool:
        func_call_ids = trace.get_func_call_ids(func_name)
        for func_call_id in func_call_ids:
            event = trace.query_func_call_event(func_call_id)
            assert not isinstance(
                event, IncompleteFuncCallEvent
            ), "why would we hypothesize on incomplete events (incomplete func calls are typically outermost functions)?"
            process_id = event.pre_record["process_id"]
            thread_id = event.pre_record["thread_id"]

            # populate the function_times
            if (process_id, thread_id) not in function_times:
                function_times[(process_id, thread_id)] = {}

            function_times[(process_id, thread_id)][func_call_id] = {
                "start": event.pre_record["time"],
                "end": event.post_record["time"],
                "function": func_name,
            }

            # populate the function_id_map
            if (process_id, thread_id) not in function_id_map:
                function_id_map[(process_id, thread_id)] = {}
            if func_name not in function_id_map[(process_id, thread_id)]:
                function_id_map[(process_id, thread_id)][func_name] = []
            function_id_map[(process_id, thread_id)][func_name].append(func_call_id)

            # populate the listed_events
            if (process_id, thread_id) not in listed_events:
                listed_events[(process_id, thread_id)] = []
            listed_events[(process_id, thread_id)].extend(
                [event.pre_record, event.post_record]
            )

    # sort the listed_events
    for (process_id, thread_id), events_list in listed_events.items():
        listed_events[(process_id, thread_id)] = sorted(
            events_list, key=lambda x: x["time"]
        )

    return function_times, function_id_map, listed_events


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

    return paths


class FunctionLeadRelation(Relation):
    """FunctionLeadRelation is a relation that checks if one function covers another function.

    say function A and function B are two functions in the trace, we say function A covers function B when
    every time function A is called, a function B invocation follows.
    """

    @staticmethod
    def infer(trace: Trace) -> Tuple[List[Invariant], List[FailedHypothesis]]:
        """Infer Invariants for the FunctionCoverRelation."""

        logger = logging.getLogger(__name__)

        # 1. Pre-process all the events
        print("Start preprocessing....")
        function_times: Dict[Tuple[str, str], Dict[str, Dict[str, Any]]] = {}
        function_id_map: Dict[Tuple[str, str], Dict[str, List[str]]] = {}
        listed_events: Dict[Tuple[str, str], List[dict[str, Any]]] = {}
        function_pool: Set[Any] = set()

        # If the trace contains no function, safely exists infer process
        function_pool = set(get_func_names_to_deal_with(trace))
        if len(function_pool) == 0:
            logger.warning(
                "No relevant function calls found in the trace, skipping the analysis"
            )
            return [], []

        function_times, function_id_map, listed_events = get_func_data_per_PT(
            trace, function_pool
        )
        print("End preprocessing")

        # 2. Check if two function on the same level for each thread and process
        def check_same_level(funcA: str, funcB: str, process_id: str, thread_id: str):
            if funcA == funcB:
                return False

            if funcA not in function_id_map[(process_id, thread_id)]:
                return False

            if funcB not in function_id_map[(process_id, thread_id)]:
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

        for (process_id, thread_id), _ in tqdm(
            listed_events.items(), ascii=True, leave=True, desc="Groups Processed"
        ):
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
                positive_examples=ExampleList({EXP_GROUP_NAME}),
                negative_examples=ExampleList({EXP_GROUP_NAME}),
            )
            for (func_A, func_B), _ in valid_relations.items()
        }
        print("End generating hypo")

        # 4. Add positive and negative examples
        print("Start adding examples...")
        for (process_id, thread_id), events_list in tqdm(
            listed_events.items(), ascii=True, leave=True, desc="Group"
        ):

            for (func_A, func_B), _ in tqdm(
                valid_relations.items(),
                desc="Function Pair",
            ):

                if func_A not in same_level_func[(process_id, thread_id)]:
                    continue

                if func_B not in same_level_func[(process_id, thread_id)][func_A]:
                    continue

                flag_A = None
                # flag_B = None
                pre_record_A = []
                # pre_record_B = []

                for event in events_list:
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
                        neg.add_group(EXP_GROUP_NAME, pre_record_A)
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
                        pos.add_group(EXP_GROUP_NAME, pre_record_A)
                        hypothesis_with_examples[
                            (func_A, func_B)
                        ].positive_examples.add_example(pos)

                        flag_A = None
                        pre_record_A = []

                if flag_A is not None:
                    flag_A = None
                    neg = Example()
                    neg.add_group(EXP_GROUP_NAME, pre_record_A)
                    hypothesis_with_examples[
                        (func_A, func_B)
                    ].negative_examples.add_example(neg)
                    pre_record_A = []
        print("End adding examples")

        # 5. Precondition inference
        brief_moode = False
        if_merge = True

        failed_hypothesis = []

        if not brief_moode:
            # Do complete precondition inference
            print("Start precondition inference...")
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
                        relation=FunctionLeadRelation,
                        params=[param for param in merged_value],
                        precondition=key,
                        text_description="Merged FunctionLeadRelation in Ordered List",
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

        function_times: Dict[Tuple[str, str], Dict[str, Dict[str, Any]]] = {}
        function_id_map: Dict[Tuple[str, str], Dict[str, List[str]]] = {}
        listed_events: Dict[Tuple[str, str], List[dict[str, Any]]] = {}

        inv_triggered = False
        # If the trace contains no function, return vacuous true result
        func_names = trace.get_func_names()
        if len(func_names) == 0:
            print("No function calls found in the trace, skipping the checking")
            return CheckerResult(
                trace=None,
                invariant=inv,
                check_passed=True,
                triggered=False,
            )

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

        function_pool = list(set(function_pool).intersection(func_names))

        # YUXUAN ASK: if function_pool is not stictly subset of func_names, should we directly return false?

        if len(function_pool) == 0:
            print(
                "No relevant function calls found in the trace, skipping the checking"
            )
            return CheckerResult(
                trace=None,
                invariant=inv,
                check_passed=True,
                triggered=False,
            )

        function_times, function_id_map, listed_events = get_func_data_per_PT(
            trace, function_pool
        )

        def check_same_level(funcA: str, funcB: str, process_id: str, thread_id: str):
            if funcA == funcB:
                return False

            if funcA not in function_id_map[(process_id, thread_id)]:
                return False

            if funcB not in function_id_map[(process_id, thread_id)]:
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

            for (process_id, thread_id), events_list in listed_events.items():
                assert isinstance(process_id, str) and isinstance(thread_id, str)

                funcA = func_A.api_full_name
                funcB = func_B.api_full_name

                if not check_same_level(funcA, funcB, process_id, thread_id):
                    continue

                # check
                flag_A = None
                pre_recordA = None
                for event in events_list:

                    if event["type"] != "function_call (pre)":
                        continue

                    if funcA == event["function"]:
                        if flag_A is None:
                            flag_A = event["time"]
                            pre_recordA = event
                            continue
                        if inv.precondition.verify([events_list], EXP_GROUP_NAME):
                            inv_triggered = True
                            return CheckerResult(
                                trace=[pre_recordA, event],
                                invariant=inv,
                                check_passed=False,
                                triggered=True,
                            )
                    if funcB == event["function"]:
                        flag_A = None
                        pre_recordA = None

        return CheckerResult(
            trace=None,
            invariant=inv,
            check_passed=True,
            triggered=inv_triggered,
        )
