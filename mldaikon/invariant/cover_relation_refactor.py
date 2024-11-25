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
from mldaikon.invariant.lead_relation import (
    get_func_data_per_PT,
    get_func_names_to_deal_with,
)
from mldaikon.invariant.precondition import find_precondition
from mldaikon.trace.trace import Trace

EXP_GROUP_NAME = "func_cover"


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
        # for existing_path in paths[:]:
        #     if is_subset(existing_path, new_path):
        #         paths.remove(existing_path)
        #     if is_subset(new_path, existing_path):
        #         return
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


class FunctionCoverRelation(Relation):
    """FunctionCoverRelation is a relation that checks if one function covers another function.

    say function A and function B are two functions in the trace, we say function A covers function B when
    every time function B is called, a function A invocation exists before it.
    """

    @staticmethod
    def generate_hypothesis(trace) -> list[Hypothesis]:
        """Generate hypothesis for the FunctionCoverRelation on trace."""

        logger = logging.getLogger(__name__)

        # 1. Pre-process all the events
        print("Start preprocessing....")
        function_times: Dict[Tuple[str, str], Dict[str, Dict[str, Any]]] = {}
        function_id_map: Dict[Tuple[str, str], Dict[str, List[str]]] = {}
        listed_events: Dict[Tuple[str, str], List[dict[str, Any]]] = {}
        function_pool: Set[Any] = set()

        # If the trace contains no function, return []
        function_pool = set(get_func_names_to_deal_with(trace))
        if len(function_pool) == 0:
            logger.warning(
                "No relevant function calls found in the trace, skipping the analysis"
            )
            return []

        function_times, function_id_map, listed_events = get_func_data_per_PT(
            trace, function_pool
        )
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
                    relation=FunctionCoverRelation,
                    params=[
                        APIParam(func_A),
                        APIParam(func_B),
                    ],
                    precondition=None,
                    text_description=f"FunctionCoverRelation between {func_A} and {func_B}",
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
                flag_B = None
                pre_record_A = []
                pre_record_B = []

                for event in events_list:
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
                            neg.add_group(EXP_GROUP_NAME, pre_record_B)
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
                            neg.add_group(EXP_GROUP_NAME, [event])
                            hypothesis_with_examples[
                                (func_A, func_B)
                            ].negative_examples.add_example(neg)
                        else:
                            pos = Example()
                            pos.add_group(EXP_GROUP_NAME, pre_record_A)
                            hypothesis_with_examples[
                                (func_A, func_B)
                            ].positive_examples.add_example(pos)

                        pre_record_B = [event]
        print("End adding examples")

        return list(hypothesis_with_examples.values())

    @staticmethod
    def collect_examples(trace, hypothesis):
        """Generate examples for a hypothesis on trace."""
        inv = hypothesis.invariant

        function_times: Dict[Tuple[str, str], Dict[str, Dict[str, Any]]] = {}
        function_id_map: Dict[Tuple[str, str], Dict[str, List[str]]] = {}
        listed_events: Dict[Tuple[str, str], List[dict[str, Any]]] = {}

        # If the trace contains no function, return orginal hypothesis
        func_names = trace.get_func_names()
        if len(func_names) == 0:
            print("No function calls found in the trace, skipping the collecting")
            return

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

        if len(function_pool) == 0:
            print(
                "No relevant function calls found in the trace, skipping the collecting"
            )
            return

        print("Start fetching data for collecting...")
        function_times, function_id_map, listed_events = get_func_data_per_PT(
            trace, function_pool
        )
        print("End fetching data for collecting...")

        def check_same_level(funcA: str, funcB: str, process_id, thread_id):
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

        print("Starting collecting iteration...")
        for i in tqdm(range(invariant_length - 1)):
            func_A = inv.params[i]
            func_B = inv.params[i + 1]

            assert isinstance(func_A, APIParam) and isinstance(
                func_B, APIParam
            ), "Invariant parameters should be string."

            for (process_id, thread_id), events_list in listed_events.items():
                funcA = func_A.api_full_name
                funcB = func_B.api_full_name

                if not check_same_level(funcA, funcB, process_id, thread_id):
                    continue

                # check
                flag_A = None
                flag_B = None
                pre_record_A = []
                pre_record_B = []

                for event in events_list:
                    if event["type"] != "function_call (pre)":
                        continue

                    if funcA == event["function"]:
                        flag_A = event["time"]
                        flag_B = None
                        pre_record_A = [event]

                    if funcB == event["function"]:
                        if flag_B is not None:
                            neg = Example()
                            neg.add_group(EXP_GROUP_NAME, pre_record_B)
                            hypothesis.negative_examples.add_example(neg)
                            pre_record_B = [event]
                            flag_B = event["time"]
                            continue

                        flag_B = event["time"]
                        if flag_A is None:
                            neg = Example()
                            neg.add_group(EXP_GROUP_NAME, [event])
                            hypothesis.negative_examples.add_example(neg)
                        else:
                            pos = Example()
                            pos.add_group(EXP_GROUP_NAME, pre_record_A)
                            hypothesis.positive_examples.add_example(pos)

                        pre_record_B = [event]

        print("End collecting iteration...")

    @staticmethod
    def infer(trace: Trace) -> Tuple[List[Invariant], List[FailedHypothesis]]:
        """Infer Invariants for the FunctionCoverRelation."""

        all_hypotheses = FunctionCoverRelation.generate_hypothesis(trace)

        # for hypothesis in all_hypotheses:
        #     FunctionCoverRelation.collect_examples(trace, hypothesis)

        if_merge = True

        print("Start precondition inference...")
        failed_hypothesis = []
        for hypothesis in all_hypotheses.copy():
            preconditions = find_precondition(
                hypothesis, trace
            )
            if preconditions is not None:
                hypothesis.invariant.precondition = (
                    preconditions
                )
            else:
                failed_hypothesis.append(
                    FailedHypothesis(hypothesis)
                )
                all_hypotheses.remove(hypothesis)
        print("End precondition inference")

        if not if_merge:
            return (
                list(
                    [hypo.invariant for hypo in all_hypotheses]
                ),
                failed_hypothesis,
            )
        print("End precondition inference")

            # 6. Merge invariants
        print("Start merging invariants...")
        relation_pool: Dict[
            GroupedPreconditions | None, List[Tuple[APIParam, APIParam]]
        ] = {}
        # relation_pool contains all binary relations classified by GroupedPreconditions (key)
        for hypothesis in all_hypotheses:
            param0 = hypothesis.invariant.params[0]
            param1 = hypothesis.invariant.params[1]

            assert(isinstance(param0, APIParam) and isinstance(param1, APIParam))
            if (
                hypothesis.invariant.precondition
                not in relation_pool
            ):
                relation_pool[
                    hypothesis.invariant.precondition
                ] = []
            relation_pool[
                hypothesis.invariant.precondition
            ].append((param0, param1))

        merged_relations: Dict[
            GroupedPreconditions | None, List[List[APIParam]]
        ] = {}

        for key, values in tqdm(relation_pool.items(), desc="Merging Invariants"):
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

        print("Start fetching data for checking...")
        function_times, function_id_map, listed_events = get_func_data_per_PT(
            trace, function_pool
        )
        print("End fetching data for checking...")

        def check_same_level(funcA: str, funcB: str, process_id, thread_id):
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

        print("Starting checking iteration...")
        for i in tqdm(range(invariant_length - 1)):
            func_A = inv.params[i]
            func_B = inv.params[i + 1]

            assert isinstance(func_A, APIParam) and isinstance(
                func_B, APIParam
            ), "Invariant parameters should be string."

            for (process_id, thread_id), events_list in listed_events.items():
                funcA = func_A.api_full_name
                funcB = func_B.api_full_name

                if not check_same_level(funcA, funcB, process_id, thread_id):
                    continue

                # check
                flag_B = None
                pre_recordB = None
                for event in events_list:
                    if event["type"] != "function_call (pre)":
                        continue

                    if funcA == event["function"]:
                        flag_B = None
                        pre_recordB = None

                    if funcB == event["function"]:
                        if flag_B is not None:
                            if inv.precondition.verify([events_list], EXP_GROUP_NAME):
                                inv_triggered = True
                                print(
                                    "The relation "
                                    + funcA
                                    + " covers "
                                    + funcB
                                    + " is violated!\n"
                                )
                                return CheckerResult(
                                    trace=[pre_recordB, event],
                                    invariant=inv,
                                    check_passed=False,
                                    triggered=True,
                                )
                        flag_B = event["time"]
                        pre_recordB = event

        # FIXME: triggered is always False for passing invariants
        return CheckerResult(
            trace=None,
            invariant=inv,
            check_passed=True,
            triggered=inv_triggered,
        )

    @staticmethod
    def get_precondition_infer_keys_to_skip(hypothesis: Hypothesis) -> list[str]:
        return []
