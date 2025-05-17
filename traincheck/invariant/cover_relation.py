import logging
from itertools import permutations
from typing import Any, Dict, List, Set, Tuple

from tqdm import tqdm

from traincheck.invariant.base_cls import (
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
from traincheck.invariant.lead_relation import (
    check_same_level,
    get_func_data_per_PT,
    get_func_names_to_deal_with,
)
from traincheck.invariant.precondition import find_precondition
from traincheck.trace.trace import Trace
from traincheck.trace.trace_pandas import TracePandas

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
        assert isinstance(trace, TracePandas)
        if trace.function_pool is not None:
            function_pool = trace.function_pool
        else:
            function_pool = set(get_func_names_to_deal_with(trace))
            trace.function_pool = function_pool

        if len(function_pool) == 0:
            logger.warning(
                "No relevant function calls found in the trace, skipping the analysis"
            )
            return []
        if (
            trace.function_times is not None
            and trace.function_id_map is not None
            and trace.listed_events is not None
        ):
            function_times = trace.function_times
            function_id_map = trace.function_id_map
            listed_events = trace.listed_events
        else:
            function_times, function_id_map, listed_events = get_func_data_per_PT(
                trace, function_pool
            )
            trace.function_times = function_times
            trace.function_id_map = function_id_map
            trace.listed_events = listed_events
        print("End preprocessing")

        print("Start same level checking...")
        same_level_func: Dict[Tuple[str, str], Dict[str, Any]] = {}
        valid_relations: Dict[Tuple[str, str], bool] = {}

        if (
            trace.same_level_func_cover is not None
            and trace.valid_relations_cover is not None
        ):
            same_level_func = trace.same_level_func_cover
            valid_relations = trace.valid_relations_cover
        else:
            for (process_id, thread_id), _ in tqdm(
                listed_events.items(), ascii=True, leave=True, desc="Groups Processed"
            ):
                same_level_func[(process_id, thread_id)] = {}
                for func_A, func_B in tqdm(
                    permutations(function_pool, 2),
                    ascii=True,
                    leave=True,
                    desc="Combinations Checked",
                    total=len(function_pool) ** 2,
                ):
                    if check_same_level(
                        func_A,
                        func_B,
                        process_id,
                        thread_id,
                        function_id_map,
                        function_times,
                    ):
                        if func_A not in same_level_func[(process_id, thread_id)]:
                            same_level_func[(process_id, thread_id)][func_A] = []
                        same_level_func[(process_id, thread_id)][func_A].append(func_B)
                        valid_relations[(func_A, func_B)] = True
            trace.same_level_func_cover = same_level_func
            trace.valid_relations_cover = valid_relations
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

                if func_B not in same_level_func[(process_id, thread_id)]:
                    continue

                if func_A not in same_level_func[(process_id, thread_id)][func_B]:
                    # all B invocations are negative examples
                    for event in events_list:
                        if (
                            event["type"] == "function_call (pre)"
                            and event["function"] == func_B
                        ):
                            example = Example()
                            example.add_group(EXP_GROUP_NAME, [event])
                            hypothesis_with_examples[
                                (func_A, func_B)
                            ].negative_examples.add_example(example)
                    continue

                flag_A = None
                for event in events_list:
                    if event["type"] != "function_call (pre)":
                        continue

                    if func_A == event["function"]:
                        flag_A = event["time"]

                    if func_B == event["function"]:
                        example = Example()
                        example.add_group(EXP_GROUP_NAME, [event])
                        if flag_A is None:
                            hypothesis_with_examples[
                                (func_A, func_B)
                            ].negative_examples.add_example(example)
                        else:
                            hypothesis_with_examples[
                                (func_A, func_B)
                            ].positive_examples.add_example(example)
                            flag_A = None  # reset flag_A

        print("End adding examples")

        return list(hypothesis_with_examples.values())

    @staticmethod
    def collect_examples(trace, hypothesis):
        """Generate examples for a hypothesis on trace."""

        logger = logging.getLogger(__name__)

        # 1. Pre-process all the events
        print("Start preprocessing....")
        function_times: Dict[Tuple[str, str], Dict[str, Dict[str, Any]]] = {}
        function_id_map: Dict[Tuple[str, str], Dict[str, List[str]]] = {}
        listed_events: Dict[Tuple[str, str], List[dict[str, Any]]] = {}
        function_pool: Set[Any] = set()

        # If the trace contains no function, return []
        assert isinstance(trace, TracePandas)
        if trace.function_pool is not None:
            function_pool = trace.function_pool
        else:
            function_pool = set(get_func_names_to_deal_with(trace))
            trace.function_pool = function_pool

        if len(function_pool) == 0:
            logger.warning(
                "No relevant function calls found in the trace, skipping the analysis"
            )
            return
        if (
            trace.function_times is not None
            and trace.function_id_map is not None
            and trace.listed_events is not None
        ):
            function_times = trace.function_times
            function_id_map = trace.function_id_map
            listed_events = trace.listed_events
        else:
            function_times, function_id_map, listed_events = get_func_data_per_PT(
                trace, function_pool
            )
            trace.function_times = function_times
            trace.function_id_map = function_id_map
            trace.listed_events = listed_events
        print("End preprocessing")

        print("Start same level checking...")
        same_level_func: Dict[Tuple[str, str], Dict[str, Any]] = {}
        valid_relations: Dict[Tuple[str, str], bool] = {}

        if (
            trace.same_level_func_cover is not None
            and trace.valid_relations_cover is not None
        ):
            same_level_func = trace.same_level_func_cover
            valid_relations = trace.valid_relations_cover
        else:
            for (process_id, thread_id), _ in tqdm(
                listed_events.items(), ascii=True, leave=True, desc="Groups Processed"
            ):
                same_level_func[(process_id, thread_id)] = {}
                for func_A, func_B in tqdm(
                    permutations(function_pool, 2),
                    ascii=True,
                    leave=True,
                    desc="Combinations Checked",
                    total=len(function_pool) ** 2,
                ):
                    if check_same_level(
                        func_A,
                        func_B,
                        process_id,
                        thread_id,
                        function_id_map,
                        function_times,
                    ):
                        if func_A not in same_level_func[(process_id, thread_id)]:
                            same_level_func[(process_id, thread_id)][func_A] = []
                        same_level_func[(process_id, thread_id)][func_A].append(func_B)
                        valid_relations[(func_A, func_B)] = True
            trace.same_level_func_cover = same_level_func
            trace.valid_relations_cover = valid_relations
        print("End same level checking")

        inv = hypothesis.invariant

        function_pool_temp = []

        invariant_length = len(inv.params)
        for i in range(invariant_length):
            func = inv.params[i]
            assert isinstance(
                func, APIParam
            ), "Invariant parameters should be APIParam."
            function_pool_temp.append(func.api_full_name)

        function_pool = list(set(function_pool).intersection(function_pool_temp))

        if len(function_pool) == 0:
            print(
                "No relevant function calls found in the trace, skipping the collecting"
            )
            return

        print("Starting collecting iteration...")
        for i in tqdm(range(invariant_length - 1)):
            param_A = inv.params[i]
            param_B = inv.params[i + 1]

            assert isinstance(param_A, APIParam) and isinstance(
                param_B, APIParam
            ), "Invariant parameters should be string."
            func_A = param_A.api_full_name
            func_B = param_B.api_full_name

            for (process_id, thread_id), events_list in listed_events.items():

                if func_B not in same_level_func[(process_id, thread_id)]:
                    continue

                if func_A not in same_level_func[(process_id, thread_id)][func_B]:
                    # all B invocations are negative examples
                    for event in events_list:
                        if (
                            event["type"] == "function_call (pre)"
                            and event["function"] == func_B
                        ):
                            example = Example()
                            example.add_group(EXP_GROUP_NAME, [event])
                            hypothesis.negative_examples.add_example(example)
                    continue

                # check
                flag_A = None
                for event in events_list:
                    if event["type"] != "function_call (pre)":
                        continue

                    if func_A == event["function"]:
                        flag_A = event["time"]

                    if func_B == event["function"]:
                        example = Example()
                        example.add_group(EXP_GROUP_NAME, [event])
                        if flag_A is None:
                            hypothesis.negative_examples.add_example(example)
                        else:
                            hypothesis.positive_examples.add_example(example)
                            flag_A = None  # reset flag_A

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
            preconditions = find_precondition(hypothesis, [trace])
            if preconditions is not None:
                hypothesis.invariant.precondition = preconditions
            else:
                failed_hypothesis.append(
                    FailedHypothesis(hypothesis, "Precondition not found")
                )
                all_hypotheses.remove(hypothesis)
        print("End precondition inference")

        if not if_merge:
            return (
                list([hypo.invariant for hypo in all_hypotheses]),
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
            param_A = hypothesis.invariant.params[0]
            param_B = hypothesis.invariant.params[1]

            assert isinstance(param_A, APIParam) and isinstance(param_B, APIParam)
            if hypothesis.invariant.precondition not in relation_pool:
                relation_pool[hypothesis.invariant.precondition] = []
            relation_pool[hypothesis.invariant.precondition].append((param_A, param_B))

        merged_relations: Dict[GroupedPreconditions | None, List[List[APIParam]]] = {}

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

        logger = logging.getLogger(__name__)

        # 1. Pre-process all the events
        print("Start preprocessing....")
        function_times: Dict[Tuple[str, str], Dict[str, Dict[str, Any]]] = {}
        function_id_map: Dict[Tuple[str, str], Dict[str, List[str]]] = {}
        listed_events: Dict[Tuple[str, str], List[dict[str, Any]]] = {}
        function_pool: Set[Any] = set()

        # If the trace contains no function, return []
        assert isinstance(trace, TracePandas)

        # caching the function_pool results
        if trace.function_pool is not None:
            function_pool = trace.function_pool
        else:
            function_pool = set(get_func_names_to_deal_with(trace))
            trace.function_pool = function_pool

        if len(function_pool) == 0:
            logger.warning(
                "No relevant function calls found in the trace, skipping the analysis"
            )
            return CheckerResult(
                trace=None,
                invariant=inv,
                check_passed=True,
                triggered=False,
            )

        if (
            trace.function_times is not None
            and trace.function_id_map is not None
            and trace.listed_events is not None
        ):
            function_times = trace.function_times
            function_id_map = trace.function_id_map
            listed_events = trace.listed_events
        else:
            function_times, function_id_map, listed_events = get_func_data_per_PT(
                trace, function_pool
            )
            trace.function_times = function_times
            trace.function_id_map = function_id_map
            trace.listed_events = listed_events
        print("End preprocessing")

        print("Start same level checking...")
        same_level_func: Dict[Tuple[str, str], Dict[str, Any]] = {}
        valid_relations: Dict[Tuple[str, str], bool] = {}

        if (
            trace.same_level_func_cover is not None
            and trace.valid_relations_cover is not None
        ):
            same_level_func = trace.same_level_func_cover
            valid_relations = trace.valid_relations_cover
        else:
            for (process_id, thread_id), _ in tqdm(
                listed_events.items(), ascii=True, leave=True, desc="Groups Processed"
            ):
                same_level_func[(process_id, thread_id)] = {}
                for func_A, func_B in tqdm(
                    permutations(function_pool, 2),
                    ascii=True,
                    leave=True,
                    desc="Combinations Checked",
                    total=len(function_pool) ** 2,
                ):
                    if check_same_level(
                        func_A,
                        func_B,
                        process_id,
                        thread_id,
                        function_id_map,
                        function_times,
                    ):
                        if func_A not in same_level_func[(process_id, thread_id)]:
                            same_level_func[(process_id, thread_id)][func_A] = []
                        same_level_func[(process_id, thread_id)][func_A].append(func_B)
                        valid_relations[(func_A, func_B)] = True
            trace.same_level_func_cover = same_level_func
            trace.valid_relations_cover = valid_relations
        print("End same level checking")

        inv_triggered = False

        function_pool_temp = []

        invariant_length = len(inv.params)
        for i in range(invariant_length):
            func = inv.params[i]
            assert isinstance(
                func, APIParam
            ), "Invariant parameters should be APIParam."
            function_pool_temp.append(func.api_full_name)

        function_pool = set(function_pool).intersection(set(function_pool_temp))  # type: ignore

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

        print("Starting checking iteration...")
        for i in tqdm(range(invariant_length - 1)):
            param_A = inv.params[i]
            param_B = inv.params[i + 1]

            assert isinstance(param_A, APIParam) and isinstance(
                param_B, APIParam
            ), "Invariant parameters should be string."

            func_A = param_A.api_full_name
            func_B = param_B.api_full_name

            for (process_id, thread_id), events_list in listed_events.items():
                if func_B not in same_level_func[(process_id, thread_id)]:
                    continue

                if func_A not in same_level_func[(process_id, thread_id)][func_B]:
                    # no A invoked, all B should be invalid
                    for event in events_list:
                        if (
                            event["type"] == "function_call (pre)"
                            and event["function"] == func_B
                        ):
                            if not inv.precondition.verify(
                                [event], EXP_GROUP_NAME, trace
                            ):
                                continue

                            inv_triggered = True
                            return CheckerResult(
                                trace=[event],
                                invariant=inv,
                                check_passed=False,
                                triggered=True,
                            )
                    continue

                # check
                unmatched_A_exist = False
                for event in events_list:
                    if event["type"] != "function_call (pre)":
                        continue

                    if func_A == event["function"]:
                        unmatched_A_exist = True

                    if func_B == event["function"]:
                        if not inv.precondition.verify([event], EXP_GROUP_NAME, trace):
                            continue

                        inv_triggered = True
                        if unmatched_A_exist is False:
                            inv_triggered = True
                            return CheckerResult(
                                trace=[event],
                                invariant=inv,
                                check_passed=False,
                                triggered=True,
                            )
                        else:
                            unmatched_A_exist = False  # consumed the last A, a new A should be found before the next B

        # FIXME: triggered is always False for passing invariants
        return CheckerResult(
            trace=None,
            invariant=inv,
            check_passed=True,
            triggered=inv_triggered,
        )

    @staticmethod
    def get_precondition_infer_keys_to_skip(hypothesis: Hypothesis) -> list[str]:
        return ["function"]
