import logging
from itertools import permutations
from typing import Any, Dict, Iterable, List, Set, Tuple

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
from traincheck.invariant.precondition import find_precondition
from traincheck.trace.trace import Trace
from traincheck.trace.trace_pandas import TracePandas

EXP_GROUP_NAME = "func_lead"
MAX_FUNC_NUM_CONSECUTIVE_CALL = 4  # ideally this should be proportional to the number of training and testing iterations in the trace


def check_same_level(
    func_A: str,
    func_B: str,
    process_id: str,
    thread_id: str,
    function_id_map,
    function_times,
):
    """Check if func_A and func_B are at the same level in the call stack.
    By "same level", func_A and func_B are not always nested within each other (no caller-callee relationships).
    The nested functions are filtered out in the preprocessing step.

    Args:
        func_A (str): function name A
        func_B (str): function name B
        process_id (str): process id
        thread_id (str): thread id
        function_id_map: a map from (process_id, thread_id) to function name to all function call ids of that function,
            the ids should be sorted by the time of the function call
        function_times: a map from (process_id, thread_id) to function call id to start and end times of that function call
            the times should be sorted by the time of the function call

    Returns:
        bool: True if func_A and func_B are at the same level, False otherwise
    """

    if func_A == func_B:
        return False

    if func_B not in function_id_map[(process_id, thread_id)]:
        return False

    if func_A not in function_id_map[(process_id, thread_id)]:
        return False

    for idA in function_id_map[(process_id, thread_id)][func_A]:
        for idB in function_id_map[(process_id, thread_id)][func_B]:
            preA = function_times[(process_id, thread_id)][idA]["start"]
            postA = function_times[(process_id, thread_id)][idA]["end"]
            preB = function_times[(process_id, thread_id)][idB]["start"]
            postB = function_times[(process_id, thread_id)][idB]["end"]
            if preA > postB or preB > postA:
                # if preA < postB, it means that A is called before B is finished
                # if preB < postA, it means that B is called before A is finished
                # in both cases, A and B are not always nested within each other
                return True

    return False


def get_func_names_to_deal_with(trace: Trace) -> List[str]:
    """Get all functions in the trace."""
    function_pool: Set[str] = set()

    # get all functions in the trace
    all_func_names = trace.get_func_names()

    # filtering 1: remove private functions
    for func_name in all_func_names:
        if "._" in func_name:
            continue
        function_pool.add(func_name)

    # filtering 2: remove functions that have consecutive calls less than FUNC_CALL_FILTERING_THRESHOLD
    for func_name in function_pool.copy():
        max_num_consecutive_call = trace.get_max_num_consecutive_call_func(func_name)
        if max_num_consecutive_call > MAX_FUNC_NUM_CONSECUTIVE_CALL:
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
    events = trace.events

    filtered_events = events[events["function"].isin(function_pool)]

    events = filtered_events

    group_by_events = events.groupby(["process_id", "thread_id"])

    for group_events in tqdm(group_by_events):
        (process_id, thread_id), evs = group_events
        sorted_group_events = evs.sort_values(by="time")
        if (process_id, thread_id) not in function_id_map:
            function_id_map[(process_id, thread_id)] = {}

        if (process_id, thread_id) not in function_times:
            function_times[(process_id, thread_id)] = {}

        for _, event in sorted_group_events.iterrows():
            if event["function"] in function_pool:
                if event["function"] not in function_id_map[(process_id, thread_id)]:
                    function_id_map[(process_id, thread_id)][event["function"]] = []
                func_id = event["func_call_id"]
                function_id_map[(process_id, thread_id)][event["function"]].append(
                    func_id
                )

                if event["type"] == "function_call (pre)":
                    if func_id not in function_times[(process_id, thread_id)]:
                        function_times[(process_id, thread_id)][func_id] = {}
                    function_times[(process_id, thread_id)][func_id]["start"] = event[
                        "time"
                    ]
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
                # populate the listed_events
                if (process_id, thread_id) not in listed_events:
                    listed_events[(process_id, thread_id)] = []
                listed_events[(process_id, thread_id)].extend([event.to_dict()])

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


class FunctionLeadRelation(Relation):
    """FunctionLeadRelation is a relation that checks if one function Leads another function.

    say function A and function B are two functions in the trace, we say function A leads function B when
    every time function A is called, a function B invocation follows.
    """

    @staticmethod
    def generate_hypothesis(trace) -> list[Hypothesis]:
        """Generate hypothesis for the FunctionLeadRelation on trace."""
        logger = logging.getLogger(__name__)

        # 1. Pre-process all the events
        print("Start preprocessing....")
        function_times: Dict[Tuple[str, str], Dict[str, Dict[str, Any]]] = {}
        function_id_map: Dict[Tuple[str, str], Dict[str, List[str]]] = {}
        listed_events: Dict[Tuple[str, str], List[dict[str, Any]]] = {}
        function_pool: Set[Any] = set()

        # If the trace contains no function, safely exists infer process
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
            trace.same_level_func_lead is not None
            and trace.valid_relations_lead is not None
        ):
            same_level_func = trace.same_level_func_lead
            valid_relations = trace.valid_relations_lead
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
            trace.same_level_func_lead = same_level_func
            trace.valid_relations_lead = valid_relations
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
                    # no B is invoked in this process and thread. All A invocations are negative examples
                    for event in events_list:
                        if (
                            event["type"] == "function_call (pre)"
                            and event["function"] == func_A
                        ):
                            example = Example()
                            example.add_group(EXP_GROUP_NAME, [event])
                            hypothesis_with_examples[
                                (func_A, func_B)
                            ].negative_examples.add_example(example)
                    continue

                time_last_unmatched_A = None
                last_pre_record_A = None
                last_example = None
                hypothesis = hypothesis_with_examples[(func_A, func_B)]
                for event in events_list:
                    if event["type"] != "function_call (pre)":
                        continue

                    if func_A == event["function"]:
                        if time_last_unmatched_A:
                            # the last A has not been followed by a B, a negative example:
                            assert last_example
                            hypothesis.negative_examples.add_example(last_example)

                        time_last_unmatched_A = event["time"]
                        last_pre_record_A = event
                        last_example = Example()
                        last_example.add_group(EXP_GROUP_NAME, [last_pre_record_A])

                    if func_B == event["function"]:
                        if time_last_unmatched_A:
                            assert (
                                last_example
                            ), "Raising an alarm for an A without B, but A's record is None, likely a bug"
                            hypothesis.positive_examples.add_example(last_example)
                            time_last_unmatched_A = None

                if time_last_unmatched_A is not None:
                    assert last_example
                    hypothesis.negative_examples.add_example(last_example)

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

        # If the trace contains no function, safely exists infer process
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
            trace.same_level_func_lead is not None
            and trace.valid_relations_lead is not None
        ):
            same_level_func = trace.same_level_func_lead
            valid_relations = trace.valid_relations_lead
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
            trace.same_level_func_lead = same_level_func
            trace.valid_relations_lead = valid_relations
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

        function_pool = set(function_pool).intersection(function_pool_temp)

        if len(function_pool) == 0:
            print(
                "No relevant function calls found in the trace, skipping the collecting"
            )
            return

        print("Starting collecting iteration...")
        for i in range(invariant_length - 1):
            param_A = inv.params[i]
            param_B = inv.params[i + 1]

            assert isinstance(param_A, APIParam) and isinstance(
                param_B, APIParam
            ), "Invariant parameters should be string."

            func_A = param_A.api_full_name
            func_B = param_B.api_full_name
            for (process_id, thread_id), events_list in listed_events.items():

                if func_A not in same_level_func[(process_id, thread_id)]:
                    # func_A is not invoked in this process and thread, no need to check
                    continue

                if func_B not in same_level_func[(process_id, thread_id)][func_A]:
                    # no B is invoked in this process and thread. All A invocations are negative examples
                    for event in events_list:
                        if (
                            event["type"] == "function_call (pre)"
                            and event["function"] == func_A
                        ):
                            last_example = Example()
                            last_example.add_group(EXP_GROUP_NAME, [event])
                            hypothesis.negative_examples.add_example(last_example)
                    continue

                time_last_unmatched_A = None
                last_pre_record_A = None
                last_example = None
                for event in events_list:
                    if event["type"] != "function_call (pre)":
                        continue

                    if func_A == event["function"]:
                        if time_last_unmatched_A:
                            # the last A has not been followed by a B, a negative example:
                            assert last_example
                            hypothesis.negative_examples.add_example(last_example)

                        time_last_unmatched_A = event["time"]
                        last_pre_record_A = event
                        last_example = Example()
                        last_example.add_group(EXP_GROUP_NAME, [last_pre_record_A])

                    if func_B == event["function"]:
                        if time_last_unmatched_A:
                            assert (
                                last_example
                            ), "Raising an alarm for an A without B, but A's record is None, likely a bug"
                            hypothesis.positive_examples.add_example(last_example)
                            time_last_unmatched_A = None

                if time_last_unmatched_A is not None:
                    hypothesis.negative_examples.add_example(last_example)

    @staticmethod
    def infer(trace: Trace) -> Tuple[List[Invariant], List[FailedHypothesis]]:
        """Infer Invariants for the FunctionLeadrRelation."""

        all_hypotheses = FunctionLeadRelation.generate_hypothesis(trace)

        # for hypothesis in all_hypotheses:
        #     FunctionLeadRelation.collect_examples(trace, hypothesis)

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

        if_merge = True

        if not if_merge:
            return (
                list([hypo.invariant for hypo in all_hypotheses]),
                failed_hypothesis,
            )

        # 6. Merge invariants
        print("Start merging invariants...")
        relation_pool: Dict[
            GroupedPreconditions | None, List[Tuple[APIParam, APIParam]]
        ] = {}
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
                    relation=FunctionLeadRelation,
                    params=[param for param in merged_value],
                    precondition=key,
                    text_description="Merged FunctionLeadRelation in Ordered List",
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

        # If the trace contains no function, safely exists infer process
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
            trace.same_level_func_lead is not None
            and trace.valid_relations_lead is not None
        ):
            same_level_func = trace.same_level_func_lead
            valid_relations = trace.valid_relations_lead
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
            trace.same_level_func_lead = same_level_func
            trace.valid_relations_lead = valid_relations
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

        function_pool = set(function_pool).intersection(set(function_pool_temp))

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
        for i in range(invariant_length - 1):
            param_A = inv.params[i]
            param_B = inv.params[i + 1]

            assert isinstance(param_A, APIParam) and isinstance(
                param_B, APIParam
            ), "Invariant parameters should be string."

            func_A = param_A.api_full_name
            func_B = param_B.api_full_name
            for (process_id, thread_id), events_list in listed_events.items():

                if func_A not in same_level_func[(process_id, thread_id)]:
                    # func_A is not invoked in this process and thread, no need to check
                    continue

                if func_B not in same_level_func[(process_id, thread_id)][func_A]:
                    # all A invocations in this process and thread are negative examples
                    # directly find the first A and return the result
                    for event in events_list:
                        if event["type"] != "function_call (pre)":
                            continue

                        if func_A == event["function"]:
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
                    # if we have not returned in this branch, lets check the next process and thread
                    continue

                # check
                has_B_showup_for_last_A = True  # initialize the flag to True
                last_A_pre_record = None
                for event in events_list:

                    if event["type"] != "function_call (pre)":
                        continue

                    if func_A == event["function"]:
                        if not inv.precondition.verify([event], EXP_GROUP_NAME, trace):
                            continue

                        inv_triggered = True

                        if has_B_showup_for_last_A:
                            # check passed for the last A, reset the flag
                            has_B_showup_for_last_A = False
                            last_A_pre_record = event
                            continue
                        else:
                            # we encountered an new A, but the last A has not been followed by a B
                            assert last_A_pre_record is not None
                            return CheckerResult(
                                trace=[last_A_pre_record],
                                invariant=inv,
                                check_passed=False,
                                triggered=True,
                            )
                    if func_B == event["function"]:
                        has_B_showup_for_last_A = True

        return CheckerResult(
            trace=None,
            invariant=inv,
            check_passed=True,
            triggered=inv_triggered,
        )

    @staticmethod
    def get_precondition_infer_keys_to_skip(hypothesis: Hypothesis) -> list[str]:
        return ["function"]
