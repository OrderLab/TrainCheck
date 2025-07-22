import logging
from itertools import permutations
from typing import Any, Dict, Iterable, List, Set, Tuple

from tqdm import tqdm

from traincheck.instrumentor.tracer import TraceLineType
from traincheck.invariant.base_cls import (
    APIParam,
    CheckerResult,
    Example,
    ExampleList,
    FailedHypothesis,
    GroupedPreconditions,
    Hypothesis,
    Invariant,
    OnlineCheckerResult,
    Param,
    Relation,
)
from traincheck.invariant.precondition import find_precondition
from traincheck.onlinechecker.utils import Checker_data, set_meta_vars_online
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
    listed_events: Dict[Tuple[str, str], List[dict[str, Any]]] = {}

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


def get_func_A_B_events(events_list: List[dict[str, Any]], func_A: str, func_B: str):
    events_A = [event for event in events_list if event["function"] == func_A]
    events_A_pre = [
        event for event in events_A if event["type"] == "function_call (pre)"
    ]
    events_A_post = [
        event
        for event in events_A
        if event["type"] == "function_call (post)"
        or event["type"] == "function_call (post) (exception)"
    ]
    events_B = [event for event in events_list if event["function"] == func_B]
    events_B_pre = [
        event for event in events_B if event["type"] == "function_call (pre)"
    ]
    events_B_post = [
        event
        for event in events_B
        if event["type"] == "function_call (post)"
        or event["type"] == "function_call (post) (exception)"
    ]
    return (events_A_pre, events_A_post, events_B_pre, events_B_post)


def get_post_func_event(events_list: List[dict[str, Any]], func_call_id: str):
    event_posts = [
        event for event in events_list if event["func_call_id"] == func_call_id
    ]
    assert event_posts is not None, "Post event not found"
    event_post = event_posts[0]
    return event_post


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

                # find all A and B events in the current process and thread
                events_A_pre, events_A_post, events_B_pre, events_B_post = (
                    get_func_A_B_events(events_list, func_A, func_B)
                )

                event_A_idx = 0
                event_B_idx = 0

                pre_event_A_idx = None
                pre_event_A_time = None

                last_example = None

                for event_A_pre in events_A_pre:
                    invocation_id = event_A_pre["func_call_id"]
                    event_A_post = get_post_func_event(events_A_post, invocation_id)
                    pre_event_A_idx = event_A_idx
                    pre_event_A_time = event_A_post["time"]
                    event_A_idx += 1
                    example = Example()
                    example.add_group(EXP_GROUP_NAME, [event_A_pre])
                    last_example = example
                    break

                assert pre_event_A_idx is not None
                assert pre_event_A_time is not None
                assert last_example is not None

                if event_A_idx >= len(events_A_pre):
                    max_time = events_B_post["time"].max()
                    if pre_event_A_time <= max_time:
                        hypothesis_with_examples[
                            (func_A, func_B)
                        ].positive_examples.add_example(last_example)
                    else:
                        hypothesis_with_examples[
                            (func_A, func_B)
                        ].negative_examples.add_example(last_example)

                    continue

                for event_A_pre in events_A_pre[event_A_idx:]:
                    invocation_id = event_A_pre["func_call_id"]
                    example = Example()
                    example.add_group(EXP_GROUP_NAME, [event_A_pre])

                    if event_B_idx >= len(events_B_pre):
                        # If we have exhausted all B events, skip the rest of A events
                        break

                    event_A_post = get_post_func_event(events_A_post, invocation_id)

                    if event_A_pre["time"] <= pre_event_A_time:
                        if last_example is not None:
                            hypothesis_with_examples[
                                (func_A, func_B)
                            ].negative_examples.add_example(last_example)

                        pre_event_A_idx = event_A_idx
                        pre_event_A_time = event_A_post["time"]
                        event_A_idx += 1
                        last_example = example
                        continue

                    found_B_after_A = False
                    # First A post time <= B pre time  <= B post time <= next A pre time
                    while event_B_idx < len(events_B_pre):
                        event_B_pre = events_B_pre[event_B_idx]
                        event_B_time = event_B_pre["time"]

                        if event_B_time > event_A_pre["time"]:
                            break

                        if event_B_time <= pre_event_A_time:
                            event_B_idx += 1
                            continue

                        B_invocation_id = event_B_pre["func_call_id"]
                        event_B_post = get_post_func_event(
                            events_B_post, B_invocation_id
                        )
                        if event_B_post["time"] > event_A_pre["time"]:
                            event_B_idx += 1
                            continue

                        found_B_after_A = True
                        event_B_idx += 1
                        break

                    if last_example is not None:
                        if found_B_after_A:
                            # Check if there's a B event after the current A event
                            hypothesis_with_examples[
                                (func_A, func_B)
                            ].positive_examples.add_example(last_example)
                        else:
                            hypothesis_with_examples[
                                (func_A, func_B)
                            ].negative_examples.add_example(last_example)

                    pre_event_A_idx = event_A_idx
                    pre_event_A_time = event_A_post["time"]
                    event_A_idx += 1
                    last_example = example
                # add the rest of the A events as negative examples
                for event_A_pre in events_A_pre[event_A_idx:]:
                    example = Example()
                    example.add_group(EXP_GROUP_NAME, [event_A_pre])
                    hypothesis_with_examples[
                        (func_A, func_B)
                    ].negative_examples.add_example(example)

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

                    # find all A and B events in the current process and thread
                events_A_pre, events_A_post, events_B_pre, events_B_post = (
                    get_func_A_B_events(events_list, func_A, func_B)
                )
                # print(f"Found {len(events_A_pre)} A events and {len(events_B_pre)} B events")

                event_A_idx = 0
                event_B_idx = 0

                pre_event_A_idx = None
                pre_event_A_time = None

                last_example = None

                for event_A_pre in events_A_pre:
                    invocation_id = event_A_pre["func_call_id"]
                    event_A_post = get_post_func_event(events_A_post, invocation_id)
                    pre_event_A_idx = event_A_idx
                    pre_event_A_time = event_A_post["time"]
                    event_A_idx += 1
                    example = Example()
                    example.add_group(EXP_GROUP_NAME, [event_A_pre])
                    last_example = example
                    break

                if event_A_idx >= len(events_A_pre):
                    max_time = events_B_post["time"].max()
                    if pre_event_A_time <= max_time:
                        hypothesis[(func_A, func_B)].positive_examples.add_example(
                            last_example
                        )
                    else:
                        hypothesis[(func_A, func_B)].negative_examples.add_example(
                            last_example
                        )

                    continue

                assert pre_event_A_idx is not None
                assert pre_event_A_time is not None

                for event_A_pre in events_A_pre[event_A_idx:]:
                    invocation_id = event_A_pre["func_call_id"]
                    example = Example()
                    example.add_group(EXP_GROUP_NAME, [event_A_pre])

                    if event_B_idx >= len(events_B_pre):
                        # If we have exhausted all B events, skip the rest of A events
                        break

                    event_A_post = get_post_func_event(events_A_post, invocation_id)

                    if event_A_pre["time"] <= pre_event_A_time:
                        hypothesis[(func_A, func_B)].negative_examples.add_example(
                            last_example
                        )

                        pre_event_A_idx = event_A_idx
                        pre_event_A_time = event_A_post["time"]
                        event_A_idx += 1
                        last_example = example
                        continue

                    found_B_after_A = False
                    # First A post time <= B pre time  <= B post time <= next A pre time
                    while event_B_idx < len(events_B_pre):
                        event_B_pre = events_B_pre[event_B_idx]
                        event_B_time = event_B_pre["time"]

                        if event_B_time > event_A_pre["time"]:
                            break

                        if event_B_time <= pre_event_A_time:
                            event_B_idx += 1
                            continue

                        B_invocation_id = event_B_pre["func_call_id"]
                        event_B_post = get_post_func_event(
                            events_B_post, B_invocation_id
                        )
                        if event_B_post["time"] > event_A_pre["time"]:
                            event_B_idx += 1
                            continue

                        found_B_after_A = True
                        event_B_idx += 1
                        break

                    if found_B_after_A:
                        # Check if there's a B event after the current A event
                        hypothesis[(func_A, func_B)].positive_examples.add_example(
                            last_example
                        )
                    else:
                        hypothesis[(func_A, func_B)].negative_examples.add_example(
                            last_example
                        )

                    pre_event_A_idx = event_A_idx
                    pre_event_A_time = event_A_post["time"]
                    event_A_idx += 1
                    last_example = example
                # add the rest of the A events as negative examples
                for event_A_pre in events_A_pre[event_A_idx:]:
                    example = Example()
                    example.add_group(EXP_GROUP_NAME, [event_A_pre])
                    hypothesis[(func_A, func_B)].negative_examples.add_example(example)

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

                events_A_pre, events_A_post, events_B_pre, events_B_post = (
                    get_func_A_B_events(events_list, func_A, func_B)
                )

                event_A_idx = 0
                event_B_idx = 0

                pre_event_A = None
                pre_event_A_time = None

                for event_A_pre in events_A_pre:
                    if not inv.precondition.verify(
                        [event_A_pre], EXP_GROUP_NAME, trace
                    ):
                        event_A_idx += 1
                        continue
                    inv_triggered = True
                    invocation_id = event_A_pre["func_call_id"]
                    event_A_post = get_post_func_event(events_A_post, invocation_id)
                    pre_event_A = event_A_pre
                    pre_event_A_time = event_A_post["time"]
                    event_A_idx += 1
                    break

                for event_A_pre in events_A_pre[event_A_idx:]:
                    if not inv.precondition.verify(
                        [event_A_pre], EXP_GROUP_NAME, trace
                    ):
                        continue

                    if pre_event_A_time is not None:
                        if event_A_pre["time"] <= pre_event_A_time:
                            assert pre_event_A is not None
                            return CheckerResult(
                                trace=[pre_event_A],
                                invariant=inv,
                                check_passed=False,
                                triggered=True,
                            )

                    event_A_post = get_post_func_event(
                        events_A_post, event_A_pre["func_call_id"]
                    )

                    found_B_after_A = False
                    while event_B_idx < len(events_B_pre):
                        event_B_pre = events_B_pre[event_B_idx]
                        event_B_time = event_B_pre["time"]

                        if event_B_time > event_A_pre["time"]:
                            break

                        if event_B_time <= pre_event_A_time:
                            event_B_idx += 1
                            continue

                        B_invocation_id = event_B_pre["func_call_id"]
                        event_B_post = get_post_func_event(
                            events_B_post, B_invocation_id
                        )
                        if event_B_post["time"] > event_A_pre["time"]:
                            event_B_idx += 1
                            continue

                        found_B_after_A = True
                        event_B_idx += 1
                        break

                    if not found_B_after_A:
                        assert pre_event_A is not None
                        return CheckerResult(
                            trace=[pre_event_A],
                            invariant=inv,
                            check_passed=False,
                            triggered=True,
                        )
                    pre_event_A_time = event_A_post["time"]
                    pre_event_A = event_A_pre

        return CheckerResult(
            trace=None,
            invariant=inv,
            check_passed=True,
            triggered=inv_triggered,
        )

    @staticmethod
    def _get_identifying_params(inv: Invariant) -> list[Param]:
        params = []
        for i in range(len(inv.params) - 1):
            params.append(inv.params[i])
        return params

    @staticmethod
    def _get_variables_to_check(inv):
        return None

    @staticmethod
    def _get_apis_to_check(inv: Invariant):
        api_name_list = []
        for param in inv.params:
            assert isinstance(param, APIParam)
            api_name_list.append(param.api_full_name)
        return api_name_list

    @staticmethod
    def _get_api_args_map_to_check(inv):
        return None

    @staticmethod
    def online_check(
        check_relation_first: bool,
        inv: Invariant,
        trace_record: dict,
        checker_data: Checker_data,
    ):
        if trace_record["type"] != TraceLineType.FUNC_CALL_PRE:
            return OnlineCheckerResult(
                trace=None,
                invariant=inv,
                check_passed=True,
            )

        assert inv.precondition is not None, "Invariant should have a precondition."

        checker_param = APIParam(trace_record["function"])
        lead_param = None
        for i in range(len(inv.params)):
            if inv.params[i] == checker_param:
                if i == len(inv.params) - 1:
                    lead_param = None
                    break
                lead_param = inv.params[i + 1]
                break
        if lead_param is None:
            return OnlineCheckerResult(
                trace=None,
                invariant=inv,
                check_passed=True,
            )

        assert isinstance(lead_param, APIParam)

        process_id = trace_record["process_id"]
        thread_id = trace_record["thread_id"]
        func_name = trace_record["function"]
        ptname = (process_id, thread_id, func_name)

        start_time = None
        end_time = trace_record["time"]

        with checker_data.lock:
            [trace_record] = set_meta_vars_online([trace_record], checker_data)

        if not inv.precondition.verify([trace_record], EXP_GROUP_NAME, None):
            return OnlineCheckerResult(
                trace=None,
                invariant=inv,
                check_passed=True,
            )

        with checker_data.lock:
            for func_id, func_event in checker_data.pt_map[ptname].items():
                if func_event.post_record is None:
                    continue
                time = func_event.post_record["time"]
                if time >= end_time:
                    continue
                if not inv.precondition.verify(
                    set_meta_vars_online([func_event.pre_record], checker_data),
                    EXP_GROUP_NAME,
                    None,
                ):
                    continue
                if start_time is None or time > start_time:
                    start_time = time

        if start_time is None:
            return OnlineCheckerResult(
                trace=None,
                invariant=inv,
                check_passed=True,
            )

        lead_func_name = lead_param.api_full_name
        lead_ptname = (process_id, thread_id, lead_func_name)
        with checker_data.lock:
            if lead_ptname in checker_data.pt_map:
                for func_id, func_event in checker_data.pt_map[lead_ptname].items():
                    if func_event.pre_record is None or func_event.post_record is None:
                        continue
                    pre_time = func_event.pre_record["time"]
                    post_time = func_event.post_record["time"]
                    if pre_time >= start_time and post_time <= end_time:
                        return OnlineCheckerResult(
                            trace=None,
                            invariant=inv,
                            check_passed=True,
                        )

        return OnlineCheckerResult(
            trace=[trace_record],
            invariant=inv,
            check_passed=False,
        )

    @staticmethod
    def get_precondition_infer_keys_to_skip(hypothesis: Hypothesis) -> list[str]:
        return ["function"]
