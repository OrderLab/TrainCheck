from itertools import combinations
from typing import Any, Dict, Iterable, List, Set, Tuple

from tqdm import tqdm

from traincheck.invariant.base_cls import (  # GroupedPreconditions,
    APIParam,
    CheckerResult,
    Example,
    ExampleList,
    FailedHypothesis,
    Hypothesis,
    Invariant,
    Relation,
)
from traincheck.invariant.precondition import find_precondition
from traincheck.trace.trace import Trace
from traincheck.utils import safe_isnan

EXP_GROUP_NAME = "distinct_arg"
MAX_FUNC_NUM_CONSECUTIVE_CALL = 6
IOU_THRESHHOLD = 0.1  # pre-defined threshhold for IOU


def calculate_IOU(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    iou = len(intersection) / len(union) if len(union) != 0 else 0
    return iou


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


def get_event_data_per_function_per_step(trace: Trace, function_pool: Set[Any]):
    listed_arguments: Dict[
        str, Dict[int, Dict[Tuple[str, str], List[dict[str, Any]]]]
    ] = {}
    for func_name in function_pool.copy():
        func_call_ids = trace.get_func_call_ids(func_name)
        keep_this_func = False
        for func_call_id in func_call_ids:
            event = trace.query_func_call_event(func_call_id)
            if "args" not in event.pre_record:
                continue

            if "meta_vars.step" not in event.pre_record:
                # assumed to be in the initialization phase
                event.pre_record["meta_vars.step"] = -1

            keep_this_func = True
            process_id = event.pre_record["process_id"]
            thread_id = event.pre_record["thread_id"]
            step = event.pre_record["meta_vars.step"]
            if func_name not in listed_arguments:
                listed_arguments[func_name] = {}
                listed_arguments[func_name][step] = {}
                listed_arguments[func_name][step][(process_id, thread_id)] = []

            if step not in listed_arguments[func_name]:
                listed_arguments[func_name][step] = {}
                listed_arguments[func_name][step][(process_id, thread_id)] = []

            if (process_id, thread_id) not in listed_arguments[func_name][step]:
                listed_arguments[func_name][step][(process_id, thread_id)] = []

            listed_arguments[func_name][step][(process_id, thread_id)].append(
                event.pre_record
            )

        if not keep_this_func:
            function_pool.remove(func_name)

    return function_pool, listed_arguments


def get_event_list(trace: Trace, function_pool: Iterable[str]):
    listed_events: List[dict[str, Any]] = []
    # for all func_ids, get their corresponding events
    for func_name in function_pool:
        func_call_ids = trace.get_func_call_ids(func_name)
        for func_call_id in func_call_ids:
            event = trace.query_func_call_event(func_call_id)
            listed_events.extend(
                # [event.pre_record, event.post_record]
                [event.pre_record]
            )

    # sort the listed_events
    # for (process_id, thread_id), events_list in listed_events.items():
    #     listed_events[(process_id, thread_id)] = sorted(
    #         events_list, key=lambda x: x["time"]
    #     )

    return listed_events


def compare_argument(value1, value2, IOU_criteria=True):
    if type(value1) != type(value2):
        return False
    if isinstance(value1, list):
        if (
            IOU_criteria
            and all(isinstance(item, int) for item in value1)
            and all(isinstance(item, int) for item in value2)
        ):
            return calculate_IOU(value1, value2) >= IOU_THRESHHOLD
        if len(value1) != len(value2):
            return False
        for idx, val in enumerate(value1):
            if not compare_argument(val, value2[idx]):
                return False
        return True
    if isinstance(value1, dict):
        if len(value1) != len(value2):
            return False
        for key in value1:
            if key not in value2:
                return False
            if not compare_argument(value1[key], value2[key]):
                return False
        return True
    if isinstance(value1, float):
        return abs(value1 - value2) < 1e-8
    return value1 == value2


def is_arguments_list_same(args1: list, args2: list):
    if safe_isnan(args1) or safe_isnan(args2):
        return False
    if len(args1) != len(args2):
        return False
    for index in args1:
        arg1 = args1[index]
        if index in args2:
            arg2 = args2[index]
        else:
            return False
        if not compare_argument(arg1, arg2):
            return False
    return True


# class APIArgsParam(Param):
#     def __init__(
#         self, api_full_name: str, arg_name: str
#     ):
#         self.api_full_name = api_full_name
#         self.arg_name = arg_name

#     def __eq__(self, other):
#         if isinstance(other, APIArgsParam):
#             return self.api_full_name == other.api_full_name and self.arg_name == other.arg_name
#         return False

#     def __hash__(self):
#         return hash(self.api_full_name + self.arg_name)

#     def __str__(self):
#         return f"{self.api_full_name} {self.arg_name}"

#     def __repr__(self):
#         return self.__str__()


class DistinctArgumentRelation(Relation):
    """
    DistinctArgumentRelation defines a relation where a function is expected to
    take different argument values across threads or processes within the same training step.
    """

    @staticmethod
    def generate_hypothesis(trace) -> list[Hypothesis]:
        """Generate hypothesis for the DistinctArgumentRelation on trace."""
        # 1. Pre-process all the events
        print("Start preprocessing....")
        listed_arguments: Dict[
            str, Dict[int, Dict[Tuple[str, str], List[dict[str, Any]]]]
        ] = {}
        function_pool: Set[Any] = set()

        function_pool = set(get_func_names_to_deal_with(trace))

        function_pool, listed_arguments = get_event_data_per_function_per_step(
            trace, function_pool
        )
        print("End preprocessing")

        # If there is no filtered function, return [], []
        if not function_pool:
            return []

        # This is just for test.
        # function_pool = set()
        # function_pool.add("torch.nn.init.normal_")

        # 2. Generating hypothesis
        print("Start generating hypo...")
        hypothesis_with_examples = {
            func_name: Hypothesis(
                invariant=Invariant(
                    relation=DistinctArgumentRelation,
                    params=[APIParam(func_name)],
                    precondition=None,
                    text_description=f"{func_name} has distinct input arguments on difference PT for each step",
                ),
                positive_examples=ExampleList({EXP_GROUP_NAME}),
                negative_examples=ExampleList({EXP_GROUP_NAME}),
            )
            for func_name in function_pool
        }
        print("End generating hypo")

        # 3. Add positive and negative examples
        print("Start adding examples...")
        for func_name in tqdm(function_pool):
            flag = False
            for step, records in listed_arguments[func_name].items():
                for PT_pair1, PT_pair2 in combinations(records.keys(), 2):
                    for event1 in records[PT_pair1]:
                        for event2 in records[PT_pair2]:
                            if not is_arguments_list_same(
                                event1["args"], event2["args"]
                            ):
                                flag = True
                                pos = Example()
                                pos.add_group(EXP_GROUP_NAME, [event1, event2])
                                hypothesis_with_examples[
                                    func_name
                                ].positive_examples.add_example(pos)
                            else:
                                neg = Example()
                                neg.add_group(EXP_GROUP_NAME, [event1, event2])
                                hypothesis_with_examples[
                                    func_name
                                ].negative_examples.add_example(neg)

                for PT_pair in records.keys():
                    for event1, event2 in combinations(records[PT_pair], 2):
                        if not is_arguments_list_same(event1["args"], event2["args"]):
                            flag = True
                            pos = Example()
                            pos.add_group(EXP_GROUP_NAME, [event1, event2])
                            hypothesis_with_examples[
                                func_name
                            ].positive_examples.add_example(pos)
                        else:
                            neg = Example()
                            neg.add_group(EXP_GROUP_NAME, [event1, event2])
                            hypothesis_with_examples[
                                func_name
                            ].negative_examples.add_example(neg)

            if not flag:
                hypothesis_with_examples.pop(func_name)

        print("End adding examples")

        return list(hypothesis_with_examples.values())

    @staticmethod
    def collect_examples(trace, hypothesis):
        """Generate examples for a hypothesis on trace."""
        inv = hypothesis.invariant

        # 1. Pre-process all the events
        print("Start preprocessing....")
        listed_arguments: Dict[
            str, Dict[int, Dict[Tuple[str, str], List[dict[str, Any]]]]
        ] = {}
        function_pool: Set[Any] = set()
        func = inv.params[0]

        assert isinstance(func, APIParam), "Invariant parameters should be APIParam."

        func_name = func.api_full_name
        function_pool.add(func_name)

        function_pool, listed_arguments = get_event_data_per_function_per_step(
            trace, function_pool
        )

        print("End preprocessing")

        if not function_pool:
            return

        for step, records in listed_arguments[func_name].items():
            for PT_pair1, PT_pair2 in combinations(records.keys(), 2):
                for event1 in records[PT_pair1]:
                    for event2 in records[PT_pair2]:
                        if not is_arguments_list_same(event1["args"], event2["args"]):
                            pos = Example()
                            pos.add_group(EXP_GROUP_NAME, [event1, event2])
                            hypothesis.positive_examples.add_example(pos)
                        else:
                            neg = Example()
                            neg.add_group(EXP_GROUP_NAME, [event1, event2])
                            hypothesis.negative_examples.add_example(neg)

            for PT_pair in records.keys():
                for event1, event2 in combinations(records[PT_pair], 2):
                    if not is_arguments_list_same(event1["args"], event2["args"]):
                        pos = Example()
                        pos.add_group(EXP_GROUP_NAME, [event1, event2])
                        hypothesis.positive_examples.add_example(pos)
                    else:
                        neg = Example()
                        neg.add_group(EXP_GROUP_NAME, [event1, event2])
                        hypothesis.negative_examples.add_example(neg)

    @staticmethod
    def infer(trace: Trace) -> Tuple[List[Invariant], List[FailedHypothesis]]:
        """Infer Invariants for the DistinctArgumentRelation."""
        all_hypotheses = DistinctArgumentRelation.generate_hypothesis(trace)

        # for hypothesis in all_hypotheses:
        #     DistinctArgumentRelation.collect_examples(trace, hypothesis)

        # 4. Precondition inference
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

        return (
            list([hypo.invariant for hypo in all_hypotheses]),
            failed_hypothesis,
        )

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

        # 1. Pre-process all the events
        print("Start preprocessing....")
        listed_arguments: Dict[
            str, Dict[int, Dict[Tuple[str, str], List[dict[str, Any]]]]
        ] = {}
        function_pool: Set[Any] = set()
        func = inv.params[0]

        assert isinstance(func, APIParam), "Invariant parameters should be APIParam."

        func_name = func.api_full_name
        function_pool.add(func_name)

        function_pool, listed_arguments = get_event_data_per_function_per_step(
            trace, function_pool
        )

        if not function_pool:
            return CheckerResult(
                trace=None,
                invariant=inv,
                check_passed=True,
                triggered=False,
            )

        events_list = get_event_list(trace, function_pool)
        print("End preprocessing")

        if not inv.precondition.verify(events_list, EXP_GROUP_NAME, trace):
            return CheckerResult(
                trace=None,
                invariant=inv,
                check_passed=True,
                triggered=False,
            )

        for step, records in listed_arguments[func_name].items():
            for PT_pair1, PT_pair2 in combinations(records.keys(), 2):
                for event1 in records[PT_pair1]:
                    for event2 in records[PT_pair2]:
                        if is_arguments_list_same(event1["args"], event2["args"]):
                            return CheckerResult(
                                trace=[event1, event2],
                                invariant=inv,
                                check_passed=False,
                                triggered=True,
                            )

            for PT_pair in records.keys():
                for event1, event2 in combinations(records[PT_pair], 2):
                    if is_arguments_list_same(event1["args"], event2["args"]):
                        return CheckerResult(
                            trace=[event1, event2],
                            invariant=inv,
                            check_passed=False,
                            triggered=True,
                        )

        return CheckerResult(
            trace=None,
            invariant=inv,
            check_passed=True,
            triggered=True,
        )

    @staticmethod
    def get_precondition_infer_keys_to_skip(hypothesis: Hypothesis) -> list[str]:
        return []
