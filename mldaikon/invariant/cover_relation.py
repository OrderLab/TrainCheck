import logging
from itertools import combinations
from typing import Any, Dict, List, Tuple

from tqdm import tqdm

from mldaikon.invariant.base_cls import (
    APIParam,
    CheckerResult,
    Example,
    ExampleList,
    Hypothesis,
    Invariant,
    Relation,
)
from mldaikon.invariant.precondition import find_precondition
from mldaikon.trace.trace import Trace


class FunctionCoverRelation(Relation):

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
                (events["function"].str.starts_with("torch.optim"))
                | (events["function"].str.starts_with("torch.nn"))
                | (events["function"].str.starts_with("torch.autograd"))
            )
            & (~events["function"].str.contains("._"))
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
        group_name = "func"
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
                flag_B = None
                pre_record_A = []
                pre_record_B = []

                for event in tqdm(sorted_group_events.iter_rows(named=True)):
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
                            neg.add_group("func", pre_record_B)
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
                            neg.add_group("func", [event])
                            hypothesis_with_examples[
                                (func_A, func_B)
                            ].negative_examples.add_example(neg)
                        else:
                            pos = Example()
                            pos.add_group("func", pre_record_A)
                            hypothesis_with_examples[
                                (func_A, func_B)
                            ].positive_examples.add_example(pos)

                        pre_record_B = [event]

        # 5. Precondition inference
        hypos_to_delete = []
        for hypo in hypothesis_with_examples:
            logger.debug(
                f"Finding Precondition for {hypo}: {hypothesis_with_examples[hypo].invariant.text_description}"
            )
            preconditions = find_precondition(hypothesis_with_examples[hypo])
            logger.debug(f"Preconditions for {hypo}:\n{str(preconditions)}")

            if preconditions is not None:
                hypothesis_with_examples[hypo].invariant.precondition = preconditions
            else:
                logger.debug(f"Precondition not found for {hypo}")
                hypos_to_delete.append(hypo)

        for hypo in hypos_to_delete:
            del hypothesis_with_examples[hypo]

        return list([hypo.invariant for hypo in hypothesis_with_examples.values()])

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
        assert len(inv.params) == 2, "Invariant should have exactly two parameters."
        assert inv.precondition is not None, "Invariant should have a precondition."

        funcA = inv.params[0]
        funcB = inv.params[1]

        assert isinstance(funcA, APIParam) and isinstance(
            funcB, APIParam
        ), "Invariant parameters should be string."

        all_functions = trace.get_func_names()

        if funcB not in all_functions:

            return CheckerResult(
                trace=None,
                invariant=inv,
                check_passed=True,
            )

        # check
        events = trace.events
        flag_A = None
        flag_B = None
        for event in events.iter_rows(named=True):

            if funcA == event["function"]:
                flag_A = event["time"]
                flag_B = None

            if funcB == event["function"]:
                if flag_B is not None:
                    if inv.precondition.verify(trace, "func"):
                        return CheckerResult(
                            trace=trace.events,
                            invariant=inv,
                            check_passed=False,
                        )

                flag_B = event["time"]
                if flag_A is None:
                    if inv.precondition.verify([trace], "func"):
                        return CheckerResult(
                            trace=trace.events,
                            invariant=inv,
                            check_passed=False,
                        )

        return CheckerResult(
            trace=trace.events,
            invariant=inv,
            check_passed=True,
        )
