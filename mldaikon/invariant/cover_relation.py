import json
from collections import defaultdict, Counter

import logging
import time
from itertools import combinations

from tqdm import tqdm

from mldaikon.config import config
from mldaikon.invariant.base_cls import (
    CheckerResult,
    Example,
    ExampleList,
    Hypothesis,
    Invariant,
    Relation,
    VarTypeParam,
)
from mldaikon.invariant.precondition import find_precondition
from mldaikon.trace.trace import Liveness, Trace, read_trace_file

def read_function_pool(file_path):
    with open(file_path, 'r') as file:
        functions = file.readlines()
    return set(func.strip() for func in functions)


class FunctionCoverRelation(Relation):

    @staticmethod
    def infer(trace: Trace) -> list[Invariant]:
        """Infer Invariants for the FunctionCoverRelation."""

        logger = logging.getLogger(__name__)

        #pre-process
        function_pool = read_function_pool('function_pool.txt')
        function_times = defaultdict(dict)
        function_id_map = defaultdict(list)

        events = trace.events

        required_columns = {'function', 'func_call_id', 'type', 'time'}
        if not required_columns.issubset(events.columns):
            raise ValueError(f"Missing column: {required_columns - set(events.columns)}")
        
        # for event in events.iter_rows(named=True):
        #     if event['function'] in function_pool:
        #         func_id = event['func_call_id']
        #         function_id_map[event['function']].append(func_id)

        #         if event['type'] == 'function_call (pre)':
        #             function_times[func_id]['start'] = event['time']
        #             function_times[func_id]['function'] = event['function']
        #         elif event['type'] in ['function_call (post)', 'function_call (post) (exception)']:
        #             function_times[func_id]['end'] = event['time']
        
        # group_by_time = []
        # for func_id, times in function_times.items():
        #     if 'start' in times and 'end' in times:
        #         group_by_time.append((times['start'], 'start', times['function']))
        #         group_by_time.append((times['end'], 'end', times['function']))

        # group_by_time.sort()

        def check_same_level(funcA: str, funcB: str):
            if funcA == funcB:
                return False
            
            for idA in function_id_map[funcA]:
                for idB in function_id_map[funcB]:
                    preA = function_times[idA]['start']
                    postA = function_times[idA]['end']
                    preB = function_times[idB]['start']
                    postB = function_times[idB]['end']
                    if preB >= postA:
                        break
                    if postB <= preA:
                        continue
                    return False
            return True
        
        same_level_func = defaultdict(list)

        for funcA in function_pool:
            for funcB in function_pool:
                if check_same_level(funcA, funcB):
                    same_level_func[funcA].append(funcB)

        active_counts = Counter()
        valid_relations = defaultdict(lambda: True)

        for func_A in function_pool:
            for func_B in same_level_func[func_A]:
                valid_relations[(func_A, func_B)] = True
        
        #Generating hypothesis
        group_name = "func"
        hypothesis_with_examples = {
            (func_A, func_B): Hypothesis(
                invariant=Invariant(
                    relation=FunctionCoverRelation,
                    params=[
                        func_A,
                        func_B,
                    ],
                    precondition=None,
                    text_description=f"FunctionCoverRelation between {func_A} and {func_B}",
                ),
                positive_examples=ExampleList({group_name}),
                negative_examples=ExampleList({group_name}),
            )
            for (func_A, func_B), _ in valid_relations.items()
        }

        # for time, event_type, function in group_by_time:
        #     if event_type == 'start':
        #         active_counts[function] += 1
        #         for other_function in same_level_func[function]:
        #             if active_counts[other_function] < active_counts[function]:
        #                 valid_relations[(other_function, function)] = False

        #add positive and negative examples
        for (func_A, func_B), _ in valid_relations.items():
            flag_A = None
            flag_B = None
            pre_record_A = []
            pre_record_B = []
            for event in events.iter_rows(named=True):
                # pre_record_A.append(event)
                # pre_record_B.append(event)

                if func_A == event['function']:
                    flag_A = event['time']
                    flag_B = None
                    pre_record_A = [event]

                if func_B == event['function']:
                    if flag_B != None:
                        valid_relations[(func_A, func_B)] = False
                        # neg = Example()
                        # neg.add_group("func", pre_record_B)
                        # hypothesis_with_examples[(func_A, func_B)].negative_examples.add_example(neg)
                        # pre_record_B = [event]
                        continue

                    flag_B = event['time']
                    if flag_A == None:
                        valid_relations[(func_A, func_B)] = False
                        # neg = Example()
                        # neg.add_group("func", pre_record_A)
                        # hypothesis_with_examples[(func_A, func_B)].negative_examples.add_example(neg)
                    else:
                        pos = Example()
                        pos.add_group("func", pre_record_A)
                        hypothesis_with_examples[(func_A, func_B)].positive_examples.add_example(pos)
                    
                    pre_record_B = [event]

        #precondition inference
        hypos_to_delete = []
        for hypo in hypothesis_with_examples:
            logger.debug(
                f"Finding Precondition for {hypo}: {hypothesis_with_examples[hypo].invariant.text_description}"
            )
            preconditions = find_precondition(
                hypothesis_with_examples[hypo]
            )
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
        return None
    

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

        logger = logging.getLogger(__name__)

        funcA = inv.params[0]
        funcB = inv.params[1]

        assert isinstance(funcA, str) and isinstance(
            funcB, str
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

            if funcA == event['function']:
                flag_A = event['time']
                flag_B = None

            if funcB == event['function']:
                if flag_B != None:
                    if inv.precondition.verify(trace, "func"):
                        return CheckerResult(
                                trace=trace,
                                invariant=inv,
                                check_passed=False,
                            )

                flag_B = event['time']
                if flag_A == None:
                    if inv.precondition.verify(trace, "func"):
                        return CheckerResult(
                                trace=trace,
                                invariant=inv,
                                check_passed=False,
                            )
        
        return CheckerResult(
                trace=trace,
                invariant=inv,
                check_passed=True,
            )
                
                    



        

        

