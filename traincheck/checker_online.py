from traincheck.invariant import Invariant, read_inv_file
from tqdm import tqdm
from traincheck.invariant.base_cls import (
    APIParam,
    Arguments,
    CheckerResult,
    Example,
    ExampleList,
    FailedHypothesis,
    Hypothesis,
    Invariant,
    Param,
    Relation,
    VarNameParam,
    VarTypeParam,
    calc_likelihood,
    construct_api_param,
    construct_var_param_from_var_change,
    is_signature_empty,
)
import json
from traincheck.trace import MDNONEJSONEncoder

import time
import datetime
import os
import queue
from traincheck.trace.types import VarInstId
from traincheck.config import config
import re
from traincheck.trace.types import Liveness
from traincheck.instrumentor.tracer import TraceLineType
from typing import NamedTuple
import threading
from traincheck.trace.types import AttrState
import logging
import argparse
from traincheck.onlinechecker.utils import Checker_data, timing_info, lock
from traincheck.onlinechecker.streamhandler_filesystem import run_stream_monitor
import signal
import sys

OBSERVER = None 
KILLING_PROCESS = (
    False  # True indicates that SIGTERM has been sent to the running process
)
NUM_VIOLATIONS = 0
FAILED_INV = dict()

ORIGINAL_SIGINT_HANDLER = signal.getsignal(signal.SIGINT)
ORIGINAL_SIGTERM_HANDLER = signal.getsignal(signal.SIGTERM)

def handle_SIGINT(signum, frame):
    global KILLING_PROCESS

    print("Received SIGINT")
    if KILLING_PROCESS:
        exit(130)
        return
    KILLING_PROCESS = True
    stop_checker()
    # if callable(ORIGINAL_SIGINT_HANDLER):
    #     ORIGINAL_SIGINT_HANDLER(signum, frame)
    exit(130)


def handle_SIGTERM(signum, frame):
    global KILLING_PROCESS

    print("Received SIGTERM")
    if KILLING_PROCESS:
        exit(143)
        return
    KILLING_PROCESS = True
    stop_checker()
    if callable(ORIGINAL_SIGTERM_HANDLER):
        ORIGINAL_SIGTERM_HANDLER(signum, frame)
    else:
        exit(143)

curr_excepthook = sys.excepthook


def kill_running_process_on_except(typ, value, tb):
    stop_checker()
    curr_excepthook(typ, value, tb)


def register_hook_closing_program():
    signal.signal(signal.SIGTERM, handle_SIGTERM)
    signal.signal(signal.SIGINT, handle_SIGINT)
    sys.excepthook = kill_running_process_on_except

def sort_inv_file(invariants):
    logger = logging.getLogger(__name__)
    logger.info("Reading invariants from file: %s", invariants)
    invs = read_inv_file(invariants)
    logger.info("Total %d invariants read from file: %s", len(invs), invariants)
    logger.info("Sorting invariants by parameters")
    param_to_invs : dict[Param, list[Invariant]] = {}
    vartype_to_invs : dict[str, dict[str, list[Invariant]]] = {}
    needed_vars = set()
    needed_apis = set()
    needed_args_map = set()
    for inv in invs:
        assert (
            inv.precondition is not None
        ), "Invariant precondition is None. It should at least be 'Unconditional' or an empty list. Please check the invariant file and the inference process."
        params = inv.relation.get_mapping_key(inv)
        needed_var, needed_api, needed_args_api = inv.relation.get_needed_data(inv)
        if needed_var is not None:
            needed_vars.update(needed_var)
        if needed_api is not None:
            needed_apis.update(needed_api)
        if needed_args_api is not None:
            needed_args_map.update(needed_args_api)
        for param in params:
            if isinstance(param, VarTypeParam):
                if param.var_type not in vartype_to_invs:
                    vartype_to_invs[param.var_type] = {}
                if param.attr_name not in vartype_to_invs[param.var_type]:
                    vartype_to_invs[param.var_type][param.attr_name] = []
                vartype_to_invs[param.var_type][param.attr_name].append(inv)
            else:
                if param not in param_to_invs:
                    param_to_invs[param] = []
                param_to_invs[param].append(inv)
    logger.info("Sorting done.")
    needed_data = (needed_vars, needed_apis, needed_args_map)
    return param_to_invs, vartype_to_invs, needed_data 

def get_voilated_pair_hash(trace_pair):
    from traincheck.invariant.base_cls import make_hashable
    h1 = hash(make_hashable(trace_pair[0]))
    h2 = hash(make_hashable(trace_pair[1]))
    return tuple(sorted((h1, h2), reverse=True))

def check(invariants, traces, trace_folders, output_dir: str):
    global OBSERVER
    global NUM_VIOLATIONS
    global FAILED_INV
    register_hook_closing_program()
    logger = logging.getLogger(__name__)
    logger.info("Starting online checker")
    logger.addHandler(logging.StreamHandler())
    param_to_invs, vartype_to_invs, needed_data = sort_inv_file(invariants)
    checker_data = Checker_data(needed_data)
    OBSERVER = run_stream_monitor(traces, trace_folders, checker_data)
    output_file = os.path.join(output_dir, "failed.log")
    violated_paris = dict()
    while True:
        trace_record = checker_data.check_queue.get()
        if checker_data.check_queue.empty():
            logger.debug("Check queue is empty")
        if trace_record is None:
            continue
            
        with checker_data.cond:
            while True:
                if trace_record["time"] > checker_data.min_read_time:
                    logger.debug("Wait for the different trace file to catch up")
                    checker_data.cond.wait()
                    logger.debug("Woke up from wait")
                else:
                    break

        if "var_name" in trace_record and trace_record["var_name"] is not None:
            varid = VarInstId(trace_record["process_id"], trace_record["var_name"], trace_record["var_type"])
            if varid.var_type in vartype_to_invs:
                for attr_name, invs in vartype_to_invs[varid.var_type].items():
                    attr_name = config.VAR_ATTR_PREFIX + attr_name
                    if attr_name in trace_record and trace_record[attr_name] is not None:
                        for inv in invs:
                            try:
                                start = time.perf_counter()
                                result = inv.relation.online_check(True, inv, trace_record, checker_data)
                                end = time.perf_counter()
                                duration = end - start
                                with lock:
                                    name = "online_check"
                                    if name not in timing_info:
                                        timing_info[name] = []
                                    timing_info[name].append(duration)
                                if not result.check_passed:
                                    violated_pair = get_voilated_pair_hash(result.trace)
                                    if inv not in violated_paris:
                                        violated_paris[inv] = set()
                                    if violated_pair not in violated_paris[inv]:
                                        violated_paris[inv].add(violated_pair)
                                    else:
                                        continue
                                    if inv not in FAILED_INV:
                                        FAILED_INV[inv] = 0
                                    FAILED_INV[inv] += 1
                                    NUM_VIOLATIONS += 1
                                    result.set_id_and_detection_time(NUM_VIOLATIONS, time.monotonic_ns())
                                    logger.error(f"Voilated id {NUM_VIOLATIONS}:\nInvariant {inv} violated near time {trace_record['time']}")
                                    with open(output_file, "a") as f:
                                        json.dump(result.to_dict(), f, indent=4, cls=MDNONEJSONEncoder)
                                        f.write("\n")
                            except Exception as e:
                                logger.error(f"Error when checking invariant {inv.text_description} with trace {trace_record}: {e}")
                                # TODO: delete raise
                                raise e
                                
        elif "func_call_id" in trace_record and trace_record["func_call_id"] is not None:   
            apiparam = APIParam(trace_record["function"])
            if apiparam in param_to_invs:
                for inv in param_to_invs[apiparam]:
                    try:
                        start = time.perf_counter()
                        result = inv.relation.online_check(True, inv, trace_record, checker_data)
                        end = time.perf_counter()
                        duration = end - start
                        with lock:
                            name = "online_check"
                            if name not in timing_info:
                                timing_info[name] = []
                            timing_info[name].append(duration)
                        if not result.check_passed:
                            if inv not in FAILED_INV:
                                FAILED_INV[inv] = 0
                            FAILED_INV[inv] += 1
                            NUM_VIOLATIONS += 1
                            result.set_id_and_detection_time(NUM_VIOLATIONS, time.monotonic_ns())
                            logger.error(f"Voilated id {NUM_VIOLATIONS}:\nInvariant {inv} violated near time {trace_record['time']}")
                            with open(output_file, "a") as f:
                                json.dump(result.to_dict(), f, indent=4, cls=MDNONEJSONEncoder)
                                f.write("\n")
                    except Exception as e:
                        logger.error(f"Error when checking invariant {inv.text_description} with trace {trace_record}: {e}")
                        # TODO: delete raise
                        raise e
                        

def stop_checker():
    global OBSERVER
    if OBSERVER is None:
        return
    
    OBSERVER.stop()
    OBSERVER.join()

    global NUM_VIOLATIONS
    global FAILED_INV

    logger = logging.getLogger(__name__)
    logger.info("Checker stopped")
    logger.info(f"Total {NUM_VIOLATIONS} violations found")
    logger.info(f"Total {len(FAILED_INV)} invariants violated:")
    # for inv, count in failed_inv.items():
    #     logger.info(f"Invariant {inv} violated {count} times")
    logger.info(f"Violations are stored")
    for name, times in timing_info.items():
        print(f"{name}: called {len(times)} times, total time = {sum(times)}s, avg time = {sum(times)/len(times):.6f}s")
    

def main():
    parser = argparse.ArgumentParser(
        description="(Online) Invariant Checker for ML Pipelines in Python"
    )
    parser.add_argument(
        "-t",
        "--traces",
        nargs="+",
        required=False,
        help="Traces files to infer invariants on",
    )
    parser.add_argument(
        "-f",
        "--trace-folders",
        nargs="+",
        help='Folders containing traces files to infer invariants on. Trace files should start with "trace_" or "proxy_log.json"',
    )
    parser.add_argument(
        "-i",
        "--invariants",
        nargs="+",
        required=True,
        help="Invariants files to check on traces",
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--check-relation-first",
        action="store_true",
        help="""Check the relation first, otherwise, the precondition will be checked first. 
            Enabling this flag will make the checker slower, but enables the checker to catch 
            the cases where the invariant still holds even if the precondition is not satisfied, 
            which opens opportunity for precondition refinement. Note that the precondition 
            refinement algorithm is not implemented yet.""",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        help="Output folder to store the results, defaulted to traincheck_checker_results_{timestamp}/",
    )

    args = parser.parse_args()

    # check if either traces or trace folders are provided
    if args.traces is None and args.trace_folders is None:
        # print help message if neither traces nor trace folders are provided
        parser.print_help()
        parser.error(
            "Please provide either traces or trace folders to infer invariants"
        )

    if args.invariants is None:
        parser.print_help()
        parser.error("Please provide exactly one invariant file to check")

    if args.debug:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO

    time_now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    ## DEBUG
    time_now = f"{time_now}_relation_first_{args.check_relation_first}"
    # set logging to a file
    logging.basicConfig(
        filename=f"traincheck_onlinechecker_{time_now}.log",
        level=log_level,
    )

    logger = logging.getLogger(__name__)
    # log all the arguments
    logger.info("Checker started with Arguments:")
    for arg, val in vars(args).items():
        logger.info("%s: %s", arg, val)

    if not args.output_dir:
        args.output_dir = f"traincheck_onlinechecker_results_{time_now}"
    os.makedirs(args.output_dir, exist_ok=True)

    # copy the invariants to the output folder
    for inv_file in args.invariants:
        os.system(f"cp {inv_file} {args.output_dir}/invariants.json")
    
    check(args.invariants, args.traces, args.trace_folders, args.output_dir)
    # print(aaaaa)
    # check("/Users/universe/Documents/univer/study/MLSYS/OrderLab/TrainCheck/firsttest/invariants_test.json", "/Users/universe/Documents/univer/study/MLSYS/OrderLab/TrainCheck/firsttest/traincheck_mnist_trace")
    # check("/Users/universe/Documents/univer/study/MLSYS/OrderLab/TrainCheck/firsttest/invariants_test.json", "/Users/universe/Documents/univer/study/MLSYS/OrderLab/TrainCheck/firsttest/traincheck_84911_trace")
    # check("/Users/universe/Documents/univer/study/MLSYS/OrderLab/TrainCheck/firsttest/invariants_test.json", "/Users/universe/Documents/univer/study/MLSYS/OrderLab/TrainCheck/firsttest/test")
    # check("/Users/universe/Documents/univer/study/MLSYS/TrainCheck/firsttest/invariants.json", "/Users/universe/Documents/univer/study/MLSYS/TrainCheck/firsttest/traincheck_mnist_trace")
    # check("/Users/universe/Documents/univer/study/MLSYS/OrderLab/TrainCheck/test_for_con/invariants_deepspeed-1801-fp16.json", "/Users/universe/Documents/univer/study/MLSYS/OrderLab/TrainCheck/test_for_con/trace_deepspeed-1801")
    # check("/Users/universe/Documents/univer/study/MLSYS/OrderLab/TrainCheck/test_for_con/invariants_deepspeed-1801-fp16.json", "/Users/universe/Documents/univer/study/MLSYS/OrderLab/TrainCheck/test_for_con/trace_test/simulated")
    # check("/Users/universe/Documents/univer/study/MLSYS/OrderLab/TrainCheck/test_for_con/invariants_deepspeed-1801-fp16.json", "/Users/universe/Documents/univer/study/MLSYS/OrderLab/TrainCheck/test_for_con/trace_test2/simulated")
    # check("/Users/universe/Documents/univer/study/MLSYS/OrderLab/TrainCheck/test_for_con/invariants_deepspeed-1801-fp16.json", "/Users/universe/Documents/univer/study/MLSYS/OrderLab/TrainCheck/test_for_con/trace_test3")
    # check("/Users/universe/Documents/univer/study/MLSYS/OrderLab/TrainCheck/test_for_co_le/invariants_mmpretrain-702.json", "/Users/universe/Documents/univer/study/MLSYS/OrderLab/TrainCheck/test_for_co_le/trace_mmpretrain-702_test")
    # check("/Users/universe/Documents/univer/study/MLSYS/OrderLab/TrainCheck/test_for_co_le/invariants_pytorch-51800.json", "/Users/universe/Documents/univer/study/MLSYS/OrderLab/TrainCheck/test_for_co_le/trace_pytorch-51800")
    # check("/Users/universe/Documents/univer/study/MLSYS/OrderLab/TrainCheck/test_for_da/invariants_transformers-17877.json", "/Users/universe/Documents/univer/study/MLSYS/OrderLab/TrainCheck/test_for_da/trace_transformers-17877")
    # check("/Users/universe/Documents/univer/study/MLSYS/OrderLab/TrainCheck/firsttest/invariants_test.json", "/Users/universe/Documents/univer/study/MLSYS/OrderLab/TrainCheck/firsttest/traincheck_84911_trace")
    # check("/Users/universe/Documents/univer/study/MLSYS/OrderLab/TrainCheck/TrainCheck-Evaluation-Workloads-main/silent-issue-detection/invariants_transformers-33844.json", "/Users/universe/Documents/univer/study/MLSYS/OrderLab/TrainCheck/test_for_con/test")
                
if __name__ == "__main__":
    main()