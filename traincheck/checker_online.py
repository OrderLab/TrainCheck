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
from watchdog.observers import Observer
from watchdog.observers.polling import PollingObserver
from watchdog.events import FileSystemEventHandler
import os
from traincheck.trace.utils import flatten_dict, replace_none_with_md_none
import queue
from traincheck.trace.types import VarInstId
from traincheck.config import config
import re
from traincheck.trace.types import Liveness
from traincheck.instrumentor.tracer import TraceLineType
from typing import NamedTuple
import threading
from traincheck.trace.types import AttrState
from traincheck.trace.types import (
    HighLevelEvent,
    ALL_EVENT_TYPES,
    FuncCallEvent,
    FuncCallExceptionEvent,
    IncompleteFuncCallEvent,
    VarChangeEvent,
)

def sort_inv_file(invariants: str):
    invs = read_inv_file(invariants)
    param_to_invs : dict[Param, list[Invariant]] = {}
    vartype_to_invs : dict[str, dict[str, list[Invariant]]] = {}
    needed_vars = set()
    needed_apis = set()
    needed_args_map = set()
    for inv in invs:
        assert (
            inv.precondition is not None
        ), "Invariant precondition is None. It should at least be 'Unconditional' or an empty list. Please check the invariant file and the inference process."
        # TODO: improve code quality
        params = inv.relation.get_mapping_key(inv)
        needed_var = inv.relation.get_needed_variables(inv)
        needed_api = inv.relation.get_needed_api(inv)
        needed_args_api = inv.relation.needed_args_map(inv)
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


    # with open("./test.txt", "w") as f:     
    #     for param, invs_ in param_to_invs.items():
    #         if isinstance(param, APIParam):
    #             f.write(param.api_full_name)
    #         elif isinstance(param, VarNameParam):
    #             f.write(param.var_name)
    #         elif isinstance(param, VarTypeParam):
    #             f.write(param.var_type)
    #         for inv in invs_:
    #             f.write(json.dumps(inv.to_dict(), cls=MDNONEJSONEncoder))
    #             f.write("\n")
    return param_to_invs, vartype_to_invs, needed_vars, needed_apis, needed_args_map

class OnlineFuncCallEvent(FuncCallEvent):
    def __init__(self, func_name):
        self.func_name = func_name
        self.pre_record = None
        self.post_record = None
        self.exception = None

        self.args = None
        self.kwargs = None
        self.return_values = None


    def get_traces(self):
        return [self.pre_record, self.post_record]

    def __hash__(self) -> int:
        return super().__hash__()

    def __eq__(self, other) -> bool:
        return super().__eq__(other)

# ! NOTE: not thread safe
class Checker_data:
    def __init__(self, needed_vars, needed_apis,  needed_args_map):
        self.needed_vars = needed_vars
        self.needed_apis = needed_apis
        self.needed_args_map = needed_args_map

        self.check_queue = queue.Queue()
        self.varid_map = {}
        self.type_map = {}
        self.pt_map = {}
        self.process_to_vars = {}
        self.args_map = {}

        self.read_time_map = {}
        self.min_read_time = None
        self.min_read_path = None
        self.read_lock = threading.Lock()
        self.cond = threading.Condition(self.read_lock)

class StreamLogHandler(FileSystemEventHandler):
    def __init__(self, file_path, checker_data: Checker_data):
        self.file_path = file_path
        self.fp = open(file_path, 'r')

        self.queue = checker_data.check_queue

        self.varid_map = checker_data.varid_map
        self.type_map = checker_data.type_map
        self.pt_map = checker_data.pt_map
        self.process_to_vars = checker_data.process_to_vars
        self.args_map = checker_data.args_map
        self.min_read_time = checker_data.min_read_time
        self.read_lock = checker_data.read_lock
        self.cond = checker_data.cond
        self.checker_data = checker_data

        self._save_initial_content()

        self.fp.seek(0, 2) 

    def _set_maps(self, trace_record):
        if "var_name" in trace_record and trace_record["var_name"] is not None:
            varid = VarInstId(trace_record["process_id"], trace_record["var_name"], trace_record["var_type"])
            var_name = trace_record["var_name"]
            var_type = trace_record["var_type"]
            if var_name in self.checker_data.needed_vars or var_type in self.checker_data.needed_vars:
                if varid not in self.varid_map:
                    self.varid_map[varid] = {}

                if varid.process_id not in self.process_to_vars:
                    self.process_to_vars[varid.process_id] = set()

                self.process_to_vars[varid.process_id].add(varid)
                
                for attr_name, value in trace_record.items():
                    if value is None:
                        continue

                    if attr_name.startswith(config.VAR_ATTR_PREFIX):
                        attr_name = attr_name[len(config.VAR_ATTR_PREFIX):]
                    else:
                        continue

                    from traincheck.invariant.base_cls import make_hashable

                    curr_value = make_hashable(value)
                    if any(
                        [
                            re.match(pattern, attr_name) is not None
                            for pattern in config.PROP_ATTR_PATTERNS
                        ]
                    ):
                        continue

                    if attr_name not in self.varid_map[varid]:
                        self.varid_map[varid][attr_name] = [
                            AttrState(
                                curr_value,
                                Liveness(trace_record["time"], None),
                                [trace_record],
                            )    
                        ]
                    else:
                            
                        self.varid_map[varid][attr_name][-1].liveness.end_time = trace_record["time"]
                        self.varid_map[varid][attr_name].append(
                            AttrState(
                                curr_value,
                                Liveness(trace_record["time"], None),
                                [trace_record],
                            )
                        )
                            
                if trace_record["var_type"] is not None:
                    if trace_record["var_type"] not in self.type_map:
                        self.type_map[trace_record["var_type"]] = set()
                    self.type_map[trace_record["var_type"]].add(varid)
        elif "func_call_id" in trace_record and trace_record["func_call_id"] is not None:   
            process_id = trace_record["process_id"]
            thread_id = trace_record["thread_id"]
            ptid = (process_id, thread_id)
            func_call_id = trace_record["func_call_id"]
            function_name = trace_record["function"]
            ptname = (process_id, thread_id, function_name)
            if function_name in self.checker_data.needed_apis:
                if ptname not in self.pt_map:
                    self.pt_map[ptname] = {}
                if func_call_id not in self.pt_map[ptname]:
                    self.pt_map[ptname][func_call_id] = OnlineFuncCallEvent(function_name)
                if trace_record["type"] == TraceLineType.FUNC_CALL_PRE:
                    self.pt_map[ptname][func_call_id].pre_record = trace_record
                    self.pt_map[ptname][func_call_id].args = trace_record["args"]
                    self.pt_map[ptname][func_call_id].kwargs = trace_record["kwargs"]
                elif trace_record["type"] == TraceLineType.FUNC_CALL_POST:
                    self.pt_map[ptname][func_call_id].post_record = trace_record
                    self.pt_map[ptname][func_call_id].return_values = trace_record["return_values"]
                elif trace_record["type"] == TraceLineType.FUNC_CALL_POST_EXCEPTION:
                    self.pt_map[ptname][func_call_id].post_record = trace_record
                    self.pt_map[ptname][func_call_id].exception = trace_record["exception"]

            if trace_record["type"] == TraceLineType.FUNC_CALL_PRE:
                if function_name in self.checker_data.needed_args_map:
                    if "args" in trace_record:
                        if "meta_vars.step" not in trace_record:
                            trace_record["meta_vars.step"] = -1
                        step = trace_record["meta_vars.step"]
                        if function_name not in self.args_map:
                            self.args_map[function_name] = {}
                        if step not in self.args_map[function_name]:
                            self.args_map[function_name][step] = {}
                        if ptid not in self.args_map[function_name][step]:
                            self.args_map[function_name][step][ptid] = []
                        self.args_map[function_name][step][ptid].append(trace_record)

        self.queue.put(trace_record)

        with self.checker_data.cond:
            self.checker_data.read_time_map[self.file_path] = trace_record["time"]
            recalc_needed = (
               self.checker_data.min_read_path == self.file_path
                or self.checker_data.min_read_time is None
            )
            if recalc_needed:
                pre_min_read_time = self.checker_data.min_read_time
                self.checker_data.min_read_path, self.checker_data.min_read_time = min(
                    self.checker_data.read_time_map.items(), default=(None, None))
                if pre_min_read_time != self.checker_data.min_read_time:
                    self.checker_data.cond.notify_all()
            


    def _handle_line(self, lines):
        for line in lines:
            trace_record = None
            try:
                flat_dict = flatten_dict(
                        json.loads(line, object_hook=replace_none_with_md_none),
                        skip_fields=["args", "kwargs", "return_values"],
                    )
                trace_record = flat_dict
                self._set_maps(trace_record)

            except Exception as e:
                # TODO: log
                print(line)
                raise e


    def _save_initial_content(self):
        self.fp.seek(0)
        lines = self.fp.readlines()
        if not lines:
            return
        
        self._handle_line(lines)
        print("ok")

        # TODO: remove, just for check correctness
        # time_now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # dir_name = "test_for_watch_dog"
        # os.makedirs(dir_name, exist_ok=True)
        # if self.file_path.endswith(".json"):
        #     file_name = os.path.join(dir_name, f"log_init_proxy_{time_now}.txt")
        # else:
        #     file_name = os.path.join(dir_name, f"log_init_api_{time_now}.txt")
        # with open(file_name, 'w') as f:
        #     f.writelines(lines)

    def on_modified(self, event):
        if os.path.abspath(event.src_path) != os.path.abspath(self.file_path):
            return

        self._handle_line(self.fp)

        # TODO: remove, just for check correctness
        # time_now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # dir_name = "test_for_watch_dog"
        # os.makedirs(dir_name, exist_ok=True)
        # if self.file_path.endswith(".json"):
        #     file_name = os.path.join("test_for_watch_dog", f"log_proxy_{time_now}.txt")
        # else:
        #     file_name = os.path.join("test_for_watch_dog", f"log_api_{time_now}.txt")
        # with open(file_name, 'a') as f:
        #     for line in self.fp:
        #         f.write(line)



def run_stream_monitor(log_paths, checker_data: Checker_data):
    observer = PollingObserver()
    handlers = []


    # TODO: Keep it the same as the trace reading in the checker.py
    for file in os.listdir(log_paths):
        if file.startswith("trace_") or file.endswith("proxy_log.json"):
            file_path = os.path.join(log_paths, file)
            handler = StreamLogHandler(file_path, checker_data)
            handlers.append(handler)
            watch_dir = os.path.dirname(file_path)
            observer.schedule(handler, path=watch_dir, recursive=False)
            print(f"Watching: {file_path}")

    observer.start()
    return observer
 

def check(invariants: str, log_paths: str):
    param_to_invs, vartype_to_invs, needed_vars, needed_apis, needed_args_map = sort_inv_file(invariants)
    checker_data = Checker_data(needed_vars, needed_apis, needed_args_map)
    observer = run_stream_monitor(log_paths, checker_data)
    num = 0
    failed_inv = set()
    violated_paris = {}
    try:
        while True:
            trace_record = checker_data.check_queue.get()
            if checker_data.check_queue.empty():
                print("queue empty")
                # time.sleep(20)
            if trace_record is None:
                continue
            
            with checker_data.cond:
                while True:
                    if trace_record["time"] > checker_data.min_read_time:
                        print("Wait")
                        checker_data.cond.wait()
                        print("Wake up")
                    else:
                        break

            if "var_name" in trace_record and trace_record["var_name"] is not None:
                varid = VarInstId(trace_record["process_id"], trace_record["var_name"], trace_record["var_type"])
                if varid.var_type in vartype_to_invs:
                    # print(f"matched var_type: {varid.var_type}")
                    for attr_name, invs in vartype_to_invs[varid.var_type].items():
                        attr_name = config.VAR_ATTR_PREFIX + attr_name
                        if attr_name in trace_record and trace_record[attr_name] is not None:
                            # print(f"matched attr_name: {attr_name}")
                            for inv in invs:
                                # print(inv.text_description)
                                # result = inv.relation.online_check(True, inv, trace_record, checker_data)
                                try:
                                    result = inv.relation.online_check(True, inv, trace_record, checker_data)
                                    if not result:
                                        # trace_record1, trace_record2, attr_name = result
                                        # if trace_record1["process_id"] > trace_record2["process_id"]:
                                        #     trace_record1, trace_record2 = trace_record2, trace_record1
                                        # pair = (trace_record1["var_name"], trace_record1["time"], trace_record1["process_id"], trace_record2["var_name"], trace_record2["process_id"], trace_record2["time"])
                                        # if pair not in violated_paris:
                                        #     violated_paris[pair] = 0
                                        # violated_paris[pair] += 1
                                        num += 1
                                        # print(trace_record.process_id, trace_record.time)
                                        print(f"Violated invariant: {inv.text_description}")
                                        failed_inv.add(inv)
                                except Exception as e:
                                    print(inv)
                                    # raise e

            elif "func_call_id" in trace_record and trace_record["func_call_id"] is not None:   
                apiparam = APIParam(trace_record["function"])
                if apiparam in param_to_invs:
                    for inv in param_to_invs[apiparam]:
                        # print(inv.text_description)
                        try:
                            result = inv.relation.online_check(True, inv, trace_record, checker_data)
                            if not result:
                                num += 1
                                print(f"Violated invariant: {inv.text_description}")
                                failed_inv.add(inv)
                        except Exception as e:
                            print(inv)
                            print(inv.text_description)
                            raise e
                        

    except KeyboardInterrupt:
        observer.stop()
        print(f"Total violated times: {num}")
        print(f"Total violated invariants: {len(failed_inv)}")
        # for pair, count in violated_paris.items():
        #     print(f"Pair: {pair}, Count: {count}")
    observer.join()


        
    

def main():
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
    check("/Users/universe/Documents/univer/study/MLSYS/OrderLab/TrainCheck/firsttest/invariants_test.json", "/Users/universe/Documents/univer/study/MLSYS/OrderLab/TrainCheck/firsttest/traincheck_84911_trace")
    # check("/Users/universe/Documents/univer/study/MLSYS/OrderLab/TrainCheck/TrainCheck-Evaluation-Workloads-main/silent-issue-detection/invariants_transformers-33844.json", "/Users/universe/Documents/univer/study/MLSYS/OrderLab/TrainCheck/test_for_con/test")
                
if __name__ == "__main__":
    main()