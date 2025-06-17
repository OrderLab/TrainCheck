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

def sort_inv_file(invariants: str):
    invs = read_inv_file(invariants)
    param_to_invs : dict[Param, list[Invariant]] = {}
    vartype_to_invs : dict[str, dict[str, list[Invariant]]] = {}
    for inv in invs:
        assert (
            inv.precondition is not None
        ), "Invariant precondition is None. It should at least be 'Unconditional' or an empty list. Please check the invariant file and the inference process."
        params = inv.relation.get_mapping_key(inv)
        # TODO: param_to_invs 细分为 contain, lead, cover
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
    return param_to_invs, vartype_to_invs


class Trace_record:
    def __init__(self, flat_dict=None):
        self.func_call_id = None
        self.thread_id = None
        self.process_id = None
        self.meta_vars_step = None
        self.type = None
        self.function = None
        self.is_bound_method = None
        self.obj_id = None
        self.args = None
        self.kwargs = None
        self.time = None
        self.return_values = None
        self.exception = None
        self.exception_msg = None
        self.meta_vars_stage = None
        self.proxy_obj_names = None
        self.var_name = None
        self.var_type = None
        self.mode = None
        self.dump_loc = None
        # TODO: attributes and meta_vars not use dict
        self.attributes = {
            "_ML_DAIKON_data_ID": None,
            "data": None,
            "dtype": None,
            "grad": None,
            "grad_fn": None,
            "is_cpu": None,
            "is_cuda": None,
            "is_ipu": None,
            "is_leaf": None,
            "is_meta": None,
            "is_mkldnn": None,
            "is_mps": None,
            "is_mtia": None,
            "is_nested": None,
            "is_ort": None,
            "is_quantized": None,
            "is_sparse": None,
            "is_sparse_csr": None,
            "is_vulkan": None,
            "is_xla": None,
            "is_xpu": None,
            "itemsize": None,
            "name": None,
            "nbytes": None,
            "ndim": None,
            "requires_grad": None,
            "retains_grad": None,
            "shape": None,
            "_ML_DAIKON_grad_ID": None
        }
        # TODO: check contain all meta data, and this is the same for many trace record
        self.meta_vars = {
            "step": None,
            "stage": None,
            "_TENSOR_MODEL_PARALLEL_GROUP": None,
            "_PIPELINE_MODEL_PARALLEL_GROUP": None,
            "_MODEL_PARALLEL_GROUP": None,
            "_EMBEDDING_GROUP": None,
            "_DATA_PARALLEL_GROUP": None
        }
        self.args = None
        self.kwargs = None
        self.return_values = None
        if flat_dict:
            self._load_from_flat_dict(flat_dict)

    def _load_from_flat_dict(self, flat_dict):
        for key, value in flat_dict.items():
            if key.startswith("attributes."):
                attr_key = key[len("attributes."):]
                self.attributes[attr_key] = value
            elif key.startswith("meta_vars."):
                meta_key = key[len("meta_vars."):]
                self.meta_vars[meta_key] = value
                
            elif hasattr(self, key):
                setattr(self, key, value)
            # TODO: else log

# ! NOTE: this is different from the one in traincheck/trace/types.py
class AttrState:
    def __init__(self, value: type, liveness: Liveness, trace_record: Trace_record):
        self.value: type = value
        self.liveness: Liveness = liveness
        self.trace_record = [trace_record]

    def __str__(self):
        return f"Value: {self.value}, Liveness: {self.liveness}"

    def __eq__(self, other):
        return self.value == other.value and self.liveness == other.liveness

# ! NOTE: this is different from the one in traincheck/trace/types.py
class FuncCallEvent:
    def __init__(self):
        self.pre_record = None
        self.post_record = None
    

# ! NOTE: not thread safe
class Checker_data:
    def __init__(self):
        self.trace_records = []
        self.check_queue = queue.Queue()
        self.varid_map = {}
        self.type_map = {}
        self.pt_map = {}
        self.process_to_vars = {}

        self.min_read_time = {}
        self.read_lock = threading.Lock()
        self.cond = threading.Condition(self.read_lock)

class StreamLogHandler(FileSystemEventHandler):
    def __init__(self, file_path, checker_data: Checker_data):
        self.file_path = file_path
        self.fp = open(file_path, 'r')

        self.trace_records = []
        self.queue = checker_data.check_queue

        # TODO: these map should not belong to this class
        self.varid_map = checker_data.varid_map
        self.type_map = checker_data.type_map
        self.pt_map = checker_data.pt_map
        self.process_to_vars = checker_data.process_to_vars
        self.min_read_time = checker_data.min_read_time
        self.read_lock = checker_data.read_lock
        self.cond = checker_data.cond
        self.checker_data = checker_data

        self._save_initial_content()

        self.fp.seek(0, 2) 

    def _set_maps(self, trace_record):
        if trace_record.var_type is not None or trace_record.var_name is not None:
            varid = VarInstId(trace_record.process_id, trace_record.var_name, trace_record.var_type)
            if varid not in self.varid_map:
                self.varid_map[varid] = {}

            if varid.process_id not in self.process_to_vars:
                self.process_to_vars[varid.process_id] = set()

            self.process_to_vars[varid.process_id].add(varid)
            
            if trace_record.attributes is not None:
                for attr_name, value in trace_record.attributes.items():
                    if value is None:
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
                                Liveness(trace_record.time, None),
                                trace_record,
                            )    
                        ]
                    else:
                        
                        self.varid_map[varid][attr_name][-1].liveness.end_time = trace_record.time
                        self.varid_map[varid][attr_name].append(
                            AttrState(
                                curr_value,
                                Liveness(trace_record.time, None),
                                trace_record,
                            )
                        )
                        

            if trace_record.var_type is not None:
                if trace_record.var_type not in self.type_map:
                    self.type_map[trace_record.var_type] = set()
                self.type_map[trace_record.var_type].add(varid)
        elif trace_record.func_call_id is not None:   
            process_id = trace_record.process_id
            thread_id = trace_record.thread_id
            ptid = (process_id, thread_id)
            if ptid not in self.pt_map:
                self.pt_map[ptid] = {}
            if trace_record.function not in self.pt_map[ptid]:
                self.pt_map[ptid][trace_record.function] = {}
            if trace_record.func_call_id not in self.pt_map[ptid][trace_record.function]:
                self.pt_map[ptid][trace_record.function][trace_record.func_call_id] = FuncCallEvent()
            if trace_record.type == TraceLineType.FUNC_CALL_PRE:
                self.pt_map[ptid][trace_record.function][trace_record.func_call_id].pre_record = trace_record
            elif trace_record.type == TraceLineType.FUNC_CALL_POST:
                self.pt_map[ptid][trace_record.function][trace_record.func_call_id].post_record = trace_record
            elif trace_record.type == TraceLineType.FUNC_CALL_POST_EXCEPTION:
                self.pt_map[ptid][trace_record.function][trace_record.func_call_id].post_record = trace_record


        self.queue.put(trace_record)

        with self.checker_data.cond:
            self.checker_data.min_read_time[self.file_path] = trace_record.time
            self.checker_data.cond.notify_all()
            


    def _handle_line(self, lines):
        for line in lines:
            trace_record = None
            try:
                flat_dict = flatten_dict(
                        json.loads(line, object_hook=replace_none_with_md_none),
                        skip_fields=["args", "kwargs", "return_values"],
                    )
                trace_record = Trace_record(flat_dict)
                self.trace_records.append(trace_record)
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
    param_to_invs, vartype_to_invs = sort_inv_file(invariants)
    checker_data = Checker_data()
    observer = run_stream_monitor(log_paths, checker_data)
    num = 0
    failed_inv = set()
    try:
        while True:
            trace_record = checker_data.check_queue.get()
            if checker_data.check_queue.empty():
                print("queue empty")
                # time.sleep(20)
            if trace_record is None:
                continue
                       
            # with checker_data.cond:
            #     if checker_data.min_read_time is None or trace_record.time >= checker_data.min_read_time[1]:
            #         print("TOO QUICK")
            #         # print(checker_data.min_read_time)
            #         # print(f"trace_record time: {trace_record.time}")
            #         print(f"Waiting")
            #         checker_data.cond.wait() 
            #         print("Wake up")
 
            # print("check trace record")
            if trace_record.var_type is not None or trace_record.var_name is not None:
                varid = VarInstId(trace_record.process_id, trace_record.var_name, trace_record.var_type)
                if varid.var_type in vartype_to_invs:
                    # print(f"matched var_type: {varid.var_type}")
                    for attr_name, invs in vartype_to_invs[varid.var_type].items():
                        if attr_name in trace_record.attributes and trace_record.attributes[attr_name] is not None:
                            # print(f"matched attr_name: {attr_name}")
                            for inv in invs:
                                print(inv.text_description)
                                result = inv.relation.online_check(True, inv, trace_record, checker_data)
                                if not result:
                                    num += 1
                                    print(f"Violated invariant: {inv.text_description}")
            elif trace_record.func_call_id is not None:
                apiparam = APIParam(trace_record.function)
                if apiparam in param_to_invs:
                    for inv in param_to_invs[apiparam]:
                        print(inv.text_description)
                        with checker_data.cond:
                            while True:
                                min_time = None
                                for _ , min_read_time in checker_data.min_read_time.items():
                                    if min_time is None or min_time > min_read_time:
                                        min_time = min_read_time
                                if trace_record.time > min_time:
                                    print("Wait")
                                    checker_data.cond.wait()
                                    print("Wake up")
                                else:
                                    break
                        result = inv.relation.online_check(True, inv, trace_record, checker_data)
                        if not result:
                            num += 1
                            print(f"Violated invariant: {inv.text_description}")
                            failed_inv.add(inv)

    except KeyboardInterrupt:
        observer.stop()
        print(f"Total violated trace: {num}")
        print(f"Total violated invariant: {len(failed_inv)}")
    observer.join()


        
    

def main():
    # print(aaaaa)
    # check("/Users/universe/Documents/univer/study/MLSYS/TrainCheck/firsttest/invariants_test.json", "/Users/universe/Documents/univer/study/MLSYS/TrainCheck/firsttest/traincheck_mnist_trace")
    # check("/Users/universe/Documents/univer/study/MLSYS/TrainCheck/firsttest/invariants_test.json", "/Users/universe/Documents/univer/study/MLSYS/TrainCheck/firsttest/traincheck_84911_trace")
    check("/Users/universe/Documents/univer/study/MLSYS/TrainCheck/firsttest/invariants_test.json", "/Users/universe/Documents/univer/study/MLSYS/TrainCheck/firsttest/test1")
    # check("/Users/universe/Documents/univer/study/MLSYS/TrainCheck/firsttest/invariants.json", "/Users/universe/Documents/univer/study/MLSYS/TrainCheck/firsttest/traincheck_mnist_trace")
    # check("/Users/universe/Documents/univer/study/MLSYS/TrainCheck/test_for_con/invariants_deepspeed-1801-fp16.json", "/Users/universe/Documents/univer/study/MLSYS/TrainCheck/test_for_con/trace_deepspeed-1801")
    # check("/Users/universe/Documents/univer/study/MLSYS/TrainCheck/test_for_con/invariants_deepspeed-1801-fp16.json", "/Users/universe/Documents/univer/study/MLSYS/TrainCheck/test_for_con/trace_test")
                
if __name__ == "__main__":
    main()