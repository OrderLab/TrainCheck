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

class StreamLogHandler(FileSystemEventHandler):
    def __init__(self, file_path):
        self.file_path = file_path
        self.fp = open(file_path, 'r')

        self._save_initial_content()

        self.fp.seek(0, 2) 

    def _save_initial_content(self):
        self.fp.seek(0)
        lines = self.fp.readlines()
        if not lines:
            return
        
        # TODO: remove, just for check correctness
        time_now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        dir_name = "test_for_watch_dog"
        os.makedirs(dir_name, exist_ok=True)
        if self.file_path.endswith(".json"):
            file_name = os.path.join(dir_name, f"log_init_proxy_{time_now}.txt")
        else:
            file_name = os.path.join(dir_name, f"log_init_api_{time_now}.txt")
        with open(file_name, 'w') as f:
            f.writelines(lines)

    def on_modified(self, event):
        if os.path.abspath(event.src_path) != os.path.abspath(self.file_path):
            return

        # TODO: remove, just for check correctness
        time_now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        dir_name = "test_for_watch_dog"
        os.makedirs(dir_name, exist_ok=True)
        if self.file_path.endswith(".json"):
            file_name = os.path.join("test_for_watch_dog", f"log_proxy_{time_now}.txt")
        else:
            file_name = os.path.join("test_for_watch_dog", f"log_api_{time_now}.txt")
        with open(file_name, 'a') as f:
            for line in self.fp:
                f.write(line)

def sort_inv_file(invariants: str):
    invs = read_inv_file(invariants)
    param_to_invs : dict[Param, list[Invariant]] = {}
    print(len(invs))
    for inv in invs:
        assert (
            inv.precondition is not None
        ), "Invariant precondition is None. It should at least be 'Unconditional' or an empty list. Please check the invariant file and the inference process."
        params = inv.relation.get_mapping_key(inv)
        for param in params:
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
    return param_to_invs

def run_stream_check(log_paths):
    observer = PollingObserver()
    handlers = []


    # TODO: Keep it the same as the trace reading in the checker.py
    for file in os.listdir(log_paths):
        if file.startswith("trace_") or file.endswith("proxy_log.json"):
            file_path = os.path.join(log_paths, file)
            handler = StreamLogHandler(file_path)
            handlers.append(handler)
            watch_dir = os.path.dirname(file_path)
            observer.schedule(handler, path=watch_dir, recursive=False)
            print(f"Watching: {file_path}")

    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

def check(invariants: str, log_paths: str):
    param_to_invs = sort_inv_file(invariants)
    run_stream_check(log_paths)

def main():
    # print(aaaaa)
    check("/Users/universe/Documents/univer/study/MLSYS/TrainCheck/firsttest/invariants.json", "test")
                
if __name__ == "__main__":
    main()