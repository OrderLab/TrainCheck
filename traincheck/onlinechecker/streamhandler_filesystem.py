from watchdog.observers import Observer
from watchdog.observers.polling import PollingObserver
from watchdog.events import FileSystemEventHandler
import os
import queue
import re
from traincheck.config import config
import logging
import datetime
import time
import json

from traincheck.trace.utils import flatten_dict, replace_none_with_md_none
from traincheck.trace.types import Liveness
from traincheck.instrumentor.tracer import TraceLineType
from traincheck.trace.types import VarInstId, AttrState, Liveness

from .utils import Checker_data, OnlineFuncCallEvent, timing_info, lock


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
        self.needed_vars = checker_data.needed_vars
        self.needed_apis = checker_data.needed_apis
        self.needed_args_map = checker_data.needed_args_map

        self.min_read_time = checker_data.min_read_time
        self.lock = checker_data.lock
        self.cond = checker_data.cond
        self.checker_data = checker_data
        
        logger = logging.getLogger(__name__)
        self.logger = logger

        self._save_initial_content()

        self.fp.seek(0, 2) 

    def _save_initial_content(self):
        self.logger.info(f"Processing initial content from {self.file_path}")
        self.fp.seek(0)
        lines = self.fp.readlines()
        if not lines:
            return
        
        self._handle_line(lines)
        self.logger.info(f"Initial content from {self.file_path} processed.")

    def on_modified(self, event):
        if os.path.abspath(event.src_path) != os.path.abspath(self.file_path):
            return
        time = datetime.datetime.now()
        self.logger.info(f"File {self.file_path} modified at {time}")
        self._handle_line(self.fp)

    def _handle_line(self, lines):
        start = time.perf_counter()
        for line in lines:
            trace_record = None
            try:
                flat_dict = flatten_dict(
                        json.loads(line, object_hook=replace_none_with_md_none),
                        skip_fields=["args", "kwargs", "return_values"],
                    )
                trace_record = flat_dict
                self._set_maps(trace_record)
                self.queue.put(trace_record)

            except Exception as e:
                self.logger.error(f"Error processing line in {self.file_path}: {e}. Line content: {line}")
                continue

        end = time.perf_counter()
        duration = end - start
        with lock:
            name = self.file_path + "handle_line"
            if name not in timing_info:
                timing_info[name] = []
            timing_info[name].append(duration)

    def _set_maps(self, trace_record):
        if "var_name" in trace_record and trace_record["var_name"] is not None:
            self._set_var_map(trace_record)
        elif "func_call_id" in trace_record and trace_record["func_call_id"] is not None:   
            self._set_func_map(trace_record)
        
        self._set_read_time(trace_record)

    def _set_var_map(self, trace_record):
        with self.lock:
            var_name = trace_record["var_name"]
            var_type = trace_record["var_type"]
            if var_name in self.needed_vars or var_type in self.needed_vars:
                varid = VarInstId(trace_record["process_id"], trace_record["var_name"], trace_record["var_type"])
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

    def _set_func_map(self, trace_record):
        with self.lock:
            function_name = trace_record["function"]
            process_id = trace_record["process_id"]
            thread_id = trace_record["thread_id"]
            ptid = (process_id, thread_id)
            func_call_id = trace_record["func_call_id"]
            ptname = (process_id, thread_id, function_name)
            trace_type = trace_record["type"]
            if function_name in self.needed_apis:
                if ptname not in self.pt_map:
                    self.pt_map[ptname] = {}
                if func_call_id not in self.pt_map[ptname]:
                    self.pt_map[ptname][func_call_id] = OnlineFuncCallEvent(function_name)
                if trace_type == TraceLineType.FUNC_CALL_PRE:
                    self.pt_map[ptname][func_call_id].pre_record = trace_record
                    self.pt_map[ptname][func_call_id].args = trace_record["args"]
                    self.pt_map[ptname][func_call_id].kwargs = trace_record["kwargs"]
                elif trace_type == TraceLineType.FUNC_CALL_POST:
                    self.pt_map[ptname][func_call_id].post_record = trace_record
                    self.pt_map[ptname][func_call_id].return_values = trace_record["return_values"]
                elif trace_type == TraceLineType.FUNC_CALL_POST_EXCEPTION:
                    self.pt_map[ptname][func_call_id].post_record = trace_record
                    self.pt_map[ptname][func_call_id].exception = trace_record["exception"]

            if trace_type == TraceLineType.FUNC_CALL_PRE:
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
            
    def _set_read_time(self, trace_record):
        with self.cond:
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


def run_stream_monitor(traces, trace_folders, checker_data: Checker_data):
    logger = logging.getLogger(__name__)
    observer = PollingObserver()
    handlers = []
    if traces is not None:
        file_path = os.path.abspath(traces[0])
        handler = StreamLogHandler(file_path, checker_data)
        handlers.append(handler)
        watch_dir = os.path.dirname(file_path)
        observer.schedule(handler, path=watch_dir, recursive=False)
        logger.info(f"Watching: {file_path}")

    if trace_folders is not None:
        for trace_folder in trace_folders:
            for file in os.listdir(trace_folder):
                if file.startswith("trace_") or file.endswith("proxy_log.json"):
                    file_path = os.path.join(trace_folder, file)
                    handler = StreamLogHandler(file_path, checker_data)
                    handlers.append(handler)
                    watch_dir = os.path.dirname(file_path)
                    observer.schedule(handler, path=watch_dir, recursive=False)
                    logger.info(f"Watching: {file_path}")

    observer.start()
    return observer