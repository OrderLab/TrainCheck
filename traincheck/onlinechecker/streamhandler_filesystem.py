import json
import logging
import os
import re
import time

from watchdog.events import FileSystemEventHandler
from watchdog.observers.polling import PollingObserver

from traincheck.config import config
from traincheck.instrumentor.tracer import TraceLineType
from traincheck.instrumentor.types import PTID
from traincheck.trace.types import AttrState, ContextManagerState, Liveness, VarInstId
from traincheck.trace.utils import (
    BindedFuncInput,
    bind_args_kwargs_to_signature,
    flatten_dict,
    load_signature_from_class_method_name,
    replace_none_with_md_none,
)
from traincheck.utils import safe_isnan

from .utils import Checker_data, OnlineFuncCallEvent


class StreamLogHandler(FileSystemEventHandler):
    """A file system handler to monitor the trace log file changes."""

    def __init__(self, file_path, checker_data: Checker_data):
        self.file_path = file_path
        self.fp = open(file_path, "r")

        self.queue = checker_data.check_queue

        self.varid_map = checker_data.varid_map
        self.type_map = checker_data.type_map
        self.pt_map = checker_data.pt_map
        self.process_to_vars = checker_data.process_to_vars
        self.args_map = checker_data.args_map

        self.context_map = checker_data.context_map
        self.init_map = checker_data.init_map

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
        self.logger.debug(f"File {self.file_path} modified at {time.monotonic_ns()}")
        self._handle_line(self.fp)

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
                self.queue.put(trace_record)

            except Exception as e:
                self.logger.error(
                    f"Error processing line in {self.file_path}: {e}. Line content: {line}"
                )
                continue

    def _set_maps(self, trace_record):
        """Set the variable map and function call map based on the trace record."""
        if "var_name" in trace_record and trace_record["var_name"] is not None:
            self._set_var_map(trace_record)
        elif (
            "func_call_id" in trace_record and trace_record["func_call_id"] is not None
        ):
            self._set_func_map(trace_record)

        self._set_read_time(trace_record)

    def _set_var_map(self, trace_record):
        with self.lock:
            var_name = trace_record["var_name"]
            var_type = trace_record["var_type"]
            if var_name in self.needed_vars or var_type in self.needed_vars:
                varid = VarInstId(
                    trace_record["process_id"],
                    trace_record["var_name"],
                    trace_record["var_type"],
                )
                if varid not in self.varid_map:
                    self.varid_map[varid] = {}

                if varid.process_id not in self.process_to_vars:
                    self.process_to_vars[varid.process_id] = set()

                self.process_to_vars[varid.process_id].add(varid)

                for attr_name, value in trace_record.items():
                    if value is None:
                        continue

                    if attr_name.startswith(config.VAR_ATTR_PREFIX):
                        attr_name = attr_name[len(config.VAR_ATTR_PREFIX) :]
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
                        self.varid_map[varid][attr_name] = []
                    else:
                        self.varid_map[varid][attr_name][-1].liveness.end_time = (
                            trace_record["time"]
                        )

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
                    # TODO: check whether dict is necessary here, can be a list?
                    self.pt_map[ptname][func_call_id] = OnlineFuncCallEvent(
                        function_name
                    )
                if trace_type == TraceLineType.FUNC_CALL_PRE:
                    self.pt_map[ptname][func_call_id].pre_record = trace_record
                    self.pt_map[ptname][func_call_id].args = trace_record["args"]
                    self.pt_map[ptname][func_call_id].kwargs = trace_record["kwargs"]
                elif trace_type == TraceLineType.FUNC_CALL_POST:
                    self.pt_map[ptname][func_call_id].post_record = trace_record
                    self.pt_map[ptname][func_call_id].return_values = trace_record[
                        "return_values"
                    ]
                elif trace_type == TraceLineType.FUNC_CALL_POST_EXCEPTION:
                    self.pt_map[ptname][func_call_id].post_record = trace_record
                    self.pt_map[ptname][func_call_id].exception = trace_record[
                        "exception"
                    ]

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

            if (
                ".__enter__" in function_name
                or ".__exit__" in function_name
                or ".__init__" in function_name
            ):
                if (
                    "torch.autograd.grad_mode" not in function_name
                    and "torch.autograd.profiler.record_function" not in function_name
                ):
                    self._set_context_map(trace_record)

    def _set_context_map(self, trace_record):
        function_name = trace_record["function"]
        process_id = trace_record["process_id"]
        thread_id = trace_record["thread_id"]
        ptid = PTID(process_id, thread_id)
        ptname = (process_id, thread_id, function_name)
        trace_type = trace_record["type"]
        if ".__init__" in function_name and trace_type == TraceLineType.FUNC_CALL_PRE:
            context_manager_name = function_name.removesuffix(".__init__")
            ptname = (process_id, thread_id, context_manager_name)
            if ptname not in self.init_map:
                self.init_map[ptname] = []
            self.init_map[ptname].append(trace_record)

        elif (
            ".__enter__" in function_name and trace_type == TraceLineType.FUNC_CALL_POST
        ):
            context_manager_name = function_name.removesuffix(".__enter__")
            ptname = (process_id, thread_id, context_manager_name)
            closest_init_record = None
            closest_init_time = None
            if ptname in self.init_map:
                for init_record in reversed(self.init_map[ptname]):
                    if init_record["time"] < trace_record["time"]:
                        if (
                            closest_init_time is None
                            or init_record["time"] > closest_init_time
                        ):
                            closest_init_time = init_record["time"]
                            closest_init_record = init_record

            start_time = trace_record["time"]
            args = closest_init_record["args"]
            kwargs = closest_init_record["kwargs"]

            if not safe_isnan(args):
                signature = load_signature_from_class_method_name(
                    closest_init_record["function"]
                )

                binded_args_and_kwargs = bind_args_kwargs_to_signature(
                    args, kwargs, signature
                )
            else:
                # create an empty BindedFuncInput if args is NaN, as it indicates
                # that we did not record the args and kwargs for this function call
                binded_args_and_kwargs = BindedFuncInput({})

            if ptid not in self.context_map:
                self.context_map[ptid] = {}
            if context_manager_name not in self.context_map[ptid]:
                self.context_map[ptid][context_manager_name] = []
            self.context_map[ptid][context_manager_name].append(
                ContextManagerState(
                    name=context_manager_name,
                    ptid=ptid,
                    liveness=Liveness(start_time, None),
                    input=binded_args_and_kwargs,
                )
            )
        elif ".__exit__" in function_name and trace_type == TraceLineType.FUNC_CALL_PRE:
            context_manager_name = function_name.removesuffix(".__exit__")
            contextmanagerstate = None
            if ptid in self.context_map:
                if context_manager_name in self.context_map[ptid]:
                    for state in reversed(self.context_map[ptid][context_manager_name]):
                        if state.liveness.start_time < trace_record["time"]:
                            break
                        if state.liveness.end_time is not None:
                            break
                        contextmanagerstate = state
            if contextmanagerstate is not None:
                contextmanagerstate.liveness.end_time = trace_record["time"]

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
                    self.checker_data.read_time_map.items(), default=(None, None)
                )
                if pre_min_read_time != self.checker_data.min_read_time:
                    self.checker_data.cond.notify_all()


def run_stream_monitor(traces, trace_folders, checker_data: Checker_data):
    """Run the stream monitor to watch the trace files and folders."""
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
