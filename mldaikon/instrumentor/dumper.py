import datetime
import json  # consider using ORJSON for better performance?
import logging
import os
import threading
from queue import Empty, Queue

from mldaikon.instrumentor.types import PTID

DEBUG = os.environ.get("ML_DAIKON_DEBUG", False)


# per process & thread logging
stop_event = threading.Event()
monitoring_thread = None
trace_API_dumper_queues: dict[PTID, Queue] = {}
trace_VAR_dumper_queues: dict[PTID, Queue] = {}

# per process logging
instrumentation_loggers: dict[int, logging.Logger] = {}


def monitor_main_thread(main_thread, stop_event):
    main_thread.join()  # Wait for the main thread to finish
    print("Main thread has finished or encountered an exception")
    stop_event.set()  # Signal the logging threads to stop


def trace_dumper(task_queue: Queue, trace_file_name: str, stop_event: threading.Event):
    with open(trace_file_name, "w") as f:
        while True:
            try:
                trace = task_queue.get(timeout=0.5)
            except Empty:
                if stop_event.is_set():
                    break
                continue
            trace_str = json.dumps(trace)
            f.write(f"{trace_str}\n")
            task_queue.task_done()


def get_trace_API_dumper_queue():
    global monitoring_thread
    if monitoring_thread is None:
        monitoring_thread = threading.Thread(
            target=monitor_main_thread, args=(threading.main_thread(), stop_event)
        )
        monitoring_thread.start()

    pid = os.getpid()
    tid = threading.current_thread().ident

    ptid = PTID(pid, tid)
    if ptid in trace_API_dumper_queues:
        return trace_API_dumper_queues[ptid]

    output_dir = os.getenv("ML_DAIKON_OUTPUT_DIR")
    assert (
        output_dir is not None
    ), "ML_DAIKON_OUTPUT_DIR is not set, examine the instrumented code to see if os.environ['ML_DAIKON_OUTPUT_DIR'] is set in the main function"

    trace_queue = Queue()
    trace_file_name = f"trace_API_{pid}_{tid}.log"
    trace_file_full_path = os.path.join(output_dir, trace_file_name)
    log_thread = threading.Thread(
        target=trace_dumper, args=(trace_queue, trace_file_full_path, stop_event)
    )
    log_thread.start()

    trace_API_dumper_queues[ptid] = trace_queue
    return trace_queue


def get_trace_VAR_dumper_queue():
    global monitoring_thread
    if monitoring_thread is None:
        monitoring_thread = threading.Thread(
            target=monitor_main_thread, args=(threading.main_thread(), stop_event)
        )
        monitoring_thread.start()

    pid = os.getpid()
    tid = threading.current_thread().ident

    ptid = PTID(pid, tid)
    if ptid in trace_VAR_dumper_queues:
        return trace_VAR_dumper_queues[ptid]

    output_dir = os.getenv("ML_DAIKON_OUTPUT_DIR")
    assert (
        output_dir is not None
    ), "ML_DAIKON_OUTPUT_DIR is not set, examine the instrumented code to see if os.environ['ML_DAIKON_OUTPUT_DIR'] is set in the main function"

    trace_queue = Queue()
    trace_file_name = f"trace_VAR_{pid}_{tid}.log"
    trace_file_full_path = os.path.join(output_dir, trace_file_name)
    log_thread = threading.Thread(
        target=trace_dumper, args=(trace_queue, trace_file_full_path, stop_event)
    )
    log_thread.start()

    trace_VAR_dumper_queues[ptid] = trace_queue
    return trace_queue


def dump_trace_API(trace: dict):
    """add a timestamp (unix) to the trace and dump it to the trace log file"""
    trace_queue = get_trace_API_dumper_queue()
    trace["time"] = datetime.datetime.now().timestamp()
    trace_queue.put(trace)


def dump_trace_VAR(trace: dict):
    """add a timestamp (unix) to the trace and dump it to the trace log file"""
    trace_queue = get_trace_VAR_dumper_queue()
    if "time" not in trace:
        trace["time"] = datetime.datetime.now().timestamp()
    trace_queue.put(trace)


def get_instrumentation_logger_for_process():
    pid = os.getpid()
    output_dir = os.getenv("ML_DAIKON_OUTPUT_DIR")
    assert (
        output_dir is not None
    ), "ML_DAIKON_OUTPUT_DIR is not set, examine the instrumented code to see if os.environ['ML_DAIKON_OUTPUT_DIR'] is set in the main function"

    if pid in instrumentation_loggers:
        return instrumentation_loggers[pid]

    logger = logging.getLogger(f"instrumentation_{pid}")
    if DEBUG:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    log_file = f"instrumentation_{pid}.log"
    file_handler = logging.FileHandler(os.path.join(output_dir, log_file))
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(file_handler)
    instrumentation_loggers[pid] = logger
    return logger
