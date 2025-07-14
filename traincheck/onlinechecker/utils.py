import queue
import threading

from traincheck.trace.types import (
    FuncCallEvent,
)

class Checker_data:
    """Data structure for online checker threads. Holds the needed data and the queue for processing.
    """
    def __init__(self, needed_data):
        needed_vars, needed_apis, needed_args_map = needed_data
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
        self.lock = threading.Lock()
        self.cond = threading.Condition(self.lock)

class OnlineFuncCallEvent(FuncCallEvent):
    """A function call event for online checking."""
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

# use for time analysis
# timing_info = {}
# lock = threading.Lock()

# def profile_section(name):
#     def decorator(func):
#         def wrapper(*args, **kwargs):
#             start = time.perf_counter()
#             result = func(*args, **kwargs)
#             end = time.perf_counter()
#             duration = end - start
#             with lock:
#                 if name not in timing_info:
#                     timing_info[name] = []
#                 timing_info[name].append(duration)
#             return result
#         return wrapper
#     return decorator