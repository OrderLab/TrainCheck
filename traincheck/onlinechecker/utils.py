import queue
import threading

class Checker_data:
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