import copy
import queue
import threading

from traincheck.instrumentor.types import PTID
from traincheck.trace.types import (
    FuncCallEvent,
    VarChangeEvent,
    VarInstId,
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

        self.context_map = {}
        self.init_map = {}

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

def get_var_ids_unchanged_but_causally_related(
    func_call_id: str,
    var_type: str | None = None,
    attr_name: str | None = None,
    trace_record: dict = None,
    checker_data: Checker_data = None,
) -> list[VarInstId]:
    """Find all variables that are causally related to a function call but not changed within the function call.

    Casually related vars: Variables are accessed or modified by the object that the function call is bound to.
    """
    related_vars = get_causally_related_vars(func_call_id, trace_record, checker_data)
    changed_vars = query_var_changes_within_func_call(
        func_call_id, var_type, attr_name, trace_record, checker_data
    )

    related_vars_not_changed = []
    if var_type is not None:
        related_vars = {
            var_id for var_id in related_vars if var_id.var_type == var_type
        }
        changed_vars = [
            var_change
            for var_change in changed_vars
            if var_change[0].var_type == var_type
        ]
    if attr_name is not None:
        changed_vars = [
            var_change
            for var_change in changed_vars
            if var_change[1] == attr_name
        ]

    for var_id in related_vars:
        if any([var_change[0] == var_id for var_change in changed_vars]):
            continue
        related_vars_not_changed.append(var_id)
    return related_vars_not_changed

def get_causally_related_vars(
    func_call_id, trace_record, checker_data
) -> set[VarInstId]:
    """Find all variables that are causally related to a function call.
    By causally related, we mean that the variables have been accessed or modified by the object (with another method) that the function call is made on.
    """

    ptid = (trace_record["process_id"], trace_record["thread_id"])
    process_id = trace_record["process_id"]
    thread_id = trace_record["thread_id"]
    func_name = trace_record["function"]
    func_id = trace_record["func_call_id"]
    ptname = (process_id, thread_id, func_name)
    with checker_data.lock:
        func_call_pre_event = checker_data.pt_map[ptname][func_id].pre_record

        if func_call_pre_event is None:
            raise ValueError(
                f"Function call pre-event not found for func_call_id: {func_call_id}"
            )

    assert func_call_pre_event[
        "is_bound_method"
    ], f"Causal relation extraction is only supported for bound methods, got {func_call_pre_event['function']} which is not"

    obj_id = func_call_pre_event["obj_id"]

    causally_related_var_ids: set[VarInstId] = set()

    with checker_data.lock:
        for _, calls in checker_data.pt_map.items():
            for call_id, record in calls.items():
                if (
                    record.pre_record["obj_id"] == obj_id
                    and record.pre_record["time"] < func_call_pre_event["time"]
                ):
                    assert (
                        record.pre_record["process_id"]
                        == func_call_pre_event["process_id"]
                    ), "Related function call is on a different process."
                    assert (
                        record.pre_record["thread_id"]
                        == func_call_pre_event["thread_id"]
                    ), "Related function call is on a different thread."

                    for var_name, var_type in record.pre_record["proxy_obj_names"]:
                        if var_name == "" and var_type == "":
                            continue
                        causally_related_var_ids.add(
                            VarInstId(
                                record.pre_record["process_id"], var_name, var_type
                            )
                        )

    return causally_related_var_ids

def query_var_changes_within_func_call(
    func_call_id: str,
    var_type: str,
    attr_name: str,
    trace_record: dict,
    checker_data: Checker_data,
) -> list[VarChangeEvent]:
    """Extract all variable change events from the trace, within the duration of a specific function call."""
    process_id = trace_record["process_id"]
    thread_id = trace_record["thread_id"]
    func_name = trace_record["function"]
    func_id = trace_record["func_call_id"]
    ptname = (process_id, thread_id, func_name)
    with checker_data.lock:
        func_call_event = checker_data.pt_map[ptname][func_id]
        pre_record = func_call_event.pre_record

        post_record = func_call_event.post_record

        start_time = pre_record["time"]
        end_time = post_record["time"]

    return query_var_changes_within_time_and_process(
        (start_time, end_time),
        var_type,
        attr_name,
        trace_record["process_id"],
        checker_data,
    )


def query_var_changes_within_time_and_process(
    time_range: tuple[int | float, int | float],
    var_type: str,
    attr_name: str,
    process_id: int,
    checker_data: Checker_data,
) -> list[VarChangeEvent]:
    """Extract all variable change events from the trace, within a specific time range and process."""
    events = []
    with checker_data.lock:
        for varid in checker_data.type_map[var_type]:
            for i in reversed(range(1, len(checker_data.varid_map[varid][attr_name]))):

                change_time = checker_data.varid_map[varid][attr_name][
                    i
                ].liveness.start_time
                if change_time <= time_range[0]:
                    break
                if change_time > time_range[1]:
                    continue
                new_state = checker_data.varid_map[varid][attr_name][i]
                old_state = checker_data.varid_map[varid][attr_name][i - 1]
                if new_state.value == old_state.value:
                    continue
                events.append((varid, attr_name))
    return events


def get_var_raw_event_before_time(
    var_id: VarInstId, time: int, checker_data: Checker_data
) -> list[dict]:
    """Get all original trace records of a variable before the specified time."""

    raw_events = []
    with checker_data.lock:
        for attr_name, records in checker_data.varid_map[var_id].items():
            for record in records:
                if record.liveness.start_time < time:
                    raw_events.append(record.traces[-1])

    return raw_events

def get_meta_vars_online(
    time: float, precess_id:int, thread_id:int, checker_data: Checker_data
):
    ptid = PTID(precess_id, thread_id)
    active_context_managers = []
    meta_vars = {}

    if ptid not in checker_data.context_map:
        return None
    context_managers = checker_data.context_map[ptid]
    for context_manager_name, context_manager_states in context_managers.items():
        for context_manager_state in reversed(context_manager_states):
            if context_manager_state.liveness.start_time <= time \
                and (
                    context_manager_state.liveness.end_time is None
                    or context_manager_state.liveness.end_time >= time
                ):
                active_context_managers.append(context_manager_state)
    
    prefix = "context_managers"
    for _, context_manager in enumerate(active_context_managers):
        meta_vars[f"{prefix}.{context_manager.name}"] = context_manager.to_dict()[
            "input"
        ]

    return meta_vars

def set_meta_vars_online(
        records: list, checker_data: Checker_data
):
    earliest_time = None
    earliest_process_id = None
    earliest_thread_id = None
    for record in records:
        if earliest_time is None or record["time"] < earliest_time:
            earliest_time = record["time"]
            earliest_process_id = record["process_id"]
            earliest_thread_id = record["thread_id"]
    meta_vars = get_meta_vars_online(
        earliest_time, earliest_process_id, earliest_thread_id, checker_data
    )

    if meta_vars:
        for key in meta_vars:
            for i in range(len(records)):
                records[i][f"meta_vars.{key}"] = meta_vars[key]
    return records
    

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