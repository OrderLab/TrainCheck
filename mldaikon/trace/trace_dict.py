import logging
import re
import json
import pandas as pd
from pandas import json_normalize
from tqdm import tqdm

from mldaikon.config import config
from mldaikon.instrumentor.tracer import TraceLineType
from mldaikon.trace.types import (
    AttrState,
    FuncCallEvent,
    FuncCallExceptionEvent,
    Liveness,
    VarChangeEvent,
    VarInstId,
)
from collections import defaultdict

logger = logging.getLogger(__name__)

def get_attr_name(col_name: str) -> str:
    if config.VAR_ATTR_PREFIX not in col_name:
        raise ValueError(f"{col_name} does not contain the tracker_var_field_prefix.")
    return col_name[len(config.VAR_ATTR_PREFIX) :]

class TraceCache:
    def __init__(self, api_trace_file_path, var_trace_file_path, truncate_incomplete_func_calls=True):
        self.var_ids = None
        self.var_insts = None
        self.var_changes = None
        self.api_trace_file_path = api_trace_file_path
        self.var_trace_file_path = var_trace_file_path
        self.events = defaultdict(lambda: defaultdict(list))
        self.vars = defaultdict(lambda: defaultdict(list))
        self.read_log_files(api_trace_file_path, mode = "api")  # Updated to handle list of file paths
        self.read_log_files(var_trace_file_path, mode = "var")  # Updated to handle list of file paths
        if truncate_incomplete_func_calls:
            self._rm_incomplete_trailing_func_calls()

    def load_log_data(self, log_data, mode):
        for entry in log_data:
            process_id = entry["process_id"]
            thread_id = entry["thread_id"]
            if mode == "api":
                self.events[process_id][thread_id].append(entry)
            elif mode == "var":
                self.vars[process_id][thread_id].append(entry)

    def read_log_file(self, file_path, mode):
        with open(file_path, 'r') as file:
            for line in file:
                try:
                    log_data = json.loads(line.strip())
                    self.load_log_data([log_data], mode)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON on line: {line}")
                    print(f"Error message: {e}")

    def read_log_files(self, file_paths, mode):
        """Reads one or more log files depending on if file_paths is a list or a single path."""
        if isinstance(file_paths, list):
            for file_path in file_paths:
                self.read_log_file(file_path, mode)
        else:
            self.read_log_file(file_paths, mode)



    def get_start_time(self, process_id=None, thread_id=None) -> float:
        """Get the start time of the trace. If process_id or thread_id is provided, 
        the start time of the specific process or thread will be returned."""
    
        if process_id is not None and thread_id is not None:
            return min(self.events[process_id][thread_id])

        if process_id is not None:
            start_times = [min(times) for times in self.events[process_id].values()]
            return min(start_times)

        if thread_id is not None:
            start_times = []
            for process in self.events.values():
                if thread_id in process:
                    start_times.extend(process[thread_id])
            return min(start_times)

        start_times = []
        for process in self.events.values():
            for times in process.values():
                start_times.extend(times)
        return min(start_times)

    def get_end_time(self, process_id=None, thread_id=None) -> float:
        """Get the start time of the trace. If process_id or thread_id is provided, 
        the start time of the specific process or thread will be returned."""
    
        if process_id is not None and thread_id is not None:
            return max(self.events[process_id][thread_id])

        if process_id is not None:
            start_times = [max(times) for times in self.events[process_id].values()]
            return max(start_times)

        if thread_id is not None:
            start_times = []
            for process in self.events.values():
                if thread_id in process:
                    start_times.extend(process[thread_id])
            return max(start_times)

        start_times = []
        for process in self.events.values():
            for times in process.values():
                start_times.extend(times)
        return max(start_times)

    def _rm_incomplete_trailing_func_calls(self):
        """Remove incomplete trailing function calls from the trace."""
        logger = logging.getLogger(__name__)


        func_call_map = defaultdict(list)
        for process_id, threads in self.events.items():
            for thread_id, records in threads.items():
                for record in records:
                    func_call_map[record["func_call_id"]].append(record)


        incomplete_func_call_ids = [
            func_call_id
            for func_call_id, records in func_call_map.items()
            if len(records) == 1
        ]

        for func_call_id in incomplete_func_call_ids:
            row = func_call_map[func_call_id][0]
            assert (
                row["type"] == "function_call (pre)"
            ), "Incomplete function call is not a pre-call event."
            logger.warning(f"Incomplete function call detected: {row}")
            process_id = row["process_id"]
            thread_id = row["thread_id"]


            outermost_func_call_pre = min(
                (r for r in self.events[process_id][thread_id] if r["type"] == "function_call (pre)"),
                key=lambda r: r["time"],
                default=None,
            )

            if outermost_func_call_pre is None:
                continue


            outermost_func_call_post = next(
                (
                    r
                    for r in self.events[process_id][thread_id]
                    if r["func_call_id"] == outermost_func_call_pre["func_call_id"]
                    and r["type"] == "function_call (post)"
                ),
                None,
            )

            if row["func_call_id"] == outermost_func_call_pre["func_call_id"]:
                logger.warning(
                    f"The outermost function call is incomplete: {outermost_func_call_pre['function']} with id {outermost_func_call_pre['func_call_id']}. Will treat it as a complete function call."
                )
                continue

            if outermost_func_call_post is not None:
                assert (
                    thread_id != outermost_func_call_pre["thread_id"]
                ), f"""Incomplete function call (func_call_id: {row['func_call_id']}) (name: {row["function"]}) is not on a different thread than outermost function (func_call_id: {outermost_func_call_pre['func_call_id']}) (name: {outermost_func_call_pre["function"]}) on process {process_id}. Please investigate."""

                if row["time"] > outermost_func_call_post["time"] - config.INCOMPLETE_FUNC_CALL_SECONDS_TO_OUTERMOST_POST:
                    logger.warning(f"Removing incomplete function call: {row}")
                    self.events[process_id][thread_id] = [
                        r for r in self.events[process_id][thread_id]
                        if r["func_call_id"] != row["func_call_id"]
                    ]
                else:
                    raise ValueError(
                        f"Incomplete function call is not close enough to the outermost function call post event: {row}"
                    )
            else:
                self.events[process_id][thread_id] = [
                    r for r in self.events[process_id][thread_id]
                    if r["time"] < row["time"]
                ]

    def get_process_ids(self):
        return list(self.events.keys())
    
    def get_thread_ids(self):
        thread_ids = set()
        for process_id in self.events:
            for thread_id in self.events[process_id]:
                thread_ids.add(thread_id)

        return list(thread_ids)
    
    def get_func_call_ids(self, process_id, thread_id):
        func_call_ids = set()
        for process_id in self.events:
            for thread_id in self.events[process_id]:
                for event in self.events[process_id][thread_id]:
                    func_call_ids.add(event["func_call_id"])

        return list(func_call_ids)
        
    def get_func_names(self):
        func_names = set()
        for process_id in self.events:
            for thread_id in self.events[process_id]:
                for event in self.events[process_id][thread_id]:
                    func_names.add(event["function"])

        return list(func_names)
    

    def get_func_is_bound_method(self, func_name: str) -> bool:
        """Check if a function is bound to a class (i.e. method of an object).

        Args:
            func_name (str): The name of the function.

        Returns:
            bool: True if the function is bound to a class, False otherwise.

        Raises:
            AssertionError: If the boundness information is not found for the function.
        """
        is_bound_method = []

        # 遍历 self.events 的嵌套字典结构
        for process_id, threads in self.events.items():
            for thread_id, records in threads.items():
                for record in records:
                    if record["function"] == func_name and record["is_bound_method"] is not None:
                        is_bound_method.append(record["is_bound_method"])

        if not is_bound_method:
            raise AssertionError(f"Boundness information not found for {func_name}")

        # 检查所有 is_bound_method 是否一致
        assert all(
            is_bound_method[0] == is_bound_method[i]
            for i in range(1, len(is_bound_method))
        ), f"Boundness information is not consistent for {func_name}"

        return is_bound_method[0]


    def get_causally_related_vars(self, func_call_id) -> set[VarInstId]:
        """Find all variables that are causally related to a function call.
        By causally related, we mean that the variables have been accessed or modified by the object (with another method) that the function call is made on.
        """
        
        # 查找特定 func_call_id 的 pre-call 事件
        func_call_pre_event = None
        for process_id, threads in self.events.items():
            for thread_id, records in threads.items():
                for record in records:
                    if (
                        record["type"] == TraceLineType.FUNC_CALL_PRE
                        and record["func_call_id"] == func_call_id
                    ):
                        func_call_pre_event = record
                        break
                if func_call_pre_event:
                    break
            if func_call_pre_event:
                break
        
        if func_call_pre_event is None:
            raise ValueError(f"Function call pre-event not found for func_call_id: {func_call_id}")

        # 确保该方法是绑定到某个对象上的
        assert func_call_pre_event[
            "is_bound_method"
        ], f"Causal relation extraction is only supported for bound methods, got {func_call_pre_event['function']} which is not"

        obj_id = func_call_pre_event[
            "obj_id"
        ]  # 获取函数绑定对象的 ID

        # 查找在该函数调用之前与此对象相关的所有函数调用
        causally_related_var_ids: set[VarInstId] = set()

        for process_id, threads in self.events.items():
            for thread_id, records in threads.items():
                for record in records:
                    if (
                        record["type"] == TraceLineType.FUNC_CALL_PRE
                        and record["obj_id"] == obj_id
                        and record["time"] < func_call_pre_event["time"]
                    ):
                        assert (
                            record["process_id"] == func_call_pre_event["process_id"]
                        ), "Related function call is on a different process."
                        assert (
                            record["thread_id"] == func_call_pre_event["thread_id"]
                        ), "Related function call is on a different thread."
                        
                        for var_name, var_type in record["proxy_obj_names"]:
                            if var_name == "" and var_type == "":
                                continue
                            causally_related_var_ids.add(VarInstId(process_id, var_name, var_type))

        return causally_related_var_ids


    def get_var_ids_unchanged_but_causally_related(
        self,
        func_call_id: str,
        var_type: str | None = None,
        attr_name: str | None = None,
    ) -> list[VarInstId]:
        """Find all variables that are causally related to a function call but not changed within the function call.

        Causally related vars: Variables are accessed or modified by the object that the function call is bound to.
        """

        # 获取与此函数调用相关的变量
        related_vars = self.get_causally_related_vars(func_call_id)

        # 获取在函数调用中发生变化的变量
        changed_vars = self.query_var_changes_within_func_call(func_call_id)

        # 过滤条件
        if var_type is not None:
            related_vars = {var_id for var_id in related_vars if var_id.var_type == var_type}
            changed_vars = [var_change for var_change in changed_vars if var_change.var_id.var_type == var_type]

        if attr_name is not None:
            changed_vars = [var_change for var_change in changed_vars if var_change.attr_name == attr_name]

        # 找出相关但未发生变化的变量
        related_vars_not_changed = []
        for var_id in related_vars:
            if not any(var_change.var_id == var_id for var_change in changed_vars):
                related_vars_not_changed.append(var_id)

        return related_vars_not_changed


    def get_var_ids(self) -> list[VarInstId]:
        """Find all variables (uniquely identified by name, type, and process id) from the trace."""
        
        # 如果 var_ids 已经计算过，直接返回
        if self.var_ids is not None:
            return self.var_ids

        variables_set = set()  # 用于存储唯一的变量

        # 遍历 self.events 的嵌套字典结构
        for process_id, threads in self.vars.items():
            for thread_id, records in threads.items():
                for record in records:
                    var_name = record.get("var_name")
                    var_type = record.get("var_type")
                    if var_name and var_type:
                        variables_set.add((process_id, var_name, var_type))

        # 将唯一的变量转换为 VarInstId 对象并存储在 self.var_ids 中
        self.var_ids = [VarInstId(process_id, var_name, var_type) for process_id, var_name, var_type in variables_set]

        return self.var_ids


    def get_var_insts(self) -> dict[VarInstId, dict[str, list[AttrState]]]:
        """Index and get all variable instances from the trace.

        Returns:
            dict[VarInstId, dict[str, list[AttrState]]]: A dictionary mapping variable instances to their attributes and their states.
            {
                VarInstId(process_id, var_name, var_type): {
                    attr_name: [AttrState(value, liveness, traces), ...],
                    ...
                },
            }
        """

        if self.var_insts is not None:
            return self.var_insts

        var_ids = self.get_var_ids()
        if len(var_ids) == 0:
            logger.warning("No variables found in the trace.")
            return {}

        var_insts = {}

        for var_id in tqdm(var_ids, desc="Indexing Variable Instances"):
            # 获取相关的状态变化事件，并按时间排序
            state_changes = []
            for process_id, threads in self.vars.items():
                if process_id != var_id.process_id:
                    continue
                for thread_id, records in threads.items():
                    for record in records:
                        if (
                            record["var_name"] == var_id.var_name
                            and record["var_type"] == var_id.var_type
                            and record["type"] == TraceLineType.STATE_CHANGE
                        ):
                            state_changes.append(record)

            # 按时间排序
            state_changes.sort(key=lambda x: x["time"])

            # 初始化此变量的属性值
            attr_values = {}
            for state_change in state_changes:
                for col, value in state_change.items():
                    if col.startswith(config.VAR_ATTR_PREFIX):
                        attr_name = get_attr_name(col)

                        # 跳过不需要处理的属性
                        if any(re.match(pattern, attr_name) for pattern in config.PROP_ATTR_PATTERNS) or any(
                            isinstance(value, _type) for _type in config.PROP_ATTR_TYPES
                        ):
                            continue

                        if attr_name not in attr_values:
                            attr_values[attr_name] = [
                                AttrState(
                                    value,
                                    Liveness(state_change["time"], None),
                                    [state_change],
                                )
                            ]
                        else:
                            if attr_values[attr_name][-1].value != value:
                                attr_values[attr_name][-1].liveness.end_time = state_change["time"]
                                attr_values[attr_name].append(
                                    AttrState(
                                        value,
                                        Liveness(state_change["time"], None),
                                        [state_change],
                                    )
                                )
                            else:
                                attr_values[attr_name][-1].traces.append(state_change)

            # 设置最后一个状态变化的结束时间
            for attr_name in attr_values:
                if attr_values[attr_name][-1].liveness.end_time is None:
                    attr_values[attr_name][-1].liveness.end_time = self.get_end_time()

            var_insts[var_id] = attr_values

        # 检查是否有缺少跟踪信息的变量
        for var_id in var_insts:
            for attr in var_insts[var_id]:
                for value in var_insts[var_id][attr]:
                    if len(value.traces) == 0:
                        print(f"Warning: No traces found for {var_id} {attr}")

        self.var_insts = var_insts
        return self.var_insts

    def get_var_raw_event_before_time(self, var_id: VarInstId, time: int) -> list[dict]:
        """Get all original trace records of a variable before the specified time."""

        raw_events = []

        # 遍历嵌套字典结构
        for process_id, threads in self.vars.items():
            if process_id != var_id.process_id:
                continue
            for thread_id, records in threads.items():
                for record in records:
                    if (
                        record["var_name"] == var_id.var_name
                        and record["var_type"] == var_id.var_type
                        and record["time"] < time
                    ):
                        raw_events.append(record)

        return raw_events

    def get_var_changes(self) -> list[VarChangeEvent]:
        """Get all variable change events from the trace.

        Essentially, this function will comprise consecutive states of the same variable attribute as a single change event.

        Returns:
            list[VarChangeEvent]: A list of all variable change events.
        """

        if self.var_changes is not None:
            return self.var_changes

        var_insts = self.get_var_insts()  # 获取所有变量实例及其状态

        self.var_changes = []

        # 遍历所有变量实例
        for var_id, attributes in var_insts.items():
            for attr_name, attr_states in attributes.items():
                for i in range(1, len(attr_states)):
                    change_time = attr_states[i].liveness.start_time
                    old_state = attr_states[i - 1]
                    new_state = attr_states[i]
                    
                    assert change_time is not None, (
                        f"Start time not found for {var_id} {attr_name} {new_state.value}"
                    )

                    # 创建并添加 VarChangeEvent 对象
                    self.var_changes.append(
                        VarChangeEvent(
                            var_id=var_id,
                            attr_name=attr_name,
                            change_time=change_time,
                            old_state=old_state,
                            new_state=new_state,
                        )
                    )

        return self.var_changes


    def query_var_changes_within_time(
        self, time_range: tuple[int, int]
    ) -> list[VarChangeEvent]:
        """Extract all variable change events from the trace, within a specific time range."""
        
        # 确保时间范围有效
        if not isinstance(time_range, tuple) or len(time_range) != 2:
            raise ValueError("time_range must be a tuple of two integers")
        if time_range[0] > time_range[1]:
            raise ValueError("Invalid time range: start time must be less than or equal to end time")

        # 获取所有变量变化事件
        var_changes = self.get_var_changes()

        # 过滤出在指定时间范围内的变化事件
        filtered_var_changes = [
            var_change
            for var_change in var_changes
            if time_range[0] <= var_change.change_time <= time_range[1]
        ]

        return filtered_var_changes

    def query_var_changes_within_time_and_process(
        self, time_range: tuple[int | float, int | float], process_id: int
    ) -> list[VarChangeEvent]:
        """Extract all variable change events from the trace, within a specific time range and process."""

        # 输入验证
        if not isinstance(time_range, tuple) or len(time_range) != 2:
            raise ValueError("time_range must be a tuple of two numbers (int or float).")
        if time_range[0] > time_range[1]:
            raise ValueError("Invalid time range: start time must be less than or equal to end time.")
        
        # 获取所有变量变化事件
        var_changes = self.get_var_changes()

        # 过滤出在指定时间范围内且属于特定进程的变化事件
        filtered_var_changes = [
            var_change
            for var_change in var_changes
            if time_range[0] <= var_change.change_time <= time_range[1]
            and var_change.var_id.process_id == process_id
        ]

        return filtered_var_changes

    def query_var_changes_within_func_call(
        self, func_call_id: str
    ) -> list[VarChangeEvent]:
        """Extract all variable change events from the trace, within the duration of a specific function call."""
        
        # 获取函数调用的 pre-call 记录
        pre_record = self.get_pre_func_call_record(func_call_id)

        # 获取函数调用的 post-call 记录
        post_record = self.get_post_func_call_record(func_call_id)

        # 定义函数调用的开始时间
        start_time = pre_record["time"]

        # 如果没有 post-call 记录，使用线程的结束时间，否则使用 post-call 的时间
        if post_record is None:
            end_time = (
                self.get_end_time(pre_record["process_id"], pre_record["thread_id"])
                + 0.001
            )
        else:
            end_time = post_record["time"]

        # 在指定时间范围和进程内查询变量变化事件
        return self.query_var_changes_within_time_and_process(
            (start_time, end_time), process_id=pre_record["process_id"]
        )

    def get_pre_func_call_record(self, func_call_id: str) -> dict:
        """Get the pre-call record of a function given its func_call_id."""

        pre_record = None

        # 遍历嵌套字典结构查找匹配的 pre-call 记录
        for process_id, threads in self.events.items():
            for thread_id, records in threads.items():
                for record in records:
                    if (
                        record["func_call_id"] == func_call_id
                        and record["type"] == TraceLineType.FUNC_CALL_PRE
                    ):
                        if pre_record is not None:
                            raise AssertionError(
                                f"Multiple pre-call events found for {func_call_id}, expected 1"
                            )
                        pre_record = record

        # 确保找到且仅找到一个 pre-call 记录
        if pre_record is None:
            raise AssertionError(
                f"No pre-call event found for {func_call_id}, expected 1"
            )

        return pre_record

    def get_post_func_call_record(self, func_call_id: str) -> dict | None:
        """Get the post call record of a function given its func_call_id.
        Returns None if the post call event is not found and the pre-call event is the outermost function call.
        """

        post_record = None

        # 遍历嵌套字典结构查找匹配的 post-call 记录
        for process_id, threads in self.events.items():
            for thread_id, records in threads.items():
                for record in records:
                    if (
                        record["func_call_id"] == func_call_id
                        and record["type"] in (
                            TraceLineType.FUNC_CALL_POST,
                            TraceLineType.FUNC_CALL_POST_EXCEPTION,
                        )
                    ):
                        if post_record is not None:
                            raise AssertionError(
                                f"Multiple post-call events found for {func_call_id}, expected 1"
                            )
                        post_record = record

        # 检查是否找到了 post-call 记录
        if post_record is None:
            logger.warning(f"No post call event found for {func_call_id}")

            # 获取 pre-call 记录
            pre_record = self.get_pre_func_call_record(func_call_id)

            # 查找最外层函数调用的 pre-call 记录
            outermost_func_call_pre = None
            for process_id, threads in self.events.items():
                for thread_id, records in threads.items():
                    for record in records:
                        if (
                            record["type"] == TraceLineType.FUNC_CALL_PRE
                            and record["process_id"] == pre_record["process_id"]
                            and record["thread_id"] == pre_record["thread_id"]
                        ):
                            outermost_func_call_pre = record
                            break
                    if outermost_func_call_pre:
                        break
                if outermost_func_call_pre:
                    break

            # 检查 pre-call 记录是否为最外层函数调用
            if outermost_func_call_pre and pre_record["func_call_id"] == outermost_func_call_pre["func_call_id"]:
                return None
            else:
                raise ValueError(
                    f"No post call event found for {func_call_id}, but it is not the outermost function call."
                )

        return post_record

    def query_func_call_events_within_time(
        self,
        time_range: tuple[int | float, int | float],
        process_id: int,
        thread_id: int,
    ) -> list[FuncCallEvent | FuncCallExceptionEvent]:
        """Extract all function call events from the trace, within a specific time range, process, and thread."""

        func_call_events: list[FuncCallEvent | FuncCallExceptionEvent] = []
        func_call_records = []

        # 手动筛选满足条件的函数调用记录
        for proc_id, threads in self.events.items():
            if proc_id != process_id:
                continue
            for th_id, records in threads.items():
                if th_id != thread_id:
                    continue
                for record in records:
                    if (
                        record["type"] in {
                            TraceLineType.FUNC_CALL_PRE,
                            TraceLineType.FUNC_CALL_POST,
                            TraceLineType.FUNC_CALL_POST_EXCEPTION,
                        }
                        and time_range[0] < record["time"] < time_range[1]
                    ):
                        func_call_records.append(record)

        # 按 func_call_id 分组
        func_call_groups = defaultdict(list)
        for record in func_call_records:
            func_call_groups[record["func_call_id"]].append(record)

        for func_call_id, records in func_call_groups.items():
            assert (
                len(records) == 2
            ), f"Function call records count is not 2 for {func_call_id}, found {len(records)}"

            # 按时间排序，以确保第一个记录是 pre-record
            records.sort(key=lambda r: r["time"])
            pre_record = records[0]
            post_record = records[1]

            assert (
                pre_record["type"] == TraceLineType.FUNC_CALL_PRE
            ), f"First record for {func_call_id} is not pre, got {pre_record['type']}"

            assert post_record["type"] in {
                TraceLineType.FUNC_CALL_POST,
                TraceLineType.FUNC_CALL_POST_EXCEPTION,
            }, f"Second record for {func_call_id} is not post, got {post_record['type']}"

            # 根据 post_record 类型构建事件对象
            if post_record["type"] == TraceLineType.FUNC_CALL_POST:
                func_call_events.append(
                    FuncCallEvent(pre_record["function"], pre_record, post_record)
                )
            else:
                func_call_events.append(
                    FuncCallExceptionEvent(pre_record["function"], pre_record, post_record)
                )

        return func_call_events


    def query_high_level_events_within_func_call(
        self, func_call_id: str
    ) -> list[FuncCallEvent | FuncCallExceptionEvent | VarChangeEvent]:
        """Extract all high-level events (function calls and variable changes) within a specific function call."""

        # 获取函数调用的 pre-call 记录
        pre_record = self.get_pre_func_call_record(func_call_id)

        # 获取函数调用的 post-call 记录
        post_record = self.get_post_func_call_record(func_call_id)

        # 如果没有 post-call 记录，使用线程的结束时间作为范围的结束
        if post_record is None:
            logger.warning(f"Post call event not found for {func_call_id}")
            end_time = (
                self.get_end_time(pre_record["process_id"], pre_record["thread_id"]) + 0.001
            )
            time_range = (pre_record["time"], end_time)
        else:
            time_range = (pre_record["time"], post_record["time"])

        process_id = pre_record["process_id"]
        thread_id = pre_record["thread_id"]

        # 查询指定时间范围内的所有函数调用事件
        high_level_func_call_events = self.query_func_call_events_within_time(
            time_range, process_id, thread_id
        )

        # 查询特定函数调用期间的所有变量变化事件
        high_level_var_change_events = self.query_var_changes_within_func_call(
            func_call_id
        )

        # 返回函数调用事件和变量变化事件的组合
        return high_level_func_call_events + high_level_var_change_events



    def get_time_precentage(self, time: int) -> float:
        return (time - self.get_start_time()) / (
            self.get_end_time() - self.get_start_time()
        )