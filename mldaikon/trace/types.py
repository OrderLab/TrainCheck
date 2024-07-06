from abc import abstractmethod
from dataclasses import dataclass
from typing import NamedTuple

from mldaikon.instrumentor.tracer import TraceLineType


class VarInstId(NamedTuple):
    process_id: int
    var_name: str
    var_type: str

    # def __str__(self):
    #     return f"VarInstId: {self.process_id}, {self.var_name}, {self.var_type}"

    # def __repr__(self) -> str:
    #     return super().__repr__()

    # def __hash__(self) -> int:
    #     return hash(str(self.__dict__))

    # def __eq__(self, other) -> bool:
    #     return self.__dict__ == other.__dict__


class Liveness:
    def __init__(self, start_time: int | None, end_time: int | None):
        self.start_time = start_time
        self.end_time = end_time

    def __str__(self):
        return f"Start Time: {self.start_time}, End Time: {self.end_time}, Duration: {self.end_time - self.start_time}"


class AttrState:
    def __init__(self, value: type, liveness: Liveness, traces: list[dict]):
        self.value: type = value
        self.liveness: Liveness = liveness
        self.traces = traces

    def __str__(self):
        return f"Value: {self.value}, Liveness: {self.liveness}"


"""High-level events to be extracted from the low-level trace events (a low-level event is a single line in a trace file)."""


@dataclass
class HighLevelEvent(object):
    """Base class for high-level events. A high-level event is an conceptual event that is extracted from the low-level trace events (each line in the trace).
    For example, a function call event is a high-level event that is extracted from the low-level trace events of 'function_call (pre)' and 'function_call (post)'.
    """

    @abstractmethod
    def get_traces(self):
        pass

    def __hash__(self) -> int:
        # return hash value based on the fields of the class
        return hash(str(self.__dict__))

    def __eq__(self, other) -> bool:
        # compare the fields of the class
        return self.__dict__ == other.__dict__


@dataclass
class FuncCallEvent(HighLevelEvent):
    """A function call event."""

    def __init__(self, func_name: str, pre_record: dict, post_record: dict):
        self.func_name = func_name
        self.pre_record = pre_record
        self.post_record = post_record
        assert (
            pre_record["type"] == TraceLineType.FUNC_CALL_PRE
            and post_record["type"] == TraceLineType.FUNC_CALL_POST
        )

    def __str__(self):
        return f"FuncCallEvent: {self.func_name}"

    def get_traces(self):
        return [self.pre_record, self.post_record]


@dataclass
class FuncCallExceptionEvent(HighLevelEvent):
    def __init__(self, func_name: str, pre_record: dict, post_record: dict):
        self.func_name = func_name
        self.pre_record = pre_record
        self.post_record = post_record
        assert (
            pre_record["type"] == TraceLineType.FUNC_CALL_PRE
            and post_record["type"] == TraceLineType.FUNC_CALL_POST_EXCEPTION
        )

    def __str__(self):
        return f"FuncCallExceptionEvent: {self.func_name}"

    def get_traces(self):
        return [self.pre_record, self.post_record]


@dataclass
class VarChangeEvent(HighLevelEvent):
    def __init__(
        self,
        var_id: VarInstId,
        attr_name: str,
        change_time: int,
        old_state: AttrState,
        new_state: AttrState,
    ):
        self.var_id = var_id
        self.attr_name = attr_name
        self.change_time = change_time
        self.old_state = old_state
        self.new_state = new_state

    def __str__(self):
        return f"VarChangeEvent: {self.var_id}, {self.attr_name}, {self.change_time}, {self.old_state}, {self.new_state}"

    def get_traces(self):
        return self.old_state.traces + self.new_state.traces
