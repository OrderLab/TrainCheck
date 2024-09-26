from abc import abstractmethod
from typing import NamedTuple

from mldaikon.instrumentor.tracer import TraceLineType


class MD_NONE:
    def __hash__(self) -> int:
        return hash(None)

    def __eq__(self, o: object) -> bool:
        return type(o) == MD_NONE

    def to_dict(self):
        """Return a serializable dictionary representation of the object."""
        return None


class VarInstId(NamedTuple):
    process_id: int
    var_name: str
    var_type: str


class Liveness:
    def __init__(self, start_time: float | None, end_time: float | None):
        self.start_time = start_time
        self.end_time = end_time

    def __str__(self):
        return f"Start Time: {self.start_time}, End Time: {self.end_time}, Duration: {self.end_time - self.start_time}"

    def __eq__(self, other):
        return self.start_time == other.start_time and self.end_time == other.end_time

    def __hash__(self) -> int:
        return hash(str(self.__dict__))


class AttrState:
    def __init__(self, value: type, liveness: Liveness, traces: list[dict]):
        self.value: type = value
        self.liveness: Liveness = liveness
        self.traces = traces

    def __str__(self):
        return f"Value: {self.value}, Liveness: {self.liveness}"

    def __eq__(self, other):
        return self.value == other.value and self.liveness == other.liveness

    def __hash__(self) -> int:
        return hash(str(self.__dict__))


"""High-level events to be extracted from the low-level trace events (a low-level event is a single line in a trace file)."""


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

    def __hash__(self) -> int:
        return super().__hash__()

    def __eq__(self, other) -> bool:
        return super().__eq__(other)


class IncompleteFuncCallEvent(HighLevelEvent):
    """An outermost function call event, but without the post record."""

    def __init__(self, func_name: str, pre_record: dict, potential_end_time: float):
        self.func_name = func_name
        self.pre_record = pre_record
        self.potential_end_time = potential_end_time
        assert pre_record["type"] == TraceLineType.FUNC_CALL_PRE

    def __str__(self):
        return f"IncompleteFuncCallEvent: {self.func_name}"

    def get_traces(self):
        return [self.pre_record]

    def __hash__(self) -> int:
        return super().__hash__()

    def __eq__(self, other) -> bool:
        return super().__eq__(other)


class FuncCallExceptionEvent(HighLevelEvent):
    def __init__(self, func_name: str, pre_record: dict, post_record: dict):
        self.func_name = func_name
        self.pre_record = pre_record
        self.post_record = post_record
        self.exception = post_record["exception"]
        assert (
            pre_record["type"] == TraceLineType.FUNC_CALL_PRE
            and post_record["type"] == TraceLineType.FUNC_CALL_POST_EXCEPTION
        )

    def __str__(self):
        return f"FuncCallExceptionEvent: {self.func_name}"

    def get_traces(self):
        return [self.pre_record, self.post_record]

    def __hash__(self) -> int:
        return super().__hash__()

    def __eq__(self, other) -> bool:
        return super().__eq__(other)


class VarChangeEvent(HighLevelEvent):
    def __init__(
        self,
        var_id: VarInstId,
        attr_name: str,
        change_time: float,
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

    def __hash__(self) -> int:
        return super().__hash__()

    def __eq__(self, other) -> bool:
        return super().__eq__(other)


ALL_EVENT_TYPES = [
    FuncCallEvent,
    IncompleteFuncCallEvent,
    FuncCallExceptionEvent,
    VarChangeEvent,
]
