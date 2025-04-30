import json
from abc import abstractmethod
from typing import NamedTuple

from traincheck.instrumentor.tracer import TraceLineType
from traincheck.instrumentor.types import PTID


class MD_NONE:
    def __hash__(self) -> int:
        return hash(None)

    def __eq__(self, o: object) -> bool:
        return type(o) == MD_NONE or o is None

    def __repr__(self):
        return "None"

    def __str__(self):
        return "None"

    def to_dict(self):
        """Return a serializable dictionary representation of the object."""
        return None

    @staticmethod
    def json_encoder(d):
        if type(d) == MD_NONE:
            return None
        return d

    @staticmethod
    def replace_with_none(list_or_dict):
        if isinstance(list_or_dict, list):
            for i, value in enumerate(list_or_dict):
                if isinstance(value, MD_NONE):
                    list_or_dict[i] = None
                elif isinstance(value, (list, dict)):
                    MD_NONE.replace_with_none(value)
        elif isinstance(list_or_dict, dict):
            for key, value in list_or_dict.items():
                if isinstance(value, MD_NONE):
                    list_or_dict[key] = None
                elif isinstance(value, (list, dict)):
                    MD_NONE.replace_with_none(value)


# Custom JSON Encoder
class MDNONEJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        # Convert MD_NONE to None
        if isinstance(obj, MD_NONE):
            return None
        return super().default(obj)


class MDNONEJSONDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, obj):
        if obj is None:
            return MD_NONE()
        return obj


class MD_NULL:
    def __eq__(self, o: object) -> bool:
        return type(o) == MD_NULL

    def to_dict(self):
        """Return a serializable dictionary representation of the object."""
        raise NotImplementedError("MD_NULL cannot be serialized")


class VarInstId(NamedTuple):
    process_id: int
    var_name: str
    var_type: str


class Liveness:
    def __init__(self, start_time: float | None, end_time: float | None):
        self.start_time = start_time
        self.end_time = end_time

    def is_alive(self, time: float) -> bool:
        assert self.start_time is not None and self.end_time is not None
        return self.start_time <= time <= self.end_time

    def __str__(self):
        return f"Start Time: {self.start_time}, End Time: {self.end_time}, Duration: {self.end_time - self.start_time}"

    def __repr__(self) -> str:
        return self.__str__()

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

    def __repr__(self):
        return self.__str__()


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

        # TODO: use the Arguments class to replace self.args and self.kwargs
        self.args: dict[str, dict[str, dict[str, object]]] = pre_record[
            "args"
        ]  # lists of [type -> attr_name -> value]
        self.kwargs: dict[str, dict[str, object]] = pre_record[
            "kwargs"
        ]  # key --> attr_name -> value
        self.return_values: (
            dict[str, dict[str, object]] | list[dict[str, dict[str, object]]]
        ) = post_record[
            "return_values"
        ]  # key --> attr_name -> value

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

        # TODO: replace self.args and self.kwargs with the Arguments class
        self.args = pre_record["args"]
        self.kwargs = pre_record["kwargs"]

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

        # TODO: replace self.args and self.kwargs with the Arguments class
        self.args = pre_record["args"]
        self.kwargs = pre_record["kwargs"]

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


class BindedFuncInput:
    def __init__(self, binded_args_and_kwargs: dict[str, dict]):
        self.binded_args_and_kwargs = binded_args_and_kwargs

    def get_available_args(self):
        return self.binded_args_and_kwargs.keys()

    def get_arg(self, arg_name):
        return self.binded_args_and_kwargs[arg_name]

    def get_arg_type(self, arg_name):
        return list(self.binded_args_and_kwargs[arg_name].keys())[0]

    def get_arg_value(self, arg_name):
        return list(self.binded_args_and_kwargs[arg_name].values())[0]

    def to_dict_for_precond_inference(self):
        # flat this object later, get rid of the type annotation
        # return {arg_name: arg_value}
        return {
            arg_name: list(arg_value.values())[0]
            for arg_name, arg_value in self.binded_args_and_kwargs.items()
        }

    def __str__(self) -> str:
        return str(self.binded_args_and_kwargs)

    def __repr__(self) -> str:
        return self.__str__()


class ContextManagerState:
    def __init__(
        self, name: str, ptid: PTID, liveness: Liveness, input: BindedFuncInput
    ):
        self.name = name
        self.ptid = ptid
        self.liveness = liveness
        self.input = input

    def to_dict(self):
        # flat this object later.
        return {
            "name": self.name,
            "process_id": self.ptid.pid,
            "thread_id": self.ptid.tid,
            "start_time": self.liveness.start_time,
            "end_time": self.liveness.end_time,
            "input": self.input.to_dict_for_precond_inference(),
        }

    def __str__(self):
        return f"ContextManagerState: {self.name}, {self.ptid}, {self.liveness}, {self.input}"

    def __repr__(self):
        return self.__str__()
