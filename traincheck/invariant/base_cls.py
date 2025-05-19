from __future__ import annotations

import abc
import importlib
import inspect
import json
import logging
import math
from enum import Enum
from typing import Any, Hashable, Iterable, Optional, Type

import traincheck.config.config as config
from traincheck.instrumentor.dumper import var_to_serializable
from traincheck.invariant.symbolic_value import (
    GENERALIZED_TYPES,
    check_generalized_value_match,
)
from traincheck.trace.trace import Trace, VarInstId
from traincheck.trace.types import (
    MD_NONE,
    FuncCallEvent,
    FuncCallExceptionEvent,
    HighLevelEvent,
    IncompleteFuncCallEvent,
    MDNONEJSONDecoder,
    VarChangeEvent,
)
from traincheck.utils import safe_isnan


class Meta_NOT_SET(type):
    def __str__(cls):
        return "NOT SET"

    def __repr__(cls):
        return "NOT SET"


class _NOT_SET(metaclass=Meta_NOT_SET):
    pass


FUNC_SIGNATURE_OBJS: dict[str, inspect.Signature | None] = {}
STAGE_KEY = "meta_vars.stage"


def load_function_obj(func_name: str) -> Any:
    """Load the function object from the provided function name."""
    # need to load the function's parent module
    # find the module name up to the last dot that's prior to a lowercase letter
    try:
        func_paths = func_name.split(".")
        module_name = func_paths[0]
        for i in range(1, len(func_paths) - 1):
            module_name += "." + func_paths[i]
            if func_paths[i + 1][0].isupper():  # indicates the start of the class name
                break

        left_over_paths = func_paths[i + 1 :]

        module = importlib.import_module(module_name)
        for path in left_over_paths:
            module = getattr(module, path)

        func_obj = module
        assert callable(
            func_obj
        ), f"Function {func_name} is not callable, check loading logic."
        return func_obj
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.debug(
            f"Failed to load the object for the function: {func_name}, error: {e}"
        )
        return None


def load_function_signature(func_name: str) -> inspect.Signature | None:
    if func_name in FUNC_SIGNATURE_OBJS:
        return FUNC_SIGNATURE_OBJS[func_name]

    # need to load the function's parent module
    # find the module name up to the last dot that's prior to a lowercase letterq
    try:
        func_obj = load_function_obj(func_name)
        if func_obj is None:
            raise ValueError(f"Failed to load the function object for {func_name}")
        FUNC_SIGNATURE_OBJS[func_name] = inspect.signature(func_obj)
        return FUNC_SIGNATURE_OBJS[func_name]
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.debug(
            f"Failed to load the signature for the function: {func_name}, error: {e}"
        )
        FUNC_SIGNATURE_OBJS[func_name] = (
            None  # failed to load the signature, mark it here to avoid repeated attempts
        )
        return None


def is_signature_empty(func_name: str) -> bool:
    signature = load_function_signature(func_name)
    if signature is None:
        return True
    return len(signature.parameters) == 0


class Arguments:
    def __init__(
        self,
        args: dict[str, dict[str, Any]],
        kwargs: dict[str, dict[str, Any]],
        func_name: str,
        consider_default_values: bool = False,
    ):
        """Difference with BindedFuncInput is that this class does not handle
        default values and only works with the provided args and kwargs.

        Ideally these two classes should be merged into one, but for now we are keeping them separate due to
        engineering time constraints.
        """

        self.args = args
        self.kwargs = kwargs
        self.func_name = func_name

        self.arguments: dict[str, dict[str, Any]]

        self.signature = load_function_signature(func_name)
        if self.signature is None:
            logger = logging.getLogger(__name__)
            logger.debug(
                f"Failed to load the signature for the function: {func_name}, can only work on kwargs."
            )
            self.arguments = kwargs.copy()
        elif all(
            param.kind
            in [inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD]
            for param in self.signature.parameters.values()
        ):
            logger = logging.getLogger(__name__)
            logger.warning(
                f"Function {func_name} has overly-general signature (only *args or **kwargs), can only work on kwargs."
            )
            self.arguments = kwargs.copy()
        else:
            # check if *args exists in the signature
            allow_unmatched_args = False
            if any(
                param.kind == inspect.Parameter.VAR_POSITIONAL
                for param in self.signature.parameters.values()
            ):
                allow_unmatched_args = True
                self.unknown_args = []

            self.arguments = kwargs.copy()
            signature_params = list(self.signature.parameters.keys())
            for i_str, arg in args.items():
                i = int(i_str)
                if i < len(signature_params):
                    self.arguments[signature_params[i]] = arg
                elif allow_unmatched_args:
                    self.unknown_args.append(arg)
                else:
                    raise ValueError(
                        f"Too many positional arguments for function {func_name}, expecting {len(signature_params)} ({signature_params}) but got {len(args)}"  # type: ignore
                    )

            if consider_default_values:
                for param_name, param in self.signature.parameters.items():
                    if (
                        param_name not in self.arguments
                        and param.default != inspect.Parameter.empty
                    ):
                        self.arguments[param_name] = var_to_serializable(param.default)

            if consider_default_values:
                for param_name, param in self.signature.parameters.items():
                    if (
                        param_name not in self.arguments
                        and param.default != inspect.Parameter.empty
                    ):
                        self.arguments[param_name] = var_to_serializable(param.default)

    def to_dict(self) -> dict:
        return {
            "args": self.arguments,
            "func_name": self.func_name,
        }

    @staticmethod
    def from_dict(arguments_dict: dict) -> Arguments:
        return Arguments(
            kwargs=arguments_dict["args"],
            func_name=arguments_dict["func_name"],
            args={},
        )

    def merge_with(self, other: Arguments) -> Arguments:
        # do a intersection of the provided arguments
        merged_args = {k: v for k, v in self.arguments.items() if k in other.arguments}

        # for each specific arg, merge the divergent value
        for k, v in self.arguments.items():
            if k not in merged_args:
                continue

            # merging rule #1: consistency ==
            if v != other.arguments[k]:
                del merged_args[k]
            # merging rule #2: if the value is a generalized type, then we can merge it
            # >, <, >=, <= for numerical types
            # TODO

        return Arguments(args={}, kwargs=merged_args, func_name=self.func_name)

    def is_empty(self) -> bool:
        return len(self.arguments) == 0

    def check_for_violation(self, other: Arguments) -> bool:
        # every key in self should be in other, and should have the same value
        for k, v in self.arguments.items():
            if k not in other.arguments:
                return True
            if v != other.arguments[k]:
                return True
        return False

    def to_instr_opts_dict(self) -> dict:
        """returns a dictionary specifying which arguments to dump in the trace

        dictionary schema:
        {
            # for positional arguments
            (arg_idx, arg_name): [
                {"attr_name": {"attr_name"}},
                "attr_name",
                ...
            ]
            # for keyword arguments
            arg_name: [
                xxx
            ]
        }

        TODO: Refactor the returned value with a formally defined schema. For example we currently use None to indicate that the argument is a primitive type or just dump the field as is. This is not a good practice.
        """

        instr_opts: dict[str | int, None | dict] = {}
        all_arg_names = []
        if self.signature is not None:
            all_arg_names = list(self.signature.parameters.keys())
        for arg_name in self.arguments:
            param_key: str
            param_idx: int = -1
            try:
                arg_idx = all_arg_names.index(arg_name)
                # get the actual parameter
                assert self.signature is not None
                param = self.signature.parameters[arg_name]

                if param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD:
                    param_key = arg_name
                    param_idx = arg_idx
                elif param.kind == inspect.Parameter.KEYWORD_ONLY:
                    param_key = arg_name
                else:
                    logging.warning(
                        f"Function {self.func_name} have a non-positional_or_keyword parameter {arg_name}, kind {param.kind}, this is not supported for selective dumping, and will be skipped entirely."
                    )
                    continue
            except ValueError:
                # likely the signature cannot be loaded or in this case the signature is *args, **kwargs thus we cannot determine the index
                param_key = arg_name

            arg_dict: dict[str, Any] = self.arguments[arg_name]
            # there will only be one key to the above dictionary, which is a str representing the type of the value
            arg_value = list(arg_dict.values())[0]
            if isinstance(arg_value, dict) and arg_value:
                # if arg_value is a non-empty dict, arg is an object that has nested attributes

                # replace every non-dict value (leaf nodes) with None
                arg_instr_opt = arg_value.copy()

                def replace_leaf_nodes(d):
                    for k, v in d.items():
                        if isinstance(v, dict):
                            replace_leaf_nodes(v)
                        else:
                            d[k] = None

                replace_leaf_nodes(arg_instr_opt)
                instr_opts[param_key] = arg_instr_opt
                if param_idx != -1:
                    instr_opts[param_idx] = arg_instr_opt
            else:
                # arg is just a simple, primitive type
                instr_opts[param_key] = None
                if param_idx != -1:
                    instr_opts[param_idx] = None

        return instr_opts

    def __eq__(self, other) -> bool:
        if not isinstance(other, Arguments):
            return False
        return self.arguments == other.arguments

    def __hash__(self) -> int:
        return hash(make_hashable(self.arguments))

    def __str__(self):
        return str(self.arguments)

    def __repr__(self):
        return self.__str__()


class Param:
    # param_type: str  # ["func", "var_type", "var_name"]

    def __hash__(self) -> int:
        return hash(make_hashable(self.to_dict()))

    def __eq__(self, other) -> bool:
        if not isinstance(other, Param):
            return False
        return self.to_dict() == other.to_dict()

    def to_dict(self):
        ret = {
            "param_type": self.__class__.__name__,
        }
        self_state = self.__dict__
        for field, value in self_state.items():
            if value == _NOT_SET:
                continue
            if isinstance(value, Exception):
                ret[field] = (
                    f"Exception: {type(value)}, msg: {value}"  # TODO: hack, this is not seralizable back to python Exceptions
                )
                continue
            if isinstance(value, Arguments):
                ret[field] = value.to_dict()
                continue
            # try if the value is seralizable
            try:
                json.dumps({field: value})
                ret[field] = value
            except TypeError:
                if hasattr(value, "to_dict"):
                    ret[field] = value.to_dict()
                else:
                    ret[field] = f"NOT SERIALIZABLE: {str(value)}"

        return ret

    @staticmethod
    def from_dict(param_dict: dict) -> Param:
        for param_type in Param.__subclasses__():
            if param_type.__name__ == param_dict["param_type"]:
                args = {k: v for k, v in param_dict.items() if k != "param_type"}
                # if any of the v is null, convert to MD_NONE
                for k, v in args.items():
                    if v is None:
                        args[k] = MD_NONE()
                    elif k == "arguments":
                        args[k] = Arguments.from_dict(v)
                    elif "value" in k:
                        args[k] = make_hashable(v)
                return param_type(**args)
        raise ValueError(f"Unknown param type: {param_dict['param_type']}")

    def check_event_match(self, event: HighLevelEvent) -> bool:
        "Check if the high level event contains the required information for the param."
        raise NotImplementedError("check_event_match method is not implemented yet.")

    def get_customizable_field_names(self) -> set[str]:
        """Returns the field names that can be customized for the param."""
        raise NotImplementedError(
            "get_customizable_field_names method is not implemented yet."
        )

    def get_customized_fields(self) -> dict[str, type]:
        """Returns the fields that can be customized for the param."""
        raise NotImplementedError(
            "get_customized_fields method should not be called on the base class."
        )

    # @abc.abstractmethod
    def with_no_customization(self) -> Param:
        raise NotImplementedError(
            "with_no_customization method should not be called on the base class"
        )


class APIParam(Param):
    def __init__(
        self,
        api_full_name: str,
        exception: Exception | Type[_NOT_SET] = _NOT_SET,
        arguments: Arguments | Type[_NOT_SET] = _NOT_SET,
    ):
        self.api_full_name = api_full_name
        self.exception = exception
        self.arguments = arguments

    def check_event_match(self, event: HighLevelEvent) -> bool:
        if not isinstance(
            event, (FuncCallEvent, FuncCallExceptionEvent, IncompleteFuncCallEvent)
        ):
            return False

        # TODO: Handle Stop Iteration Exception!!!
        matched = event.func_name == self.api_full_name
        if self.exception != _NOT_SET:
            matched = (
                matched
                and isinstance(event, FuncCallExceptionEvent)
                and event.exception == self.exception
            )
        else:
            matched = matched and not isinstance(event, FuncCallExceptionEvent)

        if not matched:
            return False

        # check the arguments if they are provided
        if isinstance(self.arguments, Arguments):
            # current_args should not violate the provided arguments (i.e., self.arguments should be a subset of current_args)
            current_args = Arguments(event.args, event.kwargs, event.func_name)
            matched = matched and not self.arguments.check_for_violation(current_args)

        return matched

    def with_no_customization(self) -> APIParam:
        return APIParam(self.api_full_name)

    def get_necessary_fields(self) -> dict[str, str]:
        return {
            "api_full_name": self.api_full_name,
        }

    def get_customizable_field_names(self) -> set[str]:
        return {"exception"}

    def get_customized_fields(self) -> dict[str, type]:
        if self.exception == _NOT_SET:
            return {}

        return {
            "exception": Exception,
        }

    def __eq__(self, other):
        if isinstance(other, APIParam):
            return self.api_full_name == other.api_full_name
        return False

    def __hash__(self):
        return hash(self.api_full_name)

    def __str__(self):
        return f"{self.api_full_name} with exception: {self.exception}"

    def __repr__(self):
        return self.__str__()

    @staticmethod
    def from_dict(param_dict: dict) -> APIParam:
        args = {k: v for k, v in param_dict.items() if k != "param_type"}
        # if any of the v is null, convert to MD_NONE
        for k, v in args.items():
            if v is None:
                args[k] = MD_NONE()
        return APIParam(**args)


class VarTypeParam(Param):
    def __init__(
        self,
        var_type: str,
        attr_name: str,
        pre_value: Any = _NOT_SET,
        post_value: Any = _NOT_SET,
        const_value: Any = _NOT_SET,
    ):
        self.var_type = var_type
        self.attr_name = attr_name

        """ TODO:
        Can we use symbolic values here (for the below parameterizable params)? For example, instead of giving a specific value, we can state that the pre_value should be a non-null value.
        This can be useful in cases where the exact value is not known, but the type of the value is known.
        """

        ## === optional parametrized values ===
        # for the APIContainRelation relation
        self.pre_value = pre_value
        self.post_value = post_value

        # for the VarPeriodicChangeRelation relation
        self.const_value = const_value  # CHECKING OF THIS VALUE CAN ONLY BE DONE IN THE RELATION'S EVALUATE METHOD

    def check_event_match(self, event: HighLevelEvent) -> bool:
        """Checks whether the event is a candidate described by the param.
        Note that, only var_type and attr_name are checked here.
        The parameterized values (e.g. pre_value and const_value) should be checked in the relation's evaluate method.

        TODO: potential value of using higher level events is that we can capture higher level information. For example,
        `zero_grad` does assign zero to the gradients of the model everytime, but call to `zero_grad` is only correct if the previous value
        of the gradients was not zero. This information can be captured in the higher level event.
        """
        if not isinstance(event, VarChangeEvent):
            return False
        var_attr_matched = (
            event.var_id.var_type == self.var_type and event.attr_name == self.attr_name
        )

        if not var_attr_matched:
            return False

        if self.const_value != _NOT_SET:
            logger = logging.getLogger(__name__)
            logger.warning(
                "Const value is set for VarTypeParam, this should be checked in the relation's evaluate method instead of the check_event_match method."
            )

        pre_and_post_value_matched = True
        if self.pre_value != _NOT_SET:
            if self.pre_value != event.old_state.value:
                if self.pre_value in GENERALIZED_TYPES:
                    pre_and_post_value_matched = (
                        pre_and_post_value_matched
                        and check_generalized_value_match(
                            self.pre_value, event.old_state.value
                        )
                    )
                else:
                    return False

        if self.post_value != _NOT_SET:
            if self.post_value != event.new_state.value:
                if self.post_value in GENERALIZED_TYPES:
                    pre_and_post_value_matched = (
                        pre_and_post_value_matched
                        and check_generalized_value_match(
                            self.post_value, event.new_state.value
                        )
                    )
                else:
                    return False

        return pre_and_post_value_matched

    def check_var_id_match(self, var_id: VarInstId) -> bool:
        return var_id.var_type == self.var_type

    def with_no_customization(self) -> VarTypeParam:
        return VarTypeParam(self.var_type, self.attr_name)

    def get_necessary_fields(self) -> dict[str, str]:
        return {
            "var_type": self.var_type,
            "attr_name": self.attr_name,
        }

    def get_customizable_field_names(self) -> set[str]:
        return {"pre_value", "post_value", "const_value"}

    def get_customized_fields(self) -> dict[str, type]:
        fields = {}
        for attr in ["pre_value", "post_value", "const_value"]:
            if getattr(self, attr) != _NOT_SET:
                fields[attr] = getattr(self, attr)
        return fields

    def __str__(self) -> str:
        return f"{self.var_type} {self.attr_name}, pre_value: {self.pre_value}, post_value: {self.post_value}, const_value: {self.const_value}"

    def __repr__(self) -> str:
        return self.__str__()


class VarNameParam(Param):
    def __init__(
        self,
        var_type: str,
        var_name: str,
        attr_name: str,
        pre_value: Any = _NOT_SET,
        post_value: Any = _NOT_SET,
        const_value: Any = _NOT_SET,
    ):
        self.var_type = var_type
        self.var_name = var_name
        self.attr_name = attr_name

        ## === optional parametrized values ===
        # for the APIContainRelation relation
        self.pre_value = pre_value
        self.post_value = post_value

        # for the VarPeriodicChangeRelation and ConsistencyTransientVarsRelation
        self.const_value = const_value

    def check_event_match(self, event: HighLevelEvent) -> bool:
        """Checks whether the event is a candidate described by the param.
        Note that, only var_type and attr_name are checked here.
        The parameterized values (e.g. pre_value and const_value) should be checked in the relation's evaluate method.

        TODO: potential value of using higher level events is that we can capture higher level information. For example,
        `zero_grad` does assign zero to the gradients of the model everytime, but call to `zero_grad` is only correct if the previous value
        of the gradients was not zero. This information can be captured in the higher level event.
        """
        if not isinstance(event, VarChangeEvent):
            return False
        var_attr_matched = (
            event.var_id.var_type == self.var_type and event.attr_name == self.attr_name
        )

        if self.const_value != _NOT_SET:
            logger = logging.getLogger(__name__)
            logger.warning(
                "Const value is set for VarNameParam, this should be checked in the relation's evaluate method instead of the check_event_match method."
            )

        pre_and_post_value_matched = True
        if self.pre_value != _NOT_SET:
            if self.pre_value != event.old_state.value:
                if self.pre_value in GENERALIZED_TYPES:
                    pre_and_post_value_matched = (
                        pre_and_post_value_matched
                        and check_generalized_value_match(
                            self.pre_value, event.old_state.value
                        )
                    )
                else:
                    return False

        if self.post_value != _NOT_SET:
            if self.post_value != event.new_state.value:
                if self.post_value in GENERALIZED_TYPES:
                    pre_and_post_value_matched = (
                        pre_and_post_value_matched
                        and check_generalized_value_match(
                            self.post_value, event.new_state.value
                        )
                    )
                else:
                    return False

        return var_attr_matched and pre_and_post_value_matched

    def check_var_id_match(self, var_id: VarInstId) -> bool:
        return var_id.var_type == self.var_type and var_id.var_name == self.var_name

    def with_no_customization(self) -> VarNameParam:
        return VarNameParam(self.var_type, self.var_name, self.attr_name)

    def get_necessary_fields(self) -> dict[str, str]:
        return {
            "var_type": self.var_type,
            "var_name": self.var_name,
            "attr_name": self.attr_name,
        }

    def get_customizable_field_names(self) -> set[str]:
        return {"pre_value", "post_value", "const_value"}

    def get_customized_fields(self) -> dict[str, Any]:
        fields = {}
        for attr in ["pre_value", "post_value", "const_value"]:
            if getattr(self, attr) != _NOT_SET:
                fields[attr] = getattr(self, attr)
        return fields

    def __str__(self) -> str:
        return f"{self.var_type} {self.var_name} {self.attr_name}, pre_value: {self.pre_value}, post_value: {self.post_value}, const_value: {self.const_value}"

    def __repr__(self) -> str:
        return self.__str__()


class InputOutputParam(Param):
    def __init__(
        self,
        name: Optional[str],
        index: Optional[int],
        type: str,
        additional_path: tuple[str] | None,
        api_name: Optional[str],
        is_input: bool,  # not input means output
    ):
        self.name = name
        self.index = index
        self.type = type
        self.additional_path = additional_path
        self.api_name = api_name
        self.is_input = is_input

    def check_event_match(self, event: HighLevelEvent) -> bool:
        raise NotImplementedError("check_event_match method is not implemented yet.")

    def get_value_from_list_of_tensors(self, list_of_tensors: list) -> Any:
        assert (
            self.index is not None
        ), "Index should be when calling get_value_from_list_of_tensors"
        assert (
            self.additional_path is not None
        ), "Additional path should be None when calling get_value_from_list_of_tensors"
        # print("index", self.index)
        tensor = list_of_tensors[self.index]
        value = tensor
        for additional_path in self.additional_path:
            value = value[additional_path]
        return value

    def get_value_from_arguments(self, arguments: Arguments) -> Any:
        assert (
            self.name is not None
        ), "Name should be when calling get_value_from_arguments"
        assert self.additional_path is None or isinstance(
            self.additional_path, MD_NONE
        ), f"Additional path should be None when calling get_value_from_arguments, got {self.additional_path}"

        if self.name in arguments.arguments:
            arg = arguments.arguments[self.name]
            if self.additional_path and not isinstance(self.additional_path, MD_NONE):
                for path in self.additional_path:
                    if path not in arg:
                        raise ValueError("Arg cannot be found.")
                    arg = arg[path]
            return list(arg.values())[0]
        else:
            raise ValueError(f"Name {self.name} not found in the arguments.")


def construct_api_param(
    event: FuncCallEvent | FuncCallExceptionEvent | IncompleteFuncCallEvent,
) -> APIParam:
    if isinstance(event, FuncCallEvent):
        return APIParam(event.func_name)
    if isinstance(event, IncompleteFuncCallEvent):
        return APIParam(event.func_name)

    if isinstance(event, FuncCallExceptionEvent):
        return APIParam(event.func_name, event.exception)

    raise ValueError(f"Invalid event type: {type(event)}")


def construct_var_param_from_var_change(
    event: VarChangeEvent,
) -> VarTypeParam | VarNameParam:
    """NOTE as of its current form APICONTAINRELATION can invoke this"""

    pre_value = event.old_state.value
    new_value = event.new_state.value

    if config.VAR_INV_TYPE == "type":
        return VarTypeParam(
            event.var_id.var_type,
            event.attr_name,
            pre_value=make_hashable(pre_value),
            post_value=make_hashable(new_value),
            # const_value=None, # TODO
        )
    elif config.VAR_INV_TYPE == "name":
        return VarNameParam(
            event.var_id.var_type,
            event.var_id.var_name,
            event.attr_name,
            pre_value=make_hashable(pre_value),
            post_value=make_hashable(new_value),
            # const_value=None, # TODO
        )

    raise ValueError(f"Invalid VAR_INV_TYPE: {config.VAR_INV_TYPE}")


class PreconditionClauseType(Enum):
    CONSTANT = "constant"
    CONSISTENT = "consistent"
    UNEQUAL = "unequal"
    EXIST = "exist"


PT = PreconditionClauseType


class PreconditionClause:
    def __init__(
        self,
        prop_name: str,
        prop_dtype: type | None,
        _type: PT,
        additional_path: tuple[str] | None,
        values: set | None,
    ):
        """A class to represent a single clause in a precondition. A clause is a property that should hold for the hypothesis to be valid.

        Args:
        - prop_name: The name of the property
        - prop_dtype: The data type of the property
        - _type: The type of the precondition (constant, consistent, unequal, exist), see `PreconditionClauseType`
        - additional_path: The additional path to the property in the trace. This is provided if prop_name refers to a dictionary in the trace (TODO: can we extend it to lists as well?)
        - values: The values that the property should hold. This is a set of values that the property should hold. This is only checked for CONSTANT clauses.
        """
        assert _type in [
            PT.CONSISTENT,
            PT.CONSTANT,
            PT.UNEQUAL,
            PT.EXIST,
        ], f"Invalid Precondition type {_type}"

        # for EXIST AND UNEQUAL, THE values and prop_dtype do not need to be set
        if _type in [PT.CONSTANT]:
            if prop_dtype is None:
                assert values == {None}, "Values should be None for prop_dtype None"
            else:
                assert (
                    values is not None and len(values) > 0 and prop_dtype is not None
                ), "Values should be provided for constant or consistent preconditions"

        self.prop_name = prop_name
        self.prop_dtype = prop_dtype
        self.type = _type
        self.additional_path = additional_path
        self.values = values if isinstance(values, set) else {values}

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        if self.type in [PT.CONSTANT]:
            return f"Prop: {self.prop_name}, Type: {self.type}, Values: {self.values}"
        return f"Prop: {self.prop_name}, Type: {self.type}"

    def to_dict(self) -> dict:
        clause_dict: dict[str, str | tuple] = {
            "type": self.type.value,
            "prop_name": self.prop_name,
            "additional_path": self.additional_path if self.additional_path else "None",
            "prop_dtype": self.prop_dtype.__name__ if self.prop_dtype else "None",
        }
        if self.type in [PT.CONSTANT]:
            clause_dict["values"] = tuple(self.values)
        return clause_dict

    @staticmethod
    def from_dict(clause_dict: dict) -> PreconditionClause:
        prop_name = clause_dict["prop_name"]
        _type = PT(clause_dict["type"])
        prop_dtype = eval(clause_dict["prop_dtype"])
        additional_path = (
            tuple(clause_dict["additional_path"])
            if clause_dict["additional_path"] != "None"
            else None
        )

        values = None
        if _type in [PT.CONSTANT]:
            assert "values" in clause_dict, "Values not found in the clause"
            assert isinstance(clause_dict["values"], list), "Values should be a list"
            values = set(clause_dict["values"])
        return PreconditionClause(prop_name, prop_dtype, _type, additional_path, values)

    def __eq__(self, other):
        if not isinstance(other, PreconditionClause):
            return False

        if self.type == PT.CONSISTENT and other.type == PT.CONSISTENT:
            return (
                self.prop_name == other.prop_name
                and self.prop_dtype == other.prop_dtype
                and self.type == other.type
                and self.additional_path == other.additional_path
            )

        return (
            self.prop_name == other.prop_name
            and self.prop_dtype == other.prop_dtype
            and self.type == other.type
            and self.additional_path == other.additional_path
            and self.values == other.values
        )

    def __hash__(self):
        if self.type == PT.CONSISTENT:
            return hash(
                (
                    self.prop_name,
                    self.prop_dtype,
                    self.type,
                    tuple(self.additional_path) if self.additional_path else None,
                )
            )
        return hash(
            (
                self.prop_name,
                self.prop_dtype,
                self.type,
                make_hashable(self.values) if self.values else None,
                make_hashable(self.additional_path) if self.additional_path else None,
            )
        )

    def verify(self, example: list) -> bool:
        assert isinstance(example, list)
        assert len(example) > 0

        def get_prop_from_record(record: dict) -> tuple[Any, bool]:
            if self.prop_name not in record:
                return None, False

            current_value = record[self.prop_name]
            if not self.additional_path:
                return current_value, True

            for path in self.additional_path:
                if path not in current_value:
                    return None, False
                current_value = current_value[path]

            return current_value, True

        prop_values_seen = set()
        for i in range(len(example)):
            value, found = get_prop_from_record(example[i])
            if not found or safe_isnan(value):
                return False

            if not isinstance(value, Hashable) and self.type != PT.EXIST:
                # print(
                #     f"ERROR: Property {prop_name} is not hashable, skipping this property as we cannot deal with non-hashable properties yet."
                # )
                return False

            if self.type != PT.EXIST:
                prop_values_seen.add(value)

        if self.type == PT.CONSTANT:
            if len(prop_values_seen) == 1 and tuple(prop_values_seen)[0] in self.values:
                return True
            return False

        if self.type == PT.CONSISTENT:
            if len(prop_values_seen) == 1:
                return True
            return False

        if self.type == PT.UNEQUAL:
            if len(prop_values_seen) == len(example):
                return True
            return False

        if self.type == PT.EXIST:
            # as long as we didn't return above due to the property not being found, we can return True
            return True

        raise ValueError(f"Invalid Precondition type {self.type}")


class Precondition:
    """A class to represent a precondition for a hypothesis. A precondition is a set of `PreconditionClause` objects that should hold for the hypothesis to be valid.
    Currently the `Precondition` object is a conjunction of the `PreconditionClause` objects.
    """

    def __init__(self, clauses: list[PreconditionClause]):
        """A precondition is a conjunction of clauses."""
        self.clauses = clauses

    def verify(self, example: list) -> bool:
        and_result = True
        for clause in self.clauses:
            and_result = and_result and clause.verify(example)
            if not and_result:
                return False
        return True

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        output = "** Start of Precondition **\n"
        for clause in self.clauses:
            output += str(clause) + "\n"
        output += "** End of Preconditions **"
        return output

    def implies(self, other) -> bool:
        """When self is True, other should also be True."""

        ## all the clauses in other should be in self

        # TODO: handle merging for CONSTANT and CONSISTENT clauses

        for clause in other.clauses:
            if clause not in self.clauses:
                return False

        return True

    def add_clause(self, clause: PreconditionClause):
        if clause not in self.clauses:
            self.clauses.append(clause)

    def to_dict(self) -> dict:
        return {"clauses": [clause.to_dict() for clause in self.clauses]}

    @staticmethod
    def from_dict(precondition_dict: dict) -> Precondition:
        clauses = []
        for clause_dict in precondition_dict["clauses"]:
            clauses.append(PreconditionClause.from_dict(clause_dict=clause_dict))
        return Precondition(clauses=clauses)

    def __eq__(self, other) -> bool:
        if not isinstance(other, Precondition):
            return False
        return make_hashable(self.clauses) == make_hashable(other.clauses)

    def __hash__(self) -> int:
        return hash(make_hashable(self.clauses))


class UnconditionalPrecondition(Precondition):
    def __init__(self):
        super().__init__([])
        self.stage_clause = None

    def verify(self, example: list) -> bool:
        return True

    def __repr__(self) -> str:
        return "Unconditional Precondition"

    def __str__(self) -> str:
        return "Unconditional Precondition"

    def add_clause(self, clause: PreconditionClause):
        raise ValueError("Cannot add clause to Unconditional Precondition")

    def implies(self, other) -> bool:
        # Unconditional Precondition cannot imply any other preconditions as it is always True
        return False

    def to_dict(self) -> dict:
        return {"clauses": "Unconditional"}

    def __eq__(self, other) -> bool:
        return isinstance(other, UnconditionalPrecondition)

    def __hash__(self) -> int:
        return hash("Unconditional")


class Preconditions:
    def __init__(self, preconditions: list[Precondition], inverted: bool = False):
        """Preconditions is a disjunction of preconditions. If inverted is True, then a NOT operation is applied to the entire disjunction of preconditions."""
        self.preconditions = preconditions
        self.inverted = inverted

    def verify(self, example: list) -> bool:
        or_result = False
        for precondition in self.preconditions:
            or_result = or_result or precondition.verify(example)
            if or_result:
                break

        if self.inverted:
            return not or_result
        return or_result

    def to_dict(self) -> dict:
        return {
            "inverted": self.inverted,
            "preconditions": [
                precondition.to_dict() for precondition in self.preconditions
            ],
        }

    def __str__(self):
        output = ""
        if self.inverted:
            output += "NOT (\n"
        for i, precondition in enumerate(self.preconditions):
            output += str(precondition) + "\n"
            if i != len(self.preconditions) - 1:
                output += " OR "
        if self.inverted:
            output += ")"
        return output

    def __repr__(self):
        return self.__str__()

    def __eq__(self, value):
        if not isinstance(value, Preconditions):
            return False
        return (
            make_hashable(self.preconditions) == make_hashable(value.preconditions)
            and self.inverted == value.inverted
        )

    def __hash__(self):
        return hash((make_hashable(self.preconditions), self.inverted))

    @staticmethod
    def from_dict(preconditions_dict: dict) -> Preconditions:
        preconditions: list[Precondition | UnconditionalPrecondition] = []
        for precondition_dict in preconditions_dict["preconditions"]:
            if precondition_dict["clauses"] == "Unconditional":
                assert (
                    len(preconditions_dict["preconditions"]) == 1
                ), "Unconditional precondition should be the only precondition"
                preconditions.append(UnconditionalPrecondition())
            else:
                preconditions.append(Precondition.from_dict(precondition_dict))
        return Preconditions(preconditions, preconditions_dict["inverted"])

    def is_unconditional(self) -> bool:
        return all(
            [
                isinstance(precondition, UnconditionalPrecondition)
                for precondition in self.preconditions
            ]
        )

    def __iter__(self):
        return iter(self.preconditions)

    def __len__(self):
        return len(self.preconditions)


class GroupedPreconditions:
    def __init__(self, grouped_preconditions: dict[str, Preconditions]):
        self.grouped_preconditions = grouped_preconditions

    def verify(
        self, records: list, group_name: str, trace_for_metavars: Trace | None = None
    ) -> bool:
        """Verify the preconditions for the given group name.
        Args:
        - records: The list of trace records for a single example to verify the preconditions on
        - group_name: The name of the group to verify the preconditions on
        - trace_for_metavars: The trace to use for querying additional metavars that might have not been indexed in the example, e.g. the context managers.
        """
        assert group_name in self.grouped_preconditions, f"Group {group_name} not found"
        assert len(records) > 0, "Records should not be empty"

        if trace_for_metavars is not None:
            earliest_time = records[0]["time"]
            pid = records[0]["process_id"]
            tid = records[0]["thread_id"]
            meta_vars = trace_for_metavars.get_meta_vars(earliest_time, pid, tid)
            # update every trace with the meta_vars
            if meta_vars:
                for key in meta_vars:
                    for i in range(len(records)):
                        records[i][f"meta_vars.{key}"] = meta_vars[key]

        return self.grouped_preconditions[group_name].verify(records)

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        output = "====================== Start of Grouped Precondition ======================\n"
        for group_name, preconditions in self.grouped_preconditions.items():
            output += f"Group: {group_name}\n"
            output += str(preconditions) + "\n"
        output += (
            "====================== End of Grouped Precondition ======================"
        )
        return output

    def to_dict(self) -> dict:
        return {
            group_name: preconditions.to_dict()
            for group_name, preconditions in self.grouped_preconditions.items()
        }

    def get_group(self, group_name: str) -> Preconditions:
        assert group_name in self.grouped_preconditions, f"Group {group_name} not found"
        return self.grouped_preconditions[group_name]

    def get_group_names(self) -> set[str]:
        return set(self.grouped_preconditions.keys())

    def add_stage_info(self, valid_stages: set[str], all_stages: set[str]):
        # construct a CONSTANT clause for the stage
        stage_clause = PreconditionClause(
            prop_name=STAGE_KEY,
            prop_dtype=str,
            _type=PT.CONSTANT,
            additional_path=None,
            values=valid_stages,
        )

        invalid_stages = all_stages - valid_stages

        inverted_state_clause = PreconditionClause(
            prop_name=STAGE_KEY,
            prop_dtype=str,
            _type=PT.CONSTANT,
            additional_path=None,
            values=invalid_stages,
        )

        # add the stage clause to all the preconditions, if UNCONDITIONAL, then swap it with the stage clause
        for group_name, preconditions in self.grouped_preconditions.items():
            if preconditions.is_unconditional():
                self.grouped_preconditions[group_name] = Preconditions(
                    [Precondition([stage_clause])], inverted=False
                )
            else:
                if not preconditions.inverted:
                    for precondition in preconditions:
                        precondition.add_clause(stage_clause)
                else:
                    preconditions.preconditions.append(
                        Precondition([inverted_state_clause])
                    )

    def __eq__(self, other) -> bool:
        if not isinstance(other, GroupedPreconditions):
            return False
        return sorted(self.grouped_preconditions.items()) == sorted(
            other.grouped_preconditions.items()
        )

    def __hash__(self) -> int:
        items = tuple(
            (k, tuple(v)) for k, v in sorted(self.grouped_preconditions.items())
        )
        return hash(items)

    @staticmethod
    def from_dict(precondition_dict: dict) -> GroupedPreconditions:
        grouped_preconditions: dict[str, Preconditions] = {}
        for group_name, preconditions in precondition_dict.items():
            grouped_preconditions[group_name] = Preconditions.from_dict(preconditions)
        return GroupedPreconditions(grouped_preconditions)

    def is_group_unconditional(self, group_name: str) -> bool:
        assert group_name in self.grouped_preconditions, f"Group {group_name} not found"
        return self.grouped_preconditions[group_name].is_unconditional()

    def is_unconditional(self) -> bool:
        return all(
            [
                precondition.is_unconditional()
                for precondition in self.grouped_preconditions.values()
            ]
        )


class Invariant:
    def __init__(
        self,
        relation: Type[Relation],
        params: list[Param],
        precondition: GroupedPreconditions | None,
        text_description: str | None = None,
        num_positive_examples: int | None = None,
        num_negative_examples: int | None = None,
    ):
        self.relation = relation
        self.params = params
        self.precondition = precondition
        self.text_description = text_description

        # optional values for the number of positive and negative examples -- potentially for invariant amendement at runtime
        self.num_positive_examples = num_positive_examples
        self.num_negative_examples = num_negative_examples

    def __str__(self) -> str:
        return f"""Relation: {self.relation}\nParam Selectors: {self.params}\nPrecondition: {self.precondition}\nText Description: {self.text_description}"""

    def __hash__(self) -> int:
        self_dict = self.to_dict(_dumping_for_failed_cases=True)
        # remove num_positive_examples, num_negative_examples, and text description as they are optional
        self_dict.pop("num_positive_examples", None)
        self_dict.pop("num_negative_examples", None)
        self_dict.pop("text_description", None)
        return hash(make_hashable(self_dict))

    def __eq__(self, other) -> bool:
        if not isinstance(other, Invariant):
            return False
        return hash(self) == hash(other)

    def to_dict(self, _dumping_for_failed_cases=False) -> dict:

        # when normally dumping the invariants, the precondition must be set (as only invariants that have preconditions are dumped)
        if not _dumping_for_failed_cases:
            assert (
                self.precondition is not None
            ), f"Invariant precondition is NOT SET, check the infer function of {self.relation.__name__} (invariant text description: {self.text_description})"

            return {
                "text_description": self.text_description,
                "relation": self.relation.__name__,
                "params": [param.to_dict() for param in self.params],
                "precondition": self.precondition.to_dict(),
                "num_positive_examples": self.num_positive_examples,
                "num_negative_examples": self.num_negative_examples,
            }
        else:
            # assert (
            #     self.precondition is None
            # ), "Precondition should be None for failed cases"
            # HACK: NEED TO CHECK ALL FAILED INVARIANTS THAT HAVE PRECONDITIONS
            return {
                "text_description": self.text_description,
                "relation": self.relation.__name__,
                "params": [param.to_dict() for param in self.params],
                "precondition": "Failed",
                "num_positive_examples": self.num_positive_examples,
                "num_negative_examples": self.num_negative_examples,
            }

    @staticmethod
    def from_dict(invariant_dict: dict) -> Invariant:
        relation = Relation.from_name(invariant_dict["relation"])
        text_description = invariant_dict["text_description"]
        params = [
            Param.from_dict(param_dict) for param_dict in invariant_dict["params"]
        ]
        precondition = GroupedPreconditions.from_dict(invariant_dict["precondition"])
        num_positive_examples = invariant_dict["num_positive_examples"]
        num_negative_examples = invariant_dict["num_negative_examples"]
        return Invariant(
            relation,
            params,
            precondition,
            text_description,
            num_positive_examples,
            num_negative_examples,
        )

    def check(self, trace: Trace, check_relation_first: bool) -> CheckerResult:
        assert (
            self.precondition is not None
        ), "Invariant precondition is None. It should at least be 'Unconditional' or an empty list. Please check the invariant file and the inference process."

        logging.getLogger(__name__).info(
            f"Checking invariant: {self.text_description} of relation {self.relation}"
        )
        print(
            f"Checking invariant: {self.text_description} of relation {self.relation}"
        )
        return self.relation.static_check_all(trace, self, check_relation_first)


class CheckerResult:
    def __init__(
        self,
        trace: Optional[list[dict]],
        invariant: Invariant,
        check_passed: bool,
        triggered: bool,
    ):
        if trace is None:
            assert check_passed, "Check passed should be True for None trace"
        else:
            assert len(trace) > 0, "Trace should not be empty"
        self.trace = trace
        self.invariant = invariant
        self.check_passed = check_passed
        self.triggered = triggered

    def __str__(self) -> str:
        return f"Trace: {self.trace}\nInvariant: {self.invariant}\nResult: {self.check_passed}"

    def __repr__(self):
        return self.__str__()

    def get_min_time(self):
        if not hasattr(self, "min_time"):
            self.min_time = min([x["time"] for x in self.trace])
        return self.min_time

    def get_max_time(self):
        if not hasattr(self, "max_time"):
            self.max_time = max([x["time"] for x in self.trace])
        return self.max_time

    def calc_and_set_time_precentage(self, min_time, max_time):
        if self.check_passed:
            # don't do anything if the check passed
            return 1.0

        detection_time = self.get_max_time()
        assert (
            min_time <= detection_time <= max_time
        ), f"Detection time {detection_time} not in range [{min_time}, {max_time}]"
        self.time_precentage = (detection_time - min_time) / (max_time - min_time)
        return self.time_precentage

    def to_dict(self):
        """Convert the CheckerResult object to a json serializable dictionary."""
        result_dict = {
            "invariant": self.invariant.to_dict(),
            "check_passed": self.check_passed,
            "triggered": self.triggered,
        }

        if not self.check_passed:
            assert hasattr(
                self, "time_precentage"
            ), "Time percentage NOT SET for failed check, please call calc_and_set_time_precentage before converting to dict"

            trace = self.trace.copy()
            MD_NONE.replace_with_none(trace)

            result_dict.update(
                {
                    "detection_time": self.get_max_time(),  # the time when the invariant was detected, using max_time as the invariant cannot be checked before the
                    "detection_time_percentage": self.time_precentage,
                    "trace": trace,
                }
            )

        return result_dict


def make_hashable(value):
    """Recursively convert a value into a hashable form."""
    if isinstance(value, dict):
        # Convert dictionary into a tuple of sorted (key, hashable_value) pairs
        return frozenset((key, make_hashable(val)) for key, val in value.items())
    elif isinstance(value, (list, set, tuple)):
        # Convert lists, sets, and tuples into a tuple of hashable values
        return tuple(make_hashable(item) for item in value)
    else:
        # Return the value as-is if it is already hashable (e.g., int, str)
        return value


class Example:
    def __init__(self, trace_groups: dict[str, list[dict]] | None = None):
        self.trace_groups: dict[str, list[dict]] = trace_groups or {}

    def add_group(self, group_name: str, trace: list):
        assert group_name not in self.trace_groups, f"Group {group_name} already exists"
        self.trace_groups[group_name] = trace

    def get_group(self, group_name: str) -> list[dict]:
        return self.trace_groups[group_name]

    def get_group_names(self) -> set[str]:
        return set(self.trace_groups.keys())

    def __iter__(self):
        return iter(self.trace_groups)

    def __str__(self):
        return f"Example with Groups: {self.trace_groups.keys()}"

    def __repr__(self):
        return f"Example with Groups: {self.trace_groups.keys()}"

    def __hash__(self) -> int:
        hashable_trace_groups = make_hashable(self.trace_groups)
        return hash(hashable_trace_groups)

    def __eq__(self, value: object) -> bool:
        return hash(self) == hash(value)


class ExampleList:
    def __init__(self, group_names: set[str]):
        self.group_names = group_names
        self.examples: list[Example] = []

    def add_example(self, example: Example):
        if len(self.group_names) == 0:
            assert len(self.examples) == 0
            self.group_names = example.get_group_names()
        else:
            assert (
                example.get_group_names() == self.group_names
            ), f"Example groups do not match the expected group names, expected: {self.group_names}, got: {set(example.trace_groups.keys())}"
        self.examples.append(example)

    def get_group_from_examples(self, group_name: str) -> list[list[dict]]:
        return [example.get_group(group_name) for example in self.examples]

    def get_group_names(self) -> set[str]:
        assert (
            len(self.group_names) != 0
        ), "This example has not be initialized yet, please check implementation"
        return self.group_names

    def __len__(self):
        return len(self.examples)

    @staticmethod
    def from_iterable_of_examples(
        input: Iterable[Example], preset_group_names: set[str]
    ) -> ExampleList:
        examples = []
        for exp in input:
            assert preset_group_names == exp.get_group_names()
            examples.append(exp)

        example_list = ExampleList(preset_group_names)
        example_list.examples = examples
        return example_list


# def calc_likelihood(num_pos_exps: int, num_neg_exps: int) -> float:
#     assert (
#         num_pos_exps > 0
#     ), "No positive examples found for the hypothesis, check the inference process, calc_likelihood should only be called after the example collection process"

#     # calculate the likelihood with smoothing factors
#     likelihood = (num_pos_exps + 1) / (
#         num_pos_exps + num_neg_exps + 2
#     )  # alpha = 1, beta = 1 (Posterior Likelihood)

#     return likelihood


def calc_likelihood(num_pos_exps: int, num_neg_exps: int) -> float:
    assert (
        num_pos_exps > 0
    ), "No positive examples found for the hypothesis, check the inference process."

    # Scale the difference between positive and negative examples
    # You can tune the scaling factor (lambda) to control sensitivity
    scale_factor = 0.1  # You can adjust this to make the function more or less sensitive to differences
    likelihood = 1 / (1 + math.exp(-scale_factor * (num_pos_exps - num_neg_exps)))

    return likelihood


class Hypothesis:
    def __init__(
        self,
        invariant: Invariant,
        positive_examples: ExampleList,
        negative_examples: ExampleList,
    ):
        self.invariant = invariant
        self.positive_examples = positive_examples
        self.negative_examples = negative_examples
        self.valid_only_for_stages = False
        self.valid_stages: set[str] = set()

    def get_invariant(self, all_stages: set[str]) -> Invariant:
        """
        Get the invariant from the hypothesis.
        This method should be used instead of directly accessing hypothesis.invariant
        as it will set the number of positive and negative examples in the invariant,
        and handle stage-based inference.
        """

        inv = self.invariant
        inv.num_positive_examples = len(self.positive_examples)
        inv.num_negative_examples = len(self.negative_examples)

        if self.valid_only_for_stages:
            assert (
                inv.precondition is not None
            ), "Invariant precondition has to be first inferred"
            inv.precondition.add_stage_info(self.valid_stages, all_stages)
        return inv

    def add_stage_info(self, stage: str | set[str], all_stages: set[str]):
        self.valid_only_for_stages = True
        if isinstance(stage, str):
            self.valid_stages.add(stage)
        else:
            self.valid_stages.update(stage)

        if self.valid_stages == all_stages:
            self.valid_only_for_stages = False
        else:
            assert self.valid_stages.issubset(all_stages), "Invalid stages provided"

    def _print_debug(self):
        return f"Hypothesized Invariant: {self.invariant}\n# Positive examples: {len(self.positive_examples)}\n# Negative examples: {len(self.negative_examples)}"

    def calc_likelihood(self):
        return calc_likelihood(len(self.positive_examples), len(self.negative_examples))

    def __str__(self):
        return self._print_debug()

    def __repr__(self) -> str:
        return self._print_debug()

    def __hash__(self):
        return hash(self.invariant)

    def __eq__(self, other):
        if not isinstance(other, Hypothesis):
            return False
        return hash(self) == hash(other)


class FailedHypothesis:
    def __init__(self, hypothesis: Hypothesis, failed_reason: str):
        self.failed_reason = failed_reason
        self.hypothesis = hypothesis

    def to_dict(self):
        return {
            "failed_reason": self.failed_reason,
            "invariant": self.hypothesis.invariant.to_dict(
                _dumping_for_failed_cases=True
            ),
            "num_positive_examples": len(self.hypothesis.positive_examples),
            "num_negative_examples": len(self.hypothesis.negative_examples),
        }


class Relation(abc.ABC):

    @staticmethod
    def generate_hypothesis(trace: Trace) -> list[Hypothesis]:
        """Given a trace, should return a list of hypothesis with positive/negative examples collected on the current trace
        args:
            trace: Trace
                A trace to generate the hypothesis on.
        """
        raise NotImplementedError("generate_hypothesis method is not implemented yet.")

    @staticmethod
    def collect_examples(trace: Trace, hypothesis: Hypothesis):
        """Given a trace and a hypothesis, should collect positive and negative examples for the hypothesis.
        args:
            trace: Trace
                A trace to collect the examples on.
            hypothesis: Hypothesis
                The hypothesis to collect the examples for.
        """
        raise NotImplementedError("collect_examples method is not implemented yet.")

    @staticmethod
    @abc.abstractmethod
    def infer(trace) -> tuple[list[Invariant], list[FailedHypothesis]]:
        """Given a trace, should return a boolean value indicating
        whether the relation holds or not.

        args:
            trace: str
                A trace to infer the relation on.
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def evaluate(value_group: list) -> bool:
        """Given a group of values, should return a boolean value
        indicating whether the relation holds or not.

        args:
            value_group: list
                A list of values to evaluate the relation on. The length of the list
                should be equal to the number of variables in the relation.
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def get_precondition_infer_keys_to_skip(hypothesis: Hypothesis) -> list[str]:
        """Given a hypothesis, should return a list of keys to skip in the infer process.

        args:
            hypothesis: Hypothesis
                The hypothesis to get the keys to skip for.
        """
        pass

    @staticmethod
    def from_name(relation_name: str) -> Type[Relation]:
        """Given a relation name, should return the relation class.

        args:
            relation_name: str
                The name of the relation.
        """
        for type_relation in Relation.__subclasses__():
            if type_relation.__name__ == relation_name:
                return type_relation

        raise ValueError(f"Relation {relation_name} not found")

    @staticmethod
    @abc.abstractmethod
    def static_check_all(
        trace: Trace, inv: Invariant, check_relation_first: bool
    ) -> CheckerResult:
        """Given a trace and an invariant, should return a boolean value
        indicating whether the invariant holds on the trace.

        args:
            trace: Trace
                A trace to check the invariant on.
            inv: Invariant
                The invariant to check on the trace.
        """
        pass


def read_inv_file(file_path: str | list[str]) -> list[Invariant]:
    if isinstance(file_path, str):
        file_path = [file_path]
    invs = []
    for file in file_path:
        with open(file, "r") as f:
            for line in f:
                inv_dict = json.loads(line, cls=MDNONEJSONDecoder)
                inv = Invariant.from_dict(inv_dict)
                invs.append(inv)
    return invs
