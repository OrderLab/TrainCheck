import importlib
import inspect
import json
import logging
from collections.abc import MutableMapping

from traincheck.instrumentor.dumper import var_to_serializable
from traincheck.trace.types import MD_NONE, BindedFuncInput
from traincheck.utils import typename


def _flatten_dict_gen(d, parent_key, sep, skip_fields=None):
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if skip_fields and k in skip_fields:
            yield k, v
        elif isinstance(v, MutableMapping) and v != {}:
            yield from flatten_dict(v, new_key, sep=sep).items()
        else:
            yield new_key, v


def flatten_dict(
    d: MutableMapping, parent_key: str = "", sep: str = ".", skip_fields=None
):
    return dict(_flatten_dict_gen(d, parent_key, sep, skip_fields))


def replace_none_with_md_none(obj):
    # print(obj)
    for child in obj:
        if obj[child] is None:
            obj[child] = MD_NONE()
        if isinstance(obj[child], dict):
            obj[child] = replace_none_with_md_none(obj[child])
    return obj


def read_jsonlines_flattened_with_md_none(file_path: str):
    docs = []
    logger = logging.getLogger(__name__)
    with open(file_path, "r") as f:
        for line in f.readlines():
            try:
                docs.append(
                    flatten_dict(
                        json.loads(line, object_hook=replace_none_with_md_none),
                        skip_fields=["args", "kwargs", "return_values"],
                    )
                )
            except Exception as e:
                logger.fatal(f"Failed to read line {line} due to {e}.")
                print(line)
                raise e
    return docs


def bind_args_kwargs_to_signature(
    args, kwargs, signature: inspect.Signature
) -> BindedFuncInput:
    """Bind dumped args and kwargs to the signature of the function.

    Args:
        args (list[dict]): List of dictionaries. Each dictionary describes an argument as {type of provided_value: [{attr: value}]} if the value is not a primitive type.
        kwargs (dict): Dictionary of keyword arguments, {kwarg_name: {type of provided_value: [{attr: value}]}}.
        signature (inspect.Signature): Signature of the function.

    Returns:
        BindedFuncInput: Binded function input.
        The dictionary is in the format of {arg_name: {type of the provided value: [{attr: value}] | the value itself}}.
    """

    # NOTE: we have to implement our own binding instead of using inspect.Signature.bind is because during the tracing we might not record everything (e.g. for tensors).
    bind_args_and_kwargs: dict[str, dict] = (
        {}
    )  # {arg_name: {type of the provided value: [{attr: value}] | the value itself}}

    # let's consume all the args first
    for idx, arg_name in enumerate(signature.parameters.keys()):
        # NOTE: when the first the argument is `self`, we usually can't properly serialize it during tracing. We don't skip it but still probably the consumer of this function should handle it properly.
        if idx >= len(args):
            break
        bind_args_and_kwargs[arg_name] = args[str(idx)]

    # then consume the kwargs
    for kwarg_name, kwarg in kwargs.items():
        assert (
            kwarg_name not in bind_args_and_kwargs
        ), f"Duplicate kwarg {kwarg_name} found."
        bind_args_and_kwargs[kwarg_name] = kwarg

    # then assign the default values to the rest of the arguments
    unbinded_arg_names = set(signature.parameters.keys()) - set(
        bind_args_and_kwargs.keys()
    )
    for arg_name in unbinded_arg_names:
        assert (
            signature.parameters[arg_name].default != inspect.Parameter.empty
        ), f"Argument {arg_name} is not binded and has no default value."
        default_val = signature.parameters[arg_name].default
        bind_args_and_kwargs[arg_name] = {
            typename(default_val, is_runtime=True): var_to_serializable(
                default_val
            )  # the trace is dumped with is_runtime=True, so there is no need to set it to False and take some performance overhead.
        }

    assert len(bind_args_and_kwargs) == len(
        signature.parameters
    ), f"Number of binded arguments {len(bind_args_and_kwargs)} does not match the number of arguments in the signature {len(signature.parameters)}."
    return BindedFuncInput(bind_args_and_kwargs)


def load_signature_from_class_method_name(name) -> inspect.Signature:
    # the func_name should be in the format of "module_name1.module_name2....class_name.method_name".
    parent_module_name = ".".join(name.split(".")[:-2])
    parent_class = name.split(".")[-2]
    method_name = name.split(".")[-1]
    module = importlib.import_module(parent_module_name)
    class_obj = getattr(module, parent_class)

    assert hasattr(
        class_obj, method_name
    ), f"Method {method_name} not found in class {parent_class}."
    method = getattr(class_obj, method_name)

    assert callable(method), f"Function {name} is not callable."
    return inspect.signature(method)
