import importlib
import inspect
import json
import logging
from collections.abc import MutableMapping

from mldaikon.trace.types import MD_NONE


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
        # flat this object later.
        raise NotImplementedError()


def bind_args_kwargs_to_signature(
    args, kwargs, signature: inspect.Signature
) -> BindedFuncInput:
    """Bind dumped args and kwargs to the signature of the function.

    Args:
        args (list[dict]): List of dictionaries. Each dictionary describes an argument as {type of provided_value: [{attr: value}]} if the value is not a primitive type.
        kwargs (dict): Dictionary of keyword arguments, {kwarg_name: {type of provided_value: [{attr: value}]}}.
        signature (inspect.Signature): Signature of the function.
    """

    # NOTE: we have to implement our own binding instead of using inspect.Signature.bind is because during the tracing we might not record everything (e.g. for tensors).

    bind_args_and_kwargs: dict[str, dict] = (
        {}
    )  # {arg_name: {type of the provided value: [{attr: value}] | the value itself}}

    for idx, arg_name in enumerate(signature.parameters.keys()):
        if arg_name == "self":
            # NOTE: I am not sure if the arg dumping process will dump the self argument or not. Let's assert all args are binded below to guard against uncertain behavior.
            continue
        # let's consume all the args first
        bind_args_and_kwargs[arg_name] = args[idx]

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
        bind_args_and_kwargs[arg_name] = signature.parameters[arg_name].default

    assert len(bind_args_and_kwargs) == len(
        signature.parameters
    ), f"Number of binded arguments {len(bind_args_and_kwargs)} does not match the number of arguments in the signature {len(signature.parameters)}."
    return BindedFuncInput(bind_args_and_kwargs)


def load_signature_from_func_name(func_name) -> inspect.Signature:
    # the func_name should be in the format of "module_name1.module_name2.....func_name" also the module_name1.module_name2 should be importable.
    root_module_name = func_name.split(".")[0]
    module = importlib.import_module(root_module_name)

    for sub_module_name in func_name.split(".")[1:]:
        module = getattr(module, sub_module_name)

    assert callable(module), f"Function {func_name} is not callable."
    return inspect.signature(module)
