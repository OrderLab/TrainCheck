import inspect
import json
from typing import Dict

import torch

from mldaikon.instrumentor.tracer import meta_vars
from mldaikon.proxy_wrapper.hash import tensor_hash
from mldaikon.proxy_wrapper.proxy_basics import is_proxied
from mldaikon.proxy_wrapper.proxy_config import (
    attribute_black_list,
    delta_dump_config,
    exclude_file_names,
    meta_var_black_list,
    primitive_types,
    tensor_dump_format,
)
from mldaikon.proxy_wrapper.utils import print_debug

delta_dump = delta_dump_config["delta_dump"]
delta_dump_attributes = delta_dump_config["delta_dump_attributes"]
delta_dump_meta_var = delta_dump_config["delta_dump_meta_var"]


class Singleton(type):

    _instances: Dict[type, type] = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class SkippedDumpingObj:
    def __init__(self, obj):
        self._obj = obj

    def __repr__(self):
        return f"Skipped Dumping Object: ({self._obj})"


class json_dumper(metaclass=Singleton):
    # singleton pattern for shared state
    _shared_state = False

    def __init__(self, json_file_path):
        self.json_file = open(json_file_path, "a")

    def dump_json(
        self,
        process_id,
        thread_id,
        time,
        meta_vars,
        variable_name,
        var_type,
        var_value,
        change_type,
        var_attributes,
        stack_trace=None,
    ):

        if (
            var_type == "method"
            or var_type == "function"
            or var_type in primitive_types
        ):
            return
        if (
            stack_trace
            == """[["/home/ziming/miniconda3/lib/python3.11/site-packages/torch/optim/adam.py", 92, "if p.grad is not None:"], ["/home/ziming/miniconda3/lib/python3.11/site-packages/torch/optim/adam.py", 157, "has_complex = self._init_group("], ["/home/ziming/miniconda3/lib/python3.11/site-packages/torch/optim/optimizer.py", 76, "ret = func(self, *args, **kwargs)"], ["/home/ziming/miniconda3/lib/python3.11/site-packages/torch/optim/optimizer.py", 385, "out = func(*args, **kwargs)"], ["/data/ziming/ml-daikon/_ml_daikon_instrumented_84911.py", 100, "optimizer.step()"], ["/data/ziming/ml-daikon/_ml_daikon_instrumented_84911.py", 141, "model_transfer, res = train(num_epochs, data_transfer, model_transfer, optimizer_transfer, criterion_transfer, use_cuda, f'results/{num_epochs}_{lr}')"]]"""
        ):
            import pdb

            pdb.set_trace()
            return

        data = {
            # "value": var_value,
            "var_name": variable_name,
            "var_type": var_type,
            "mode": change_type,  # "new", "update"
            "stack_trace": stack_trace,
            "process_id": process_id,
            "thread_id": thread_id,
            "time": time,
            "meta_vars": json.dumps(str(meta_vars)),
            "attributes": var_attributes,
        }
        json_data = json.dumps(data)

        self.json_file.write(json_data + "\n")

    def __del__(self):
        self.close()

    def close(self):
        self.json_file.close()

    def create_instance(self):
        return json_dumper(self.json_file.name)


def tensor_stats(tensor):
    min = float(tensor.min().item())
    max = float(tensor.max().item())
    mean = float(tensor.mean().item())
    std = float(tensor.std().item())
    shape = tuple(int(x) for x in tensor.size())
    return {
        "min": min,
        "max": max,
        "mean": mean,
        "std": std,
        "shape": shape,
    }


def dump_tensor(value):
    param_list = None
    if isinstance(value, torch.Tensor):
        if tensor_dump_format["dump_tensor_version"]:
            # import pdb; pdb.set_trace()
            param_list = value._version
        # dump out the tensor data to a list and flatten it to a 1D list
        if tensor_dump_format["dump_tensor_statistics"]:
            param_list = tensor_stats(value)
        if tensor_dump_format["dump_tensor_hash"]:
            param_list = tensor_hash(value)
        else:
            param_list = value.detach().flatten().tolist()

    return param_list


def dump_attributes(obj, value):
    result = {}
    if not hasattr(value, "__dict__"):
        return result

    # if the object is a proxy object, get the original object
    obj_dict = value.__dict__
    if is_proxied(value):
        value = obj_dict["_obj"]

    # currently only dump primitive types, tensors and nn.Module
    attr_names = [name for name in dir(value) if not name.startswith("__")]
    dump_tensor_version = tensor_dump_format["dump_tensor_version"]
    if dump_tensor_version and isinstance(value, torch.nn.parameter.Parameter):
        result = value._version
        return result

    for attr_name in attr_names:
        # don't track the attr_name starts with a _ (private variable)
        if attr_name.startswith("_"):
            continue
        if attr_name in attribute_black_list:
            continue
        try:
            attr = getattr(value, attr_name)
            if attr is None:
                result[attr_name] = None
            if type(attr) in primitive_types:
                result[attr_name] = attr

            elif isinstance(attr, torch.Tensor):
                result[attr_name] = dump_tensor(attr)

            elif isinstance(attr, torch.nn.parameter.Parameter):
                result[attr_name] = attr.__class__.__name__ + "(Parameter)"
                result[attr_name] = dump_tensor(attr.data)

            elif isinstance(attr, torch.nn.Module):
                result[attr_name] = attr.__class__.__name__ + "(nn.Module)"
                # dump out all tensors inside the nn.Module
                for name, param in attr.named_parameters():
                    result[attr_name] += f"\n{name}: {dump_tensor(param)}"
        except Exception as e:  # noqa
            print_debug(
                lambda: f"Failed to get attribute {attr_name} of object type {type(value)}, skipping it. Error: {e}."  # noqa
            )

    if delta_dump and delta_dump_attributes:
        # if they have common keys, only dump when old value is different from the new value
        old_value = obj.__dict__.get("old_value", {})
        # store the old value of the attribute
        store_old_value(obj, result)
        if old_value is not None:
            result = {
                key: value
                for key, value in result.items()
                if key not in old_value or old_value[key] != value
            }
    return result


def dump_meta_vars(obj, level=20, proxy_file_path=""):
    frame = inspect.currentframe()
    while (
        frame.f_code.co_filename == proxy_file_path
        or frame.f_code.co_filename == __file__
    ):
        frame = frame.f_back

    important_vars = {}
    # get the file name list inside the repo
    i = 0
    while i < level and frame is not None:
        if frame.f_code.co_filename in exclude_file_names:
            frame = frame.f_back
            continue

        frame_vars = frame.f_locals

        important_vars.update(
            {
                key: frame_vars[key]
                for key in frame_vars
                # Ziming: only dump primitive types, block the var name on the black list
                if isinstance(frame_vars[key], (int, float, str, bool))
                and key not in meta_var_black_list
                and key not in important_vars
            }
        )

        frame = frame.f_back
        if frame is None:
            break
        frame_vars = frame.f_locals
        i += 1
    dumped_meta_vars = concat_dicts(important_vars, meta_vars)

    if delta_dump and delta_dump_meta_var:
        # if they have common keys, only dump when old value is different from the new value
        old_value = obj.__dict__.get("old_meta_vars", {})
        # store the old value of the meta_var
        store_old_value_meta_var(obj, meta_vars=dumped_meta_vars)
        if old_value is not None:
            dumped_meta_vars = {
                key: value
                for key, value in dumped_meta_vars.items()
                if key not in old_value or old_value[key] != value
            }
    return dumped_meta_vars


def concat_dicts(dict1, dict2):
    return {**dict1, **dict2}


def torch_serialize(obj, dump_module_tensors=False):
    if isinstance(obj, (int, float, str, bool)):
        return obj
    if isinstance(obj, torch.Tensor):
        new_value = str(dump_tensor(obj))
        return new_value
    if isinstance(obj, torch.nn.Module):
        new_value = obj.__class__.__name__ + "(nn.Module)"
        if dump_module_tensors:
            # dump out all tensors inside the nn.Module
            for name, param in obj.named_parameters():
                new_value += f"\n{name}: {dump_tensor(param)}"
        return new_value
    else:
        try:
            json.dumps(obj)
        except TypeError:
            obj = str(obj)
        return obj


def store_old_value(obj, result):
    # set the current snapshot as the "old_value" attribute of the object
    if delta_dump:
        obj_dict = obj.__dict__
        assert is_proxied(obj), "The object is not a proxied object"
        if delta_dump_attributes:
            import copy

            obj_dict["old_value"] = copy.deepcopy(result)


def store_old_value_meta_var(obj, meta_vars=None):
    # save the current meta_var of the function stack
    if delta_dump:
        obj_dict = obj.__dict__
        assert is_proxied(obj), "The object is not a proxied object"
        if delta_dump_meta_var:
            if meta_vars is None:
                obj_dict["old_meta_vars"] = dump_meta_vars()
            else:
                obj_dict["old_meta_vars"] = meta_vars
