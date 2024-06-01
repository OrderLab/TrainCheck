import json
import time
import torch
import inspect
from mldaikon.proxy_wrapper.config import (
    meta_var_black_list,
    attribute_black_list,
    exclude_file_names,
)
from mldaikon.proxy_wrapper.utils import print_debug
from mldaikon.instrumentor.tracer import meta_vars


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class json_dumper(metaclass=Singleton):
    # singleton pattern for shared state
    _shared_state = False

    def __init__(self, json_file_path):
        self.json_file = open(json_file_path, "a")

    def dump_json(
        self,
        process_id,
        thread_id,
        meta_vars,
        variable_name,
        var_type,
        var_value,
        change_type,
        var_attributes,
        stack_trace=None,
    ):
        primitive_types = {int, float, str, bool}
        if (
            var_type == "method"
            or var_type == "function"
            or var_type in primitive_types
        ):
            return
        data = {
            # "value": var_value,
            "var_name": variable_name,
            "var_type": var_type,
            "process_id": process_id,
            "thread_id": thread_id,
            "time": time.time(),
            "meta_vars": json.dumps(str(meta_vars)),
            "attributes": var_attributes,
            "mode": change_type,  # "new", "update"
            "stack_trace": stack_trace,
        }
        json_data = json.dumps(data)

        self.json_file.write(json_data + "\n")

    def __del__(self):
        self.close()

    def close(self):
        self.json_file.close()
        self.json_file.write("]\n")

    def create_instance(self):
        return json_dumper(self.json_file.name)


def dump_tensor(value):
    param_list = None
    # if isinstance(value, torch.Tensor):
    #     # dump the min, max, mean of the tensor to check whether the tensor is updated
    #     min = float(value.min().item())
    #     max = float(value.max().item())
    #     mean = float(value.mean().item())

    #     shape = tuple(int(x) for x in value.size())
    #     result = {
    #         "min": min,
    #         "max": max,
    #         "mean": mean,
    #         "shape": shape,
    #     }
    if isinstance(value, torch.Tensor):
        # dump out the tensor data to a list
        param_list = value.detach().tolist()
        # # HACK: if the param_list is 2 dimensional, then add a dummy dimension to make it 2D
        if not isinstance(param_list[0], list):
            param_list = [param_list]
    return param_list


def dump_attributes(obj):
    result = {}
    if not hasattr(obj, "__dict__"):
        return result

    # if the object is a proxy object, get the original object
    obj_dict = obj.__dict__
    if "is_proxied_obj" in obj_dict:
        obj = obj_dict["_obj"]._obj

    # currently only dump primitive types, tensors and nn.Module
    primitive_types = {int, float, str, bool}
    attr_names = [name for name in dir(obj) if not name.startswith("__")]

    for attr_name in attr_names:
        # don't track the attr_name starts with a _ (private variable)
        if attr_name.startswith("_"):
            continue
        if attr_name in attribute_black_list:
            continue
        try:
            attr = getattr(obj, attr_name)
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
        except Exception as e:
            print_debug(
                f"Failed to get attribute {attr_name} of object type {type(obj)}, skipping it. Error: {e}"
            )
    return result


def dump_meta_vars(level=8, proxy_file_path=""):
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
    return concat_dicts(important_vars, meta_vars)


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
