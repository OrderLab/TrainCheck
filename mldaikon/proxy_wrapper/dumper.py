import json
import time
import torch
import inspect
from mldaikon.proxy_wrapper.utils import print_debug


class json_dumper:

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
        var_attributes,
    ):
        data = {
            "process_id": process_id,
            "thread_id": thread_id,
            "time": time.time(),
            "meta_vars": json.dumps(str(meta_vars)),
            "var_name": variable_name,
            "var_type": var_type,
            "attributes": var_attributes,
            "value": var_value,
        }
        json_data = json.dumps(data)

        self.json_file.write(json_data + "\n")

    def close(self):
        self.json_file.close()

    def create_instance(self):
        return json_dumper(self.json_file.name)


def dump_tensor(value):
    min = float(value.min().item())
    max = float(value.max().item())
    shape = tuple(int(x) for x in value.size())
    result = {
        "min": min,
        "max": max,
        "shape": shape,
    }
    return result


def dump_attributes(obj):
    result = {}
    if not hasattr(obj, "__dict__"):
        return result
    # # currently only trace tensor object
    if not isinstance(obj, torch.Tensor):
        return result
    obj_dict = obj.__dict__
    if "is_proxied_obj" in obj_dict:
        obj = obj_dict["_obj"]._obj

    primitive_types = {int, float, str, bool}
    attr_names = [name for name in dir(obj) if not name.startswith("__")]

    for attr_name in attr_names:
        try:
            attr = getattr(obj, attr_name)
            if type(attr) in primitive_types:
                result[attr_name] = str(attr)
        except Exception as e:
            print_debug(
                f"Failed to get attribute {attr_name} of object {obj}, skipping it. Error: {e}"
            )
    return result


def dump_meta_vars(level=8, proxy_file_path=""):
    frame = inspect.currentframe().f_back
    while frame.f_code.co_filename == proxy_file_path:
        frame = frame.f_back
    frame_vars = frame.f_locals
    important_vars = {}
    for i in range(level):
        important_vars.update(
            {
                key: frame_vars[key]
                for key in frame_vars
                # Ziming: only get primitive types for now
                if isinstance(frame_vars[key], (int, float, str, bool))
            }
        )

        frame = frame.f_back
        if frame is None:
            break
        frame_vars = frame.f_locals
    return important_vars


def torch_serialize(obj):
    if isinstance(obj, (int, float, str, bool)):
        return obj
    if isinstance(obj, torch.Tensor):
        new_value = str(dump_tensor(obj))
        return new_value
    if isinstance(obj, torch.nn.Module):
        new_value = obj.__class__.__name__ + "(nn.Module)"
        return new_value
    else:
        try:
            json.dumps(obj)
        except TypeError:
            obj = str(obj)
        return obj
