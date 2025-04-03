import json
from typing import Dict

from mldaikon.instrumentor.dumper import convert_var_to_dict
from mldaikon.instrumentor.tracer import TraceLineType
from mldaikon.instrumentor.tracer import get_meta_vars as tracer_get_meta_vars
from mldaikon.proxy_wrapper.proxy_basics import is_proxied
from mldaikon.proxy_wrapper.proxy_config import delta_dump_config, primitive_types

delta_dump = delta_dump_config["delta_dump"]
delta_dump_attributes = delta_dump_config["delta_dump_attributes"]
delta_dump_meta_var = delta_dump_config["delta_dump_meta_var"]


class Singleton(type):

    _instances: Dict[type, type] = {}

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
        time,
        meta_vars,
        var_name,
        var_type,
        change_type,
        var_attributes,
        dump_loc=None,
    ):

        if (
            var_type == "method"
            or var_type == "function"
            or var_type in primitive_types
        ):
            return

        data = {
            "var_name": var_name,
            "var_type": var_type,
            "mode": change_type,  # "new", "update"
            "dump_loc": dump_loc,
            "process_id": process_id,
            "thread_id": thread_id,
            "time": time,
            "meta_vars": meta_vars,
            "attributes": var_attributes,
            "type": TraceLineType.STATE_CHANGE,
        }

        json_data = json.dumps(data)

        self.json_file.write(json_data + "\n")

    def __del__(self):
        self.close()

    def close(self):
        self.json_file.close()

    def create_instance(self):
        return json_dumper(self.json_file.name)


def dump_attributes(obj, value):
    result = {}
    if not hasattr(value, "__dict__"):
        return result

    # if the object is a proxy object, get the original object
    obj_dict = value.__dict__
    if is_proxied(value):
        value = obj_dict["_obj"]

    result = convert_var_to_dict(value)

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


def get_meta_vars(obj):
    all_meta_vars = tracer_get_meta_vars()

    if delta_dump and delta_dump_meta_var:
        # if they have common keys, only dump when old value is different from the new value
        old_value = obj.__dict__.get("old_meta_vars", {})
        # store the old value of the meta_var
        store_old_value_meta_var(obj, meta_vars=all_meta_vars)
        if old_value is not None:
            all_meta_vars = {
                key: value
                for key, value in all_meta_vars.items()
                if key not in old_value or old_value[key] != value
            }
    return all_meta_vars


def concat_dicts(dict1, dict2):
    return {**dict1, **dict2}


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
                obj_dict["old_meta_vars"] = get_meta_vars(obj)
            else:
                obj_dict["old_meta_vars"] = meta_vars
