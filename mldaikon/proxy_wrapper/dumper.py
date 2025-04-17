import json
from typing import Dict

from traincheck.instrumentor.dumper import convert_var_to_dict
from traincheck.instrumentor.tracer import TraceLineType
from traincheck.instrumentor.tracer import get_meta_vars as tracer_get_meta_vars
from traincheck.proxy_wrapper.proxy_basics import is_proxied
from traincheck.proxy_wrapper.proxy_config import primitive_types


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
    return result


def get_meta_vars(obj):
    all_meta_vars = tracer_get_meta_vars()

    return all_meta_vars


def concat_dicts(dict1, dict2):
    return {**dict1, **dict2}
