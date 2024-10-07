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
                        skip_fields=["args", "kwargs", "return_value"],
                    )
                )
            except Exception as e:
                logger.fatal(f"Failed to read line {line} due to {e}.")
                print(line)
                raise e
    return docs
