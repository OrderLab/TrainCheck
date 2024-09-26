import logging
import json
from collections.abc import MutableMapping

from mldaikon.trace.types import MD_NONE



def _flatten_dict_gen(d, parent_key, sep):
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            yield from flatten_dict(v, new_key, sep=sep).items()
        else:
            yield new_key, v


def flatten_dict(d: MutableMapping, parent_key: str = "", sep: str = "."):
    return dict(_flatten_dict_gen(d, parent_key, sep))


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
    logger  = logging.getLogger(__name__)
    with open(file_path, "r") as f:
        for line in f.readlines():
            try:
                docs.append(
                    flatten_dict(
                        json.loads(line, object_hook=replace_none_with_md_none)
                    )
                )
            except Exception as e:
                logger.fatal(f"Failed to read line {line} due to {e}.")
                print(line)
                raise e
    return docs