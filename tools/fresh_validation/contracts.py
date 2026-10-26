"""Validated JSON boundaries shared by the correctness-study tools.

Unknown JSON and frame metadata enter as ``object``. These helpers never replace
required fields with defaults, and distinguish booleans from numeric counts.
"""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TypeAlias, cast

JSONValue: TypeAlias = (
    None | bool | int | float | str | list["JSONValue"] | dict[str, "JSONValue"]
)
JSONObject: TypeAlias = dict[str, JSONValue]


def mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{context} must be an object with string keys")
    # The check establishes the key type; read-only Mapping does not promise
    # anything about external values before the individual field validators run.
    return cast(Mapping[str, object], value)


def sequence(value: object, context: str) -> Sequence[object]:
    if not isinstance(value, list):
        raise ValueError(f"{context} must be a JSON array")
    return value


def string(value: object, context: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{context} must be a string")
    return value


def integer(value: object, context: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{context} must be an integer")
    return value


def boolean(value: object, context: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{context} must be a boolean")
    return value


def number(value: object, context: str) -> int | float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{context} must be a number")
    return value


def required(data: Mapping[str, object], key: str) -> object:
    if key not in data:
        raise ValueError(f"Missing required field: {key}")
    return data[key]


def json_value(value: object, context: str = "JSON") -> JSONValue:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, list):
        return [json_value(child, context) for child in value]
    if isinstance(value, Mapping):
        return {
            key: json_value(child, f"{context}.{key}")
            for key, child in mapping(value, context).items()
        }
    raise ValueError(f"{context} has a non-JSON value of type {type(value).__name__}")


def json_object(value: object, context: str = "JSON") -> JSONObject:
    result = json_value(value, context)
    if not isinstance(result, dict):
        raise ValueError(f"{context} must be an object")
    return result


def json_list(value: object, context: str) -> list[JSONValue]:
    result = json_value(value, context)
    if not isinstance(result, list):
        raise ValueError(f"{context} must be an array")
    return result


def read_object(path: Path) -> JSONObject:
    value: object = json.loads(path.read_text())
    return json_object(value, str(path))
