"""TrainCheck public exports, loaded when requested rather than on package import.

Explicit instrumentor imports retain their historical activation behavior.
Importing a lightweight observation helper must not patch framework functions.
"""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from traincheck.developer.annotations import (
        annotate_answer_start_token_ids,
        annotate_stage,
    )
    from traincheck.instrumentor.tracer import Instrumentor, get_all_subclasses

__all__ = [
    "Instrumentor",
    "get_all_subclasses",
    "annotate_stage",
    "annotate_answer_start_token_ids",
]

_EXPORT_MODULES: dict[str, str] = {
    "Instrumentor": "traincheck.instrumentor.tracer",
    "get_all_subclasses": "traincheck.instrumentor.tracer",
    "annotate_stage": "traincheck.developer.annotations",
    "annotate_answer_start_token_ids": "traincheck.developer.annotations",
}


def __getattr__(name: str) -> object:
    if name not in _EXPORT_MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value: object = getattr(import_module(_EXPORT_MODULES[name]), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
