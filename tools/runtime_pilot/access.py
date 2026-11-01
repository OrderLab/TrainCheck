"""Process-local TrainCheck import boundary for a control worker.

Install before dependency bootstrap, while the worker is still single-threaded.
This finder blocks normal imports, including an otherwise installed TrainCheck;
it is not an OS sandbox against code that edits import machinery, injects
``sys.modules`` entries, or executes source through a custom loader.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from importlib.abc import MetaPathFinder
from importlib.machinery import ModuleSpec
from types import ModuleType


def _is_traincheck(fullname: str) -> bool:
    return fullname == "traincheck" or fullname.startswith("traincheck.")


class TrainCheckImportBlocker(MetaPathFinder):
    """Reject only TrainCheck's exact package namespace through normal imports."""

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None,
        target: ModuleType | None = None,
    ) -> ModuleSpec | None:
        if _is_traincheck(fullname):
            raise ModuleNotFoundError(
                f"TrainCheck is unavailable in this control worker: {fullname}",
                name=fullname,
            )
        return None


def install_traincheck_blocker() -> TrainCheckImportBlocker:
    """Install first in the import search order, or reuse the owned finder.

    Existing TrainCheck namespace entries, even ``None`` placeholders, are an
    error: removing them would conceal earlier imports and their side effects.
    Reinstallation validates that invariant again and restores the same finder
    to first position if another component has moved it.
    """
    loaded = sorted(name for name in tuple(sys.modules) if _is_traincheck(name))
    if loaded:
        raise RuntimeError(
            "Cannot install TrainCheck import blocker after namespace entries "
            f"already exist: {', '.join(loaded)}"
        )
    for index, finder in enumerate(sys.meta_path):
        # Only reuse this exact implementation, not a subclass with different
        # behavior. Do not remove or replace unrelated import finders.
        if type(finder) is TrainCheckImportBlocker:
            assert isinstance(finder, TrainCheckImportBlocker)
            if index:
                del sys.meta_path[index]
                sys.meta_path.insert(0, finder)
            return finder
    blocker = TrainCheckImportBlocker()
    sys.meta_path.insert(0, blocker)
    return blocker
