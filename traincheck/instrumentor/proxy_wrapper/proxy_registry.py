import threading
from collections import deque
from collections.abc import Callable, Mapping
from typing import Literal, Protocol


class TraceableVariable(Protocol):
    def dump_trace(self, phase: str, dump_loc: str | None) -> None: ...


class RegistryEntry:
    """A tracked object and its associated observation metadata."""

    def __init__(
        self, obj: TraceableVariable, var_name: str, var_type: str, stale: bool
    ) -> None:
        self.var = obj
        self.var_name = var_name
        self.var_type = var_type
        self.stale = stale


class VarRegistry:
    """Name-indexed variables with ordered full and modified-only sampling.

    Sampling snapshots membership and order, checking type and stale status at
    each entry's turn. Additions wait for the next call; replaced entries are skipped.
    Root finalization is deferred until the outermost sampling call ends, keeping
    a record's canonical name stable throughout conversion and serialization.
    """

    def __init__(self) -> None:
        self.registry: dict[str, RegistryEntry] = {}
        # GC may request retirement synchronously while the sampling thread owns
        # the lock. Reentry only queues retirement until sampling is finished.
        self.registry_lock = threading.RLock()
        self._sampling_depth = 0
        self._pending_cleanups: deque[Callable[[], None]] = deque()
        self._draining_cleanups = False

    def add_var(self, var: TraceableVariable, var_name: str, var_type: str) -> None:
        """Register a variable or mark its current observation as modified."""
        with self.registry_lock:
            if var_name in self.registry:
                self.registry[var_name].var = var
                self.registry[var_name].var_name = var_name
                self.registry[var_name].var_type = var_type
                self.registry[var_name].stale = False
            else:
                self.registry[var_name] = RegistryEntry(
                    var, var_name, var_type, stale=False
                )

    def defer_cleanup(self, cleanup: Callable[[], None]) -> None:
        """Retire immediately when idle, or after the current sample completes."""
        with self.registry_lock:
            self._pending_cleanups.append(cleanup)
            self._drain_cleanups()

    def _drain_cleanups(self) -> None:
        # Caller holds registry_lock. Preserve queue order during nested requests.
        if self._sampling_depth or self._draining_cleanups:
            return
        self._draining_cleanups = True
        try:
            while self._pending_cleanups:
                self._pending_cleanups.popleft()()
        finally:
            self._draining_cleanups = False

    def _dump(
        self,
        phase: Literal["sample", "selective-sample"],
        dump_loc: str | None,
        dump_config: Mapping[str, object] | None,
    ) -> None:
        if dump_config is None:
            # Absence remains an error, as in the original .keys() call; it must
            # not silently become an empty filter and a successful empty sample.
            raise AttributeError("dump_config is required for registry sampling")
        to_dump_types = set(dump_config.keys())
        with self.registry_lock:
            self._sampling_depth += 1
            try:
                entries = [
                    (name, entry, entry.var) for name, entry in self.registry.items()
                ]
                for name, entry, variable in entries:
                    if (
                        self.registry.get(name) is not entry
                        or entry.var is not variable
                    ):
                        continue
                    if entry.var_type not in to_dump_types:
                        continue
                    if phase == "selective-sample" and entry.stale:
                        continue
                    # Preserve the original failure contract: the attempted entry
                    # is stale even if dumping raises; later entries stay unchanged.
                    entry.stale = True
                    variable.dump_trace(phase=phase, dump_loc=dump_loc)
            finally:
                self._sampling_depth -= 1
                self._drain_cleanups()

    def dump_sample(
        self,
        dump_loc: str | None = None,
        dump_config: Mapping[str, object] | None = None,
    ) -> None:
        """Dump all selected entries in registry order and mark attempts stale."""
        self._dump("sample", dump_loc, dump_config)

    def dump_modified(
        self,
        dump_loc: str | None = None,
        dump_config: Mapping[str, object] | None = None,
    ) -> None:
        """Dump selected entries still marked modified when their turn arrives.

        Modified means an update was attempted, not that tensor contents differ.
        Successful and failed dump attempts are marked stale until add_var marks
        them modified again. Configuration values are retained for compatibility;
        only its type-name keys currently select entries.
        """
        self._dump("selective-sample", dump_loc, dump_config)


global_registry = VarRegistry()


def get_global_registry() -> VarRegistry:
    return global_registry
