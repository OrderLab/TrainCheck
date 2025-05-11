import threading
import typing

from traincheck.utils import typename

if typing.TYPE_CHECKING:
    from .proxy import Proxy


class RegistryEntry:
    """A class to store the proxy object and its associated metadata"""

    def __init__(self, proxy: "Proxy", stale: bool):
        self.proxy = proxy
        self.stale = stale


class ProxyRegistry:
    """A helper class managing all proxy variables being tracked and allow for controlled dumps of
    the variable states.

    A variable is uniquely identified by its "name"
    """

    def __init__(self):
        self.registry: dict[str, RegistryEntry] = {}
        self.registry_lock = threading.Lock()

    def add_var(self, var: "Proxy", var_name: str):
        """Add a new proxy variable to the registry"""
        with self.registry_lock:
            self.registry[var_name] = RegistryEntry(proxy=var, stale=False)

    def dump_sample(self, dump_loc=None):
        """A complete dump of all present proxy objects

        Calling this API mark all proxy objects as stale which
        will affect the `dump_only_modified` API.
        """
        with self.registry_lock:
            for var_name, entry in self.registry.items():
                entry.stale = True
                entry.proxy.dump_trace(phase="sample", dump_loc=dump_loc)

    def dump_only_modified(self, dump_loc=None, dump_config=None):
        """Dump only the proxy variables that might be modified since last dump

        args:
            dump_loc: the location to dump the trace, an optional string to add to trace records
            dump_config: the config for dumping, each key would be the type of the variable and the value
                would be whether to dump all changed vars or just one

        ** This is a middle ground between blindly dump everything everytime v.s. fully-accurate delta dumping **
        fully-accuracy dumping is hard as for each "modifications" to the variable, you will need to compare
        the new state v.s. the old state to ensure the state has actually changed, which introduces great overhead.

        This function implements delta dumping but does not guarantee two consecutive dumps will be different,
        we only guarantee that between two dumps there has been attempts (e.g. through __setattr__ or observer)
        to modify the variable.


        Side effects:
        when calling the function, all dumped proxy vars will be marked as stale and will not be dumped next time
        unless there are new modification attempts to t
        """
        to_dump_types = set(dump_config.keys())
        with self.registry_lock:
            for var_name, entry in self.registry.items():
                var_type = typename(entry.proxy._obj, is_runtime=True)
                if var_type not in to_dump_types:
                    continue

                if entry.stale:
                    continue

                entry.stale = True
                entry.proxy.dump_trace(phase="selective-sample", dump_loc=dump_loc)
                if not dump_config[var_type]["dump_unchanged"]:
                    # remove the var from to_dump_types so that we don't dump the same type twice
                    to_dump_types.remove(var_type)


# Global dictionary to store registered objects
global_registry = ProxyRegistry()


def get_global_registry():
    return global_registry
