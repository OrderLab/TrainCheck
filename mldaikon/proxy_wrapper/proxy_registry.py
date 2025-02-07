import threading
from typing import TYPE_CHECKING

from mldaikon.proxy_wrapper import proxy_config

if TYPE_CHECKING:
    from mldaikon.proxy_wrapper.proxy import Proxy


class RegistryEntry:
    """A class to store the proxy object and its associated metadata"""

    def __init__(self, proxy: "Proxy", stale: bool, _obj=None):
        self.proxy = proxy
        self.stale = stale
        self._obj = _obj


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

    def add_unproxied_obj(self, var_name: str, _obj):
        """Add the unproxied object to the proxy variable"""
        with self.registry_lock:
            self.registry[var_name]._obj = _obj

    def access_obj(self, var_name: str):
        """Access the object of the proxy variable"""
        if proxy_config.add_obj_to_registry:
            with self.registry_lock:
                return self.registry[var_name]._obj

    def dump_sample(self, dump_loc=None):
        """A complete dump of all present proxy objects

        Calling this API mark all proxy objects as stale which
        will affect the `dump_only_modified` API.
        """
        with self.registry_lock:
            for var_name, entry in self.registry.items():
                entry.stale = True
                entry.proxy.dump_trace("sample", dump_loc=dump_loc)

    def dump_only_modified(self, dump_loc=None):
        """Dump only the proxy variables that might be modified since last dump

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
        with self.registry_lock:
            for var_name, entry in self.registry.items():
                if entry.stale:
                    entry.stale = False
                    entry.proxy.dump_trace("selective-sample", dump_loc=dump_loc)


# Global dictionary to store registered objects
global_registry = ProxyRegistry()
