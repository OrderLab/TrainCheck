import threading


class RegistryEntry:
    """A class to store the tracked object and its associated metadata"""

    def __init__(self, obj, var_name, var_type, stale):
        self.var = obj
        self.var_name = var_name
        self.var_type = var_type
        self.stale = stale


class VarRegistry:
    """A helper class managing all variables being tracked and allow for controlled dumps of
    the variable states.

    A variable is uniquely identified by its "name"
    When a variable is added to the registry, it is marked as "not stale".
    When a variable is dumped through `dump_sample` or `dump_modified`, it is marked as "stale".
    A variable is only dumped through `dump_modified` if it is not stale.

    """

    def __init__(self):
        self.registry: dict[str, RegistryEntry] = {}
        self.registry_lock = threading.Lock()

    def add_var(self, var, var_name: str, var_type: str):
        """Add a new proxy variable to the registry"""
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

    def dump_sample(self, dump_loc=None, dump_config=None):
        """A complete dump of all present proxy objects

        Calling this API mark all proxy objects as stale which
        will affect the `dump_modified` API.
        """
        to_dump_types = set(dump_config.keys())
        with self.registry_lock:
            for _, entry in self.registry.items():
                var_type = entry.var_type
                if var_type not in to_dump_types:
                    continue
                entry.stale = True
                entry.var.dump_trace(phase="sample", dump_loc=dump_loc)

    def dump_modified(self, dump_loc=None, dump_config=None):
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
        print("\nDumping from", dump_loc)
        to_dump_types = set(dump_config.keys())
        with self.registry_lock:
            for var_name, entry in self.registry.items():
                print(f"var_name: {var_name}")
                var_type = entry.var_type
                if var_type not in to_dump_types:
                    print("  Skipping variable type:", var_type)
                    continue

                if entry.stale:
                    print("  Skipping stale variable.")
                    continue

                entry.stale = True
                entry.var.dump_trace(phase="selective-sample", dump_loc=dump_loc)
        print("Done dumping modified variables.")


# Global dictionary to store registered objects
global_registry = VarRegistry()


def get_global_registry():
    return global_registry
