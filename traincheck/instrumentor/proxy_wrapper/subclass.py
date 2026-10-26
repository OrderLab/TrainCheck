import functools
import logging
import os
import threading
import weakref
from dataclasses import dataclass, field
from typing import Callable, cast

import torch
from torch import nn

import traincheck.config.config as config
from traincheck.config.config import should_disable_proxy_dumping
from traincheck.instrumentor.dumper import dump_trace_VAR
from traincheck.instrumentor.proxy_wrapper.dumper import dump_attributes, get_meta_vars
from traincheck.instrumentor.tracer import TraceLineType
from traincheck.utils import get_timestamp_ns, typename

from .proxy_basics import is_fake_tensor
from .proxy_registry import RegistryEntry, get_global_registry

SUBCLASS_HOOK_KEY = "_tc_setattr_hook"
MODEL_BINDINGS_KEY = "_tc_model_bindings"
MODEL_CLEANUP_KEY = "_tc_model_tracking_cleanup"

logger = logging.getLogger(__name__)


def in_dynamo() -> bool:
    try:
        import torch._dynamo as dynamo

        return bool(dynamo.is_compiling())
    except Exception:
        return False


@dataclass
class _ParameterBindings:
    paths: dict[object, str] = field(default_factory=dict)


@dataclass
class _ModuleTrackingHooks:
    callbacks: dict[object, Callable[[nn.Module, str], None]] = field(
        default_factory=dict
    )

    def __call__(self, owner: nn.Module, name: str) -> None:
        for callback in tuple(self.callbacks.values()):
            callback(owner, name)


class ProxyParameter(torch.nn.Parameter):
    process_id: int
    thread_id: int | None
    var_name: str

    loglevel = logging.INFO

    def __new__(
        cls,
        data,
        logdir="proxy_log.log",
        log_level=logging.INFO,
        # TODO
        # recurse=False,
        var_name="",
        var_type="",
        should_dump_trace=True,
        from_call=False,
        from_iter=False,
        # TODO
        # from_copy=False,
    ):
        if isinstance(data, ProxyParameter):
            return data

        if in_dynamo() or is_fake_tensor(data):
            # we do not proxy the parameter if we are in dynamo or the tensor is a fake tensor
            if isinstance(data, nn.Parameter):
                return data
            return nn.Parameter(data, requires_grad=data.requires_grad)

        requires_grad = getattr(data, "requires_grad", False)
        tensor_grad = getattr(data, "grad", None)

        # When wrapping an existing Parameter we need to preserve any Python level
        # attributes (e.g. hooks, user defined flags, ``grad``) so that the proxy
        # behaves identically to the original parameter. ``Parameter.__new__``
        # returns a fresh instance, so we snapshot the metadata from ``data`` and
        # replay it on the new ProxyParameter via the base Tensor ``__setattr__``
        # to avoid triggering the logging logic implemented in this class.
        snapshot: dict = {}

        if isinstance(data, nn.Parameter):
            snapshot = dict(getattr(data, "__dict__", {}))
            base_tensor = data.detach()
        elif isinstance(data, torch.Tensor):
            base_tensor = data.detach()
        else:
            base_tensor = torch.as_tensor(data)

        proxied = super().__new__(cls, base_tensor, requires_grad=requires_grad)

        if snapshot:
            tensor_setattr = torch.Tensor.__setattr__
            for name, value in snapshot.items():
                if name == "grad":
                    continue
                try:
                    tensor_setattr(proxied, name, value)
                except AttributeError:
                    # Some slots (e.g. torch internals) are read-only; skip them.
                    continue

        if tensor_grad is not None:
            torch.Tensor.__setattr__(proxied, "grad", tensor_grad)

        return proxied

    def __init__(
        self,
        data,
        logdir="proxy_log.log",
        log_level=logging.INFO,
        # TODO
        # recurse=False,
        var_name="",
        var_type="",
        should_dump_trace=True,
        from_call=False,
        from_iter=False,
        # TODO
        # from_copy=False,
    ):
        super().__init__()
        # Access proxy attribute: since we are wrapping the getattr method, we need to access the attribute directly
        self.__dict__["process_id"] = os.getpid()
        self.__dict__["thread_id"] = threading.current_thread().ident
        self.__dict__["logdir"] = logdir
        self.__dict__["log_level"] = log_level
        # TODO
        # self.__dict__["meta_vars"] = {}
        # self.__dict__["is_traincheck_proxied_obj"] = True
        self.__dict__["is_traincheck_proxyparameter"] = True
        # TODO
        # self.__dict__["recurse"] = recurse
        self.__dict__["var_name"] = var_name
        self.__dict__["var_type"] = var_type
        # TODO
        # self.__dict__["old_value"] = None
        # self.__dict__["old_meta_vars"] = None

        current_time = get_timestamp_ns()

        self.__dict__["last_update_timestamp"] = current_time
        logger.debug(f"[ProxyParameter] Created ProxyParameter: {self.var_name}")
        self.register_object()

        if should_dump_trace and not should_disable_proxy_dumping():
            if from_call:
                phase = "call"

            if from_iter:
                phase = "iter"
            # if the object is generated from getattr, then do not dump it
            else:
                phase = "update"
            self.dump_trace(phase=phase, dump_loc="initing")

    def __setattr__(self, name, value):

        super().__setattr__(name, value)
        self.update_timestamp()
        self.register_object()
        if should_disable_proxy_dumping():
            return
        self.dump_trace(
            phase="update",
            dump_loc=f"__setattr__ (attribute '{name}')",
        )

    def __deepcopy__(self, memo):
        data = self.detach().clone(memory_format=torch.preserve_format)
        data.requires_grad_(self.requires_grad)
        if in_dynamo() or is_fake_tensor(self):
            return self
        return type(self)(
            data,
            var_name=self.var_name,
        )

    def update_timestamp(self):
        # Update the timestamp of the object, should be called when the object is updated, e.g. __setattr__ and observer
        current_time = get_timestamp_ns()
        self.__dict__["last_update_timestamp"] = current_time
        # TODO:
        # Proxy.var_dict[self.__dict__["var_name"]].last_update_timestamp = current_time

    def _model_tracking_enabled(self) -> bool:
        # Direct construction has no model ownership metadata and remains active.
        # Model-owned proxies remain observable while any tracked root owns them.
        if MODEL_BINDINGS_KEY not in self.__dict__:
            return True
        bindings: object = self.__dict__[MODEL_BINDINGS_KEY]
        if not isinstance(bindings, _ParameterBindings):
            raise TypeError("ProxyParameter model bindings must be _ParameterBindings")
        return bool(bindings.paths)

    def register_object(self) -> None:
        if not self._model_tracking_enabled():
            return
        get_global_registry().add_var(
            self, self.__dict__["var_name"], self.__dict__["var_type"]
        )

    def dump_trace(self, phase: str, dump_loc: str | None) -> None:
        if config.DISABLE_WRAPPER or not self._model_tracking_enabled():
            return

        # TODO
        var_name = self.__dict__["var_name"]
        # assert var_name is not None  # '' is allowed as a var_name (root object)
        # filter_by_tensor_version = proxy_config.dump_info_config[
        #     "filter_by_tensor_version"
        # ]
        # if filter_by_tensor_version and phase == "update":
        #     if hasattr(obj, "_version"):
        #         if obj._version == Proxy.var_dict[self.__dict__["var_name"]].version:
        #             return

        last_update_timestamp = self.__dict__["last_update_timestamp"]

        # TODO
        # if not isinstance(obj, torch.nn.Module):
        dump_trace_VAR(
            {
                "process_id": self.process_id,
                "thread_id": self.thread_id,
                "time": last_update_timestamp,
                "meta_vars": get_meta_vars(self),
                "var_name": var_name,
                "var_type": "torch.nn.Parameter",
                "mode": phase,
                "dump_loc": dump_loc,
                "attributes": dump_attributes(self, self),
                "type": TraceLineType.STATE_CHANGE,
            }
        )


def proxy_parameter(
    module: nn.Module,
    logdir: str = "proxy_log.log",
    log_level: int = logging.INFO,
    parent_name: str = "",
    should_dump_trace: bool = True,
    from_call: bool = False,
    from_iter: bool = False,
) -> None:
    """Track a module tree while preserving ordinary Parameter identity.

    Exact nn.Parameter objects are promoted in place: source references, aliases,
    autograd state, and existing optimizer references remain one object. Custom
    Parameter subclasses keep the legacy out-of-place conversion. Their aliases
    within one traversal share a proxy, but retained custom-source references stay
    separate; assigning a source again converts its current state anew.

    Observation uses the first registered live root's canonical parameter path.
    Attribute assignments (including None) reconcile paths; ordinary attributes
    do not rescan the tree. Direct register_parameter/register_module calls and
    delattr remain outside this setter hook. Collecting one root preserves other
    roots' ownership. Roots and modules are held weakly; retracking a root replaces
    its prior session.
    Distinct roots should use distinct parent_name namespaces. Callers must
    serialize tracking/assignment; concurrent model mutation is unsupported.
    """
    if not isinstance(module, nn.Module):
        raise TypeError("proxy_parameter expects a torch.nn.Module")
    if in_dynamo():
        return
    if MODEL_CLEANUP_KEY in module.__dict__:
        previous_cleanup: object = module.__dict__[MODEL_CLEANUP_KEY]
        if not callable(previous_cleanup):
            raise TypeError("TrainCheck model tracking cleanup must be callable")
        previous_cleanup()

    ownership = object()
    tracked: dict[int, weakref.ReferenceType[ProxyParameter]] = {}
    hooked_modules: dict[int, weakref.ReferenceType[nn.Module]] = {}

    def bindings_for(parameter: ProxyParameter) -> _ParameterBindings:
        if MODEL_BINDINGS_KEY not in parameter.__dict__:
            parameter.__dict__[MODEL_BINDINGS_KEY] = _ParameterBindings()
        bindings: object = parameter.__dict__[MODEL_BINDINGS_KEY]
        if not isinstance(bindings, _ParameterBindings):
            raise TypeError("ProxyParameter model bindings must be _ParameterBindings")
        return bindings

    def reconcile(parameters: dict[int, ProxyParameter]) -> None:
        registry = get_global_registry()
        with registry.registry_lock:
            names: list[tuple[ProxyParameter, str | None]] = []
            for parameter in parameters.values():
                old_name: object = parameter.__dict__["var_name"]
                if not isinstance(old_name, str):
                    raise TypeError("ProxyParameter observation name must be str")
                paths = bindings_for(parameter).paths
                new_name = next(iter(paths.values())) if paths else None
                if new_name is not None and not isinstance(new_name, str):
                    raise TypeError("ProxyParameter model binding path must be str")
                if old_name != new_name:
                    entry = registry.registry.get(old_name)
                    if entry is not None and entry.var is parameter:
                        del registry.registry[old_name]
                names.append((parameter, new_name))
            # Remove all departed paths before publishing replacements, including
            # an old tied parameter moving to its surviving alias's name.
            for parameter, name in names:
                if name is None:
                    continue
                parameter.__dict__["var_name"] = name
                entry = registry.registry.get(name)
                if entry is None or entry.var is not parameter:
                    registry.registry[name] = RegistryEntry(
                        parameter, name, "torch.nn.Parameter", stale=False
                    )

    def remove_hook(owner: nn.Module) -> None:
        if SUBCLASS_HOOK_KEY not in owner.__dict__:
            return
        hook: object = owner.__dict__[SUBCLASS_HOOK_KEY]
        if not isinstance(hook, _ModuleTrackingHooks):
            raise TypeError(
                "TrainCheck model tracking hook must be _ModuleTrackingHooks"
            )
        hook.callbacks.pop(ownership, None)
        if not hook.callbacks:
            del owner.__dict__[SUBCLASS_HOOK_KEY]

    def cleanup() -> None:
        affected: dict[int, ProxyParameter] = {}
        for identity, reference in tracked.items():
            parameter = reference()
            if parameter is not None:
                bindings_for(parameter).paths.pop(ownership, None)
                affected[identity] = parameter
        reconcile(affected)
        tracked.clear()
        for reference_to_module in hooked_modules.values():
            owner = reference_to_module()
            if owner is None:
                continue
            remove_hook(owner)
        hooked_modules.clear()
        root = root_reference()
        if root is not None and root.__dict__.get(MODEL_CLEANUP_KEY) is cleanup:
            del root.__dict__[MODEL_CLEANUP_KEY]

    def release_root(reference: weakref.ReferenceType[nn.Module]) -> None:
        cleanup()

    root_reference = weakref.ref(module, release_root)
    module.__dict__[MODEL_CLEANUP_KEY] = cleanup

    def convert_parameter(parameter: nn.Parameter, name: str) -> nn.Parameter:
        if not isinstance(parameter, nn.Parameter):
            raise TypeError(
                f"Registered parameter {name!r} is not a torch.nn.Parameter"
            )
        if isinstance(parameter, ProxyParameter) or is_fake_tensor(parameter):
            return parameter
        if type(parameter) is nn.Parameter:
            # A fresh Tensor wrapper splits source and proxy autograd metadata.
            # Promotion preserves the Python object, TensorImpl, and every slot.
            torch.Tensor.__setattr__(parameter, "__class__", ProxyParameter)
            assert isinstance(parameter, ProxyParameter)
            bindings_for(parameter).paths[ownership] = name
            ProxyParameter.__init__(
                parameter,
                parameter,
                logdir,
                log_level,
                name,
                "torch.nn.Parameter",
                should_dump_trace,
                from_call,
                from_iter,
            )
            return parameter
        # Preserve the established conversion for custom Parameter subclasses;
        # their class/layout/dispatch semantics cannot safely be promoted in place.
        proxied = ProxyParameter(
            parameter,
            logdir,
            log_level,
            name,
            typename(parameter, is_runtime=True),
            should_dump_trace,
            from_call,
            from_iter,
        )
        # Autograd hook dictionaries are slots, not __dict__ entries. Sharing them
        # preserves removal by handles registered before the legacy conversion.
        torch.Tensor.__setattr__(proxied, "_backward_hooks", parameter._backward_hooks)
        if hasattr(parameter, "_post_accumulate_grad_hooks"):
            torch.Tensor.__setattr__(
                proxied,
                "_post_accumulate_grad_hooks",
                parameter._post_accumulate_grad_hooks,
            )
        return proxied

    def on_assignment(owner: nn.Module, name: str) -> None:
        if in_dynamo() or (
            name not in owner._parameters and name not in owner._modules
        ):
            return
        root = root_reference()
        if root is not None:
            synchronize(root)

    def synchronize(root: nn.Module) -> None:
        observed: dict[int, tuple[ProxyParameter, str]] = {}
        # This cache is traversal-local, so custom-source state cannot become
        # stale between assignments. Strong references prevent identity reuse.
        converted: dict[int, tuple[nn.Parameter, nn.Parameter]] = {}
        visited: dict[int, weakref.ReferenceType[nn.Module]] = {}

        def visit(current: nn.Module, path: str) -> None:
            if not isinstance(current, nn.Module):
                raise TypeError(f"Registered child {path!r} is not a torch.nn.Module")
            if id(current) in visited:
                return
            visited[id(current)] = weakref.ref(current)
            for name, parameter in list(current._parameters.items()):
                if parameter is None:
                    continue
                identity = id(parameter)
                cached = converted.get(identity)
                if cached is None:
                    proxied = convert_parameter(parameter, path + "." + name)
                    converted[identity] = (parameter, proxied)
                else:
                    assert cached[0] is parameter
                    proxied = cached[1]
                current._parameters[name] = proxied
                if isinstance(proxied, ProxyParameter) and id(proxied) not in observed:
                    observed[id(proxied)] = (proxied, path + "." + name)
            for name, child in current._modules.items():
                if child is not None:
                    visit(child, path + "." + name)
            if SUBCLASS_HOOK_KEY not in current.__dict__:
                current.__dict__[SUBCLASS_HOOK_KEY] = _ModuleTrackingHooks()
            hook: object = current.__dict__[SUBCLASS_HOOK_KEY]
            if not isinstance(hook, _ModuleTrackingHooks):
                raise TypeError(
                    "TrainCheck model tracking hook must be _ModuleTrackingHooks"
                )
            hook.callbacks[ownership] = on_assignment

        visit(root, parent_name)
        affected: dict[int, ProxyParameter] = {}
        for identity, reference in tracked.items():
            parameter = reference()
            if parameter is not None:
                if identity not in observed:
                    bindings_for(parameter).paths.pop(ownership, None)
                affected[identity] = parameter
        for identity, (parameter, path) in observed.items():
            bindings_for(parameter).paths[ownership] = path
            affected[identity] = parameter
        reconcile(affected)
        for identity, reference_to_module in hooked_modules.items():
            if identity in visited:
                continue
            owner = reference_to_module()
            if owner is not None:
                remove_hook(owner)
        hooked_modules.clear()
        hooked_modules.update(visited)
        tracked.clear()
        tracked.update(
            (identity, weakref.ref(parameter))
            for identity, (parameter, _) in observed.items()
        )

    synchronize(module)


# instrument torch.nn.Module's setattr
# PyTorch 2.2 annotates only Tensor | Module, but its native setter also accepts
# arbitrary Python attributes (for example training=False and cached masks=None).
# This cast states that existing runtime contract; it does not coerce the value.
orig_setattr = cast(
    Callable[[nn.Module, str, object], None], torch.nn.Module.__setattr__
)


@functools.wraps(orig_setattr)
def wrapped_setattr(self: nn.Module, name: str, value: object) -> None:
    # Absence identifies an untracked module. A present malformed hook is a
    # broken tracking contract, not an instruction to silently disable tracking.
    if SUBCLASS_HOOK_KEY not in self.__dict__:
        orig_setattr(self, name, value)
        return
    hook: object = self.__dict__[SUBCLASS_HOOK_KEY]
    if not callable(hook):
        raise TypeError("TrainCheck module tracking hook must be callable")
    orig_setattr(self, name, value)
    hook(self, name)


torch.nn.Module.__setattr__ = wrapped_setattr
