import functools
import logging
import os
import threading
import weakref
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
from .proxy_registry import get_global_registry

SUBCLASS_HOOK_KEY = "_tc_setattr_hook"

logger = logging.getLogger(__name__)


def in_dynamo() -> bool:
    try:
        import torch._dynamo as dynamo

        return bool(dynamo.is_compiling())
    except Exception:
        return False


class ProxyParameter(torch.nn.Parameter):
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

    def register_object(self):
        get_global_registry().add_var(
            self, self.__dict__["var_name"], self.__dict__["var_type"]
        )

    def dump_trace(self, phase, dump_loc):
        if config.DISABLE_WRAPPER:
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
    """Track a module's parameters and subsequent assignments to its module tree.

    Aliases within this tracking session share one proxy and its first registered
    name. Weak identity entries avoid retaining replaced source parameters or
    comparing Tensor values. Separate calls do not share this cache. Callers must
    serialize model tracking and assignment; concurrent model mutation is unsupported.
    """
    if not isinstance(module, nn.Module):
        raise TypeError("proxy_parameter expects a torch.nn.Module")
    if in_dynamo():
        return

    aliases: dict[
        int,
        tuple[weakref.ReferenceType[nn.Parameter], weakref.ReferenceType[nn.Parameter]],
    ] = {}

    def track_parameter(parameter: nn.Parameter, name: str) -> nn.Parameter:
        if not isinstance(parameter, nn.Parameter):
            raise TypeError(
                f"Registered parameter {name!r} is not a torch.nn.Parameter"
            )
        # Calling ProxyParameter again reruns __init__, overwriting its canonical
        # name and emitting another initialization record even for the same object.
        if isinstance(parameter, ProxyParameter):
            return parameter
        identity = id(parameter)
        cached = aliases.get(identity)  # Absence means this source has no live proxy.
        if cached is not None and cached[0]() is parameter:
            existing = cached[1]()
            if existing is not None:
                return existing
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

        # Tensor autograd hook dictionaries are slots, not entries in __dict__.
        # Share them rather than copying so existing removable handles still work.
        torch.Tensor.__setattr__(proxied, "_backward_hooks", parameter._backward_hooks)
        # Post-accumulation hooks are absent on older supported PyTorch versions.
        if hasattr(parameter, "_post_accumulate_grad_hooks"):
            torch.Tensor.__setattr__(
                proxied,
                "_post_accumulate_grad_hooks",
                parameter._post_accumulate_grad_hooks,
            )

        def forget(reference: weakref.ReferenceType[nn.Parameter]) -> None:
            current = aliases.get(identity)
            if current is not None and (
                current[0] is reference or current[1] is reference
            ):
                del aliases[identity]

        aliases[identity] = (
            weakref.ref(parameter, forget),
            weakref.ref(proxied, forget),
        )
        return proxied

    def track_module(current: nn.Module, path: str, visited: set[int]) -> None:
        if not isinstance(current, nn.Module):
            raise TypeError(f"Registered child {path!r} is not a torch.nn.Module")
        if id(current) in visited:
            return
        visited.add(id(current))
        # Include aliases rather than named_parameters' default deduplication.
        for name, parameter in list(current._parameters.items()):
            if parameter is not None:
                current._parameters[name] = track_parameter(
                    parameter, path + "." + name
                )
        for name, child in current._modules.items():
            if child is not None:
                track_module(child, path + "." + name, visited)

        def subclass_setattr_hook(owner: nn.Module, name: str) -> None:
            if in_dynamo():
                return
            # The native setter has already validated and registered the value,
            # including any replacement made by PyTorch registration hooks. Buffer
            # and ordinary Tensor assignments must never become parameters.
            if name in owner._parameters:
                parameter = owner._parameters[name]
                if parameter is not None:
                    owner._parameters[name] = track_parameter(
                        parameter, path + "." + name
                    )
            elif name in owner._modules:
                child = owner._modules[name]
                if child is not None:
                    track_module(child, path + "." + name, set())

        current.__dict__[SUBCLASS_HOOK_KEY] = subclass_setattr_hook

    track_module(module, parent_name, set())


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
