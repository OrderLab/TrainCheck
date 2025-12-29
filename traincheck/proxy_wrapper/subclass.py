import logging
import os
import threading

import torch
from torch import nn

from traincheck.config.config import should_disable_proxy_dumping
from traincheck.instrumentor.dumper import dump_trace_VAR
from traincheck.instrumentor.tracer import TraceLineType
from traincheck.proxy_wrapper.dumper import dump_attributes, get_meta_vars
from traincheck.utils import get_timestamp_ns

from .proxy_basics import is_fake_tensor

# from .proxy_registry import get_global_registry
# from .utils import print_debug


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
        # TODO
        # self.__dict__["old_value"] = None
        # self.__dict__["old_meta_vars"] = None

        current_time = get_timestamp_ns()

        self.__dict__["last_update_timestamp"] = current_time

        # print(f"init: {self.var_name}")
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
        # print(f"paremeter: {self.var_name}, name = {name}, value = {value}")
        super().__setattr__(name, value)
        self.update_timestamp()
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
        # get_global_registry().add_var(self, self.__dict__["var_name"])
        # TODO: implement the registry, we will need to make sure the registerred timestamp is updated and is consistent with the timestamp in the object
        pass

    def dump_trace(self, phase, dump_loc):
        # print(f"parameter: {self.var_name}, phase = {phase}, dump_loc = {dump_loc}")
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
    logdir="proxy_log.log",
    log_level=logging.INFO,
    # TODO
    # recurse=False,
    parent_name="",
    should_dump_trace=True,
    from_call=False,
    from_iter=False,
    # TODO
    # from_copy=False,
):
    if in_dynamo():
        return
    for name, t in list(module.named_parameters(recurse=False)):
        module._parameters[name] = ProxyParameter(
            t,
            logdir,
            log_level,
            parent_name + "." + name,
            should_dump_trace,
            from_call,
            from_iter,
        )
    for name, child in module.named_children():
        proxy_parameter(
            child,
            logdir,
            log_level,
            parent_name + "." + name,
            should_dump_trace,
            from_call,
            from_iter,
        )
