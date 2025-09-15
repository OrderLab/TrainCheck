import copy
import logging
import os
import threading
import time
import types
from typing import Dict

import torch
from torch import nn

import traincheck.config.config as general_config
import traincheck.proxy_wrapper.proxy_config as proxy_config  # HACK: cannot directly import config variables as then they would be local variables
import traincheck.proxy_wrapper.proxy_methods as proxy_methods
from traincheck.proxy_wrapper.dumper import dump_attributes, get_meta_vars
from traincheck.utils import get_timestamp_ns, typename

from .dumper import json_dumper as dumper
from .proxy_basics import unproxy_arg, unproxy_args_kwargs
from .proxy_handler import PROXY_SUPPORT_OBJ_TYPES

# from .proxy_registry import get_global_registry
from .utils import print_debug


def in_dynamo() -> bool:
    try:
        import torch._dynamo as dynamo

        return bool(dynamo.is_compiling())
    except Exception:
        return False


def is_fake_tensor(x: torch.Tensor) -> bool:
    try:
        from torch._subclasses.fake_tensor import FakeTensor  # 2.x

        if isinstance(x, FakeTensor):
            return True
    except Exception:
        pass
    if getattr(x, "fake_mode", None) is not None:
        return True
    if getattr(x, "_is_fake", False):
        return True

    return isinstance(x, torch.Tensor) and x.device.type == "meta"


class ProxyParameter(torch.nn.Parameter):
    loglevel = logging.INFO
    jsondumper = dumper(
        os.path.join(os.getenv("ML_DAIKON_OUTPUT_DIR", "."), "proxy_log.json")  # type: ignore
    )

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
        if in_dynamo() or is_fake_tensor(data):
            if isinstance(data, nn.Parameter):
                return data
            return nn.Parameter(data, requires_grad=data.requires_grad)

        if isinstance(data, ProxyParameter):
            return data

        return torch.Tensor._make_subclass(cls, data.detach(), data.requires_grad)

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
        # TODO
        # self.__dict__["recurse"] = recurse
        self.__dict__["var_name"] = var_name
        # TODO
        # self.__dict__["old_value"] = None
        # self.__dict__["old_meta_vars"] = None

        current_time = get_timestamp_ns()

        self.__dict__["last_update_timestamp"] = current_time

        print(f"init: {self.var_name}")
        if should_dump_trace:
            if from_call:
                phase = "call"

            if from_iter:
                phase = "iter"
            # if the object is generated from getattr, then do not dump it
            else:
                phase = "update"
            self.dump_trace(phase=phase, dump_loc="initing")

    def __setattr__(self, name, value):
        print(f"paremeter: {self.var_name}, name = {name}, value = {value}")
        self.dump_trace(
            phase="update",
            dump_loc=f"__setattr__ (attribute '{name}')",
        )
        return super().__setattr__(name, value)

    def __deepcopy__(self, memo):
        data = self.data
        if in_dynamo() or is_fake_tensor(self):
            return self
        return type(self)(
            data.clone(memory_format=torch.preserve_format),
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
        print(f"parameter: {self.var_name}, phase = {phase}, dump_loc = {dump_loc}")
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
        self.jsondumper.dump_json(
            process_id=self.process_id,
            thread_id=self.thread_id,
            time=last_update_timestamp,
            meta_vars=get_meta_vars(self),
            var_name=var_name,
            # TODO
            var_type="torch.nn.Parameter",
            change_type=phase,
            # TODO: verify dump_attributes
            var_attributes=dump_attributes(self, self.data),
            dump_loc=dump_loc,
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
