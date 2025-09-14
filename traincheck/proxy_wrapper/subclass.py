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


def _is_fake_like(t: torch.Tensor) -> bool:
    try:
        from torch._subclasses.fake_tensor import FakeTensor

        if isinstance(t, FakeTensor):
            return True
    except Exception:
        pass
    if getattr(t, "fake_mode", None) is not None:
        return True
    if getattr(t, "_is_fake", False):
        return True
    ty = type(t)
    if "fake" in (
        getattr(ty, "__name__", "").lower() + getattr(ty, "__module__", "").lower()
    ):
        return True
    return False


class ProxyParameter(torch.nn.Parameter):
    def __new__(cls, data, var_name=""):
        if isinstance(data, ProxyParameter):
            return data

        return torch.Tensor._make_subclass(cls, data.detach(), data.requires_grad)

    def __init__(self, data, var_name=""):
        self.__dict__["varname"] = var_name
        print(f"init: {self.varname}")
        super().__init__()

    def __setattr__(self, name, value):
        print(f"paremeter: {self.varname}, name = {name}, value = {value}")

        return super().__setattr__(name, value)

    def __deepcopy__(self, memo):
        data = self.data
        return type(self)(
            data.clone(memory_format=torch.preserve_format),
            self.requires_grad,
            var_name=self.varname,
        )


def proxy_parameter(module: nn.Module, parent_name: str = ""):
    for name, t in list(module.named_parameters(recurse=False)):
        module._parameters[name] = ProxyParameter(t, parent_name + "." + name)
    for name, child in module.named_children():
        proxy_parameter(child, parent_name + "." + name)
