import copy
import logging
import os
import threading
import time
import types
from typing import Dict

import torch

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


class ProxyObjInfo:
    def __init__(self, var_name: str, last_update_timestamp: int, version: int | None):
        self.var_name = var_name
        self.last_update_timestamp = last_update_timestamp
        self.version = version

    @staticmethod
    def construct_from_proxy_obj(proxy_obj) -> "ProxyObjInfo":
        return ProxyObjInfo(
            proxy_obj.__dict__["var_name"],
            proxy_obj.__dict__["last_update_timestamp"],
            proxy_obj._obj._version if hasattr(proxy_obj._obj, "_version") else None,
        )

    def __repr__(self):
        return f"ProxyObjInfo(var_name={self.var_name}, last_update_timestamp={self.last_update_timestamp}, version={self.version})"


def proxy_handler(
    obj,
    logdir,
    log_level,
    var_name,
    should_dump_trace,
    from_call=False,
    from_iter=False,
):
    # if list or tuple, do the same thing for each element
    if isinstance(obj, (list, tuple)):
        for element in obj:
            element = proxy_handler(
                element,
                logdir,
                log_level,
                var_name,
                should_dump_trace=should_dump_trace,
                from_call=from_call,
                from_iter=from_iter,
            )
    if isinstance(obj, types.GeneratorType):

        def generator_proxy_handler():
            for element in obj:
                yield proxy_handler(
                    element,
                    logdir,
                    log_level,
                    var_name,
                    should_dump_trace=should_dump_trace,
                    from_call=from_call,
                    from_iter=from_iter,
                )

        obj = generator_proxy_handler()
    if typename(obj, is_runtime=True).startswith("torch.distributed"):
        return obj
    for obj_type in PROXY_SUPPORT_OBJ_TYPES:
        if issubclass(type(obj), obj_type):
            proxied_obj = Proxy(
                obj,
                logdir=logdir,
                log_level=log_level,
                var_name=var_name,
                should_dump_trace=should_dump_trace,
                from_call=from_call,
                from_iter=from_iter,
            )

            # Register the object
            proxied_obj.register_object()
            return proxied_obj

    # if the object is not in handled_obj_type, then return the object directly
    return obj


class Proxy:
    var_dict: Dict[str, ProxyObjInfo] = {}
    loglevel = logging.INFO
    jsondumper = dumper(
        os.path.join(os.getenv("ML_DAIKON_OUTPUT_DIR", "."), "proxy_log.json")  # type: ignore
    )

    @staticmethod
    def proxy_parameters(module: torch.nn.Module, parent_name="", from_iter=False):
        start_time = time.perf_counter()
        num_params = 0
        for name, parameter in module.named_parameters(recurse=False):
            num_params += 1
            parameter = Proxy(  # type: ignore
                parameter, var_name=parent_name + "." + name, from_iter=from_iter
            )
            module._parameters[name] = parameter

        time_end = time.perf_counter()
        if num_params != 0:
            print(
                "logger_proxy: "
                + f"Proxied {num_params} parameters of '{parent_name + module.__class__.__name__}', duration: {time_end - start_time} seconds"
            )

    def update_timestamp(self):
        # Update the timestamp of the object, should be called when the object is updated, e.g. __setattr__ and observer
        current_time = get_timestamp_ns()
        self.__dict__["last_update_timestamp"] = current_time
        Proxy.var_dict[self.__dict__["var_name"]].last_update_timestamp = current_time

    def register_object(self):
        # get_global_registry().add_var(self, self.__dict__["var_name"])
        # TODO: implement the registry, we will need to make sure the registerred timestamp is updated and is consistent with the timestamp in the object
        pass

    def __deepcopy__(self, memo):
        # Create a new instance of the proxy object
        if isinstance(self._obj, torch.Tensor):
            new_copy = type(self)(self._obj.clone().detach(), from_copy=True)

        else:
            new_copy = type(self)(copy.deepcopy(self._obj, memo), from_copy=True)

        # Copy other attributes if necessary
        new_copy.__dict__["var_name"] = self.__dict__["var_name"]
        # check every attribute in the object
        for attr_name, attr_value in self.__dict__.items():
            if attr_name in ["_obj", "var_name"]:
                continue
            if isinstance(attr_value, torch.Tensor):
                # setattr(new_copy, attr_name, attr_value.clone().detach())
                new_copy.__dict__[attr_name] = attr_value.clone().detach()
            else:
                # setattr(new_copy, attr_name, copy.deepcopy(attr_value, memo))
                new_copy.__dict__[attr_name] = copy.deepcopy(attr_value, memo)
        return new_copy

    def dump_trace(self, phase, dump_loc):
        obj = self._obj
        var_name = self.__dict__["var_name"]
        assert var_name is not None  # '' is allowed as a var_name (root object)
        filter_by_tensor_version = proxy_config.dump_info_config[
            "filter_by_tensor_version"
        ]
        if filter_by_tensor_version and phase == "update":
            if hasattr(obj, "_version"):
                if obj._version == Proxy.var_dict[self.__dict__["var_name"]].version:
                    return

        last_update_timestamp = self.__dict__["last_update_timestamp"]

        if not isinstance(obj, torch.nn.Module):
            self.jsondumper.dump_json(
                process_id=self.process_id,
                thread_id=self.thread_id,
                time=last_update_timestamp,
                meta_vars=get_meta_vars(self),
                var_name=var_name,
                var_type=typename(obj, is_runtime=True),
                change_type=phase,
                var_attributes=dump_attributes(self, obj),
                dump_loc=dump_loc,
            )

    def __init__(
        self,
        obj,
        logdir="proxy_log.log",
        log_level=logging.INFO,
        recurse=False,
        var_name="",
        should_dump_trace=True,
        from_call=False,
        from_iter=False,
        from_copy=False,
    ):
        if from_copy:
            self.__dict__["_obj"] = obj
            return
        # Access proxy attribute: since we are wrapping the getattr method, we need to access the attribute directly
        self.__dict__["process_id"] = os.getpid()
        self.__dict__["thread_id"] = threading.current_thread().ident
        self.__dict__["logdir"] = logdir
        self.__dict__["log_level"] = log_level
        self.__dict__["meta_vars"] = {}
        self.__dict__["is_traincheck_proxied_obj"] = True
        self.__dict__["recurse"] = recurse
        self.__dict__["var_name"] = var_name
        self.__dict__["old_value"] = None
        self.__dict__["old_meta_vars"] = None

        if type(obj) is Proxy:
            print_debug(
                lambda: "logger_proxy: "
                + f"Object '{obj.__class__.__name__}' is already a proxy"
            )

            # create a shallow copy of the object
            self._obj = obj._obj
            self.__dict__["last_update_timestamp"] = obj.__dict__[
                "last_update_timestamp"
            ]
            self.__dict__["is_traincheck_proxied_obj"] = obj.__dict__[
                "is_traincheck_proxied_obj"
            ]
            self.__dict__["recurse"] = obj.__dict__["recurse"]
            self.__dict__["var_name"] = obj.__dict__["var_name"]
            self.__dict__["logdir"] = obj.__dict__["logdir"]
            self.__dict__["log_level"] = obj.__dict__["log_level"]
            self.__dict__["meta_vars"] = obj.__dict__["meta_vars"]
            self.__dict__["old_value"] = obj.__dict__["old_value"]
            self.__dict__["old_meta_vars"] = obj.__dict__["old_meta_vars"]
            return

        if isinstance(obj, torch.nn.Module):  # special handling for nn.Module
            if self.__dict__["recurse"]:
                # proxy all of its parameters
                assert not from_call
                assert not from_iter

                self.proxy_parameters(obj, parent_name=var_name)
                for name, module in obj.named_children():
                    proxy_module = Proxy(
                        module, var_name=var_name + "." + name, recurse=True
                    )
                    obj._modules[name] = proxy_module
            else:
                self.proxy_parameters(obj, parent_name=var_name, from_iter=from_iter)

        current_time = get_timestamp_ns()
        self.__dict__["_obj"] = obj

        self.__dict__["last_update_timestamp"] = current_time
        Proxy.var_dict[var_name] = ProxyObjInfo.construct_from_proxy_obj(self)

        dump_call_return = proxy_config.dump_info_config["dump_call_return"]
        dump_iter = proxy_config.dump_info_config["dump_iter"]
        if not dump_call_return and from_call:
            return
        if not dump_iter and from_iter:
            return

        if should_dump_trace:
            if from_call:
                phase = "call"

            if from_iter:
                phase = "iter"
            # if the object is generated from getattr, then do not dump it
            else:
                phase = "update"
            self.dump_trace(phase=phase, dump_loc="initing")

    @property  # type: ignore
    def __class__(self):  # type: ignore[misc]
        return self._obj.__class__

    def __call__(self, *args, **kwargs):
        print_debug(
            lambda: f"logger_proxy: Calling '{self.__class__.__name__}' for obj: '{self.__dict__['var_name']}' (type '{typename(self._obj, is_runtime=True)}')"
        )
        result = self._obj(*args, **kwargs)
        print_debug(
            lambda: f"logger_proxy: Result type of __call__ is '{typename(result, is_runtime=True)}'"
        )

        # deprecated, since we only want to keep track of the model itself, we can for-feit coverage of function invocation results
        # disable dumping for function call results as return values are dumped through API instrumentation
        # return proxy_handler(
        #     result,
        #     self.logdir,
        #     self.log_level,
        #     self.__dict__["var_name"] + "_call_result",
        #     should_dump_trace=False,
        #     from_call=True,
        # )
        return result

    def __getattr__(self, name):
        print_debug(lambda: f"logger_proxy: Accessing attribute '{name}'")

        if name == "logdir":
            return self.__dict__.get("logdir", None)  # in order to pass down the dir
        if name == "_obj":
            return self.__dict__.get("_obj", None)  # in order to pass down the dir
        if name == "__torch_function__":
            return Proxy.__torch_function__
        attr = getattr(self._obj, name)

        if isinstance(self._obj, torch.Tensor) and isinstance(attr, torch.Tensor):
            # we should not proxy sub-tensor fields for a tensor, this can cause circular reference and memory leak
            # one caveat with this is that if the code wants to operate on the sub-tensor separately, we will lose track of their updates when they are not proxied
            return attr

        if self.__dict__["var_name"] == "":
            var_name = name
        else:
            var_name = self.__dict__["var_name"] + "." + name
        return proxy_handler(
            attr, self.logdir, self.log_level, var_name, should_dump_trace=True
        )

    def __setattr__(self, name, value):

        if name == "_obj":
            self.__dict__[name] = value  # Set the attribute directly
        else:
            var_name = self.__dict__["var_name"]
            assert (
                var_name in Proxy.var_dict
            ), f"var_name {var_name} is not in var_dict, it has not been proxied yet, check Proxy.__init__() for existence of assignment into Proxy.var_dict"
            print_debug(
                lambda: f"Time elapse: {get_timestamp_ns() - self.__dict__['last_update_timestamp']}"
            )
            self.update_timestamp()

            # register the current object into the registry, set `stale` to False
            self.register_object()

            if self.__dict__["var_name"] == "":
                global_name = name
            else:
                global_name = self.__dict__["var_name"] + "." + name

            print_debug(
                lambda: f"Setting attribute '{name}' to '{value}', with global name '{global_name}'"
            )

            # if self._obj is a tensor already, then deproxify the value
            if issubclass(type(self._obj), torch.Tensor):
                setattr(self._obj, name, unproxy_arg(value))
            else:
                setattr(
                    self._obj,
                    name,
                    proxy_handler(
                        value,
                        logdir=self.logdir,
                        log_level=self.log_level,
                        var_name=global_name,
                        should_dump_trace=False,
                    ),
                )

            if general_config.should_disable_proxy_dumping():
                # do not dump update traces
                return None

            self.dump_trace(
                phase="update",
                dump_loc=f"__setattr__ (attribute '{name}')",
            )

    def __getitem__(self, key):
        # Intercept item retrieval
        print_debug(
            lambda: f"logger_proxy: Getting item with key '{key}' for object '{self.__class__.__name__}'"
        )
        return Proxy(self._obj[key])

    def __iter__(self):
        print_debug(
            lambda: f"logger_proxy: Calling __iter__ for object '{self.__class__.__name__}'"
        )
        for element in self._obj:
            yield proxy_handler(
                element,
                logdir=self.logdir,
                log_level=self.log_level,
                var_name=self.__dict__["var_name"],
                should_dump_trace=False,
                from_iter=True,
            )

    __add__ = proxy_methods.__add__
    __array__ = proxy_methods.__array__
    __bool__ = proxy_methods.__bool__
    __delattr__ = proxy_methods.__delattr__
    __delitem__ = proxy_methods.__delitem__
    __dir__ = proxy_methods.__dir__
    __float__ = proxy_methods.__float__
    __floatdiv__ = proxy_methods.__floatdiv__
    __format__ = proxy_methods.__format__
    __getreal__ = proxy_methods.__getreal__
    __iadd__ = proxy_methods.__iadd__
    __int__ = proxy_methods.__int__
    __intdiv__ = proxy_methods.__intdiv__
    __ior__ = proxy_methods.__ior__
    __len__ = proxy_methods.__len__
    __mul__ = proxy_methods.__mul__
    __or__ = proxy_methods.__or__
    __radd__ = proxy_methods.__radd__
    __repr__ = proxy_methods.__repr__
    __rfloordiv__ = proxy_methods.__rfloordiv__
    __rmul__ = proxy_methods.__rmul__
    __ror__ = proxy_methods.__ror__
    __setitem__ = proxy_methods.__setitem__
    __str__ = proxy_methods.__str__
    __sub__ = proxy_methods.__sub__
    __truediv__ = proxy_methods.__truediv__

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        # 🚨 Ensure Proxy does not interfere with PyTorch dispatch
        if kwargs is None:
            kwargs = {}

        real_args, real_kwargs = unproxy_args_kwargs(args, kwargs)
        return func(*real_args, **real_kwargs)
