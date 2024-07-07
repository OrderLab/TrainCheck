import inspect
import json
import json.encoder
import linecache
import logging
import os
import threading
import time
import types
import typing
from typing import Dict

import torch
import torch.nn.parameter

import mldaikon.proxy_wrapper.proxy_methods as proxy_methods
from mldaikon.proxy_wrapper.config import (
    debug_mode,
    dump_call_return,
    dump_iter,
    exclude_file_names,
    filter_by_tensor_version,
    proxy_log_dir,
    proxy_update_limit,
)
from mldaikon.proxy_wrapper.dumper import dump_attributes, dump_meta_vars
from mldaikon.proxy_wrapper.dumper import json_dumper as dumper
from mldaikon.proxy_wrapper.dumper import torch_serialize
from mldaikon.proxy_wrapper.proxy_basics import unproxy_arg
from mldaikon.proxy_wrapper.proxy_handler import handled_obj_type
from mldaikon.proxy_wrapper.utils import print_debug
from mldaikon.utils import typename


def get_line(filename, lineno):
    return linecache.getline(filename, lineno).strip()


def proxy_handler(
    obj,
    logdir,
    log_level,
    var_name,
    no_init_dump=False,
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
                no_init_dump=no_init_dump,
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
                    no_init_dump=no_init_dump,
                    from_call=from_call,
                    from_iter=from_iter,
                )

        obj = generator_proxy_handler()
    if typename(obj).startswith("torch.distributed"):
        return obj
    for obj_type in handled_obj_type:
        if issubclass(type(obj), obj_type):
            return Proxy(
                obj,
                logdir=logdir,
                log_level=log_level,
                var_name=var_name,
                dump_trace_info=not no_init_dump,
                from_call=from_call,
                from_iter=from_iter,
            )
    return obj


class Proxy:
    var_dict: Dict[str, typing.Any] = {}
    logger_proxy = logging.getLogger("proxy")
    logdir = "proxy_logs.log"
    loglevel = logging.INFO
    jsondumper = dumper(proxy_log_dir)
    handler = logging.FileHandler(logdir)
    handler.setLevel(loglevel)
    logger_proxy.handlers.clear()
    logger_proxy.addHandler(handler)

    empty_name_counts = 0
    non_empty_name_counts = 0

    @staticmethod
    def print_tensor(value):
        if debug_mode:
            print_debug("logger_proxy: " + f"Tensor with shape'{value.shape}'")
            print_debug("logger_proxy: " + f"Minimum value: {torch.min(value)}")
            print_debug("logger_proxy: " + f"Maximum value: {torch.max(value)}")

    @staticmethod
    def print_update(old_value, value, attr_name=None):
        if debug_mode:
            print_debug("logger_proxy: " + f"Updating the attribute '{attr_name}'")
            print_debug("logger_proxy: " + "From:")
            if type(old_value) is torch.Tensor:
                Proxy.print_tensor(old_value)
            else:
                print_debug("logger_proxy: " + f"'{old_value}'")

            print_debug("logger_proxy: " + "To:")
            if type(value) is torch.Tensor:
                Proxy.print_tensor(value)
            else:
                print_debug("logger_proxy: " + f"'{value}'")

    @staticmethod
    def proxy_parameters(module, parent_name="", from_iter=False):
        print(
            "logger_proxy: "
            + f"Proxying all parameters of '{parent_name + module.__class__.__name__}'"
        )
        for name, parameter in module.named_parameters():
            print("logger_proxy: " + f"Proxying parameter '{parent_name+name}'")
            parameter = Proxy(
                parameter, var_name=parent_name + name, from_iter=from_iter
            )
            module._parameters[name] = parameter

    @staticmethod
    def get_frame_array(frame):
        frame_array = []
        while frame:
            if frame.f_code.co_filename in exclude_file_names:
                frame = frame.f_back
                continue

            # fetch the frame info
            frame_array.append(
                (
                    frame.f_code.co_filename,
                    frame.f_lineno,
                    get_line(frame.f_code.co_filename, frame.f_lineno),
                )
            )
            frame = frame.f_back
        return frame_array

    def dump_trace(self, status):
        if Proxy.var_dict.get(self.__dict__["var_name"]) is None:
            # create
            self.__dict__["last_update_timestamp"] = 0
            Proxy.var_dict[self.__dict__["var_name"]] = self

        if (
            time.time()
            - Proxy.var_dict[self.__dict__["var_name"]].__dict__[
                "last_update_timestamp"
            ]
            > proxy_update_limit
        ):
            frame = inspect.currentframe()
            frame_array = self.get_frame_array(frame)
            dumped_frame_array = json.dumps(frame_array)
            self.dump_to_trace(self._obj, status, dumped_frame_array)
            self.__dict__["last_update_timestamp"] = time.time()

    def dump_to_trace(self, obj, status="update", dumped_frame_array=None):
        # version based filtering
        var_name = self.__dict__["var_name"]
        assert (
            var_name == self.__dict__["dumped_varname_list"]
        ), f"var_name {var_name} is not consistent with dumped_varname_list {self.__dict__['dumped_varname_list']}"
        assert var_name is not None  # '' is allowed as a var_name (root object)
        if filter_by_tensor_version and status == "update":
            if hasattr(obj, "_version"):
                if (
                    obj._version
                    == Proxy.var_dict[self.__dict__["var_name"]]._obj._version
                ):
                    return

        if not issubclass(type(obj), torch.nn.Module):
            dumped_val = str(torch_serialize(obj))
            self.jsondumper.dump_json(
                self.process_id,
                self.thread_id,
                dump_meta_vars(self, proxy_file_path=__file__),
                self.__dict__["dumped_varname_list"],
                type(obj).__name__,
                dumped_val,
                status,
                dump_attributes(self, obj),
                dumped_frame_array,
            )

    def __init__(
        self,
        obj,
        logdir="proxy_log.log",
        log_level=logging.INFO,
        is_root=False,
        var_name="",
        dump_trace_info=True,
        from_call=False,
        from_iter=False,
    ):
        # Access proxy attribute: since we are wrapping the getattr method, we need to access the attribute directly
        self.__dict__["process_id"] = os.getpid()
        self.__dict__["thread_id"] = threading.current_thread().ident
        self.__dict__["logdir"] = logdir
        self.__dict__["log_level"] = log_level
        self.__dict__["meta_vars"] = {}
        self.__dict__["last_update_timestamp"] = 0
        self.__dict__["is_ml_daikon_proxied_obj"] = True
        self.__dict__["is_root"] = is_root
        self.__dict__["var_name"] = var_name
        self.__dict__["old_value"] = None
        self.__dict__["old_meta_vars"] = None

        if type(obj) is Proxy:
            print_debug(
                "logger_proxy: "
                + f"Object '{obj.__class__.__name__}' is already a proxy"
            )

            # create a shallow copy of the object
            self._obj = obj._obj
            self.__dict__["dumped_varname_list"] = obj.__dict__["dumped_varname_list"]
            self.__dict__["last_update_timestamp"] = obj.__dict__[
                "last_update_timestamp"
            ]
            self.__dict__["is_ml_daikon_proxied_obj"] = obj.__dict__[
                "is_ml_daikon_proxied_obj"
            ]
            self.__dict__["is_root"] = obj.__dict__["is_root"]
            self.__dict__["var_name"] = obj.__dict__["var_name"]
            self.__dict__["logdir"] = obj.__dict__["logdir"]
            self.__dict__["log_level"] = obj.__dict__["log_level"]
            self.__dict__["meta_vars"] = obj.__dict__["meta_vars"]
            self.__dict__["old_value"] = obj.__dict__["old_value"]
            self.__dict__["old_meta_vars"] = obj.__dict__["old_meta_vars"]
            return

        frame = inspect.currentframe()
        frame_array = self.get_frame_array(frame)
        dumped_frame_array = json.dumps(frame_array)

        # print_debug the variable name list of the object
        print_debug("logger_proxy: " + f"Empty name counts: {Proxy.empty_name_counts}")
        print_debug(
            "logger_proxy: " + f"Non-empty name counts: {Proxy.non_empty_name_counts}"
        )
        # inherit the var_name from the parent object
        if self.__dict__["var_name"] is not None:
            current_var_name_list = self.__dict__["var_name"]
        else:
            current_var_name_list = ""
        self.__dict__["dumped_varname_list"] = current_var_name_list

        if self.__dict__["is_root"]:
            print(
                "logger_proxy: " + f"ROOT proxy object for '{obj.__class__.__name__}'"
            )
        # Ziming: here we still seperate the handling of tensor and other objects
        # however, despite the dumping logic these two are identical and could be merged

        if isinstance(obj, torch.nn.Module):  # special handling for nn.Module

            if self.__dict__["is_root"]:
                # proxy all of its parameters
                assert not from_call
                assert not from_iter

                self.proxy_parameters(obj)
                for name, module in obj.named_children():
                    # setattr(obj, name, Proxy(module, var_name=name))
                    proxy_module = Proxy(module, var_name=name)

                    obj._modules[name] = proxy_module
                    self.proxy_parameters(proxy_module, name + ".")
                    # TODO: improve the nn.Module tracing logic, currently we only trace two levels of submodules
                    # We could try to enforce a blacklist of modules that we can't trace (contain low level functions)
                    if isinstance(type(module), torch.nn.Module):
                        for subname, submodule in module.named_children():
                            proxy_submodule = Proxy(
                                submodule, var_name=name + "." + subname
                            )
                            self.proxy_parameters(
                                proxy_submodule, name + "." + subname + "."
                            )
                            module._modules[subname] = proxy_submodule

            else:
                if current_var_name_list == "":
                    self.proxy_parameters(obj, from_iter=from_iter)
                else:
                    self.proxy_parameters(
                        obj, current_var_name_list + ".", from_iter=from_iter
                    )

        current_var_name_list = current_var_name_list
        if (
            Proxy.var_dict.get(current_var_name_list) is None
        ):  # if the object is not proxied yet

            self.__dict__["_obj"] = obj
            if not dump_call_return and from_call:
                return

            if not dump_iter and from_iter:
                return

            if dump_trace_info:
                if from_call:
                    self.dump_to_trace(obj, "call", dumped_frame_array)
                if from_iter:
                    self.dump_to_trace(obj, "iter", dumped_frame_array)
                # if the object is generated from getattr, then do not dump it
                else:
                    self.dump_to_trace(obj, "new", dumped_frame_array)
                self.__dict__["last_update_timestamp"] = time.time()
                Proxy.var_dict[current_var_name_list] = self

        else:  # if the object is proxied already
            if type(obj) not in [int, float, str, bool] and obj is not None:
                print_debug(
                    "logger_proxy: "
                    + f"Object '{obj.__class__.__name__}' is already proxied"
                )

            print_debug(
                f'Time elapse: {time.time() - Proxy.var_dict[current_var_name_list].__dict__["last_update_timestamp"]}'
            )
            self.__dict__["_obj"] = obj
            if (
                time.time()
                - Proxy.var_dict[current_var_name_list].__dict__[
                    "last_update_timestamp"
                ]
                < proxy_update_limit
            ):
                return

            if not dump_call_return and from_call:
                return

            if not dump_iter and from_iter:
                return

            if from_call:
                self.dump_to_trace(obj, "call", dumped_frame_array)
            elif from_iter:
                self.dump_to_trace(obj, "iter", dumped_frame_array)
            elif dump_trace_info:
                self.dump_to_trace(obj, "update", dumped_frame_array)

            del Proxy.var_dict[current_var_name_list]
            self.__dict__["last_update_timestamp"] = time.time()
            Proxy.var_dict[current_var_name_list] = self

    @property  # type: ignore
    def __class__(self):  # type: ignore[misc]
        return self._obj.__class__

    def __call__(self, *args, **kwargs):
        print_debug(
            "logger_proxy: " + f"Go to __call__ for object '{self.__class__.__name__}'"
        )
        print_debug(
            "logger_proxy: "
            + f"Calling '{self.__class__.__name__}' for obj: '{self.__dict__['var_name']}' (type '{typename(self._obj)}')"
        )
        result = self._obj(*args, **kwargs)
        print_debug(
            "logger_proxy: " + f"Result type of __call__ is '{typename(result)}'"
        )

        return proxy_handler(
            result,
            self.logdir,
            self.log_level,
            self.__dict__["var_name"],
            from_call=True,
        )

    def __getattr__(self, name):
        print_debug("logger_proxy: " + f"Accessing attribute '{name}'")
        if name == "logdir":
            return self.__dict__.get("logdir", None)  # in order to pass down the dir
        if name == "_obj":
            return self.__dict__.get("_obj", None)  # in order to pass down the dir
        attr = getattr(self._obj, name)

        if self.__dict__["var_name"] == "":
            var_name = name
        else:
            var_name = self.__dict__["var_name"] + "." + name
        return proxy_handler(
            attr, self.logdir, self.log_level, var_name, no_init_dump=True
        )

    def __setattr__(self, name, value):

        if name == "_obj":
            self.__dict__[name] = value  # Set the attribute directly
        else:
            if Proxy.var_dict.get(self.__dict__["var_name"]) is None:
                self.__dict__["last_update_timestamp"] = 0
                Proxy.var_dict[self.__dict__["var_name"]] = self

            print_debug(
                f"Time elapse: {time.time() - Proxy.var_dict[self.__dict__['var_name']].__dict__['last_update_timestamp']}"
            )

            if self.__dict__["var_name"] == "":
                var_name = name
            else:
                var_name = self.__dict__["var_name"] + "." + name

            print_debug("<logger_proxy: " + f"Setting attribute '{name}' to '{value}'")

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
                        var_name=var_name,
                    ),
                )
            # dump frame array
            self.dump_trace("update")

    def __getitem__(self, key):
        # Intercept item retrieval
        print_debug("logger_proxy: " + f"Getting item with key '{key}'")
        return Proxy(self._obj[key])

    def __iter__(self):
        print_debug("logger_proxy: " + "Calling __iter__")
        for element in self._obj:
            yield proxy_handler(
                element,
                logdir=self.logdir,
                log_level=self.log_level,
                var_name=self.__dict__["var_name"],
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

    # max = proxy_methods.max
    # min = proxy_methods.min
    # size = proxy_methods.size

    def print_proxy_dict(self, proxy_dict):
        # for debugging purpose: print the var_dict of the proxy object
        print_debug("logger_proxy: " + "Dump Proxy Dict: ")
        for k, value in proxy_dict.items():
            if isinstance(value, torch.Tensor):
                self.print_tensor(value)
            else:
                print_debug("logger_proxy: " + f"{k}: {value}")
