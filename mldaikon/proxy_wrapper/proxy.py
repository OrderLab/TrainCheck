import copy
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

import mldaikon.config.config as general_config
import mldaikon.proxy_wrapper.proxy_config as proxy_config  # HACK: cannot directly import config variables as then they would be local variables
import mldaikon.proxy_wrapper.proxy_methods as proxy_methods
from mldaikon.instrumentor.tracer import should_dump_trace
from mldaikon.proxy_wrapper.dumper import (
    SkippedDumpingObj,
    dump_attributes,
    get_meta_vars,
)
from mldaikon.proxy_wrapper.dumper import json_dumper as dumper
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
    loglevel = logging.INFO
    jsondumper = dumper(
        os.path.join(os.getenv("ML_DAIKON_OUTPUT_DIR"), "proxy_log.json")  # type: ignore
    )

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
            if "mldaikon" in frame.f_code.co_filename:
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

    def dump_trace(
        self,
        status,
        only_record=False,
        prev_obj=None,
        prev_trace_info=None,
        disable_sampling=False,
    ):
        if not should_dump_trace(
            general_config.ENABLE_COND_DUMP,
            None,
            f"VAR: {typename(self._obj)}: {self.__dict__['var_name']}",
            None,
            None,
        ):
            # skip dumping
            return SkippedDumpingObj(self._obj)

        if Proxy.var_dict.get(self.__dict__["var_name"]) is None:
            # create
            self.__dict__["last_update_timestamp"] = 0
            Proxy.var_dict[self.__dict__["var_name"]] = self

        if (
            time.time()
            - Proxy.var_dict[self.__dict__["var_name"]].__dict__[
                "last_update_timestamp"
            ]
            > proxy_config.proxy_update_limit
            or disable_sampling
        ):
            dump_pre_and_post_trace = False
            if (
                only_record
                and status == "post_observe"
                and not isinstance(prev_obj, SkippedDumpingObj)
            ):
                assert (
                    prev_obj is not None and prev_trace_info is not None
                ), "prev_obj and prev_trace_info should not be None"
                # only dump when the object is changed

                if isinstance(prev_obj._obj, torch.Tensor) and isinstance(
                    self._obj, torch.Tensor
                ):
                    if not torch.equal(prev_obj._obj, self._obj):
                        dump_pre_and_post_trace = True
                else:
                    if prev_obj._obj != self._obj:
                        dump_pre_and_post_trace = True

                if not dump_pre_and_post_trace:
                    return None
                else:
                    current_time = time.time()
                    self.__dict__["last_update_timestamp"] = current_time
                    self.dump_to_trace(prev_obj, prev_trace_info)

            # record the trace info
            frame = inspect.currentframe()
            frame_array = self.get_frame_array(frame)
            dumped_frame_array = json.dumps(frame_array)
            current_time = time.time()
            trace_info = {
                "time": current_time,
                "status": status,
                "frame_array": dumped_frame_array,
            }

            if only_record and status == "pre_observe":
                return trace_info

            self.dump_to_trace(self._obj, trace_info)
            return None
        else:
            return SkippedDumpingObj(self._obj)

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

    def dump_to_trace(self, obj, trace_info):
        if isinstance(trace_info, SkippedDumpingObj):
            return
        # version based filtering
        if "time" in trace_info:
            current_time = trace_info["time"]
        else:
            current_time = time.time()
        if "status" in trace_info:
            status = trace_info["status"]
        else:
            status = "update"
        if "frame_array" in trace_info:
            dumped_frame_array = trace_info["frame_array"]
        else:
            raise ValueError("frame_array is not provided in trace_info")

        var_name = self.__dict__["var_name"]
        assert (
            var_name == self.__dict__["dumped_varname_list"]
        ), f"var_name {var_name} is not consistent with dumped_varname_list {self.__dict__['dumped_varname_list']}"
        assert var_name is not None  # '' is allowed as a var_name (root object)
        filter_by_tensor_version = proxy_config.dump_info_config[
            "filter_by_tensor_version"
        ]
        if filter_by_tensor_version and status == "update":
            if hasattr(obj, "_version"):
                if (
                    obj._version
                    == Proxy.var_dict[self.__dict__["var_name"]]._obj._version
                ):
                    return
        # Strong assertion: the previous type and current type of the object should be the same
        # assert typename(obj) == typename(
        #     self._obj
        # ), f"Type of the object is changed from {typename(self._obj)} to {typename(obj)}, needs careful check"

        if not issubclass(type(obj), torch.nn.Module):
            self.jsondumper.dump_json(
                process_id=self.process_id,
                thread_id=self.thread_id,
                time=current_time,
                meta_vars=get_meta_vars(self),
                var_name=self.__dict__["dumped_varname_list"],
                var_type=typename(obj),
                change_type=status,
                var_attributes=dump_attributes(self, obj),
                stack_trace=dumped_frame_array,
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
        # inherit the var_name from the parent object
        if self.__dict__["var_name"] is not None:
            current_var_name_list = self.__dict__["var_name"]
        else:
            current_var_name_list = ""
        self.__dict__["dumped_varname_list"] = current_var_name_list

        if self.__dict__["is_root"]:
            print(
                lambda: f"logger_proxy: ROOT proxy object for '{obj.__class__.__name__}'"
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
            dump_call_return = proxy_config.dump_info_config["dump_call_return"]
            dump_iter = proxy_config.dump_info_config["dump_iter"]
            if not dump_call_return and from_call:
                return
            if not dump_iter and from_iter:
                return

            current_time = time.time()
            trace_info = {
                "time": current_time,
                "frame_array": dumped_frame_array,
            }
            if dump_trace_info:
                if from_call:
                    trace_info["status"] = "call"

                if from_iter:
                    trace_info["status"] = "iter"
                # if the object is generated from getattr, then do not dump it
                else:
                    trace_info["status"] = "update"
                self.dump_to_trace(obj, trace_info)
                self.__dict__["last_update_timestamp"] = current_time
                Proxy.var_dict[current_var_name_list] = self

        else:  # if the object is proxied already
            if type(obj) not in [int, float, str, bool] and obj is not None:
                print_debug(
                    lambda: f"logger_proxy: Object '{obj.__class__.__name__}' is already proxied"
                )

            print_debug(
                lambda: f'Time elapse: {time.time() - Proxy.var_dict[current_var_name_list].__dict__["last_update_timestamp"]}'
            )
            self.__dict__["_obj"] = obj
            if (
                time.time()
                - Proxy.var_dict[current_var_name_list].__dict__[
                    "last_update_timestamp"
                ]
                < proxy_config.proxy_update_limit
            ):
                return
            dump_call_return = proxy_config.dump_info_config["dump_call_return"]
            dump_iter = proxy_config.dump_info_config["dump_iter"]
            if not dump_call_return and from_call:
                return

            if not dump_iter and from_iter:
                return

            current_time = time.time()

            trace_info = {
                "time": current_time,
                "frame_array": dumped_frame_array,
            }
            if dump_trace_info:
                if from_call:
                    trace_info["status"] = "call"
                elif from_iter:
                    trace_info["status"] = "iter"
                else:
                    trace_info["status"] = "update"

                self.dump_to_trace(obj, trace_info)

            del Proxy.var_dict[current_var_name_list]
            self.__dict__["last_update_timestamp"] = current_time
            Proxy.var_dict[current_var_name_list] = self

    @property  # type: ignore
    def __class__(self):  # type: ignore[misc]
        return self._obj.__class__

    def __call__(self, *args, **kwargs):
        print_debug(
            lambda: f"logger_proxy: Calling '{self.__class__.__name__}' for obj: '{self.__dict__['var_name']}' (type '{typename(self._obj)}')"
        )
        result = self._obj(*args, **kwargs)
        print_debug(
            lambda: f"logger_proxy: Result type of __call__ is '{typename(result)}'"
        )

        return proxy_handler(
            result,
            self.logdir,
            self.log_level,
            self.__dict__["var_name"],
            from_call=True,
        )

    def __getattr__(self, name):
        print_debug(lambda: f"logger_proxy: Accessing attribute '{name}'")
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
                lambda: f"Time elapse: {time.time() - Proxy.var_dict[self.__dict__['var_name']].__dict__['last_update_timestamp']}"
            )

            if self.__dict__["var_name"] == "":
                var_name = name
            else:
                var_name = self.__dict__["var_name"] + "." + name

            print_debug(lambda: f"Setting attribute '{name}' to '{value}'")

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
            if type(value) in proxy_config.primitive_types:
                self.dump_trace("update", disable_sampling=True)
            else:
                self.dump_trace("update")

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
        print_debug(lambda: "logger_proxy: Dump Proxy Dict: ")
        for k, value in proxy_dict.items():
            if isinstance(value, torch.Tensor):
                self.print_tensor(value)
            else:
                print_debug(lambda: f"logger_proxy: {k}: {value}")
