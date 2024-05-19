import inspect
import logging
import os
import threading
import json
import torch
import time

from mldaikon.utils import typename
from mldaikon.config.config import debug_mode, proxy_log_dir, proxy_update_limit
from mldaikon.proxy_wrapper.dumper import json_dumper as dumper
from mldaikon.proxy_wrapper.dumper import dump_tensor, dump_attributes, dump_meta_vars, torch_serialize
from mldaikon.proxy_wrapper.utils import print_debug
import mldaikon.proxy_wrapper.proxy_methods as proxy_methods

class Proxy:
    proxy_dict = {}
    # frame_dict = {} # Ziming: deprecated frame based identifier
    # tensor_frame_dict = {} # Ziming: deprecated tensor.shape based identifier
    tensor_var_dict = {}
    var_dict = {}
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
            print_debug("logger_proxy: " + f"From:")
            if type(old_value) is torch.Tensor:
                Proxy.print_tensor(old_value)
            else:
                print_debug("logger_proxy: " + f"'{old_value}'")

            print_debug("logger_proxy: " + f"To:")
            if type(value) is torch.Tensor:
                Proxy.print_tensor(value)
            else:
                print_debug("logger_proxy: " + f"'{value}'")

    def __init__(self, obj, logdir='proxy_log.log', log_level=logging.INFO):
        self.__dict__["process_id"] = os.getpid()
        self.__dict__["thread_id"] = threading.current_thread().ident
        self.__dict__["logdir"] = logdir
        self.__dict__["log_level"] = log_level
        self.__dict__["meta_vars"] = {}
        self.__dict__["last_update_timestamp"]=0
        self.__dict__["is_proxied_obj"]=True

        if type(obj) is Proxy:
            print_debug(
                "logger_proxy: "
                + f"Object '{obj.__class__.__name__}' is already a proxy"
            )
            self.__dict__["_obj"] = obj.__dict__["_obj"]

        else:
            frame = inspect.currentframe()
            var_list = []
            frame_array = []
            while frame:
                if frame.f_code.co_filename == __file__:
                    frame = frame.f_back
                else:
                    # Ziming: old line number based identifier
                    if debug_mode:
                        frame_array.append((frame.f_code.co_filename, frame.f_lineno))
                    # fetch the var_name from the stack_frame
                    current_var_name = None
                    for var_name, var_val in frame.f_locals.items():
                        if var_val is obj or (isinstance(var_val, Proxy) and var_val._obj is obj):
                            current_var_name = var_name
                            break
                    if current_var_name is not None:
                        print_debug("logger_proxy: " + f"Variable name f{current_var_name}"\
                            "of the object is found in the current frame")
                        var_list.append(current_var_name)
                    frame = frame.f_back
                    
            # print_debug the variable name list of the object
            if len(var_list) == 0:
                print_debug("logger_proxy: " + f"Empty variable name list")
                Proxy.empty_name_counts += 1
            else:
                print_debug("logger_proxy: " + f"Variable name list: {var_list}") 
                Proxy.non_empty_name_counts += 1
            print_debug("logger_proxy: " + f"Empty name counts: {Proxy.empty_name_counts}")
            print_debug("logger_proxy: " + f"Non-empty name counts: {Proxy.non_empty_name_counts}")
            
            # dumped var_list
            if len(var_list) == 0:
                current_var_name_list = "None"
            else:
                current_var_name_list = json.dumps(var_list)
            self.__dict__["dumped_varname_list"] = current_var_name_list
            
            # Ziming: here we still seperate the handling of tensor and other objects
            # however, despite the dumping logic these two are identical and could be merged
            if type(obj) is torch.Tensor:
                # init tensor
                if Proxy.tensor_var_dict.get(current_var_name_list) is None:
                    self.__dict__["_obj"] = obj
                    Proxy.tensor_var_dict[current_var_name_list] = self
                    
                    self.jsondumper.dump_json(
                        self.process_id,
                        self.thread_id,
                        dump_meta_vars(),
                        self.__dict__["dumped_varname_list"],
                        type(obj).__name__,
                        dump_tensor(obj),
                        dump_attributes(obj),
                    )
                    self.__dict__["last_update_timestamp"] = time.time()
                # update tensor
                else:
                    print_debug(
                        "logger_proxy: "
                        + f"Tensor '{current_var_name_list}' is already proxied"
                    )

                    if time.time() - Proxy.tensor_var_dict[current_var_name_list].__dict__["last_update_timestamp"] < proxy_update_limit:
                        self.__dict__["_obj"] = obj
                        return

                    self.jsondumper.dump_json(
                        self.process_id,
                        self.thread_id,
                        dump_meta_vars(),
                        self.__dict__["dumped_varname_list"],
                        type(obj).__name__,
                        dump_tensor(obj),
                        dump_attributes(obj),
                    )

                    del Proxy.tensor_var_dict[current_var_name_list]
                    self.__dict__["_obj"] = obj
                    self.__dict__["last_update_timestamp"] = time.time()
                    Proxy.tensor_var_dict[current_var_name_list] = self
            else:
                if Proxy.var_dict.get(current_var_name_list) is None:
                    new_value = str(torch_serialize(obj))
                    self.__dict__["_obj"] = obj
                    
                    self.__dict__["last_update_timestamp"] = time.time()
                    self.jsondumper.dump_json(
                        self.process_id,
                        self.thread_id,
                        dump_meta_vars(),
                        self.__dict__["dumped_varname_list"],
                        type(obj).__name__,
                        new_value,
                        dump_attributes(obj),
                    )
                    
                    Proxy.var_dict[current_var_name_list] = self
                else:
                    if not type(obj) in [int, float, str, bool] and obj is not None:
                        print_debug(
                            "logger_proxy: "
                            + f"Object '{obj.__class__.__name__}' is already proxied"
                        )

                    new_value = str(torch_serialize(obj))
                                            
                    if time.time() - Proxy.var_dict[current_var_name_list].__dict__["last_update_timestamp"] < proxy_update_limit:
                        self.__dict__["_obj"] = obj
                        return

                    self.jsondumper.dump_json(
                        self.process_id,
                        self.thread_id,
                        dump_meta_vars(),
                        self.__dict__["dumped_varname_list"],
                        type(obj).__name__,
                        new_value,
                        dump_attributes(obj),
                    )
                    
                    del Proxy.var_dict[current_var_name_list]
                    self.__dict__["_obj"] = obj
                    self.__dict__["last_update_timestamp"] = time.time()
                    Proxy.var_dict[current_var_name_list] = self

    @property
    def __class__(self):
        if "_obj" in self.__dict__:
            return self.__dict__["_obj"].__class__
        else:
            return None

    def __call__(self, *args, **kwargs):
        print_debug(
            "logger_proxy: " + f"Go to __call__ for object '{self.__class__.__name__}'"
        )
        args = tuple(arg._obj if (type(arg) is Proxy) else arg for arg in args)
        kwargs = {k: v._obj if (type(v) is Proxy) else v for k, v in kwargs.items()}

        result = self._obj(*args, **kwargs)

        # HACK: avoid proxying torch.distributed as we cannot handle ProcessGroup `in` ops in the get_group_rank & get_global_rank function
        if typename(result).startswith("torch.distributed"):
            return result

        return Proxy(result, logdir=self.logdir, log_level=self.log_level)

    def __getattr__(self, name):
        print_debug("logger_proxy: " + f"Accessing attribute '{name}'")
        if name == "logdir":
            return self.__dict__.get("logdir", None)  # in order to pass down the dir
        attr = getattr(self._obj, name)

        # HACK: avoid proxying torch.distributed as we cannot handle ProcessGroup `in` ops in the get_group_rank & get_global_rank function
        if typename(attr).startswith("torch.distributed"):
            return attr

        return Proxy(attr, logdir=self.logdir, log_level=self.log_level)

    def __setattr__(self, name, value):
        print_debug("logger_proxy: " + f"Setting attribute '{name}' to '{value}'")
        if name == "_obj":
            self.__dict__[name] = value  # Set the attribute directly
        else:
            # Intercept attribute assignment
            # old_value = getattr(self._obj, name, None)
            # old_value = str(torch_serialize(old_value))
            new_value = str(torch_serialize(value))
            self.jsondumper.dump_json(
                self.process_id,
                self.thread_id,
                dump_meta_vars(),
                self.__dict__["dumped_varname_list"],
                type(value).__name__,
                new_value,
                dump_attributes(value),
            )

            if not type(value) in [int, float, str, bool] and value is not None:
                setattr(
                    self._obj,
                    name,
                    Proxy(value, logdir=self.logdir, log_level=self.log_level),
                )
            else:
                setattr(self._obj, name, value)
                
    def __getitem__(self, key):
        # Intercept item retrieval
        print_debug("logger_proxy: " + f"Getting item with key '{key}'")
        return Proxy(self._obj[key])
    
    def __iter__(self):
        print_debug("logger_proxy: " + f"Calling __iter__")
        # HACK: avoid proxying torch.distributed as we cannot handle ProcessGroup `in` ops in the get_group_rank & get_global_rank function
        return iter(
            (
                Proxy(obj, logdir=self.logdir, log_level=self.log_level)
                if not typename(obj).startswith("torch.distributed")
                else obj
            )
            for obj in self._obj
        )

    def __next__(self):
        print_debug("logger_proxy: " + f"Calling __next__")
        result = next(self._obj)

        # HACK: avoid proxying torch.distributed as we cannot handle ProcessGroup `in` ops in the get_group_rank & get_global_rank function
        if typename(result).startswith("torch.distributed"):
            return result

        return Proxy(next(self))

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
    
    max = proxy_methods.max
    min = proxy_methods.min
    size = proxy_methods.size

    def print_proxy_dict(self, proxy_dict):
        # for debugging purpose: print the var_dict of the proxy object
        print_debug("logger_proxy: " + f"Dump Proxy Dict: ")
        for k, value in proxy_dict.items():
            if isinstance(value, torch.Tensor):
                self.print_tensor(value)
            else:
                print_debug("logger_proxy: " + f"{k}: {value}")
