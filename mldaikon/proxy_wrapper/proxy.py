import inspect
import json
import logging
import os
import threading

import torch
import time
from typing import Union, Tuple, Any, Callable, Iterator, Set, Optional, overload, TypeVar, Mapping, Dict, List
from mldaikon.utils import typename
from mldaikon.config.config import debug_mode, proxy_log_dir, proxy_update_limit
from mldaikon.proxy_wrapper.dumper import json_dumper as dumper
from mldaikon.proxy_wrapper.dumper import dump_tensor, dump_attributes, dump_meta_vars, torch_serialize
from mldaikon.proxy_wrapper.utils import print_debug
import mldaikon.proxy_wrapper.proxy_methods as proxy_methods
import linecache

def get_line(filename, lineno):
    return linecache.getline(filename, lineno).strip()

class Proxy:
    proxy_dict = {}
    frame_dict = {} # Ziming: currently used together with var name based identifier
    tensor_frame_dict = {} # Ziming: currently used together with var name based identifier
    tensor_var_dict = {} # Ziming: deprecated
    var_dict = {} # Ziming: deprecated
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

    def __init__(self, obj, logdir="proxy_log.log", log_level=logging.INFO, is_root= False ):
        self.__dict__["process_id"] = os.getpid()
        self.__dict__["thread_id"] = threading.current_thread().ident
        self.__dict__["logdir"] = logdir
        self.__dict__["log_level"] = log_level
        self.__dict__["meta_vars"] = {}
        self.__dict__["last_update_timestamp"]=0
        self.__dict__["is_proxied_obj"]=True
        self.__dict__["is_root"]=is_root

        if type(obj) is Proxy:
            print_debug(
                "logger_proxy: "
                + f"Object '{obj.__class__.__name__}' is already a proxy"
            )
            self._obj = obj._obj
            self.__dict__["dumped_varname_list"] = obj.__dict__["dumped_varname_list"]

        else:
            frame = inspect.currentframe()
            var_list = []
            frame_array = []
            while frame:
                if frame.f_code.co_filename == __file__:
                    frame = frame.f_back
                else:
                    # fetch the frame info
                    frame_array.append((frame.f_code.co_filename, frame.f_lineno, get_line(frame.f_code.co_filename, frame.f_lineno)))
                    
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
            
            dumped_frame_array = json.dumps(frame_array)  
            
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
            if self.__dict__["is_root"] == True:
                print("found the root object")
            if issubclass(type(obj), torch.nn.Module):
                # proxy all of its submodules
                if self.__dict__["is_root"] == True:
                    print("logger_proxy: " + f"Proxying all submodules of '{obj.__class__.__name__}'")
                    for name, module in obj.named_children():
                        print("logger_proxy: " + f"Proxying submodule '{name}'")
                        setattr(obj, name, Proxy(module))
                        # TODO: improve the nn.Module tracing logic, currently we only trace two levels of submodules
                        # We could try to enforce a blacklist of modules that we can't trace (contain low level functions)
                        if issubclass(type(module), torch.nn.Module):
                            for name, submodule in module.named_children():
                                print("logger_proxy: " + f"Proxying submodule '{name}'")
                                setattr(module, name, Proxy(submodule))
                                
                                        
            
            if type(obj) is torch.Tensor:
                # init tensor
                tensor_shape = obj.shape.__str__()
                current_var_name_list = current_var_name_list + dumped_frame_array + tensor_shape
                if Proxy.tensor_var_dict.get(current_var_name_list) is None:
                    self.__dict__["_obj"] = obj
                    Proxy.tensor_var_dict[current_var_name_list] = self
                    # if it is a method rather than an object, then we should not dump it
                    
                    self.jsondumper.dump_json(
                        self.process_id,
                        self.thread_id,
                        dump_meta_vars(proxy_file_path=__file__),
                        self.__dict__["dumped_varname_list"],
                        type(obj).__name__,
                        dump_tensor(obj),
                        "new",
                        dump_attributes(obj),
                        dumped_frame_array
                    )
                    self.__dict__["last_update_timestamp"] = time.time()
                # update tensor
                else:
                    print_debug(
                        "logger_proxy: "
                        + f"Tensor name: '{current_var_name_list}' is already proxied"
                    )
                    
                    print_debug(f'Time elapse: {time.time() - Proxy.tensor_var_dict[current_var_name_list].__dict__["last_update_timestamp"]}')
                    

                    if time.time() - Proxy.tensor_var_dict[current_var_name_list].__dict__["last_update_timestamp"] < proxy_update_limit:
                        self.__dict__["_obj"] = obj
                        return

                    self.jsondumper.dump_json(
                        self.process_id,
                        self.thread_id,
                        dump_meta_vars(proxy_file_path=__file__),
                        self.__dict__["dumped_varname_list"],
                        type(obj).__name__,
                        dump_tensor(obj),
                        "update",
                        dump_attributes(obj),
                        dumped_frame_array
                    )

                    del Proxy.tensor_var_dict[current_var_name_list]
                    self.__dict__["_obj"] = obj
                    self.__dict__["last_update_timestamp"] = time.time()
                    Proxy.tensor_var_dict[current_var_name_list] = self
            else:
                current_var_name_list = current_var_name_list + dumped_frame_array
                if Proxy.var_dict.get(current_var_name_list) is None:
                    new_value = str(torch_serialize(obj))
                    self.__dict__["_obj"] = obj
                    
                    self.__dict__["last_update_timestamp"] = time.time()
                    self.jsondumper.dump_json(
                        self.process_id,
                        self.thread_id,
                        dump_meta_vars(proxy_file_path=__file__),
                        self.__dict__["dumped_varname_list"],
                        type(obj).__name__,
                        new_value,
                        "new",
                        dump_attributes(obj),
                        dumped_frame_array
                    )
                    
                    Proxy.var_dict[current_var_name_list] = self
                else:
                    if not type(obj) in [int, float, str, bool] and obj is not None:
                        print_debug(
                            "logger_proxy: "
                            + f"Object '{obj.__class__.__name__}' is already proxied"
                        )

                    new_value = str(torch_serialize(obj))
                    
                    print_debug(f'Time elapse: {time.time() - Proxy.var_dict[current_var_name_list].__dict__["last_update_timestamp"]}')            
                    if time.time() - Proxy.var_dict[current_var_name_list].__dict__["last_update_timestamp"] < proxy_update_limit:
                        self.__dict__["_obj"] = obj
                        return

                    self.jsondumper.dump_json(
                        self.process_id,
                        self.thread_id,
                        dump_meta_vars(proxy_file_path=__file__),
                        self.__dict__["dumped_varname_list"],
                        type(obj).__name__,
                        new_value,
                        "update",
                        dump_attributes(obj),
                        dumped_frame_array
                    )
                    
                    del Proxy.var_dict[current_var_name_list]
                    self.__dict__["_obj"] = obj
                    self.__dict__["last_update_timestamp"] = time.time()
                    Proxy.var_dict[current_var_name_list] = self

    @property
    def __class__(self):
        return self._obj.__class__
    
    def step(self, *args, **kwargs):
        print("logger_proxy: " + f"Go to step for object '{self.__class__.__name__}'")
        self._obj.__class__.step(self, *args, **kwargs)
        
    

    def __call__(self, *args, **kwargs):
        print_debug(
            "logger_proxy: " + f"Go to __call__ for object '{self.__class__.__name__}'"
        )
        # only pass down the torch.nn.Module here
        args = tuple(arg._obj if (type(arg) is Proxy and not issubclass(arg._obj,torch.nn.Module) ) else arg for arg in args)
        kwargs = {k: v._obj if (type(v) is Proxy  and not issubclass(v._obj,torch.nn.Module)) else v for k, v in kwargs.items()}
        
        result = self._obj(*args, **kwargs)
        # HACK: avoid proxying torch.distributed as we cannot handle ProcessGroup `in` ops in the get_group_rank & get_global_rank function
        if typename(result).startswith("torch.distributed"):
            return result
        if issubclass(type(result), torch.nn.Module):
            return Proxy(result, logdir=self.logdir, log_level=self.log_level)
        return result
       

    def __getattr__(self, name):
        print_debug("logger_proxy: " + f"Accessing attribute '{name}'")
        if name == "logdir":
            return self.__dict__.get("logdir", None)  # in order to pass down the dir
        if name == "_obj":
            return self.__dict__.get("_obj", None)  # in order to pass down the dir
        attr = getattr(self._obj, name)

        # HACK: avoid proxying torch.distributed as we cannot handle ProcessGroup `in` ops in the get_group_rank & get_global_rank function
        if typename(attr).startswith("torch.distributed"):
            return attr
        # if attr is a tensor or nn.Module, return a proxy object
        if isinstance(attr, torch.Tensor): # TODO: should double check if the torch.Tensor wrapping is effective here
            return Proxy(attr, logdir=self.logdir, log_level=self.log_level)
        if issubclass(type(attr), torch.nn.Module):
            return Proxy(attr, logdir=self.logdir, log_level=self.log_level)
        
        # if attr is a bound method, return a wrapper function
        if callable(attr):
            def method(*args, **kwargs):
                result = Proxy(attr)(*args, **kwargs)
                print_debug(f"Called method '{name}' with result {result}")
                # if result is not primitive, return a proxy object
                if isinstance(result, torch.Tensor):
                    return Proxy(result, logdir=self.logdir, log_level=self.log_level)
                elif issubclass(type(result), torch.nn.Module):
                    return Proxy(result, logdir=self.logdir, log_level=self.log_level)
                return result
            return method
        return attr

    def __setattr__(self, name, value):
        print_debug("logger_proxy: " + f"Setting attribute '{name}' to '{value}'")
        if name == "_obj":
            self.__dict__[name] = value  # Set the attribute directly
        else:
            new_value = str(torch_serialize(value))
            self.jsondumper.dump_json(
                self.process_id,
                self.thread_id,
                dump_meta_vars(proxy_file_path=__file__),
                self.__dict__["dumped_varname_list"],
                type(value).__name__,
                "update",
                new_value,
                dump_attributes(value),
            )

            if value is not None:
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
        for element in self._obj:
            if typename(element).startswith("torch.distributed"):
                yield element
            else:
                yield Proxy(element, logdir=self.logdir, log_level=self.log_level)
    
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

class RootProxy(Proxy):
    def __init__(self, obj, logdir="proxy_log.log", log_level=logging.INFO):
        super().__init__(obj, logdir, log_level)
        self.__dict__["is_root"] = True