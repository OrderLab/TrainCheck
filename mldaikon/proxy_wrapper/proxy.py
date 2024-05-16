import inspect
import logging
import os
import threading
import json
import torch

from mldaikon.utils import typename
from mldaikon.config.config import debug_mode, proxy_log_dir
from .dumper import json_dumper as dumper

import torch.distributed
from torch._C._distributed_c10d import ReduceOp

#################################################
###         Proxied Torch functions

# Save the original broadcast function
original_broadcast = torch.distributed.broadcast

def broadcast(tensor, src, group=None, async_op=False):
    # Perform the original broadcast operation
    original_broadcast(tensor, src, group, async_op)

    # Wrap the first argument in a Proxy object
    tensor = Proxy(tensor, logdir='proxy_log.log', log_level=logging.INFO)

# Override the broadcast function
torch.distributed.broadcast = broadcast

original_all_reduce = torch.distributed.all_reduce

def all_reduce(tensor, op=ReduceOp.SUM, group=None, async_op=False):
    # Perform the original all_reduce operation
    original_all_reduce(tensor, op, group, async_op)

    # Wrap the first argument in a Proxy object
    tensor = Proxy(tensor, logdir='proxy_log.log', log_level=logging.INFO)

# Override the all_reduce function
torch.distributed.all_reduce = all_reduce

###########################################
##      Print Debug

def print_debug(message):
    if debug_mode:
        print(message)


def dump_tensor(value):
    min = float(value.min().item())
    max = float(value.max().item())
    shape = tuple(int(x) for x in value.size())
    result = {
        "min": min,
        "max": max,
        "shape": shape,
    }
    return result


def get_meta_vars(level=5):
    frame = inspect.currentframe()
    while frame.f_code.co_filename == __file__:
        frame = frame.f_back
        frame_vars = frame.f_locals
        # TODO: filter out the important variables
    important_vars = {}
    for i in range(level):
        important_vars.update(
            {
                key: frame_vars[key]
                for key in frame_vars
                # Ziming: only get primitive types for now
                if isinstance(frame_vars[key], (int, float, str, bool))
                # if isinstance(frame_vars[key], (int, float, str, bool, torch.Tensor))
            }
        )
        # for key, value in frame_vars.items():
        #     if isinstance(value, torch.Tensor):
        #         important_vars[key] = str(dump_tensor(value))
        frame = frame.f_back
        if frame is None:
            break
        frame_vars = frame.f_locals
    # Convert the dictionary to a JSON string and print_debug it
    # json_data = json.dumps(important_vars)
    # return json_data
    return important_vars


def torch_serialize(obj):
    if isinstance(obj, (int, float, str, bool)):
        return obj
    if isinstance(obj, torch.Tensor):
        new_value = str(dump_tensor(obj))
        return new_value
    if isinstance(obj, torch.nn.Module):
        new_value = obj.__class__.__name__ + "(nn.Module)"
        return new_value
    else:
        try:
            json.dumps(obj)
        except TypeError:
            obj = str(obj)
        return obj


class Proxy:
    proxy_dict = {}
    frame_dict = {}
    # tensor_frame_dict = {} # Ziming: deprecated tensor.shape based identifier
    tensor_var_dict = {}
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
        # handler = logging.FileHandler(logdir)
        # handler.setLevel(log_level)
        # self.logger_proxy.addHandler(handler)

        # # if the object is of type in torch.distributed namespace, we should not proxy it
        # if typename(obj).startswith("torch.distributed"):
        #     self._obj = obj
        #     return

        if not type(obj) in [int, float, str, bool] and obj is not None:
            print_debug(
                "logger_proxy: "
                + f"Go to __init__ for object '{obj.__class__.__name__}'"
            )
        else:
            print_debug("logger_proxy: " + f"Proxied premitive type '{type(obj)}'")

        if type(obj) is Proxy:
            print_debug(
                "logger_proxy: "
                + f"Object '{obj.__class__.__name__}' is already a proxy"
            )
            self._obj = obj._obj

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
                        print_debug("logger_proxy: " + f"Variable name f{current_var_name} of the object is found in the current frame")
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
            
            if type(obj) is torch.Tensor:
                if Proxy.tensor_var_dict.get(current_var_name_list) is None:
                    self.__dict__["_obj"] = obj
                    Proxy.tensor_var_dict[current_var_name_list] = self
                else:
                    
                    print_debug(
                        "logger_proxy: "
                        + f"Tensor '{current_var_name_list}' is already proxied"
                    )
                    # self.print_update(Proxy.tensor_frame_dict[current_var_name_list]._obj, obj, f"torch.Tensor")

                    self.jsondumper.dump_json(
                        self.process_id,
                        self.thread_id,
                        get_meta_vars(),
                        self.__dict__["dumped_varname_list"],
                        dump_tensor(obj),
                    )

                    del Proxy.tensor_var_dict[current_var_name_list]
                    self._obj = obj
                    Proxy.tensor_var_dict[current_var_name_list] = self
            else:
                if Proxy.frame_dict.get(tuple(frame_array)) is None:
                    new_value = str(torch_serialize(obj))

                    if hasattr(obj, "__name__"):
                        print_debug(
                            "logger_proxy: "
                            + f"Creating proxy for object '{obj.__name__}'"
                        )

                        self.jsondumper.dump_json(
                            self.process_id,
                            self.thread_id,
                            "",
                            self.__dict__["dumped_varname_list"],
                            new_value,
                        )
                    else:
                        print_debug(
                            "logger_proxy: "
                            + f"Creating proxy for object with type '{obj.__class__.__name__}'"  # FIXME: combine this with the above branch with typename(obj)
                        )

                        self.jsondumper.dump_json(
                            self.process_id,
                            self.thread_id,
                            "",
                            self.__dict__["dumped_varname_list"],
                            new_value,
                        )

                    self.__dict__["_obj"] = obj
                    # Proxy.proxy_dict[id(self._obj)] = self

                    Proxy.frame_dict[tuple(frame_array)] = self
                else:
                    if not type(obj) in [int, float, str, bool] and obj is not None:
                        print_debug(
                            "logger_proxy: "
                            + f"Object '{obj.__class__.__name__}' is already proxied"
                        )
                    # self._obj = Proxy.frame_dict[tuple(frame_array)]._obj ## attention, need to delete the original one before creating new instance
                    # obj_name = (
                    #     obj.__class__.__module__ + "." + obj.__class__.__name__
                    # )  # TODO: refactor with typename

                    # old_value = str(
                    #     torch_serialize(Proxy.frame_dict[tuple(frame_array)]._obj)
                    # )

                    new_value = str(torch_serialize(obj))

                    # self.print_update(old_value, new_value, obj_name)

                    self.jsondumper.dump_json(
                        self.process_id,
                        self.thread_id,
                        get_meta_vars(),
                        self.__dict__["dumped_varname_list"],
                        new_value,
                    )
                    del Proxy.frame_dict[tuple(frame_array)]
                    self._obj = obj
                    Proxy.frame_dict[tuple(frame_array)] = self

    @property
    def __class__(self):
        return self._obj.__class__

    def __array__(self):
        print_debug(
            "logger_proxy: " + f"Go to __array__ for object '{self.__class__.__name__}'"
        )
        return self._obj.__array__()

    # def __torch_function__(self, func, types, args=(), kwargs=None):
    #     print_debug("logger_proxy: " +
    #         f"Go to __torch_function__ for function '{func.__name__}'"
    #     )
    #     if kwargs is None:
    #         kwargs = {}

    #     # Unwrap Proxy objects in args and kwargs
    #     args = tuple(arg._obj if (type(arg) is Proxy) else arg for arg in args)
    #     kwargs = {k: v._obj if (type(v) is Proxy) else v for k, v in kwargs.items()}
    #     if hasattr(self._obj, "__torch_function__"):
    #         result = self._obj.__torch_function__(func, types, args, kwargs)
    #     else:
    #         result = func(*args, **kwargs)

    #     # Call the original function with the unwrapped args and kwargs
    #     return Proxy(result, logdir=self.logdir, log_level=self.log_level)

    def __call__(self, *args, **kwargs):
        print_debug(
            "logger_proxy: " + f"Go to __call__ for object '{self.__class__.__name__}'"
        )
        args = tuple(arg._obj if (type(arg) is Proxy) else arg for arg in args)
        kwargs = {k: v._obj if (type(v) is Proxy) else v for k, v in kwargs.items()}
        # args=[arg if isinstance(arg, Proxy) else Proxy(arg) for arg in args]
        # kwargs={k: v if isinstance(v, Proxy) else Proxy(v) for k, v in kwargs.items()}
        result = self._obj(*args, **kwargs)

        # HACK: avoid proxying torch.distributed as we cannot handle ProcessGroup `in` ops in the get_group_rank & get_global_rank function
        if typename(result).startswith("torch.distributed"):
            return result

        return Proxy(result, logdir=self.logdir, log_level=self.log_level)

    def __format__(self, format_spec):
        print_debug(
            "logger_proxy: "
            + f"Go to __format__ for object '{self.__class__.__name__}'"
        )
        # Delegate the formatting to the wrapped object
        return format(self._obj, format_spec)

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

    def __getattr__(self, name):
        print_debug("logger_proxy: " + f"Accessing attribute '{name}'")
        if name == "logdir":
            return self.__dict__.get("logdir", None)  # in order to pass down the dir
        # print_debug("logger_proxy: " +f"Accessing attribute '{name}'")
        attr = getattr(self._obj, name)

        # HACK: avoid proxying torch.distributed as we cannot handle ProcessGroup `in` ops in the get_group_rank & get_global_rank function
        if typename(attr).startswith("torch.distributed"):
            return attr

        return Proxy(attr, logdir=self.logdir, log_level=self.log_level)

    def __setattr__(self, name, value):
        print_debug("logger_proxy: " + f"Setting attribute '{name}' to '{value}'")
        if name == "_obj":
            # if type(value) is torch.Tensor:
            #     self.print_tensor(value)
            # else:
            #     print_debug("logger_proxy: " + f"Setting attribute '_obj'")
            self.__dict__[name] = value  # Set the attribute directly
        else:
            # Intercept attribute assignment
            # old_value = getattr(self._obj, name, None)
            # old_value = str(torch_serialize(old_value))
            new_value = str(torch_serialize(value))
            

            # attr_name = f"{self._obj.__class__.__module__}.{self._obj.__class__.__name__}.{name}"
            # self.print_update(old_value, value, attr_name)
            self.jsondumper.dump_json(
                self.process_id,
                self.thread_id,
                get_meta_vars(),
                self.__dict__["dumped_varname_list"],
                new_value,
            )

            if not type(value) in [int, float, str, bool] and value is not None:
                setattr(
                    self._obj,
                    name,
                    Proxy(value, logdir=self.logdir, log_level=self.log_level),
                )
            else:
                setattr(self._obj, name, value)

    def __delattr__(self, name):
        # Intercept attribute deletion
        print_debug("logger_proxy: " + f"Deleting attribute '{name}'")
        delattr(self._obj, name)

    def __getitem__(self, key):
        # Intercept item retrieval
        print_debug("logger_proxy: " + f"Getting item with key '{key}'")
        return Proxy(self._obj[key])

    def __setitem__(self, key, value):
        # Intercept item assignment
        print_debug("logger_proxy: " + f"Setting item with key '{key}' to '{value}'")
        self._obj[key] = value

    def __delitem__(self, key):
        # Intercept item deletion
        print_debug("logger_proxy: " + f"Deleting item with key '{key}'")
        del self._obj[key]

    def __add__(self, other):
        # Unwrap other if it's a Proxy
        print_debug(
            "logger_proxy: " + f"Calling __add__ for object '{self.__class__.__name__}'"
        )
        other = other._obj if isinstance(other, Proxy) else other
        return self._obj + other

    def __or__(self, other):
        print_debug(
            "logger_proxy: " + f"Calling __or__ for object '{self.__class__.__name__}'"
        )
        if isinstance(other, bool):
            # If the other operand is a boolean, convert the Proxy object to a boolean and do the bitwise OR
            return bool(self._obj) | other
        else:
            # Otherwise, do the bitwise OR on the wrapped object
            return self._obj | other

    def __ior__(self, other):
        print_debug(
            "logger_proxy: " + f"Calling __ior__ for object '{self.__class__.__name__}'"
        )
        if isinstance(other, bool):
            self._obj = bool(self._obj) | other
        else:
            self._obj |= other
        return self

    def __ror__(self, other):
        print_debug(
            "logger_proxy: " + f"Calling __ror__ for object '{self.__class__.__name__}'"
        )
        if isinstance(other, bool):
            return other | bool(self._obj)
        else:
            return other | self._obj

    def __radd__(self, other):
        print_debug(
            "logger_proxy: "
            + f"Calling __radd__ for object '{self.__class__.__name__}'"
        )
        # Unwrap other if it's a Proxy
        other = other._obj if isinstance(other, Proxy) else other
        return other + self._obj

    def __iadd__(self, other):
        print_debug(
            "logger_proxy: "
            + f"Calling __iadd__ for object '{self.__class__.__name__}'"
        )
        # Unwrap other if it's a Proxy
        other = other._obj if isinstance(other, Proxy) else other
        self._obj += other
        return self

    def __sub__(self, other):
        print_debug(
            "logger_proxy: " + f"Calling __sub__ for object '{self.__class__.__name__}'"
        )
        # Unwrap other if it's a Proxy
        other = other._obj if isinstance(other, Proxy) else other
        return self._obj - other

    def __mul__(self, other):
        print_debug(
            "logger_proxy: " + f"Calling __mul__ for object '{self.__class__.__name__}'"
        )
        # Unwrap other if it's a Proxy
        other = other._obj if isinstance(other, Proxy) else other
        return self._obj * other

    def __rmul__(self, other):
        print_debug(
            "logger_proxy: "
            + f"Calling __rmul__ for object '{self.__class__.__name__}'"
        )
        # Unwrap other if it's a Proxy
        other = other._obj if isinstance(other, Proxy) else other
        return other * self._obj

    def __truediv__(self, other):
        print_debug(
            "logger_proxy: "
            + f"Calling __truediv__ for object '{self.__class__.__name__}'"
        )
        # Unwrap other if it's a Proxy
        other = other._obj if isinstance(other, Proxy) else other
        return self._obj / other

    def __floatdiv__(self, other):
        print_debug(
            "logger_proxy: "
            + f"Calling __floatdiv__ for object '{self.__class__.__name__}'"
        )
        # Unwrap other if it's a Proxy
        other = other._obj if isinstance(other, Proxy) else other
        return self._obj // other

    def __rfloordiv__(self, other):
        print_debug(
            "logger_proxy: "
            + f"Calling __ifloordiv__ for object '{self.__class__.__name__}'"
        )
        # Unwrap other if it's a Proxy
        other = other._obj if isinstance(other, Proxy) else other
        return other // self._obj

    def __float__(self):
        print_debug(
            "logger_proxy: "
            + f"Calling __float__ for object '{self.__class__.__name__}'"
        )
        return float(self._obj)

    def __int__(self):
        print_debug(
            "logger_proxy: " + f"Calling __int__ for object '{self.__class__.__name__}'"
        )
        return int(self._obj)

    def __str__(self):
        print_debug(
            "logger_proxy: " + f"Calling __str__ for object '{self.__class__.__name__}'"
        )
        return str(self._obj)

    def __bool__(self):
        print_debug(
            "logger_proxy: "
            + f"Calling __bool__ for object '{self.__class__.__name__}'"
        )
        return bool(self._obj)

    def __repr__(self):
        print_debug(
            "logger_proxy: "
            + f"Calling __repr__ for object '{self.__class__.__name__}'"
        )
        return repr(self._obj)

    def __len__(self):
        print_debug(
            "logger_proxy: " + f"Calling __len__ for object '{self.__class__.__name__}'"
        )
        return len(self._obj)

    def __getreal__(self):
        print_debug(
            "logger_proxy: "
            + f"Calling __getreal__ for object '{self.__class__.__name__}'"
        )
        return self._obj

    def min(self):
        print_debug(
            "logger_proxy: " + f"Calling min() for object '{self.__class__.__name__}'"
        )
        return self._obj.min()

    def max(self):
        print_debug(
            "logger_proxy: " + f"Calling max() for object '{self.__class__.__name__}'"
        )
        return self._obj.max()

    def size(self):
        print_debug(
            "logger_proxy: " + f"Calling size() for object '{self.__class__.__name__}'"
        )
        return self._obj.size()

    def print_proxy_dict(self):
        print_debug("logger_proxy: " + f"Dump Proxy Dict: ")

        for k, value in Proxy.proxy_dict.items():
            if isinstance(value, torch.Tensor):
                self.print_tensor(value)
            else:
                print_debug("logger_proxy: " + f"{k}: {value}")

        print_debug("logger_proxy: " + f"Dump Frame Dict: ")
        for k, value in Proxy.frame_dict.items():
            if isinstance(value, torch.Tensor):
                self.print_tensor(value)
            else:
                print_debug("logger_proxy: " + f"{k}: {value}")
