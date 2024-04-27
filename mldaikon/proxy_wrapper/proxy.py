import inspect
import json
import logging
import os
import threading

import torch

from .dumper import json_dumper as dumper


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
                if isinstance(frame_vars[key], (int, float, str, bool, torch.Tensor))
            }
        )
        for key, value in frame_vars.items():
            if isinstance(value, torch.Tensor):
                important_vars[key] = str(dump_tensor(value))
        frame = frame.f_back
        if frame is None:
            break
        frame_vars = frame.f_locals
    # Convert the dictionary to a JSON string and print it
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
        if hasattr(obj, "__name__"):
            new_value = obj.__name__
        else:
            new_value = obj.__class__.__name__


class Proxy:
    proxy_dict = {}
    frame_dict = {}
    tensor_frame_dict = {}
    logger_proxy = logging.getLogger("proxy")
    logdir = "proxy_logs.log"
    loglevel = logging.INFO
    jsondumper = dumper("proxy_trace.json")
    handler = logging.FileHandler(logdir)
    handler.setLevel(loglevel)
    logger_proxy.handlers.clear()
    logger_proxy.addHandler(handler)

    @staticmethod
    def print_tensor(value, logging_level=logging.DEBUG):
        if logging_level == logging.INFO:
            Proxy.logger_proxy.info(f"Tensor with shape'{value.shape}'")
            Proxy.logger_proxy.info(f"Minimum value: {torch.min(value)}")
            Proxy.logger_proxy.info(f"Maximum value: {torch.max(value)}")
        else:
            Proxy.logger_proxy.debug(f"Tensor with shape'{value.shape}'")
            Proxy.logger_proxy.debug(f"Minimum value: {torch.min(value)}")
            Proxy.logger_proxy.debug(f"Maximum value: {torch.max(value)}")

    @staticmethod
    def print_update(old_value, value, attr_name=None):
        Proxy.logger_proxy.info(f"Updating the attribute '{attr_name}'")
        Proxy.logger_proxy.info(f"From:")
        if type(old_value) is torch.Tensor:
            Proxy.print_tensor(old_value, logging_level=logging.INFO)
        else:
            Proxy.logger_proxy.info(f"'{old_value}'")

        Proxy.logger_proxy.info(f"To:")
        if type(value) is torch.Tensor:
            Proxy.print_tensor(value, logging_level=logging.INFO)
        else:
            Proxy.logger_proxy.info(f"'{value}'")

    def __init__(self, obj, logdir, log_level):
        self.__dict__["process_id"] = os.getpid()
        self.__dict__["thread_id"] = threading.current_thread().ident
        self.__dict__["logdir"] = logdir
        self.__dict__["log_level"] = log_level
        self.__dict__["meta_vars"] = {}
        # handler = logging.FileHandler(logdir)
        # handler.setLevel(log_level)
        # self.logger_proxy.addHandler(handler)

        if not type(obj) in [int, float, str, bool] and obj is not None:
            self.logger_proxy.debug(
                f"Go to __init__ for object '{obj.__class__.__name__}'"
            )
        else:
            self.logger_proxy.debug(f"Proxied premitive type '{type(obj)}'")

        if type(obj) is Proxy:
            self.logger_proxy.debug(
                f"Object '{obj.__class__.__name__}' is already a proxy"
            )
            self._obj = obj._obj

        else:
            frame = inspect.currentframe()

            frame_array = []
            while frame:
                if frame.f_code.co_filename == __file__:
                    frame = frame.f_back
                else:
                    frame_array.append((frame.f_code.co_filename, frame.f_lineno))
                    frame = frame.f_back
                # print("Go up!")
            if type(obj) is torch.Tensor:
                if Proxy.tensor_frame_dict.get(tuple(frame_array)) is None:
                    self.__dict__["_obj"] = obj
                    Proxy.tensor_frame_dict[tuple(frame_array)] = {
                        tuple(obj.shape): self
                    }
                else:
                    tensor_dict = Proxy.tensor_frame_dict.get(tuple(frame_array))
                    shape = tuple(obj.shape)

                    if tensor_dict.get(shape) is None:
                        self.logger_proxy.info(
                            f"Creating proxy for Tensor with shape '{shape}'"
                        )

                        self.jsondumper.dump_json(
                            self.process_id,
                            self.thread_id,
                            get_meta_vars(),
                            f"torch.Tensor with shape {shape}",
                            {"old_value": None, "new_value": dump_tensor(obj)},
                        )
                        self.print_tensor(obj, logging.INFO)
                        self.__dict__["_obj"] = obj
                        tensor_dict[shape] = self
                    else:
                        self.logger_proxy.debug(
                            f"Tensor with shape '{shape}' is already proxied"
                        )
                        self.print_update(tensor_dict[shape]._obj, obj, f"torch.Tensor")

                        self.jsondumper.dump_json(
                            self.process_id,
                            self.thread_id,
                            get_meta_vars(),
                            f"torch.Tensor with shape {shape}",
                            {
                                "old_value": dump_tensor(tensor_dict[shape]._obj),
                                "new_value": dump_tensor(obj),
                            },
                        )

                        del tensor_dict[shape]
                        self._obj = obj
                        tensor_dict[shape] = self
            else:
                if Proxy.frame_dict.get(tuple(frame_array)) is None:
                    new_value = str(torch_serialize(obj))

                    if hasattr(obj, "__name__"):
                        self.logger_proxy.info(
                            f"Creating proxy for object '{obj.__name__}'"
                        )

                        self.jsondumper.dump_json(
                            self.process_id,
                            self.thread_id,
                            "",
                            obj.__name__,
                            {"old_value": None, "new_value": new_value},
                        )
                    else:
                        self.logger_proxy.info(
                            f"Creating proxy for object with type '{obj.__class__.__name__}'"
                        )

                        self.jsondumper.dump_json(
                            self.process_id,
                            self.thread_id,
                            "",
                            obj.__class__.__name__,
                            {"old_value": None, "new_value": new_value},
                        )

                    self.__dict__["_obj"] = obj
                    # Proxy.proxy_dict[id(self._obj)] = self

                    Proxy.frame_dict[tuple(frame_array)] = self
                else:
                    if not type(obj) in [int, float, str, bool] and obj is not None:
                        self.logger_proxy.debug(
                            f"Object '{obj.__class__.__name__}' is already proxied"
                        )
                    # self._obj = Proxy.frame_dict[tuple(frame_array)]._obj ## attention, need to delete the original one before creating new instance
                    obj_name = obj.__class__.__module__ + "." + obj.__class__.__name__

                    old_value = str(
                        torch_serialize(Proxy.frame_dict[tuple(frame_array)]._obj)
                    )

                    new_value = str(torch_serialize(obj))

                    self.print_update(old_value, new_value, obj_name)

                    self.jsondumper.dump_json(
                        self.process_id,
                        self.thread_id,
                        get_meta_vars(),
                        obj_name,
                        {"old_value": old_value, "new_value": new_value},
                    )
                    del Proxy.frame_dict[tuple(frame_array)]
                    self._obj = obj
                    Proxy.frame_dict[tuple(frame_array)] = self

    @property
    def __class__(self):
        return self._obj.__class__

    def __array__(self):
        self.logger_proxy.debug(
            f"Go to __array__ for object '{self.__class__.__name__}'"
        )
        return self._obj.__array__()

    def __torch_function__(self, func, types, args=(), kwargs=None):
        self.logger_proxy.debug(
            f"Go to __torch_function__ for function '{func.__name__}'"
        )
        if kwargs is None:
            kwargs = {}

        # Unwrap Proxy objects in args and kwargs
        args = tuple(arg._obj if (type(arg) is Proxy) else arg for arg in args)
        kwargs = {k: v._obj if (type(v) is Proxy) else v for k, v in kwargs.items()}
        result = func(*args, **kwargs)

        # Call the original function with the unwrapped args and kwargs
        return Proxy(result, logdir=self.logdir, log_level=self.log_level)

    def __call__(self, *args, **kwargs):
        self.logger_proxy.debug(
            f"Go to __call__ for object '{self.__class__.__name__}'"
        )
        args = tuple(arg._obj if (type(arg) is Proxy) else arg for arg in args)
        kwargs = {k: v._obj if (type(v) is Proxy) else v for k, v in kwargs.items()}
        # args=[arg if isinstance(arg, Proxy) else Proxy(arg) for arg in args]
        # kwargs={k: v if isinstance(v, Proxy) else Proxy(v) for k, v in kwargs.items()}
        result = self._obj(*args, **kwargs)

        return Proxy(result, logdir=self.logdir, log_level=self.log_level)

    def __format__(self, format_spec):
        self.logger_proxy.debug(
            f"Go to __format__ for object '{self.__class__.__name__}'"
        )
        # Delegate the formatting to the wrapped object
        return format(self._obj, format_spec)

    def __iter__(self):
        self.logger_proxy.debug(f"Calling __iter__")
        return iter(
            Proxy(obj, logdir=self.logdir, log_level=self.log_level)
            for obj in self._obj
        )

    def __next__(self):
        self.logger_proxy.debug(f"Calling __next__")
        return Proxy(next(self))

    def __getattr__(self, name):
        self.logger_proxy.debug(f"Accessing attribute '{name}'")
        if name == "logdir":
            return self.__dict__.get("logdir", None)  # in order to pass down the dir
        # self.logger_proxy.info(f"Accessing attribute '{name}'")
        attr = getattr(self._obj, name)

        return Proxy(attr, logdir=self.logdir, log_level=self.log_level)

    def __setattr__(self, name, value):
        self.logger_proxy.debug(f"Setting attribute '{name}' to '{value}'")
        if name == "_obj":
            if type(value) is torch.Tensor:
                self.print_tensor(value)
            else:
                self.logger_proxy.debug(f"Setting attribute '_obj'")
            self.__dict__[name] = value  # Set the attribute directly
        else:
            # Intercept attribute assignment
            old_value = getattr(self._obj, name, None)
            if isinstance(old_value, torch.Tensor):
                old_value = self.print_tensor(old_value)
            new_value = value
            if isinstance(new_value, torch.Tensor):
                new_value = self.print_tensor(new_value)

            attr_name = f"{self._obj.__class__.__module__}.{self._obj.__class__.__name__}.{name}"
            self.print_update(old_value, value, attr_name)
            self.jsondumper.dump_json(
                self.process_id,
                self.thread_id,
                get_meta_vars(),
                attr_name,
                {"old_value": old_value, "new_value": new_value},
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
        self.logger_proxy.debug(f"Deleting attribute '{name}'")
        delattr(self._obj, name)

    def __getitem__(self, key):
        # Intercept item retrieval
        self.logger_proxy.debug(f"Getting item with key '{key}'")

        return list(self._obj)[key]

    def __setitem__(self, key, value):
        # Intercept item assignment
        self.logger_proxy.info(f"Setting item with key '{key}' to '{value}'")
        self._obj[key] = value

    def __delitem__(self, key):
        # Intercept item deletion
        self.logger_proxy.debug(f"Deleting item with key '{key}'")
        del self._obj[key]

    def __add__(self, other):
        # Unwrap other if it's a Proxy
        self.logger_proxy.debug(
            f"Calling __add__ for object '{self.__class__.__name__}'"
        )
        other = other._obj if isinstance(other, Proxy) else other
        return self._obj + other

    def __or__(self, other):
        self.logger_proxy.debug(
            f"Calling __or__ for object '{self.__class__.__name__}'"
        )
        if isinstance(other, bool):
            # If the other operand is a boolean, convert the Proxy object to a boolean and do the bitwise OR
            return bool(self._obj) | other
        else:
            # Otherwise, do the bitwise OR on the wrapped object
            return self._obj | other

    def __ior__(self, other):
        self.logger_proxy.debug(
            f"Calling __ior__ for object '{self.__class__.__name__}'"
        )
        if isinstance(other, bool):
            self._obj = bool(self._obj) | other
        else:
            self._obj |= other
        return self

    def __ror__(self, other):
        self.logger_proxy.debug(
            f"Calling __ror__ for object '{self.__class__.__name__}'"
        )
        if isinstance(other, bool):
            return other | bool(self._obj)
        else:
            return other | self._obj

    def __radd__(self, other):
        self.logger_proxy.debug(
            f"Calling __radd__ for object '{self.__class__.__name__}'"
        )
        # Unwrap other if it's a Proxy
        other = other._obj if isinstance(other, Proxy) else other
        return other + self._obj

    def __iadd__(self, other):
        self.logger_proxy.debug(
            f"Calling __iadd__ for object '{self.__class__.__name__}'"
        )
        # Unwrap other if it's a Proxy
        other = other._obj if isinstance(other, Proxy) else other
        self._obj += other
        return self

    def __sub__(self, other):
        self.logger_proxy.debug(
            f"Calling __sub__ for object '{self.__class__.__name__}'"
        )
        # Unwrap other if it's a Proxy
        other = other._obj if isinstance(other, Proxy) else other
        return self._obj - other

    def __mul__(self, other):
        self.logger_proxy.debug(
            f"Calling __mul__ for object '{self.__class__.__name__}'"
        )
        # Unwrap other if it's a Proxy
        other = other._obj if isinstance(other, Proxy) else other
        return self._obj * other

    def __rmul__(self, other):
        self.logger_proxy.debug(
            f"Calling __rmul__ for object '{self.__class__.__name__}'"
        )
        # Unwrap other if it's a Proxy
        other = other._obj if isinstance(other, Proxy) else other
        return other * self._obj

    def __truediv__(self, other):
        self.logger_proxy.debug(
            f"Calling __truediv__ for object '{self.__class__.__name__}'"
        )
        # Unwrap other if it's a Proxy
        other = other._obj if isinstance(other, Proxy) else other
        return self._obj / other

    def __floatdiv__(self, other):
        self.logger_proxy.debug(
            f"Calling __floatdiv__ for object '{self.__class__.__name__}'"
        )
        # Unwrap other if it's a Proxy
        other = other._obj if isinstance(other, Proxy) else other
        return self._obj // other

    def __rfloordiv__(self, other):
        self.logger_proxy.debug(
            f"Calling __ifloordiv__ for object '{self.__class__.__name__}'"
        )
        # Unwrap other if it's a Proxy
        other = other._obj if isinstance(other, Proxy) else other
        return other // self._obj

    def __float__(self):
        self.logger_proxy.debug(
            f"Calling __float__ for object '{self.__class__.__name__}'"
        )
        return float(self._obj)

    def __int__(self):
        self.logger_proxy.debug(
            f"Calling __int__ for object '{self.__class__.__name__}'"
        )
        return int(self._obj)

    def __str__(self):
        self.logger_proxy.debug(
            f"Calling __str__ for object '{self.__class__.__name__}'"
        )
        return str(self._obj)

    def __bool__(self):
        self.logger_proxy.debug(
            f"Calling __bool__ for object '{self.__class__.__name__}'"
        )
        return bool(self._obj)

    def __repr__(self):
        self.logger_proxy.debug(
            f"Calling __repr__ for object '{self.__class__.__name__}'"
        )
        return repr(self._obj)

    def __len__(self):
        self.logger_proxy.debug(
            f"Calling __len__ for object '{self.__class__.__name__}'"
        )
        return len(self._obj)

    def __getreal__(self):
        self.logger_proxy.debug(
            f"Calling __getreal__ for object '{self.__class__.__name__}'"
        )
        return self._obj

    def min(self):
        self.logger_proxy.debug(f"Calling min() for object '{self.__class__.__name__}'")
        return self._obj.min()

    def max(self):
        self.logger_proxy.debug(f"Calling max() for object '{self.__class__.__name__}'")
        return self._obj.max()

    def size(self):
        self.logger_proxy.debug(
            f"Calling size() for object '{self.__class__.__name__}'"
        )
        return self._obj.size()

    def print_proxy_dict(self):
        self.logger_proxy.info(f"Dump Proxy Dict: ")

        for k, value in Proxy.proxy_dict.items():
            if isinstance(value, torch.Tensor):
                self.print_tensor(value)
            else:
                self.logger_proxy.info(f"{k}: {value}")

        self.logger_proxy.info(f"Dump Frame Dict: ")
        for k, value in Proxy.frame_dict.items():
            if isinstance(value, torch.Tensor):
                self.print_tensor(value)
            else:
                self.logger_proxy.info(f"{k}: {value}")
