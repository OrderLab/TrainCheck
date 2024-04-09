import inspect
import logging
import torch

def print_tensor(value):
    logging.info(f"Setting attribute '_obj' to an updated tensor with shape'{value.shape}'")
    logging.info(f"Minimum value: {torch.min(value)}")
    logging.info(f"Maximum value: {torch.max(value)}")
    if value.type() == 'torch.FloatTensor':
        logging.info(f"Standard deviation: {torch.std(value)}")

class Proxy():
    proxy_dict = {}
    frame_dict = {}
 
    def __init__(self, obj, logdir, log_level):
        
        self.__dict__['logdir'] = logdir
        self.__dict__['log_level'] = log_level

        logging.basicConfig(filename=logdir, level=log_level)
        if not type(obj) in [int, float, str, bool]  and obj is not None: 
            logging.debug(f"Go to __init__ for object '{obj.__class__.__name__}'")
        
        if type(obj) is Proxy:
            logging.info(f"Object '{obj.__class__.__name__}' is already a proxy")
            self._obj = obj._obj

        else:
            frame = inspect.currentframe()
        
            frame_array = []
            while frame:

                frame_array.append((frame.f_code.co_filename, frame.f_lineno))
                frame = frame.f_back
                # print("Go up!")
            if Proxy.frame_dict.get(tuple(frame_array)) is None:
            
                logging.info(f"Creating proxy for object '{obj.__class__.__name__}'")
    
                self._obj = obj
                # Proxy.proxy_dict[id(self._obj)] = self
                Proxy.frame_dict[tuple(frame_array)] = self
            else:
                if not type(obj) in [int, float, str, bool] and obj is not None:
                    logging.info(f"Object '{obj.__class__.__name__}' is already proxied")
                # self._obj = Proxy.frame_dict[tuple(frame_array)]._obj ## attention, need to delete the original one before creating new instance
                del Proxy.frame_dict[tuple(frame_array)]
                self._obj = obj
                Proxy.frame_dict[tuple(frame_array)] = self

    @property
    def __class__(self):
        return self._obj.__class__
    def __array__(self):
        logging.debug(f"Go to __array__ for object '{self.__class__.__name__}'")
        return self._obj.__array__()
    
        
    def __torch_function__(self, func, types, args=(), kwargs=None):
        logging.info(f"Go to __torch_function__ for function '{func.__name__}'")
        if kwargs is None:
            kwargs = {}

        # Unwrap Proxy objects in args and kwargs
        args = tuple(arg._obj if (type(arg) is Proxy) else arg for arg in args)
        kwargs = {k: v._obj if (type(v) is Proxy) else v for k, v in kwargs.items()}
        result = func(*args, **kwargs)
        
        # Call the original function with the unwrapped args and kwargs
        return Proxy(result, logdir=self.logdir, log_level = self.log_level)
        
    def __call__(self, *args, **kwargs):
        logging.info(f"Go to __call__ for object '{self.__class__.__name__}'")
        args = tuple(arg._obj if (type(arg) is Proxy) else arg for arg in args)
        kwargs = {k: v._obj if (type(v) is Proxy) else v for k, v in kwargs.items()}
        # args=[arg if isinstance(arg, Proxy) else Proxy(arg) for arg in args]
        # kwargs={k: v if isinstance(v, Proxy) else Proxy(v) for k, v in kwargs.items()}
        result =  self._obj(*args, **kwargs)

        return Proxy(result, logdir=self.logdir, log_level = self.log_level)
    
    def __format__(self, format_spec):
        logging.debug(f"Go to __format__ for object '{self.__class__.__name__}'")
        # Delegate the formatting to the wrapped object
        return format(self._obj, format_spec)
    
        
    def __iter__(self):
        logging.debug(f"Calling __iter__")
        return iter(Proxy(obj, logdir=self.logdir, log_level = self.log_level) for obj in self._obj)
    
    def __next__(self):
        logging.debug(f"Calling __next__")
        return Proxy(next(self))
    
    def __getattr__(self, name):
        logging.debug(f"Accessing attribute '{name}'")
        if name == 'logdir':
            return self.__dict__.get('logdir', None) # in order to pass down the dir
        logging.info(f"Accessing attribute '{name}'")
        attr = getattr(self._obj, name)

        return Proxy(attr, logdir=self.logdir, log_level= self.log_level)

    def __setattr__(self, name, value):
        logging.debug(f"Setting attribute '{name}' to '{value}'")
        if name == '_obj':
            if type(value) is torch.Tensor:
                print_tensor(value)
            else:
                logging.debug(f"Setting attribute '_obj'")
            self.__dict__[name] = value  # Set the attribute directly
        else:
            # Intercept attribute assignment
            if type(value) is torch.Tensor:
                print_tensor(value)
            else:
                logging.info(f"Setting attribute '{name}' to '{value}'")
            
            if not type(value) in [int, float, str, bool] and value is not None:
                setattr(self._obj, name, Proxy(value, logdir=self.logdir, log_level = self.log_level))
            else:
                setattr(self._obj, name, value)
    
    def __delattr__(self, name):
        # Intercept attribute deletion
        logging.info(f"Deleting attribute '{name}'")
        delattr(self._obj, name)
        
    def __getitem__(self, key):
        # Intercept item retrieval
        logging.info(f"Getting item with key '{key}'")
        
        return list(self._obj)[key]

    def __setitem__(self, key, value):
        # Intercept item assignment
        logging.info(f"Setting item with key '{key}' to '{value}'")
        self._obj[key] = value
        
    def __delitem__(self, key):
        # Intercept item deletion
        logging.info(f"Deleting item with key '{key}'")
        del self._obj[key]
        
    def __add__(self, other):
        # Unwrap other if it's a Proxy
        other = other._obj if isinstance(other, Proxy) else other
        return self._obj + other
    
    def __or__(self, other):
        if isinstance(other, bool):
            # If the other operand is a boolean, convert the Proxy object to a boolean and do the bitwise OR
            return bool(self._obj) | other
        else:
            # Otherwise, do the bitwise OR on the wrapped object
            return self._obj | other
        
    def __ior__(self, other):
        if isinstance(other, bool):
            self._obj = bool(self._obj) | other
        else:
            self._obj |= other
        return self
    
    def __ror__(self, other):
        if isinstance(other, bool):
            return other | bool(self._obj)
        else:
            return other | self._obj
    
    def __radd__(self, other):
        # Unwrap other if it's a Proxy
        other = other._obj if isinstance(other, Proxy) else other
        return other + self._obj
    
    def __iadd__(self, other):
        # Unwrap other if it's a Proxy
        other = other._obj if isinstance(other, Proxy) else other
        self._obj += other
        return self
    
    def __sub__(self, other):
        # Unwrap other if it's a Proxy
        other = other._obj if isinstance(other, Proxy) else other
        return self._obj - other
    
    def __mul__(self, other):
        # Unwrap other if it's a Proxy
        other = other._obj if isinstance(other, Proxy) else other
        return self._obj * other
    
    def __truediv__(self, other):
        # Unwrap other if it's a Proxy
        other = other._obj if isinstance(other, Proxy) else other
        return self._obj / other
    
    def __floatdiv__(self, other):
        # Unwrap other if it's a Proxy
        other = other._obj if isinstance(other, Proxy) else other
        return self._obj // other
    
    def __float__(self):
        return float(self._obj)
    
    def __int__(self):
        return int(self._obj)
    
    def __str__(self):
        return str(self._obj)
    
    def __bool__(self):
        return bool(self._obj)
    
    def __repr__(self):
        return repr(self._obj)
    
    def __len__(self):
        return len(self._obj)


    def print_proxy_dict():
        logging.info(f"Dump Proxy Dict: ")

        for k, value in Proxy.proxy_dict.items():
            if isinstance(value, torch.Tensor):
                print_tensor(value)
            else:
                logging.info(f"{k}: {value}")

        logging.info(f"Dump Frame Dict: ")
        for k, value in Proxy.frame_dict.items():
            if isinstance(value, torch.Tensor):
                print_tensor(value)
            else:
                logging.info(f"{k}: {value}")

        