from .utils import print_debug


def __delattr__(self, name):
    # Intercept attribute deletion
    print_debug(lambda: "logger_proxy: " + f"Deleting attribute '{name}'")
    delattr(self._obj, name)


def __setitem__(self, key, value):
    # Intercept item assignment
    print_debug(
        lambda: "logger_proxy: " + f"Setting item with key '{key}' to '{value}'"
    )
    self._obj[key] = value


def __delitem__(self, key):
    # Intercept item deletion
    print_debug(lambda: "logger_proxy: " + f"Deleting item with key '{key}'")
    del self._obj[key]


def __add__(self, other):
    # Unwrap other if it's a Proxy
    print_debug(
        lambda: "logger_proxy: "
        + f"Calling __add__ for object '{self.__class__.__name__}'"
    )
    other = other._obj if hasattr(other, "is_traincheck_proxied_obj") else other
    if isinstance(other, str):
        # If the other operand is a string, concatenate it with the string representation of the Proxy object
        return str(self._obj) + other
    return self._obj.__add__(other)


def __or__(self, other):
    print_debug(
        lambda: "logger_proxy: "
        + f"Calling __or__ for object '{self.__class__.__name__}'"
    )
    if isinstance(other, bool):
        # If the other operand is a boolean, convert the Proxy object to a boolean and do the bitwise OR
        return bool(self._obj) | other
    else:
        # Otherwise, do the bitwise OR on the wrapped object
        return self._obj | other


def __ior__(self, other):
    print_debug(
        lambda: "logger_proxy: "
        + f"Calling __ior__ for object '{self.__class__.__name__}'"
    )
    if isinstance(other, bool):
        self._obj = bool(self._obj) | other
    else:
        self._obj |= other
    return self


def __ror__(self, other):
    print_debug(
        lambda: "logger_proxy: "
        + f"Calling __ror__ for object '{self.__class__.__name__}'"
    )
    other = other._obj if hasattr(other, "is_traincheck_proxied_obj") else other
    if isinstance(other, bool):
        return other | bool(self._obj)
    return self._obj.__ror__(other)


def __radd__(self, other):
    print_debug(
        lambda: "logger_proxy: "
        + f"Calling __radd__ for object '{self.__class__.__name__}'"
    )
    # Unwrap other if it's a Proxy
    other = other._obj if hasattr(other, "is_traincheck_proxied_obj") else other
    if isinstance(other, str):
        # If the other operand is a string, concatenate it with the string representation of the Proxy object
        return other + str(self._obj)
    return self._obj.__radd__(other)


def __iadd__(self, other):
    print_debug(
        lambda: "logger_proxy: "
        + f"Calling __iadd__ for object '{self.__class__.__name__}'"
    )
    # Unwrap other if it's a Proxy
    other = other._obj if hasattr(other, "is_traincheck_proxied_obj") else other
    self._obj.__iadd__(other)
    return self


def __sub__(self, other):
    print_debug(
        lambda: "logger_proxy: "
        + f"Calling __sub__ for object '{self.__class__.__name__}'"
    )
    # Unwrap other if it's a Proxy
    other = other._obj if hasattr(other, "is_traincheck_proxied_obj") else other
    return self._obj - other


def __mul__(self, other):
    print_debug(
        lambda: "logger_proxy: "
        + f"Calling __mul__ for object '{self.__class__.__name__}'"
    )
    # Unwrap other if it's a Proxy
    other = other._obj if hasattr(other, "is_traincheck_proxied_obj") else other
    return self._obj * other


def __rmul__(self, other):
    print_debug(
        lambda: "logger_proxy: "
        + f"Calling __rmul__ for object '{self.__class__.__name__}'"
    )
    # Unwrap other if it's a Proxy
    other = other._obj if hasattr(other, "is_traincheck_proxied_obj") else other
    return other * self._obj


def __truediv__(self, other):
    print_debug(
        lambda: "logger_proxy: "
        + f"Calling __truediv__ for object '{self.__class__.__name__}'"
    )
    # Unwrap other if it's a Proxy
    other = other._obj if hasattr(other, "is_traincheck_proxied_obj") else other
    return self._obj / other


def __floatdiv__(self, other):
    print_debug(
        lambda: "logger_proxy: "
        + f"Calling __floatdiv__ for object '{self.__class__.__name__}'"
    )
    # Unwrap other if it's a Proxy
    other = other._obj if hasattr(other, "is_traincheck_proxied_obj") else other
    return self._obj // other


def __intdiv__(self, other):
    print_debug(
        lambda: "logger_proxy: "
        + f"Calling __intdiv__ for object '{self.__class__.__name__}'"
    )
    # Unwrap other if it's a Proxy
    other = other._obj if hasattr(other, "is_traincheck_proxied_obj") else other
    return self._obj // other


def __rfloordiv__(self, other):
    print_debug(
        lambda: "logger_proxy: "
        + f"Calling __ifloordiv__ for object '{self.__class__.__name__}'"
    )
    # Unwrap other if it's a Proxy
    other = other._obj if hasattr(other, "is_traincheck_proxied_obj") else other
    return other // self._obj


def __float__(self):
    print_debug(
        lambda: "logger_proxy: "
        + f"Calling __float__ for object '{self.__class__.__name__}'"
    )
    return float(self._obj)


def __int__(self):
    print_debug(
        lambda: "logger_proxy: "
        + f"Calling __int__ for object '{self.__class__.__name__}'"
    )
    return int(self._obj)


def __dir__(self):
    print_debug(
        lambda: "logger_proxy: "
        + f"Calling __dir__ for object '{self.__class__.__name__}'"
    )
    return dir(self._obj)


def __str__(self):
    print_debug(
        lambda: "logger_proxy: "
        + f"Calling __str__ for object '{self.__class__.__name__}'"
    )
    return str(self._obj)


def __bool__(self):
    print_debug(
        lambda: "logger_proxy: "
        + f"Calling __bool__ for object '{self.__class__.__name__}'"
    )
    return bool(self._obj)


def __repr__(self):
    print_debug(
        lambda: "logger_proxy: "
        + f"Calling __repr__ for object '{self.__class__.__name__}'"
    )
    return repr(self._obj)


def __len__(self):
    print_debug(
        lambda: "logger_proxy: "
        + f"Calling __len__ for object '{self.__class__.__name__}'"
    )
    return len(self._obj)


def __getreal__(self):
    print_debug(
        lambda: "logger_proxy: "
        + f"Calling __getreal__ for object '{self.__class__.__name__}'"
    )
    return self._obj


def min(self):
    print_debug(
        lambda: "logger_proxy: "
        + f"Calling min() for object '{self.__class__.__name__}'"
    )
    return self._obj.min()


def max(self):
    print_debug(
        lambda: "logger_proxy: "
        + f"Calling max() for object '{self.__class__.__name__}'"
    )
    return self._obj.max()


def size(self):
    print_debug(
        lambda: "logger_proxy: "
        + f"Calling size() for object '{self.__class__.__name__}'"
    )
    return self._obj.size()


def __array__(self):
    print_debug(
        lambda: "logger_proxy: "
        + f"Go to __array__ for object '{self.__class__.__name__}'"
    )
    return self._obj.__array__()


def __format__(self, format_spec):
    print_debug(
        lambda: "logger_proxy: "
        + f"Go to __format__ for object '{self.__class__.__name__}'"
    )
    # Delegate the formatting to the wrapped object
    return format(self._obj, format_spec)
