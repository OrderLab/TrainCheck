def is_proxied(obj):
    if hasattr(obj, "is_proxied_obj"):
        return True
    return False


def unproxy_arg(arg):

    if is_proxied(arg):
        return unproxy_arg(arg._obj)
    elif type(arg) in [list]:
        return [unproxy_arg(element) for element in arg]
    elif type(arg) in [tuple]:
        return tuple(unproxy_arg(element) for element in arg)
    else:
        return arg
