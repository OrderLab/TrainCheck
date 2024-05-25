from mldaikon.proxy_wrapper.config import debug_mode


def print_debug(message):
    if debug_mode:
        print(message)
