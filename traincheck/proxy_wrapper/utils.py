from traincheck.proxy_wrapper.proxy_config import debug_mode


def print_debug(message_func):
    if debug_mode:
        print(message_func())
