from mldaikon.config.config import debug_mode
def print_debug(message):
    if debug_mode:
        print(message)