# traverse the internal library. E.g., if you want to observe "torch.nn", you
# should change into "torch.nn" or "nn".
INTERNAL_LIBS = None

# traverse the user script. Change to the target path.
EXTERNAL_LIBS = "example_pipelines/bug_84911_traincheck.py"

# ============================== USED IN INTERNAL ==============================

# Only show the functions in the namespace.
FILTER_NAMESPACE = None

# Only show the specific function.
FILTER_FUNCTION = None

# ============================== USED IN EXTERNAL ==============================

# Maximum times to unparse a module. E.g., to parse "torch.optim.adam.Adam", it
# takes 2 times to unparse to the "torch.optim".
UNPARSE_LEVEL = 3

# Maximum depth shown in the .csv files
MAXIMUM_DEPTH = 10

# Show the hidden functions/methods or not.
SHOW_HIDDEN = False


# Only used in external.
# top-level modules that will be considered as a internal library. If you want
# to observe "torch.optim.Adam", you should add "torch" to the list.
WHITELIST_MODULES = ["torch", "torchvision", "deepspeed"]

# TODO: it might not exactly match to the path
# Only used in external.
# paths that will not be considered as a internal library because of its size
# when using the external way
BLACKLIST_PATH = []
for module in WHITELIST_MODULES:
    # try to import the module
    try:
        __import__(module)
    except ImportError:
        print(f"Cannot import {module}")
        continue
    print(f"Appending {module} to the blacklist")
    BLACKLIST_PATH.append(__import__(module).__path__[0])
