## Step 1: filer out the lines from func_level.log with the format <Node function:module_name.function_name> - function_depth, e.g. <Node function:torch.optim.lr_scheduler._format_param._copy> - 2

import importlib
import os
import re

from traincheck.proxy_wrapper.proxy_observer import add_observer_to_func


def unparse_module(module_name, level=0):
    if level > 3:
        return None
    try:
        module = importlib.import_module(f"{'.'.join(module_name.split('.')[:-1])}")
    except ModuleNotFoundError:
        module = unparse_module(".".join(module_name.split(".")[:-1]), level + 1)

    if module:
        last_name = module_name.split(".")[-1]
        try:
            func_obj = getattr(module, last_name)
            # print(f"object {last_name} found in module {'.'.join(module_name.split('.')[:-1])}")
            return func_obj
        except AttributeError:
            print(
                f"object {last_name} not found in module {'.'.join(module_name.split('.')[:-1])}"
            )
            # Ziming: from out observation this typically just mean a function call contains a local class or function call, so we can just pass
            pass
        return None
    return None


def add_observer(module_name, function_name, observe_then_unproxy=False):
    # import and get the function object use importlib
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError:
        # module could be a class here, load the class and get the function
        module = unparse_module(module_name)
        if module is None:
            print(f"error finding {function_name}: module {module_name} not found")

    try:
        # Retrieve the function or property
        cls = module  # Assume module is a class
        function = getattr(cls, function_name, None)

        # Check if it's a property before proceeding
        if isinstance(function, property):
            print(
                f"Skipping property function: {function_name} in module: {module_name}"
            )
            return

        # Apply observer to non-property functions
        print(f"Observe function: {function_name} found in module: {module}")
        setattr(
            module,
            function_name,
            add_observer_to_func(function, observe_then_unproxy),
        )

    except AttributeError:
        print(f"function {function_name} not found in module {module_name}")
        return
    # print(f'function: {function} found in module: {module}')


# read the func_level.log file
def add_observer_given_call_graph(
    log_file_path,
    depth=3,
    observe_up_to_depth=False,
    neglect_hidden_func=True,
    neglect_hidden_module=True,
    observe_then_unproxy=False,
):
    with open(log_file_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            # filter out the lines with the format <Node function:module_name.function_name> - function_depth
            if re.match(r"<Node function:.*> - \d+", line) or re.match(
                r"<Node method:.*> - \d+", line
            ):
                module_list = line.split(" ")[1].split(":")[1].split(".")[:-1]
                if neglect_hidden_module:
                    skip_flag = False
                    for module in module_list:
                        if module.startswith("_"):
                            skip_flag = True
                            break
                    if skip_flag:
                        continue
                module_name = ".".join(module_list)
                if module_name == "*":
                    continue
                function_name = (
                    line.split(" ")[1].split(":")[1].split(".")[-1].rstrip(">")
                )
                if function_name.startswith("_") and neglect_hidden_func:
                    continue
                function_depth = line.split(" ")[-1].strip()
                # save those with function_depth <= depth
                if observe_up_to_depth:
                    if int(function_depth) <= depth:
                        # print(f'module_name: {module_name}, function_name: {function_name}, function_depth: {function_depth}')
                        add_observer(module_name, function_name, observe_then_unproxy)
                else:
                    if int(function_depth) == depth:
                        # print(f'module_name: {module_name}, function_name: {function_name}, function_depth: {function_depth}')
                        add_observer(module_name, function_name, observe_then_unproxy)


if __name__ == "__main__":
    log_file_path = os.path.join(
        os.path.dirname(__file__), "func_level", "nn_func_level.log"
    )
    add_observer_given_call_graph(log_file_path)
