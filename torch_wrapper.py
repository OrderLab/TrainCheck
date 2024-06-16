import csv
import inspect
import re
from collections import defaultdict, deque

import torch
import torch.nn.modules.conv
from tqdm import tqdm


# Helper function to check if a function is a C-level function
def is_c_level_function(func):
    return not hasattr(func, "__code__")


# Helper function to get the full name of a function
def get_full_func_name(func):
    return f"{func.__module__}.{func.__name__}"


# Helper function to get all functions in a module (including submodules)
def get_functions(module, visited=None):
    if visited is None:
        visited = set()
    elif module in visited:
        return []
    visited.add(module)
    funcs = []
    # if is torch.nn.modules module, then get all the submodules

    for _, obj in inspect.getmembers(module):
        try:

            if inspect.ismodule(obj):
                funcs.extend(get_functions(obj, visited))
            elif inspect.isclass(obj):
                funcs.extend(get_functions(obj, visited))
            elif obj.__module__.startswith("torch") and (callable(obj)):
                # if is_c_level_function(obj):
                #     import pdb; pdb.set_trace()
                funcs.append((get_full_func_name(obj), obj))
        except AttributeError:
            continue
    return funcs


import logging


# Build the call graph
def build_call_graph(funcs):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler("torch_call_graph.log")
    logger.addHandler(file_handler)
    call_graph = defaultdict(list)
    func_names = {func_name: func for func_name, func in funcs}
    for func_name, func in tqdm(funcs):
        # skip C-level functions
        if is_c_level_function(func):
            continue
        try:
            source = inspect.getsource(func)
            for line in source.splitlines():
                # Ignore comments and strings
                line = re.sub(r"#.*", "", line)
                line = re.sub(r'".*?"', "", line)
                line = re.sub(r"'.*?'", "", line)
                for other_func_name in func_names:
                    if (
                        re.search(
                            r"\b{}\b".format(other_func_name.split(".")[-1]), line
                        )
                        and func_names[other_func_name] != func
                    ):
                        # skip redundant calls
                        if func_name not in call_graph[other_func_name]:
                            call_graph[other_func_name].append(func_name)
                            # print(f"{other_func_name} is called by {func_name}")
                            logger.info(f"{other_func_name} is called by {func_name}")
        except Exception:
            continue
    return call_graph


# Compute the distance to the nearest C-level function using BFS
def compute_distances(call_graph, funcs):
    distances = {func_name: float("inf") for func_name, func in funcs}
    visited = set()
    queue = deque()

    import pdb

    pdb.set_trace()

    # Initialize the queue with C-level functions
    for func_name, func in funcs:
        if is_c_level_function(func):
            distances[func_name] = 0
            queue.append(func_name)
            visited.add(func_name)

    # Perform BFS
    while queue:
        current_func_name = queue.popleft()
        current_distance = distances[current_func_name]
        for neighbor in call_graph[current_func_name]:
            if neighbor not in visited and distances[neighbor] > current_distance + 1:
                distances[neighbor] = current_distance + 1
                queue.append(neighbor)
                visited.add(neighbor)

    return distances


# Get all functions in the torch library
torch_functions = get_functions(torch)

print(len(torch_functions))
# dump all the functions to a file
with open("torch_functions.txt", "w") as f:
    for func_name, func in torch_functions:
        f.write(f"{func_name}\n")

# Build the call graph
call_graph = build_call_graph(torch_functions)

print("call_graph is built")

# Compute distances
distances = compute_distances(call_graph, torch_functions)

# Save distances to a CSV file
with open("torch_function_distances.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Function", "Distance"])
    for func_name, distance in distances.items():
        writer.writerow([func_name, distance])

print("Distances have been saved to torch_function_distances.csv")


# Example function to get the specific function name if you have the function object
def get_specific_func_name(func):
    return get_full_func_name(func)


# Example usage:
some_function = torch.nn.functional.relu
specific_func_name = get_specific_func_name(some_function)
if specific_func_name in distances:
    print(
        f"Distance to C-level function for {specific_func_name}: {distances[specific_func_name]}"
    )
else:
    print(f"{specific_func_name} not found in the distances data.")
