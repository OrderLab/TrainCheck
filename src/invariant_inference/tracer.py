import types

import torch
import inspect

def wrapper(original_function):
    def wrapped(*args, **kwargs):
        print(f"Before calling {original_function.__name__}")
        result = original_function(*args, **kwargs)
        print(f"After calling {original_function.__name__}")
        return result
    return wrapped

instrumented_modules = set()
skipped_modules = set()
skipped_functions = set()

def instrument(pymodule: types.ModuleType, depth=0):
    if pymodule in instrumented_modules or pymodule in skipped_modules:
        print(f"Depth:{depth}, Skipping module: ", pymodule.__name__)
        return 0
    
    print(f"Depth:{depth}, Instrumenting module: ", pymodule.__name__)
    instrumented_modules.add(pymodule)
    
    count_wrapped = 0
    for attr_name in dir(pymodule):
        if not hasattr(pymodule, attr_name):
            # handle __abstractmethods__ attribute
            print(f"Depth:{depth}, Skipping attribute as it does not exist: ", attr_name)
            continue

        attr = getattr(pymodule, attr_name)
        
        # skip private attributes
        if attr_name.startswith("_"):
            print(f"Depth:{depth}, Skipping private attribute: ", attr_name)
            if isinstance(attr, types.FunctionType):
                skipped_functions.add(attr)
            elif isinstance(attr, types.ModuleType):
                skipped_modules.add(attr)
            continue
        
        if isinstance(attr, types.FunctionType) or isinstance(attr, types.BuiltinFunctionType):
            if attr in skipped_functions:
                print(f"Depth: {depth}, Skipping function:", attr_name)
                continue

            print("Instrumenting function: ", attr_name)
            wrapped = wrapper(attr)
            try:
                setattr(pymodule, attr_name, wrapped)
            except Exception as e:
                # handling immutable types and attrs that have no setters
                print(f"Depth: {depth}, Skipping function {attr_name} due to error: {e}")
                continue
            count_wrapped += 1
        elif isinstance(attr, types.ModuleType):
            if attr in skipped_modules:
                print(f"Depth: {depth}, Skipping module: ", attr_name)
                continue
            if not attr.__name__.startswith('torch'):
                print(f"Depth: {depth}, Skipping module due to irrelevant name:", attr_name)
                skipped_modules.add(attr)
                continue

            print(f"Depth: {depth}, Recursing into module: ", attr_name)
            count_wrapped += instrument(attr, depth+1)

        elif inspect.isclass(attr):
            print(f"Depth: {depth}, Recursing into class: ", attr_name)
            count_wrapped += instrument(attr, depth+1)

    print(f"Depth: {depth}, Wrapped {count_wrapped} functions in module {pymodule.__name__}")
    return count_wrapped

instrument(torch)

print("calling torch.abs")
torch.abs(torch.tensor([-1., -2., 3.]))
print(torch.abs)

linearlayer = torch.nn.Linear(10, 10)
print("calling linearlayer")
linearlayer(torch.randn(10))
linearlayer.forward(torch.randn(10))
