# Description: This script imports a module dynamically and extracts information about classes and methods in the module.

import json
import inspect
import importlib
import logging
import torch

logger = logging.getLogger("module_importer")
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler("module_importer.log")
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


visited_modules = set()

def extract_module_info(module_name):
    # Check if the module has already been visited
    if module_name in visited_modules:
        return []
    visited_modules.add(module_name)

    # Import the module dynamically
    logger.info(f"Importing module: {module_name}")
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError:
        logger.error(f"Module {module_name} not found.")
        return []

    # Prepare a dictionary to store class names and methods
    module_info = []

    # Iterate through all attributes of the module
    for name in dir(module):
        attr = getattr(module, name)
        if inspect.isclass(attr) and attr.__module__ == module_name:
            # Get the class name and methods
            class_info = {
                "class_name": name,
                "class_methods": dir(attr),
                "module": module_name
            }
            module_info.append(class_info)
        elif inspect.ismodule(attr) and attr.__name__.startswith(module.__name__):
            logger.info(f"Found sub-module: {attr.__name__}")
            sub_module_info = extract_module_info(attr.__name__)
            module_info.extend(sub_module_info)

    return module_info

def save_to_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":
    module_name = "torch"
    output_file = f"{module_name}_info.json"

    module_info = extract_module_info(module_name)
    save_to_json(module_info, output_file)

    print(f"Information about {module_name} saved to {output_file}.")
