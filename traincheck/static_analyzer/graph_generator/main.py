#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    pyan.py - Generate approximate call graphs for Python programs.

    This program takes one or more Python source files, does a superficial
    analysis, and constructs a directed graph of the objects in the combined
    source, and how they define or use each other.  The graph can be output
    for rendering by e.g. GraphViz or yEd.
"""

import importlib
import logging
import os
import re
from argparse import ArgumentParser
from glob import glob

import pandas as pd

from .. import config
from .analyzer import CallGraphVisitor
from .anutils import is_hidden


def unparse_module(module_name, logger, level=0):
    if level > config.UNPARSE_LEVEL:
        return None
    try:
        module = importlib.import_module(f"{'.'.join(module_name.split('.'))}")
        module_path = "/".join(module.__file__.split("/"))
        logger.info(f"Final Path at level {level}: {module_path}")
        return module_path
    except ModuleNotFoundError:
        module_path = unparse_module(
            ".".join(module_name.split(".")[:-1]), logger, level + 1
        )
        return module_path
    except Exception as e:
        logger.info(f"Error finding {module_name}: {e}")
        module_path = unparse_module(
            ".".join(module_name.split(".")[:-1]), logger, level + 1
        )
        return module_path


def traverse_torch_dir(libname: str):
    def get_torch_path(input_path):
        torch_home = os.getenv("TORCH_HOME")
        input_path = input_path.replace(".", "/")
        if input_path.startswith("torch/"):
            input_path = input_path[6:]
        return os.path.join(torch_home, input_path)

    # traverse the directory
    filenames = []
    dirname = get_torch_path(libname)
    for dir_root, _, files in os.walk(dirname):
        for file in files:
            if file.endswith(".py"):
                full_path = os.path.join(dir_root, file)
                filenames.append(full_path)

    return filenames


def filtering(known_args, v: CallGraphVisitor):
    if known_args.function or known_args.namespace:
        if known_args.function:
            function_name = known_args.function.split(".")[-1]
            namespace = ".".join(known_args.function.split(".")[:-1])
            node = v.get_node(namespace, function_name)
        else:
            node = None

        v.filter(node=node, namespace=known_args.namespace)


def call_graph_parser_to_df(log_file_path):
    def call_graph_parser(
        log_file_path, depth, observe_up_to_depth=False, observe_then_unproxy=False
    ):
        list_of_observers = []
        with open(log_file_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                # filter out the lines with the format <Node function:module_name.function_name> - function_depth
                if re.match(r"<Node function:.*> - \d+", line) or re.match(
                    r"<Node method:.*> - \d+", line
                ):
                    # print the module_name, function_name and function_depth
                    module_list = (
                        line.split(">")[0].split(" ")[1].split(":")[1].split(".")
                    )
                    if module_list[-1] == "*":
                        continue

                    # filter out the hidden functions
                    if not config.SHOW_HIDDEN and is_hidden(module_list):
                        continue

                    function_depth = line.split(" ")[-1].strip()
                    # save those with function_depth <= depth
                    if observe_up_to_depth:
                        if int(function_depth) <= depth:
                            list_of_observers.append(".".join(module_list))
                    else:
                        if int(function_depth) == depth:
                            list_of_observers.append(".".join(module_list))
        return list_of_observers

    df = pd.DataFrame()
    for depth in range(1, config.MAXIMUM_DEPTH):
        list_of_observers = call_graph_parser(log_file_path, depth=depth)
        depth_df = pd.DataFrame(list_of_observers, columns=[f"depth_{depth}"])
        # extend the length of the original dataframe
        df = pd.concat([df, depth_df], axis=1)

    csv_path = log_file_path.replace(".log", ".csv")
    print(f"---- CSV Path: {csv_path} ----")
    df.to_csv(csv_path, index=False)


def main(cli_args=None):
    usage = """%(prog)s [--lib|--log|--verbose|--output]"""
    desc = (
        "Analyse one or more Python source files and generate an"
        "approximate call graph of the modules, classes and functions"
        " within them."
    )

    parser = ArgumentParser(usage=usage, description=desc)

    parser.add_argument(
        "--lib",
        dest="libname",
        help="filter for LIBNAME",
        metavar="LIBNAME",
        default=config.INTERNAL_LIBS,
    )

    parser.add_argument(
        "--ext",
        dest="extlib",
        help="higher-level python scripts",
        metavar="EXTLIB",
        default=config.EXTERNAL_LIBS,
    )

    parser.add_argument(
        "-o",
        "--output",
        dest="output",
        help="write function level to OUTPUT",
        metavar="OUTPUT",
        default=None,
    )

    parser.add_argument(
        "--namespace",
        dest="namespace",
        help="filter for NAMESPACE",
        metavar="NAMESPACE",
        default=config.FILTER_NAMESPACE,
    )

    parser.add_argument(
        "--function",
        dest="function",
        help="filter for FUNCTION",
        metavar="FUNCTION",
        default=config.FILTER_FUNCTION,
    )

    parser.add_argument(
        "-l", "--log", dest="logname", help="write log to LOG", metavar="LOG"
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        dest="verbose",
        help="verbose output",
    )

    parser.add_argument(
        "-V",
        "--very-verbose",
        action="store_true",
        default=False,
        dest="very_verbose",
        help="even more verbose output (mainly for debug)",
    )

    parser.add_argument(
        "--root",
        default=None,
        dest="root",
        help="Package root directory. Is inferred by default.",
    )

    known_args, _ = parser.parse_known_args(cli_args)

    # determine root
    if known_args.root is not None:
        root = os.path.abspath(known_args.root)
    else:
        root = None

    if known_args.libname is None and known_args.extlib is None:
        parser.error("The --libname or --extlib argument is required.")

    # TODO: use an int argument for verbosity
    logger = logging.getLogger(__name__)

    if known_args.very_verbose:
        logger.setLevel(logging.DEBUG)
    elif known_args.verbose:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.WARN)

    logger.addHandler(logging.StreamHandler())

    if known_args.logname:
        handler = logging.FileHandler(known_args.logname)
        logger.addHandler(handler)

    if known_args.output is None:
        output_path = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.abspath(
            os.path.join(
                output_path, "..", "func_level", f"{known_args.libname}_func_level.log"
            )
        )
    else:
        output_path = known_args.output

    if known_args.extlib is None:
        # if there is no extern library, go through the torch lib only
        filenames = traverse_torch_dir(known_args.libname)
        visitor = CallGraphVisitor(filenames, logger=logger, root=root)
        filtering(known_args, visitor)
        visitor.assign_levels()
        visitor.dump_levels(output_path=output_path, show_hidden=config.SHOW_HIDDEN)
        return

    # if there is an extern library
    ext_filenames = glob(known_args.extlib)

    # get all attributes for the extern library
    attr_output_name = ext_filenames[0].split("/")[-1].split(".")[0] + "_attr.log"

    # produce a file
    ext_visitor = CallGraphVisitor(ext_filenames, logger=logger, root=root)
    filtering(known_args, ext_visitor)
    ext_visitor.assign_levels()
    ext_visitor.dump_attributes(attr_output_name)

    func_filter_set = set()
    with open(attr_output_name, "r") as f:
        lines = f.readlines()
        for line in lines:
            if re.match(r"<Node attribute:.*> - \d+", line):
                module_list = line.split(">")[0].split(" ")[1].split(":")[1].split(".")
                logger.info(f'Add module: {".".join(module_list)}')
                func_filter_set.add(".".join(module_list))

    # unparsing the attributes
    unparsed_filenames = set()
    for func in func_filter_set:
        logger.info(f"Unparsing function: {func}")
        file = unparse_module(func, logger, level=0)
        if file is not None:
            # collect the directory name if it is an __init__.py file
            unparsed_filenames.add(
                os.path.dirname(file) if file.endswith("__init__.py") else file
            )

    # TODO: add a blacklist for unparsing
    unparsing_blacklist_files = config.BLACKLIST_PATH
    for file in unparsing_blacklist_files:
        if file in unparsed_filenames:
            unparsed_filenames.remove(file)

    # filter out the contained situations
    filtered_unparsed_filenames = set()
    for path in unparsed_filenames:
        if not any(
            path != other_path and path.startswith(other_path)
            for other_path in unparsed_filenames
        ):
            filtered_unparsed_filenames.add(path)
    unparsed_filenames = filtered_unparsed_filenames

    logger.info(f"Unparsed Filenames: {unparsed_filenames}")

    # TODO: add a whitelist for unparsing
    unparse_whitelist = config.WHITELIST_MODULES

    def unparse_processor(unparsed_filename):
        """Return a key name for the unparsed file"""
        for whitelist in unparse_whitelist:
            if whitelist in unparsed_filename:
                if unparsed_filename.endswith(".py"):
                    return unparsed_filename.split("/")[-1].split(".")[0]
                else:
                    return unparsed_filename.split("/")[-1]
        return None

    # Warning: the key name might not be unique
    unparsed_filenames = {
        unparse_processor(unparsed_filename): unparsed_filename
        for unparsed_filename in unparsed_filenames
        if unparse_processor(unparsed_filename) is not None
    }
    logger.info(f"Unparsed Filenames: {unparsed_filenames}")

    def traverse_dir(dirname):
        filenames = []
        for dir_root, _, files in os.walk(dirname):
            for file in files:
                if file.endswith(".py"):
                    full_path = os.path.join(dir_root, file)
                    filenames.append(full_path)
        return filenames

    filenames = {}
    for key, path_name in unparsed_filenames.items():
        if path_name.endswith(".py"):
            filenames[key] = [path_name]
        else:
            file_list = traverse_dir(path_name)
            filenames[key] = file_list

    for key, files in filenames.items():
        logger.info(f"Key: {key}, File number: {len(files)}")

    # process the files
    for key, file_list in filenames.items():
        visitor = CallGraphVisitor(file_list, logger=logger, root=root)
        visitor.assign_levels()
        log_file_path = os.path.join(
            os.path.dirname(output_path), f"{key}_func_level.log"
        )
        visitor.dump_levels(output_path=log_file_path, show_hidden=config.SHOW_HIDDEN)
        # output an csv
        call_graph_parser_to_df(log_file_path)


if __name__ == "__main__":
    main()
