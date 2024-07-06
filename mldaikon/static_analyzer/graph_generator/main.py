#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    pyan.py - Generate approximate call graphs for Python programs.

    This program takes one or more Python source files, does a superficial
    analysis, and constructs a directed graph of the objects in the combined
    source, and how they define or use each other.  The graph can be output
    for rendering by e.g. GraphViz or yEd.
"""

from argparse import ArgumentParser
from glob import glob
import logging
import os

from analyzer import CallGraphVisitor


def main(cli_args=None):
    usage = """%(prog)s FILENAME... [--libname|--log|--verbose|--output]"""
    desc = (
        "Analyse one or more Python source files and generate an"
        "approximate call graph of the modules, classes and functions"
        " within them."
    )

    parser = ArgumentParser(usage=usage, description=desc)

    parser.add_argument("--libname", dest="libname", help="filter for LIBNAME", metavar="LIBNAME", default=None)

    parser.add_argument("-o", "--output", dest="output", help="write function level to OUTPUT", metavar="OUTPUT", default=None)

    parser.add_argument("--namespace", dest="namespace", help="filter for NAMESPACE", metavar="NAMESPACE", default=None)

    parser.add_argument("--function", dest="function", help="filter for FUNCTION", metavar="FUNCTION", default=None)

    parser.add_argument("-l", "--log", dest="logname", help="write log to LOG", metavar="LOG")

    parser.add_argument("-v", "--verbose", action="store_true", default=False, dest="verbose", help="verbose output")

    parser.add_argument(
        "-V",
        "--very-verbose",
        action="store_true",
        default=False,
        dest="very_verbose",
        help="even more verbose output (mainly for debug)",
    )

    parser.add_argument(
        "-G",
        "--grouped-alt",
        action="store_true",
        default=False,
        dest="grouped_alt",
        help="suggest grouping by adding invisible defines edges [only useful with --no-defines]",
    )

    parser.add_argument(
        "-g",
        "--grouped",
        action="store_true",
        default=False,
        dest="grouped",
        help="group nodes (create subgraphs) according to namespace [dot only]",
    )

    parser.add_argument(
        "-e",
        "--nested-groups",
        action="store_true",
        default=False,
        dest="nested_groups",
        help="create nested groups (subgraphs) for nested namespaces (implies -g) [dot only]",
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

    if known_args.libname is not None:
        parser.error("The --libname option is not added")

    # traverse the directory
    filenames = []

    def get_torch_path(input_path):
        torch_home = os.getenv('TORCH_HOME')
        input_path = input_path.replace('.', '/')
        if input_path.startswith('torch/'):
            input_path = input_path[6:]
        return os.path.join(torch_home, input_path)

    dirname = get_torch_path(known_args.libname)
    for dir_root, _, files in os.walk(dirname):
        for file in files:
            if file.endswith('.py'):
                full_path = os.path.join(dir_root, file)
                filenames.append(full_path)

    if known_args.nested_groups:
        known_args.grouped = True

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

    v = CallGraphVisitor(filenames, logger=logger, root=root, output_path=known_args.output)

    if known_args.function or known_args.namespace:

        if known_args.function:
            function_name = known_args.function.split(".")[-1]
            namespace = ".".join(known_args.function.split(".")[:-1])
            node = v.get_node(namespace, function_name)

        else:
            node = None

        v.filter(node=node, namespace=known_args.namespace)


if __name__ == "__main__":
    main()
