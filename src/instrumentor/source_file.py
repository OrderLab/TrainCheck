import ast
from collections import namedtuple
from CONSTANTS import MODULES_TO_INSTRUMENT
import logging

logger = logging.getLogger(__name__)

"""
Methods for reading and instrumenting source files.
"""


class InsertTracerVisitor(ast.NodeTransformer):
    def __init__(self, modules_to_instrument: list[str]):
        super().__init__()
        if not modules_to_instrument:
            logger.warning(
                "modules_to_instrument is empty, not instrumenting any module."
            )
            self.modules_to_instrument = []
        else:
            self.modules_to_instrument = modules_to_instrument

    def get_instrument_node(self, module_name):
        return ast.parse(
            f"from src.instrumentor import tracer; tracer.instrumentor({module_name}).instrument()"
        ).body

    def visit_Import(self, node):
        instrument_nodes = []
        for n in node.names:
            if not (
                n.name in self.modules_to_instrument
                or n.name.split(".")[0] in self.modules_to_instrument
            ):
                logger.debug(
                    f"Skipping module {n.name} as it is not in the list of modules to instrument: {self.modules_to_instrument}."
                )
                continue
            if n.asname:
                instrument_nodes.append(self.get_instrument_node(n.asname))
            else:
                instrument_nodes.append(self.get_instrument_node(n.name))
        # let's see if there are aliases, if yes, use them
        # if not, let's use the module name directly
        return [node] + instrument_nodes

    def visit_ImportFrom(self, node):
        instrument_nodes = []
        for n in node.names:
            if not (
                node.module in self.modules_to_instrument
                or node.module.split(".")[0] in self.modules_to_instrument
            ):
                logger.debug(
                    f"Skipping module {node.module} as it is not in the list of modules to instrument: {self.modules_to_instrument}."
                )
                continue

            if n.asname:
                instrument_nodes.append(self.get_instrument_node(n.asname))
            else:
                instrument_nodes.append(self.get_instrument_node(node.module))
        return [node] + instrument_nodes


def instrument_source(source: str, modules_to_instrument: list[str]) -> str:
    """
    Instruments the given source code and returns the instrumented source code.

    **Note**: if a submodule is to be instrumented, the parent module will also be instrumented.

    """
    root = ast.parse(source)

    if modules_to_instrument is None:
        logger.warning(
            f"modules_to_instrument not provided. Using default value CONSTANTS.MODULES_TO_INSTRUMENT: {modules_to_instrument}."
        )
        modules_to_instrument = MODULES_TO_INSTRUMENT

    visitor = InsertTracerVisitor(modules_to_instrument)
    root = visitor.visit(root)
    source = ast.unparse(root)

    return source


def instrument_file(path: str, modules_to_instrument: list[str]) -> str:
    """
    Instruments the given file and returns the instrumented source code.
    """

    with open(path, "r") as file:
        source = file.read()
    # instrument the source code
    instrumented_source = instrument_source(source, modules_to_instrument)
    return instrumented_source


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Source File Instrumentor for ML Pipelines in Python"
    )
    parser.add_argument(
        "-p",
        "--path",
        type=str,
        required=True,
        help="Path to the source file to be instrumented",
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Print debug logs",
    )
    parser.add_argument(
        "-t",
        "--modules_to_instrument",
        nargs="*",
        help="Modules to be instrumented",
        default=MODULES_TO_INSTRUMENT,
    )

    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # instrument the source file
    instrumented_source = instrument_file(args.path, args.modules_to_instrument)
    print(instrumented_source)
