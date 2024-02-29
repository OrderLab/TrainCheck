import ast
from collections import namedtuple

"""
Methods for reading and instrumenting source files.
"""


class InsertTracerVisitor(ast.NodeTransformer):
    def get_instrument_node(self, module_name):
        return ast.parse(
            f"from src.instrumentor import tracer; tracer.instrumentor({module_name}).instrument()"
        ).body

    def visit_Import(self, node):
        instrument_nodes = []
        for n in node.names:
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
            if n.asname:
                instrument_nodes.append(self.get_instrument_node(n.asname))
            else:
                instrument_nodes.append(self.get_instrument_node(node.module))
        return [node] + instrument_nodes


def instrument_source(source: str) -> str:
    """
    Instruments the given source code and returns the instrumented source code.
    """
    # XXX: This is a dummy implementation. Replace this with the actual implementation.
    # TODO: please look into this https://github.com/harshitandro/Python-Instrumentation for possible implementation

    # find the lines where torch related modules are imported
    # TODO: add logging config code to the beginning of the file
    # TODO: add `from src.instrumentor import tracer` to the beginning of the file
    # TODO: for each import statement, add `tracer.instrument(module_name)` after the import statement
    # if the module_name belongs to the list of modules to be instrumented as defined in the configuration file (user)

    root = ast.parse(source)
    visitor = InsertTracerVisitor()
    root = visitor.visit(root)
    source = ast.unparse(root)

    return source


def instrument_file(path: str) -> str:
    """
    Instruments the given file and returns the instrumented source code.
    """

    with open(path, "r") as file:
        source = file.read()
    # instrument the source code
    instrumented_source = instrument_source(source)
    return instrumented_source


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Instrumentor for ML Pipelines in Python"
    )
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to the source file to be instrumented",
    )
    args = parser.parse_args()

    # instrument the source file
    instrumented_source = instrument_file(args.path)

    # write the instrumented source code to a new file
    with open(args.path, "w") as file:
        file.write(instrumented_source)
    print(f"Instrumented source code written to {args.path}")
    pass
