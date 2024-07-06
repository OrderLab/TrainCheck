import ast
import logging

from mldaikon.config.config import INSTR_MODULES_TO_INSTRUMENT

logger = logging.getLogger(__name__)

"""
Methods for reading and instrumenting source files.
"""


class InsertTracerVisitor(ast.NodeTransformer):
    def __init__(self, modules_to_instrument: list[str], scan_proxy_in_args: bool):
        super().__init__()
        if not modules_to_instrument:
            logger.warning(
                "modules_to_instrument is empty, not instrumenting any module."
            )
            self.modules_to_instrument = []
        else:
            self.modules_to_instrument = modules_to_instrument
        self.scan_proxy_in_args = scan_proxy_in_args

    def get_instrument_node(self, module_name: str):
        return ast.parse(
            f"from mldaikon.instrumentor.tracer import Instrumentor; Instrumentor({module_name}).instrument(scan_proxy_in_args={self.scan_proxy_in_args})"
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
                instrument_nodes.append(self.get_instrument_node(n.name))
        return [node] + instrument_nodes


def instrument_source(
    source: str, modules_to_instrument: list[str], scan_proxy_in_args: bool
) -> str:
    """
    Instruments the given source code and returns the instrumented source code.

    **Note**: if a submodule is to be instrumented, the parent module will also be instrumented.

    """
    root = ast.parse(source)

    if modules_to_instrument is None:
        logger.warning(
            f"modules_to_instrument not provided. Using default value CONSTANTS.INSTR_MODULES_TO_INSTRUMENT: {modules_to_instrument}."
        )
        modules_to_instrument = INSTR_MODULES_TO_INSTRUMENT

    visitor = InsertTracerVisitor(modules_to_instrument, scan_proxy_in_args)
    root = visitor.visit(root)
    source = ast.unparse(root)

    return source


def instrument_file(
    path: str,
    modules_to_instrument: list[str],
    disable_proxy_class: bool,
    scan_proxy_in_args: bool,
    proxy_module: str,
) -> str:
    """
    Instruments the given file and returns the instrumented source code.
    """

    with open(path, "r") as file:
        source = file.read()

    # instrument APIs
    instrumented_source = instrument_source(
        source, modules_to_instrument, scan_proxy_in_args
    )

    logging_code = """
import os
os.environ['MAIN_SCRIPT_NAME'] = os.path.basename(__file__).split(".")[0]    
"""
    proxy_start_code = ""

    if not disable_proxy_class:
        proxy_start_code = """
from mldaikon.proxy_wrapper.proxy import Proxy
"""
        # find the main() function
        main_func = None
        root = ast.parse(instrumented_source)
        for node in ast.walk(root):
            if isinstance(node, ast.FunctionDef) and node.name == "main":
                main_func = node
                break

        # insert code before main() execution
        if main_func:
            code_to_insert = ast.parse(
                """
            """
            )
            main_func.body = code_to_insert.body + main_func.body

        instrumented_source = ast.unparse(root)

        # insert proxy for the module
        if proxy_module != "None":
            # find where the module is constructed and insert the proxy
            root_for_proxy = ast.parse(instrumented_source)
            for node in ast.walk(root_for_proxy):
                if (
                    isinstance(node, ast.Assign)
                    and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id == proxy_module
                ):
                    node.value = ast.Call(
                        func=ast.Name(id="Proxy", ctx=ast.Load()),
                        args=[node.value],
                        keywords=[],
                    )
                    print(
                        f"Proxy inserted for module {proxy_module} at line {node.lineno}, content: {ast.unparse(node)}"
                    )
                    break

            instrumented_source = ast.unparse(root_for_proxy)

    # HACK: this is a hack to attach the logging code to the instrumented source after the __future__ imports
    instrumented_source = (
        instrumented_source.split("\n")[0]
        + logging_code
        + proxy_start_code
        + "\n".join(instrumented_source.split("\n")[1:])
    )

    return instrumented_source
