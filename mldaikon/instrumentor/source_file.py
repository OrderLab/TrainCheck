import ast
import logging

from mldaikon.config.config import INSTR_MODULES_TO_INSTRUMENT

logger = logging.getLogger(__name__)

"""
Methods for reading and instrumenting source files.
"""


class InsertTracerVisitor(ast.NodeTransformer):
    def __init__(
        self,
        modules_to_instrument: list[str],
        scan_proxy_in_args: bool,
        allow_disable_dump: bool,
        funcs_of_inv_interest: list[str] | None,
    ):
        super().__init__()
        if not modules_to_instrument:
            logger.warning(
                "modules_to_instrument is empty, not instrumenting any module."
            )
            self.modules_to_instrument = []
        else:
            self.modules_to_instrument = modules_to_instrument
        self.scan_proxy_in_args = scan_proxy_in_args
        self.allow_disable_dump = allow_disable_dump
        self.funcs_of_inv_interest = funcs_of_inv_interest

    def get_instrument_node(self, module_name: str):
        return ast.parse(
            f"from mldaikon.instrumentor.tracer import Instrumentor; Instrumentor({module_name}, scan_proxy_in_args={self.scan_proxy_in_args}, allow_disable_dump={self.allow_disable_dump}, funcs_of_inv_interest={str(self.funcs_of_inv_interest)}).instrument()"
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
    source: str,
    modules_to_instrument: list[str],
    scan_proxy_in_args: bool,
    allow_disable_dump: bool,
    funcs_of_inv_interest: list[str] | None,
) -> str:
    """
    Instruments the given source code and returns the instrumented source code.

    **Note**: if a submodule is to be instrumented, the parent module will also be instrumented.

    """
    root = ast.parse(source)

    if not modules_to_instrument:
        logger.warning(
            f"modules_to_instrument not provided. Using default value CONSTANTS.INSTR_MODULES_TO_INSTRUMENT: {modules_to_instrument}."
        )
        modules_to_instrument = INSTR_MODULES_TO_INSTRUMENT

    visitor = InsertTracerVisitor(
        modules_to_instrument,
        scan_proxy_in_args,
        allow_disable_dump,
        funcs_of_inv_interest,
    )
    root = visitor.visit(root)
    source = ast.unparse(root)

    return source


def instrument_file(
    path: str,
    modules_to_instrument: list[str],
    disable_proxy_class: bool,
    scan_proxy_in_args: bool,
    allow_disable_dump: bool,
    funcs_of_inv_interest: list[str] | None,
    proxy_module: str,
    adjusted_proxy_config: list[dict[str, int | bool]],
) -> str:
    """
    Instruments the given file and returns the instrumented source code.
    """
    auto_observer_config: dict[str, int | bool] = adjusted_proxy_config[0]
    proxy_basic_config: dict[str, int | bool] = adjusted_proxy_config[1]
    tensor_dump_format: dict[str, int | bool] = adjusted_proxy_config[2]
    delta_dump_config: dict[str, int | bool] = adjusted_proxy_config[3]

    with open(path, "r") as file:
        source = file.read()

    # instrument APIs
    instrumented_source = instrument_source(
        source,
        modules_to_instrument,
        scan_proxy_in_args,
        allow_disable_dump,
        funcs_of_inv_interest,
    )

    # logging configs
    logging_start_code = """
import os
os.environ['MAIN_SCRIPT_NAME'] = os.path.basename(__file__).split(".")[0]

import atexit
from mldaikon.instrumentor.tracer import close_logging_threads
# handle exceptions and close the logging threads
atexit.register(close_logging_threads)
"""
    # we need to close the child logging threads by sending None to the queues on the main thread
    logging_end_code = """
# no exception, close the logging threads
close_logging_threads()
"""

    ## proxy configs
    proxy_start_code = ""
    auto_observer_code = ""

    if not disable_proxy_class:
        proxy_start_code = """
from mldaikon.proxy_wrapper.proxy import Proxy
"""
        if proxy_basic_config:
            proxy_start_code += f"""
import mldaikon.proxy_wrapper.proxy_config as proxy_config
proxy_config.__dict__.update({proxy_basic_config})
"""
        if tensor_dump_format:
            proxy_start_code += f"""
from mldaikon.proxy_wrapper.proxy_config import tensor_dump_format
tensor_dump_format.update({tensor_dump_format})
"""
        if delta_dump_config:
            proxy_start_code += f"""
from mldaikon.proxy_wrapper.proxy_config import delta_dump_config
delta_dump_config.update({delta_dump_config})
"""

        if auto_observer_config["enable_auto_observer"]:
            auto_observer_code = """
import glob
import importlib
from mldaikon.proxy_wrapper.proxy_config import auto_observer_config
spec = importlib.util.find_spec('mldaikon')
if spec and spec.origin:
    mldaikon_folder = os.path.dirname(spec.origin)
    print("mldaikon folder: ", mldaikon_folder)
else:
    raise Exception("mldaikon is not installed properly")
print("auto observer enabled with observing depth: ", auto_observer_config["enable_auto_observer_depth"])
enable_auto_observer_depth = auto_observer_config["enable_auto_observer_depth"]
neglect_hidden_func = auto_observer_config["neglect_hidden_func"]
neglect_hidden_module = auto_observer_config["neglect_hidden_module"]
observe_then_unproxy = auto_observer_config["observe_then_unproxy"]
observe_up_to_depth = auto_observer_config["observe_up_to_depth"]
if observe_up_to_depth:
    print("observe up to the depth of the function call")
else:
    print("observe only the function call at the depth")
from mldaikon.static_analyzer.graph_generator.call_graph_parser import call_graph_parser

log_files = glob.glob(
    os.path.join(mldaikon_folder, "static_analyzer", "func_level", "*.log")
)
print("log_files: ", log_files)
for log_file in log_files:
    call_graph_parser(
        log_file,
        depth=enable_auto_observer_depth,
        observe_up_to_depth=observe_up_to_depth,
        neglect_hidden_func=neglect_hidden_func,
        neglect_hidden_module=neglect_hidden_module,
        observe_then_unproxy=observe_then_unproxy,
    )
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
                        keywords=[
                            ast.keyword(
                                arg="is_root", value=ast.NameConstant(value=True)
                            )
                        ],
                    )
                    print(
                        f"Proxy inserted for module {proxy_module} at line {node.lineno}, content: {ast.unparse(node)}"
                    )
                    break

            instrumented_source = ast.unparse(root_for_proxy)

    # HACK: this is a hack to attach the logging code to the instrumented source after the __future__ imports
    instrumented_source = (
        instrumented_source.split("\n")[0]
        + logging_start_code
        + proxy_start_code
        + auto_observer_code
        + "\n".join(instrumented_source.split("\n")[1:])
        + logging_end_code
    )

    return instrumented_source
