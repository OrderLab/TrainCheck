import ast
import logging
import re

from mldaikon.config.config import INSTR_MODULES_TO_INSTR

logger = logging.getLogger(__name__)

"""
Methods for reading and instrumenting source files.
"""


def get_code_head_and_tail(source: str):
    if source.startswith('"""'):
        code_head = ""
        code_tail = source
    else:
        code_head = source.split("\n")[0]
        code_tail = "\n".join(source.split("\n")[1:])
    return code_head, code_tail


class InsertTracerVisitor(ast.NodeTransformer):
    def __init__(
        self,
        modules_to_instr: list[str],
        scan_proxy_in_args: bool,
        use_full_instr: bool,
        funcs_to_instr: list[str] | None,
        API_dump_stack_trace: bool,
    ):
        super().__init__()
        if not modules_to_instr:
            logger.warning("modules_to_instr is empty, not instrumenting any module.")
            self.modules_to_instr = []
        else:
            self.modules_to_instr = modules_to_instr
        self.scan_proxy_in_args = scan_proxy_in_args
        self.use_full_instr = use_full_instr
        self.funcs_to_instr = funcs_to_instr
        self.API_dump_stack_trace = API_dump_stack_trace

    def get_instrument_node(self, module_name: str):
        return ast.parse(
            f"from mldaikon.instrumentor.tracer import Instrumentor; Instrumentor({module_name}, scan_proxy_in_args={self.scan_proxy_in_args}, use_full_instr={self.use_full_instr}, funcs_to_instr={str(self.funcs_to_instr)}, API_dump_stack_trace={self.API_dump_stack_trace}).instrument()"
        ).body

    def visit_Import(self, node):
        instrument_nodes = []
        for n in node.names:
            if not (
                n.name in self.modules_to_instr
                or n.name.split(".")[0] in self.modules_to_instr
            ):
                logger.debug(
                    f"Skipping module {n.name} as it is not in the list of modules to instrument: {self.modules_to_instr}."
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
                node.module in self.modules_to_instr
                or node.module.split(".")[0] in self.modules_to_instr
            ):
                logger.debug(
                    f"Skipping module {node.module} as it is not in the list of modules to instrument: {self.modules_to_instr}."
                )
                continue

            if n.asname:
                instrument_nodes.append(self.get_instrument_node(n.asname))
            else:
                instrument_nodes.append(self.get_instrument_node(n.name))
        return [node] + instrument_nodes


def instrument_library(
    source: str,
    modules_to_instr: list[str],
    scan_proxy_in_args: bool,
    use_full_instr: bool,
    funcs_to_instr: list[str] | None,
    API_dump_stack_trace: bool,
) -> str:
    """
    Instruments the given source code and returns the instrumented source code.

    **Note**: if a submodule is to be instrumented, the parent module will also be instrumented.

    """
    root = ast.parse(source)

    if not modules_to_instr:
        logger.warning(
            f"modules_to_instr not provided. Using default value CONSTANTS.INSTR_MODULES_TO_INSTR: {modules_to_instr}."
        )
        modules_to_instr = INSTR_MODULES_TO_INSTR

    visitor = InsertTracerVisitor(
        modules_to_instr,
        scan_proxy_in_args,
        use_full_instr,
        funcs_to_instr,
        API_dump_stack_trace,
    )
    root = visitor.visit(root)
    source = ast.unparse(root)

    return source


def instrument_model_tracker_proxy(
    source: str,
    models_to_track: list[str],
    adjusted_proxy_config: list[dict[str, int | bool | str]],
):
    auto_observer_config: dict[str, int | bool | str] = adjusted_proxy_config[0]
    proxy_basic_config: dict[str, int | bool | str] = adjusted_proxy_config[1]
    tensor_dump_format: dict[str, int | bool | str] = adjusted_proxy_config[2]
    delta_dump_config: dict[str, int | bool | str] = adjusted_proxy_config[3]

    ## proxy configs
    proxy_start_code = ""
    auto_observer_code = ""

    if proxy_basic_config:
        if "proxy_log_dir" not in proxy_basic_config:
            from mldaikon.proxy_wrapper.proxy_config import proxy_log_dir

            proxy_basic_config["proxy_log_dir"] = proxy_log_dir

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
    proxy_start_code += """
from mldaikon.proxy_wrapper.proxy import Proxy
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
from mldaikon.static_analyzer.graph_generator.call_graph_parser import add_observer_given_call_graph

log_files = glob.glob(
    os.path.join(mldaikon_folder, "static_analyzer", "func_level", "*.log")
)
print("log_files: ", log_files)
for log_file in log_files:
    add_observer_given_call_graph(
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
    root = ast.parse(source)
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

    for model in models_to_track:
        # find where the module is constructed and insert the proxy
        root_for_proxy = ast.parse(instrumented_source)
        for node in ast.walk(root_for_proxy):
            if (
                isinstance(node, ast.Assign)
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == model
            ):
                node.value = ast.Call(
                    func=ast.Name(id="Proxy", ctx=ast.Load()),
                    args=[node.value],
                    keywords=[
                        ast.keyword(arg="is_root", value=ast.Constant(value=True)),
                        ast.keyword(
                            arg="logdir",
                            value=ast.Constant(
                                value=proxy_basic_config["proxy_log_dir"]
                            ),
                        ),
                    ],
                )
                print(
                    f"Proxy inserted for module {model} at line {node.lineno}, content: {ast.unparse(node)}"
                )
                break
        instrumented_source = ast.unparse(root_for_proxy)
    code_head, code_tail = get_code_head_and_tail(instrumented_source)

    instrumented_source = code_head + proxy_start_code + auto_observer_code + code_tail

    return instrumented_source


def instrument_model_tracker_sampler(
    source: str,
    models_to_track: list[str],
):
    samplers = []
    for model in models_to_track:
        # find where the module is constructed and insert the proxy
        pattern = r"\s*" + f"{model}" + r"\s*=\s*"
        pattern_re = re.compile(pattern)

        for line_idx, line in enumerate(source.split("\n")):
            match = pattern_re.search(line)
            if match and match.start() == 0:
                break
        else:
            raise ValueError(
                f"Model {model} not found in the source code. Please check the model name and try again."
            )

        # insert the sampler after line_idx
        sampler_name = f"{model}_sampler"
        samplers.append(sampler_name)

        identation = len(line) - len(line.lstrip())
        sampler_code = line[:identation] + f"{sampler_name} = VarSampler({model})"
        source = "\n".join(
            source.split("\n")[: line_idx + 1]
            + [sampler_code]
            + source.split("\n")[line_idx + 1 :]
        )

    # iterate again, find all optimizers definitions
    for model, sampler_name in zip(models_to_track, samplers):
        # find the optimizer definition for the model
        keys = [model, "=", "optimizer"]
        for line_idx, line in enumerate(source.split("\n")):
            if all(key in line for key in keys):
                break
        else:
            raise ValueError(
                f"""Optimizer for model {model} not found in the source code.
            Please manually initialize a sampler for the model and call sampler.register_hook(optimizer) 
            for the corresponding optimizer."""
            )

        # NOTE: ideally we want to ensure that the place where we register the hook is after the optimizer is defined
        # but for now, we will just insert the hook after the optimizer definition due to our pattern to find the optimizer (see the keys variable above)
        optimizer_name = line.split("=")[0].strip()
        identation = len(line) - len(line.lstrip())
        hook_code = (
            line[:identation] + f"{sampler_name}.register_hook({optimizer_name})"
        )
        # find the identation level of the optimizer definition
        source = "\n".join(
            source.split("\n")[: line_idx + 1]
            + [hook_code]
            + source.split("\n")[line_idx + 1 :]
        )
    code_head, code_tail = get_code_head_and_tail(source)

    sampler_import_code = "from mldaikon.instrumentor import VarSampler"
    source = code_head + "\n" + sampler_import_code + "\n" + code_tail

    return source


def instrument_file(
    path: str,
    modules_to_instr: list[str],
    scan_proxy_in_args: bool,
    use_full_instr: bool,
    funcs_to_instr: list[str] | None,
    models_to_track: list[str] | None,
    model_tracker_style: str | None,
    adjusted_proxy_config: list[dict[str, int | bool | str]],
    API_dump_stack_trace: bool,
    output_dir: str,
    instr_descriptors: bool,
) -> str:
    """
    Instruments the given file and returns the instrumented source code.
    """

    with open(path, "r") as file:
        source = file.read()

    # instrument APIs
    instrumented_source = instrument_library(
        source,
        modules_to_instr,
        scan_proxy_in_args,
        use_full_instr,
        funcs_to_instr,
        API_dump_stack_trace,
    )

    # logging configs
    logging_start_code = f"""
import os
os.environ['ML_DAIKON_OUTPUT_DIR'] = "{output_dir}"
"""

    # general config update
    general_config_update = f"""
import mldaikon.config.config as general_config
general_config.INSTR_DESCRIPTORS = {instr_descriptors}
"""
    # TODO: move the INSTR_DESCRIPTORS to the instr_opts file

    if models_to_track:
        assert model_tracker_style in [
            "proxy",
            "sampler",
        ], f"Invalid model tracker style: {model_tracker_style}, must be one of ['proxy', 'sampler']"
        if model_tracker_style == "proxy":
            instrumented_source = instrument_model_tracker_proxy(
                instrumented_source,
                models_to_track,
                adjusted_proxy_config,
            )
        else:
            instrumented_source = instrument_model_tracker_sampler(
                instrumented_source,
                models_to_track,
            )

    # HACK: this is a hack to attach the logging code to the instrumented source after the __future__ imports
    code_head, code_tail = get_code_head_and_tail(instrumented_source)
    instrumented_source = (
        code_head + logging_start_code + general_config_update + code_tail
    )

    return instrumented_source
