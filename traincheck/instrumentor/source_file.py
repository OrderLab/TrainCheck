import ast
import logging
import re

from traincheck.config.config import INSTR_MODULES_TO_INSTR

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
            f"from traincheck.instrumentor.tracer import Instrumentor; Instrumentor({module_name}, scan_proxy_in_args={self.scan_proxy_in_args}, use_full_instr={self.use_full_instr}, funcs_to_instr={str(self.funcs_to_instr)}, API_dump_stack_trace={self.API_dump_stack_trace}).instrument()"
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


def instrument_model_once(source_code: str, model_name: str, mode: str) -> str:
    """
    Finds the first assignment to `model`, finds its closest parent `if` statement,
    and instruments all model assignments within other branches of that `if`.

    If no "if" statement is found, only the first assignment to `model` is instrumented.
    """
    root = ast.parse(source_code)
    parent_map = {}  # Maps child nodes to their parent nodes

    # Build parent relationships for all AST nodes
    for node in ast.walk(root):
        for child in ast.iter_child_nodes(node):
            parent_map[child] = node

    # Step 1: Find the first assignment to `model`
    first_model_assign = None
    for node in ast.walk(root):
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == model_name
            for target in node.targets
        ):
            first_model_assign = node
            break

    if not first_model_assign:
        raise ValueError(
            f"Model {model_name} not found in the source code. Please check the model name and try again."
        )

    # Step 2: Find the closest parent `if` statement
    closest_if = None
    current: ast.AST = first_model_assign
    while current in parent_map:
        current = parent_map[current]
        if isinstance(current, ast.If):
            closest_if = current
            break
    """The above code is sound with the assumption that the first assignment to `model` must be in the body of the if-statement.
        If a model is only assigned in the else branch, the definition of "the closest if statement" may not be correct.
    """

    # Step 3: Find `model` assignments in all branches of the `if`
    class ModelInstrumenter(ast.NodeTransformer):
        def __init__(self):
            self.model_name = model_name

        def visit_Assign(self, node):
            # Check if the assignment targets `model`
            if any(
                isinstance(target, ast.Name) and target.id == model_name
                for target in node.targets
            ):
                # Wrap the right-hand side in Proxy
                if mode == "proxy":
                    node.value = ast.Call(
                        func=ast.Name(id="Proxy", ctx=ast.Load()),
                        args=[node.value],
                        keywords=[
                            ast.keyword(arg="recurse", value=ast.Constant(value=True)),
                            ast.keyword(
                                arg="logdir",
                                value=ast.Attribute(
                                    value=ast.Name(id="proxy_config", ctx=ast.Load()),
                                    attr="proxy_log_dir",
                                    ctx=ast.Load(),
                                ),
                            ),
                            ast.keyword(
                                arg="var_name",
                                value=ast.Constant(value=self.model_name),
                            ),
                        ],
                    )
            return node

    if not closest_if:
        # Instrument the first assignment to `model`
        if mode == "proxy":
            ModelInstrumenter().visit(first_model_assign)
        elif mode == "sampler":
            # insert another new node after the model assignment
            var_sampler_node = ast.parse(
                f"{model_name}_sampler = VarSampler({model_name}, var_name='{model_name}')"
            ).body[0]
            root.body.insert(root.body.index(first_model_assign) + 1, var_sampler_node)
        else:
            raise ValueError(
                f"Invalid mode: {mode}. Must be one of ['proxy', 'sampler']"
            )
        ast.fix_missing_locations(root)
        return ast.unparse(root)

    else:
        all_branches = [closest_if.body, closest_if.orelse]

        while all_branches:  # Handle multiple elif cases
            branch = all_branches.pop(0)
            for stmt in branch:
                if isinstance(
                    stmt, ast.If
                ):  # If an `elif` is found, process it as a new "if"
                    all_branches.append(stmt.body)  # Add elif's body
                    all_branches.append(stmt.orelse)  # Add elif's else
                else:
                    for node in ast.walk(stmt):
                        if isinstance(node, ast.Assign) and any(
                            isinstance(target, ast.Name) and target.id == model_name
                            for target in node.targets
                        ):
                            if mode == "proxy":
                                ModelInstrumenter().visit(node)
                            elif mode == "sampler":
                                # insert another new node after the model assignment
                                var_sampler_node = ast.parse(
                                    f"{model_name}_sampler = VarSampler({model_name}, var_name='{model_name}')"
                                ).body[0]
                                stmt_idx = branch.index(stmt)
                                branch.insert(stmt_idx + 1, var_sampler_node)
                            else:
                                raise ValueError(
                                    f"Invalid mode: {mode}. Must be one of ['proxy', 'sampler']"
                                )
                            break

        ast.fix_missing_locations(root)
        return ast.unparse(root)


def get_child_parent_map(root) -> dict[ast.AST, ast.AST]:
    """
    Annotate each node with its parent node in the AST.
    This is useful for traversing the tree and modifying it later.
    """
    parent_map: dict[ast.AST, ast.AST] = {}

    for node in ast.walk(root):
        for child in ast.iter_child_nodes(node):
            if child in parent_map and not ast.unparse(child).strip() == "":
                print(
                    f"Node {ast.unparse(child)} already has a parent, {ast.unparse(parent_map[child])}"
                )
            parent_map[child] = node

    return parent_map


def instrument_all_model_assignments(
    source_code: str, model_name: str, mode: str
) -> str:
    """
    Finds all assignment statements to `model` and inserts a Proxy statement or a VarSampler statement
    after each assignment, depending on the mode.
    """
    print(
        f"Instrumenting model: {model_name}, mode: {mode}, scanning for assignments to {model_name}"
    )

    root = ast.parse(source_code)
    parent_map = get_child_parent_map(root)

    if mode == "proxy":
        instr_statement = ast.parse(
            f"{model_name} = Proxy({model_name}, recurse=True, logdir=proxy_config.proxy_log_dir, var_name='{model_name}')"
        )
    elif mode == "sampler":
        instr_statement = ast.parse(
            f"{model_name}_sampler = VarSampler({model_name}, var_name='{model_name}')"
        )
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be one of ['proxy', 'sampler']")

    # find all assignment statements to `model`
    assignments = []
    for node in ast.walk(root):
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == model_name
                for target in node.targets
            )
            or (
                isinstance(node, ast.Assign)
                and isinstance(node.targets[0], ast.Tuple)
                and any(
                    isinstance(target, ast.Name) and target.id == model_name
                    for target in node.targets[0].elts
                )
            )
        ):
            assignments.append(node)
            # insert the instrument statement right after the assignment
            instr_node = instr_statement.body[0]
            if node in parent_map:
                parent = parent_map[node]
                # print(f"Parent node: {ast.unparse(parent)}")
                print("\tInstrumenting: ", ast.unparse(node))
                if isinstance(parent, ast.For):
                    print(
                        "\t\t⬆️ Parent is a for loop, cowardly skipping instrumentation in fear of multiple models with the same 'var_name'"
                    )
                    continue
                if node in parent.body:  # type: ignore
                    idx = parent.body.index(node)  # type: ignore
                    parent.body.insert(idx + 1, instr_node)  # type: ignore
                elif isinstance(parent, ast.If) and node in parent.orelse:
                    # If the assignment is inside an else statement, insert after the assignment
                    idx = parent.orelse.index(node)
                    parent.orelse.insert(idx + 1, instr_node)
                else:
                    raise ValueError(
                        f"Node {ast.unparse(node)} not found in parent body."
                    )
            else:
                root.body.insert(root.body.index(node) + 1, instr_node)
    # Fix missing locations
    ast.fix_missing_locations(root)
    return ast.unparse(root)


def instrument_model_tracker_proxy(
    source: str,
    models_to_track: list[str],
    adjusted_proxy_config: list[dict[str, int | bool | str]],
    no_auto_var_instr: bool,
):
    auto_observer_config: dict[str, int | bool | str] = adjusted_proxy_config[0]
    proxy_basic_config: dict[str, int | bool | str] = adjusted_proxy_config[1]
    tensor_dump_format: dict[str, int | bool | str] = adjusted_proxy_config[2]

    ## proxy configs
    proxy_start_code = ""
    auto_observer_code = ""

    if proxy_basic_config:
        if "proxy_log_dir" not in proxy_basic_config:
            from traincheck.proxy_wrapper.proxy_config import proxy_log_dir

            proxy_basic_config["proxy_log_dir"] = proxy_log_dir

        proxy_start_code += f"""
import traincheck.proxy_wrapper.proxy_config as proxy_config
proxy_config.__dict__.update({proxy_basic_config})
"""
    if tensor_dump_format:
        proxy_start_code += f"""
from traincheck.proxy_wrapper.proxy_config import tensor_dump_format
tensor_dump_format.update({tensor_dump_format})
"""

    proxy_start_code += """
from traincheck.proxy_wrapper.proxy import Proxy
"""

    if auto_observer_config["enable_auto_observer"]:
        auto_observer_code = """
import glob
import importlib
from traincheck.proxy_wrapper.proxy_config import auto_observer_config
spec = importlib.util.find_spec('traincheck')
if spec and spec.origin:
    traincheck_folder = os.path.dirname(spec.origin)
    print("traincheck folder: ", traincheck_folder)
else:
    raise Exception("traincheck is not installed properly")
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
from traincheck.static_analyzer.graph_generator.call_graph_parser import add_observer_given_call_graph

log_files = glob.glob(
    os.path.join(traincheck_folder, "static_analyzer", "func_level", "*.log")
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

    if not no_auto_var_instr:
        for model in models_to_track:
            instrumented_source = instrument_all_model_assignments(
                instrumented_source, model, "proxy"
            )

    code_head, code_tail = get_code_head_and_tail(instrumented_source)
    instrumented_source = code_head + proxy_start_code + auto_observer_code + code_tail

    return instrumented_source


def instrument_model_tracker_sampler(
    source: str,
    models_to_track: list[str],
    no_auto_var_instr: bool,
):
    if not no_auto_var_instr:
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

            source = instrument_all_model_assignments(source, model, "sampler")

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
    sampler_import_code = "from traincheck.instrumentor import VarSampler"
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
    no_auto_var_instr: bool,
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

    debug_hook_code = """
from traincheck.utils import register_custom_excepthook
if os.environ.get("ML_DAIKON_DEBUG") == "1":
    print("ML_DAIKON_DEBUG is set to 1, registering custom excepthook")
    register_custom_excepthook(True)
"""

    # general config update
    general_config_update = f"""
import traincheck.config.config as general_config
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
                no_auto_var_instr,
            )
        else:
            instrumented_source = instrument_model_tracker_sampler(
                instrumented_source,
                models_to_track,
                no_auto_var_instr,
            )

    # HACK: this is a hack to attach the logging code to the instrumented source after the __future__ imports
    code_head, code_tail = get_code_head_and_tail(instrumented_source)
    instrumented_source = (
        code_head
        + logging_start_code
        + debug_hook_code
        + general_config_update
        + code_tail
    )

    return instrumented_source
