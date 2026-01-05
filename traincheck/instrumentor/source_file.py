import ast
import io
import logging
import re
import tokenize
from collections import deque
from typing import Dict, Set

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
    source_code: str, model_name: str, mode: str | None
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
    elif mode == "subclass":
        instr_statement = ast.parse(
            f"proxy_parameter({model_name}, logdir=proxy_config.proxy_log_dir, parent_name='{model_name}')"
        )

    else:
        raise ValueError(
            f"Invalid mode: {mode}. Must be one of ['proxy', 'sampler', 'subclass']"
        )

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
    model_tracker_style: str | None,
):
    auto_observer_config: dict[str, int | bool | str] = adjusted_proxy_config[0]
    proxy_basic_config: dict[str, int | bool | str] = adjusted_proxy_config[1]
    tensor_dump_format: dict[str, int | bool | str] = adjusted_proxy_config[2]

    ## proxy configs
    proxy_start_code = ""
    auto_observer_code = ""

    if proxy_basic_config:
        if "proxy_log_dir" not in proxy_basic_config:
            from traincheck.instrumentor.proxy_wrapper.proxy_config import proxy_log_dir

            proxy_basic_config["proxy_log_dir"] = proxy_log_dir

        proxy_start_code += f"""
import traincheck.instrumentor.proxy_wrapper.proxy_config as proxy_config
proxy_config.__dict__.update({proxy_basic_config})
"""
    if tensor_dump_format:
        proxy_start_code += f"""
from traincheck.instrumentor.proxy_wrapper.proxy_config import tensor_dump_format
tensor_dump_format.update({tensor_dump_format})
"""

    if model_tracker_style == "proxy":
        proxy_start_code += """
from traincheck.instrumentor.proxy_wrapper.proxy import Proxy
"""
    else:
        proxy_start_code += """
from traincheck.instrumentor.proxy_wrapper.subclass import proxy_parameter
"""

    if auto_observer_config["enable_auto_observer"]:
        auto_observer_code = """
import glob
import importlib
from traincheck.instrumentor.proxy_wrapper.proxy_config import auto_observer_config
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
                instrumented_source, model, model_tracker_style
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


def annotate_stage(
    source: str,
) -> str:
    """DEBT: Refactor the source tree exploration part with a AST-based approach"""

    def _ctx(msg: str) -> str:
        return f"[annotate_stage] {msg}"

    def has_stage(src: str, name: str) -> bool:
        return re.search(rf'annotate_stage\(\s*[\'"]{name}[\'"]\s*\)', src) is not None

    orig_has = {
        "init": has_stage(source, "init"),
        "training": has_stage(source, "training"),
        "testing": has_stage(source, "testing"),
        "checkpointing": has_stage(source, "checkpointing"),
    }
    orig_has_any = any(orig_has.values()) or ("annotate_stage(" in source)

    for stage_name, present in orig_has.items():
        if present:
            logger.info(
                _ctx(
                    f"Stage '{stage_name}' already present in source; skip adding this stage."
                )
            )

    training_lines: Set[int] = set()
    testing_lines: Set[int] = set()
    checkpointing_lines: Set[int] = set()

    q: deque = deque(maxlen=3)
    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        q.append(tok)
        if len(q) < 2:
            continue
        a = q[-3] if len(q) >= 3 else None
        b = q[-2]
        c = q[-1]

        def at_attr(name: str) -> bool:
            return (
                a is not None
                and a.type == tokenize.OP
                and a.string == "."
                and b.type == tokenize.NAME
                and b.string == name
                and c.type == tokenize.OP
                and c.string == "("
            )

        if (at_attr("train") or at_attr("step")) and not orig_has["training"]:
            training_lines.add(b.start[0])

        if (at_attr("eval") or at_attr("no_grad")) and not orig_has["testing"]:
            testing_lines.add(b.start[0])

        if at_attr("save") and not orig_has["checkpointing"]:
            checkpointing_lines.add(b.start[0])

    TRAINING_PRIORITY = 3
    TESTING_PRIORITY = 2
    CHECKPOINTING_PRIORITY = 1
    priority = {
        "training": TRAINING_PRIORITY,
        "testing": TESTING_PRIORITY,
        "checkpointing": CHECKPOINTING_PRIORITY,
    }
    line_to_stage: Dict[int, str] = {}
    for ln in checkpointing_lines:
        line_to_stage[ln] = "checkpointing"
    for ln in training_lines:
        if priority["training"] > priority.get(line_to_stage.get(ln, ""), 0):
            line_to_stage[ln] = "training"
    for ln in testing_lines:
        if priority["testing"] > priority.get(line_to_stage.get(ln, ""), 0):
            line_to_stage[ln] = "testing"

    lines = source.splitlines(keepends=True)
    new_lines: list[str] = []
    inserted_count = {
        "training": 0,
        "testing": 0,
        "checkpointing": 0,
        "init": 0,
        "import": 0,
    }
    for i, line in enumerate(lines):
        lineno = i + 1
        stage = line_to_stage.get(lineno)
        if stage:
            k = len(new_lines) - 1
            while k >= 0 and new_lines[k].strip() == "":
                k -= 1
            prev = new_lines[k] if k >= 0 else ""
            if not (
                ("annotate_stage" in prev)
                and (f'"{stage}"' in prev or f"'{stage}'" in prev)
            ):
                if (m := re.match(r"\s*", line)) is None:
                    raise ValueError("pattern not found")
                indent = m.group(0)
                new_lines.append(f'{indent}annotate_stage("{stage}")\n')
                inserted_count[stage] += 1
                logger.info(
                    _ctx(
                        f"Inserted stage '{stage}' before line {lineno}: {line.strip()}"
                    )
                )
            else:
                logger.info(
                    _ctx(
                        f"Skip inserting '{stage}' at line {lineno} (previous non-empty line already has it)."
                    )
                )
        new_lines.append(line)

    new_src = "".join(new_lines)

    def _find_annotate_import_idx(lines):
        for idx, line in enumerate(lines):
            if re.match(r"^\s*from\s+traincheck\s+import\s+annotate_stage\s*$", line):
                return idx
        return -1

    lines_list = new_src.splitlines(keepends=True)
    annot_import_idx = _find_annotate_import_idx(lines_list)

    if annot_import_idx == -1:
        insert_idx = 0
        while insert_idx < len(lines_list):
            s = lines_list[insert_idx].strip()
            if (
                lines_list[insert_idx].startswith("#!")
                or (s.startswith("#") and "coding" in s)
                or s.startswith("from __future__ import")
            ):
                insert_idx += 1
            else:
                break
        lines_list.insert(insert_idx, "from traincheck import annotate_stage\n")
        annot_import_idx = insert_idx
        inserted_count["import"] += 1
        logger.info(
            _ctx(
                f"Inserted import 'from traincheck import annotate_stage' at line {annot_import_idx + 1}."
            )
        )

    new_src = "".join(lines_list)

    if not orig_has["init"]:
        has_guard = (
            re.search(
                r'^\s*if\s+__name__\s*==\s*[\'"]__main__[\'"]\s*:\s*$', new_src, re.M
            )
            is not None
        )
        main_def = re.search(
            r"^([ \t]*)def\s+main\s*\(.*?\)\s*:\s*(?:#.*)?$", new_src, re.M
        )

        if has_guard and main_def:
            def_line_start = main_def.start()
            before_def = new_src[:def_line_start]
            def_line_idx = before_def.count("\n")
            indent = main_def.group(1)
            step = "\t" if ("\t" in indent and " " not in indent) else "    "
            body_indent = indent + step

            nl = new_src.splitlines(keepends=True)
            insert_at = def_line_idx + 1
            while insert_at < len(nl) and nl[insert_at].strip() == "":
                insert_at += 1

            def _is_triple_quote(s: str) -> bool:
                t = s.lstrip()
                return t.startswith('"""') or t.startswith("'''")

            def is_single_line_triple_quoted_string(line: str, quote: str) -> bool:
                """Return True if the line is a single-line triple-quoted string using the given quote."""
                return line.count(quote) >= 2 and line.lstrip().startswith(quote)

            if insert_at < len(nl) and _is_triple_quote(nl[insert_at]):
                quote = '"""' if nl[insert_at].lstrip().startswith('"""') else "'''"
                if is_single_line_triple_quoted_string(nl[insert_at], quote):
                    insert_at += 1
                else:
                    insert_at += 1
                    while insert_at < len(nl):
                        if quote in nl[insert_at]:
                            insert_at += 1
                            break
                        insert_at += 1

            k = insert_at - 1
            while k >= 0 and nl[k].strip() == "":
                k -= 1
            prev = nl[k] if k >= 0 else ""
            if not (("annotate_stage" in prev) and ("init" in prev)):
                nl.insert(insert_at, f'{body_indent}annotate_stage("init")\n')
                inserted_count["init"] += 1
                logger.info(
                    _ctx(
                        f"Inserted stage 'init' at start of main() body (line {insert_at + 1})."
                    )
                )
            else:
                logger.info(
                    _ctx(
                        "Skip inserting 'init' inside main(): previous non-empty line already has it."
                    )
                )
            new_src = "".join(nl)
        else:
            lines2 = new_src.splitlines(keepends=True)
            annot_import_idx = _find_annotate_import_idx(lines2)
            if annot_import_idx == -1:
                i = 0
                while i < len(lines2):
                    s = lines2[i].strip()
                    if (
                        lines2[i].startswith("#!")
                        or (s.startswith("#") and "coding" in s)
                        or s.startswith("from __future__ import")
                    ):
                        i += 1
                    else:
                        break
                while i < len(lines2):
                    s = lines2[i].strip()
                    if (
                        s.startswith("import ")
                        or s.startswith("from ")
                        or s == ""
                        or s.startswith("#")
                    ):
                        i += 1
                    else:
                        break
                insert_at = i
            else:
                insert_at = annot_import_idx + 1

            k = insert_at
            while k < len(lines2) and lines2[k].strip() == "":
                k += 1
            next_line = lines2[k] if k < len(lines2) else ""
            if not (("annotate_stage" in next_line) and ("init" in next_line)):
                lines2.insert(insert_at, 'annotate_stage("init")\n')
                inserted_count["init"] += 1
                logger.info(
                    _ctx(
                        f"Inserted stage 'init' right after annotate_stage import at line {insert_at + 1}."
                    )
                )
            else:
                logger.info(
                    _ctx(
                        "Skip inserting 'init': next non-empty line after annotate_stage import is already init."
                    )
                )

            new_src = "".join(lines2)

    if "annotate_stage(" not in new_src and not orig_has_any:
        logger.error(
            _ctx(
                "Automatic insertion failed: no annotate_stage(...) found or added. Manual insertion required."
            )
        )
        raise RuntimeError(
            _ctx("annotate_stage insertion failed; see logs for details.")
        )

    return new_src


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
    use_torch_compile: bool,
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
    # annotate stages
    instrumented_source = annotate_stage(instrumented_source)
    # logging configs
    logging_start_code = f"""
import os
os.environ['TRAINCHECK_OUTPUT_DIR'] = "{output_dir}"
"""

    debug_hook_code = """
from traincheck.utils import register_custom_excepthook
if os.environ.get("TRAINCHECK_DEBUG") == "1":
    print("TRAINCHECK_DEBUG is set to 1, registering custom excepthook")
    register_custom_excepthook(True)
"""

    # general config update
    general_config_update = f"""
import traincheck.config.config as general_config
general_config.INSTR_DESCRIPTORS = {instr_descriptors}
general_config.MODEL_TRACKER_STYLE = {model_tracker_style!r}
"""
    if use_torch_compile:
        torch_compile_config_update = """
general_config.USE_TORCH_COMPILE = True
"""
        general_config_update = general_config_update + torch_compile_config_update
    # TODO: move the INSTR_DESCRIPTORS to the instr_opts file

    if models_to_track:
        assert model_tracker_style in [
            "proxy",
            "sampler",
            "subclass",
        ], f"Invalid model tracker style: {model_tracker_style}, must be one of ['proxy', 'sampler', 'subclass']"
        if model_tracker_style == "proxy" or model_tracker_style == "subclass":
            instrumented_source = instrument_model_tracker_proxy(
                instrumented_source,
                models_to_track,
                adjusted_proxy_config,
                no_auto_var_instr,
                model_tracker_style,
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
